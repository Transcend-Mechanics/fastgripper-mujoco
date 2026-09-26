"""Read-only RTDE v2 output stream for the URSim/MuJoCo integration.

Protocol: https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html
No input recipe, robot command, register write, or force feedback is sent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket
import struct
import threading
import time


FIELDS = ("timestamp", "target_q", "target_qd", "actual_q", "actual_qd",
          "runtime_state", "speed_scaling", "actual_TCP_pose", "output_int_register_24")
TYPES = ("DOUBLE", "VECTOR6D", "VECTOR6D", "VECTOR6D", "VECTOR6D",
         "UINT32", "DOUBLE", "VECTOR6D", "INT32")
DATA_STRUCT = struct.Struct("!B25dId6di")
MAX_PACKET_BYTES = 4096
PROTOCOL_VERSION, TEXT_MESSAGE, DATA_PACKAGE = 86, 77, 85
SETUP_OUTPUTS, START = 79, 83


class RTDEError(ConnectionError):
    """The peer did not provide the supported, finite RTDE output recipe."""


@dataclass(frozen=True)
class RTDEReference:
    timestamp: float
    received_at: float
    target_q: tuple[float, ...]
    target_qd: tuple[float, ...]
    actual_q: tuple[float, ...]
    actual_qd: tuple[float, ...]
    runtime_state: int
    speed_scaling: float
    actual_TCP_pose: tuple[float, ...] = (0., 0., 0., 0., 0., 0.)
    task_phase: int = 0

    def __post_init__(self):
        for name in ("target_q", "target_qd", "actual_q", "actual_qd", "actual_TCP_pose"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != 6 or not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain six finite values")
            object.__setattr__(self, name, values)
        if not all(math.isfinite(value) for value in (self.timestamp, self.received_at, self.speed_scaling)):
            raise ValueError("RTDE timestamps and speed scaling must be finite")
        if self.timestamp < 0 or self.received_at < 0 or not 0 <= self.speed_scaling <= 1:
            raise ValueError("RTDE timestamps/scaling are outside their valid ranges")
        if not isinstance(self.runtime_state, int) or not 0 <= self.runtime_state <= 0xffffffff:
            raise ValueError("runtime_state must be UINT32")
        if not isinstance(self.task_phase, int) or not -(2**31) <= self.task_phase < 2**31:
            raise ValueError("task_phase must be INT32")


def packet(command: int, payload: bytes = b"") -> bytes:
    if len(payload) + 3 > MAX_PACKET_BYTES:
        raise RTDEError("RTDE packet exceeds maximum length")
    return struct.pack("!HB", len(payload) + 3, command) + payload


def receive_packet(connection: socket.socket, timeout: float = .5) -> tuple[int, bytes]:
    """Read one packet, including fragmented headers, with an overall deadline."""
    deadline = time.monotonic() + timeout

    def exact(size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("RTDE packet deadline exceeded")
            connection.settimeout(remaining)
            part = connection.recv(size - len(chunks))
            if not part:
                raise EOFError("RTDE connection closed")
            chunks.extend(part)
        return bytes(chunks)

    size, command = struct.unpack("!HB", exact(3))
    if not 3 <= size <= MAX_PACKET_BYTES:
        raise RTDEError(f"Invalid RTDE packet length {size}")
    return command, exact(size - 3)


def decode_reference(payload: bytes, recipe_id: int, *, received_at: float | None = None) -> RTDEReference:
    if len(payload) != DATA_STRUCT.size:
        raise RTDEError(f"Invalid RTDE data length {len(payload)}; expected {DATA_STRUCT.size}")
    values = DATA_STRUCT.unpack(payload)
    if values[0] != recipe_id:
        raise RTDEError("RTDE data recipe changed")
    try:
        return RTDEReference(timestamp=values[1], received_at=time.monotonic() if received_at is None else received_at,
                             target_q=values[2:8], target_qd=values[8:14], actual_q=values[14:20],
                             actual_qd=values[20:26], runtime_state=values[26], speed_scaling=values[27],
                             actual_TCP_pose=values[28:34], task_phase=values[34])
    except ValueError as error:
        raise RTDEError(str(error)) from error


class RTDEClient:
    """Reconnect in a background thread and expose only the latest fresh sample.

    Duplicate controller timestamps do not refresh a sample's receive time.
    Disconnects invalidate the sample immediately. `latest` returning None must
    be treated as loss of arm reference by the simulation owner.
    """

    def __init__(self, host: str, port: int = 30004, *, frequency: float = 125.,
                 timeout: float = .5, reconnect_delay: float = .5):
        if not host or not 1 <= port <= 65535:
            raise ValueError("RTDE host and port are required")
        if not math.isfinite(frequency) or not 1 <= frequency <= 500:
            raise ValueError("RTDE frequency must be 1..500 Hz")
        if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(reconnect_delay) or reconnect_delay <= 0:
            raise ValueError("RTDE timeout and reconnect delay must be positive")
        self.host, self.port, self.frequency = host, port, frequency
        self.timeout, self.reconnect_delay = timeout, reconnect_delay
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._reference: RTDEReference | None = None
        self.connected = False
        self.last_error: str | None = None
        self.connections, self.samples, self.duplicates = 0, 0, 0

    def start(self) -> RTDEClient:
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ursim-rtde", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            connection = self._socket
            self._reference = None
            self.connected = False
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout + 1.)
            if self._thread.is_alive():
                raise RuntimeError("RTDE reader did not stop")

    def __enter__(self) -> RTDEClient:
        return self.start()

    def __exit__(self, *_args) -> None:
        self.stop()

    def latest(self, max_age: float = .5, *, now: float | None = None) -> RTDEReference | None:
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("Maximum RTDE sample age must be positive")
        with self._lock:
            now = time.monotonic() if now is None else now
            reference = self._reference
            if not self.connected or reference is None or not 0 <= now - reference.received_at <= max_age:
                return None
            return reference

    def status(self) -> dict:
        with self._lock:
            reference = self._reference
            return {"connected": self.connected, "last_error": self.last_error,
                    "connections": self.connections, "samples": self.samples, "duplicates": self.duplicates,
                    "sample_age_s": None if reference is None else max(0., time.monotonic() - reference.received_at),
                    "controller_timestamp": None if reference is None else reference.timestamp}

    def _response(self, connection: socket.socket, expected: int) -> bytes:
        deadline = time.monotonic() + self.timeout
        for _ in range(16):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("RTDE negotiation deadline exceeded")
            command, payload = receive_packet(connection, remaining)
            if command == expected:
                return payload
            if command != TEXT_MESSAGE:
                raise RTDEError(f"Unexpected RTDE packet {command}; expected {expected}")
        raise RTDEError("Too many RTDE text messages during negotiation")

    def _negotiate(self, connection: socket.socket) -> int:
        connection.sendall(packet(PROTOCOL_VERSION, struct.pack("!H", 2)))
        if self._response(connection, PROTOCOL_VERSION) != b"\x01":
            raise RTDEError("URSim rejected RTDE protocol v2")
        connection.sendall(packet(SETUP_OUTPUTS, struct.pack("!d", self.frequency) + ",".join(FIELDS).encode("ascii")))
        response = self._response(connection, SETUP_OUTPUTS)
        expected = ",".join(TYPES).encode("ascii")
        if len(response) < 2 or response[0] == 0 or response[1:] != expected:
            raise RTDEError("URSim rejected the required RTDE output fields/types")
        recipe_id = response[0]
        connection.sendall(packet(START))
        if self._response(connection, START) != b"\x01":
            raise RTDEError("URSim rejected RTDE output streaming")
        return recipe_id

    def _run(self) -> None:
        while not self._stop.is_set():
            connection = None
            try:
                connection = socket.create_connection((self.host, self.port), timeout=self.timeout)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    self._socket = connection
                if self._stop.is_set():
                    break
                recipe_id = self._negotiate(connection)
                with self._lock:
                    self.connected = True
                    self.last_error = None
                    self.connections += 1
                previous_timestamp = None
                while not self._stop.is_set():
                    command, payload = receive_packet(connection, self.timeout)
                    if command == TEXT_MESSAGE:
                        continue
                    if command != DATA_PACKAGE:
                        raise RTDEError(f"Unexpected streaming RTDE packet {command}")
                    reference = decode_reference(payload, recipe_id)
                    if previous_timestamp is not None and reference.timestamp < previous_timestamp:
                        raise RTDEError("RTDE controller timestamp moved backwards")
                    with self._lock:
                        if reference.timestamp == previous_timestamp:
                            self.duplicates += 1
                        else:
                            self._reference = reference
                            self.samples += 1
                    previous_timestamp = reference.timestamp
            except (OSError, EOFError, RTDEError) as error:
                with self._lock:
                    if not self._stop.is_set():
                        self.last_error = str(error)
            finally:
                with self._lock:
                    self.connected = False
                    self._reference = None
                    self._socket = None
                if connection is not None:
                    connection.close()
            self._stop.wait(self.reconnect_delay)
