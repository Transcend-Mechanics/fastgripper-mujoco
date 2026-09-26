"""A physics-backed FastGripper device speaking raw Modbus RTU over TCP.

This is the firmware 1.0 command/status interface, not Modbus TCP (no MBAP
header). Activation accepts the model's nominal endpoint calibration; it does
not simulate the real motor's homing or electronics. Temperature is unavailable
and reads zero. Calibration, persistent configuration, and autonomous re-grasp
are deliberately unsupported and return Modbus exceptions.

The owner advances physics: under the server's shared RLock, call device.update
and controller.update before mj_step. Device time, uptime and the 2 s communication
watchdog use simulation seconds. A paused/stopped runner must explicitly stop the
device; there is no independent physics thread in this module.
"""

from __future__ import annotations

import math
import socket
import socketserver
import struct
import threading
from dataclasses import replace

from .controller import GripperController, ObjectState, Status

ACT, GTO, ATR, CLR, STOP = 1, 2, 4, 8, 16
FIRMWARE_VERSION = 0x0100


def crc16(data: bytes) -> int:
    """Modbus CRC-16, reflected polynomial 0xA001, initial value 0xFFFF."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def with_crc(data: bytes) -> bytes:
    return bytes(data) + struct.pack("<H", crc16(data))


def _word(value: float, *, signed: bool = False) -> int:
    # Match firmware lroundf (ties away from zero), then saturate diagnostics.
    integer = math.floor(value + .5) if value >= 0 else math.ceil(value - .5)
    return max(-32768 if signed else 0, min(32767 if signed else 65535, integer)) & 0xFFFF


class ModbusException(ValueError):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class BridgeDevice:
    """Command/lifecycle adapter around one existing GripperController.

    The caller owns thread synchronization. Direct register methods and TCP
    requests must use the same lock as simulation stepping. Constructing the
    adapter disables the actuator without resetting any physical state.
    """

    def __init__(self, controller: GripperController, *, slave_id: int = 9,
                 watchdog_seconds: float = 2.0):
        if not isinstance(slave_id, int) or not 1 <= slave_id <= 247:
            raise ValueError("slave_id must be in 1..247")
        if not math.isfinite(watchdog_seconds) or watchdog_seconds <= 0:
            raise ValueError("watchdog_seconds must be positive and finite")
        self.controller = controller
        self.slave_id = slave_id
        self.watchdog_seconds = float(watchdog_seconds)
        self.elapsed = 0.0
        self.last_valid_frame = 0.0
        self.activating = False
        self.releasing = False
        self.fault_code = 0
        self.watchdog_stops = 0
        self.last_stop_reason = "boot"
        self.commands = {1000: 0, 1001: 0, 1002: 100, 1003: 50}
        self.controller.reset()

    @property
    def status(self) -> int:
        if self.fault_code:
            return 4
        return 1 if self.activating else int(self.controller.status)

    def stop(self, reason: str = "stop") -> None:
        """Halt commanded motion, preserving enable state and any latched fault."""
        if self.activating:
            self.fault(5)
        else:
            self.controller.stop()
        self.activating = self.releasing = False
        self.commands[1000] = ACT if self.controller.enabled else 0
        self.last_stop_reason = str(reason)

    def fault(self, code: int = 3) -> None:
        """Latch an externally detected simulation fault; never clear implicitly."""
        if not isinstance(code, int) or not 1 <= code <= 7:
            raise ValueError("fault code must be in 1..7")
        self.fault_code = code
        self.activating = self.releasing = False
        self.controller.set_enabled(False)
        self.controller.stop()
        self.commands[1000] = 0
        self.last_stop_reason = "fault"

    def update(self, dt: float) -> None:
        """Advance lifecycle/watchdog only; do not advance controller or physics."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        self.elapsed += dt
        if self.activating:
            self.activating = False
            cfg = self.controller.config
            position = self.controller.motor_position_rad
            if not math.isfinite(position):
                self.fault(3)
            elif not min(cfg.open_motor_rad, cfg.closed_motor_rad) - cfg.settle_rad <= position <= max(cfg.open_motor_rad, cfg.closed_motor_rad) + cfg.settle_rad:
                self.fault(6)
            else:
                self.controller.set_enabled(True)
        if self.status == int(Status.MOVING) and self.elapsed - self.last_valid_frame > self.watchdog_seconds:
            self.watchdog_stops += 1
            self.stop("communication_watchdog")
        if self.releasing and not self.controller.seeking:
            self.releasing = False
            self.controller.set_enabled(False)
            self.commands[1000] = 0

    def _position_pct(self, position: float) -> int:
        cfg = self.controller.config
        pct = 100 * (position - cfg.open_motor_rad) / (cfg.closed_motor_rad - cfg.open_motor_rad)
        return _word(max(0., min(100., pct)))

    def _registers(self) -> dict[int, int]:
        c = self.controller
        cfg = c.config
        return self.commands | {
            2000: self.status,
            2001: int(c.object_state),
            2002: self.fault_code,
            2003: self._position_pct(c.motor_position_rad) if math.isfinite(c.motor_position_rad) else 0xFFFF,
            2004: self.commands[1001],
            2005: _word(c.motor_torque_nm * 100, signed=True),
            2006: 1 if c.enabled else 0,
            2007: 0,  # No simulated temperature sensor.
            2008: _word(abs(c.grip_torque_nm) * 100),
            2009: 0xFFFF if c.grip_motor_rad is None else self._position_pct(c.grip_motor_rad),
            2010: _word(c.peak_torque_nm * 100),
            2100: _word(c.motor_position_rad * 100, signed=True) if math.isfinite(c.motor_position_rad) else 0,
            2101: _word(c.motor_velocity_rad_s * 100, signed=True) if math.isfinite(c.motor_velocity_rad_s) else 0,
            2102: FIRMWARE_VERSION,
            2103: int(self.elapsed) & 0xFFFF,
            3000: self.slave_id, 3001: 2,
            3002: _word(cfg.max_speed_rad_s * 100),
            3003: _word(cfg.seek_gain * 100),
            3004: _word(cfg.torque_cap_nm * 100),
            3005: _word(cfg.open_motor_rad * 100, signed=True),
            3006: _word(cfg.closed_motor_rad * 100, signed=True),
            3007: 0, 3008: 1, 3009: 0,
        }

    @staticmethod
    def _validate_range(address: int, count: int) -> None:
        if not isinstance(address, int) or not isinstance(count, int) or not 0 <= address <= 65535 or not 1 <= count <= 60 or address + count > 65536:
            raise ModbusException(3, "invalid register range")

    def read_registers(self, address: int, count: int) -> list[int]:
        self._validate_range(address, count)
        registers = self._registers()
        try:
            values = [registers[a] for a in range(address, address + count)]
        except KeyError as exc:
            raise ModbusException(2, "unmapped register") from exc
        self.last_valid_frame = self.elapsed
        return values

    def write_registers(self, address: int, values: list[int], *, multiple: bool = True) -> None:
        """Validate the entire transaction, then apply one atomic motion intent."""
        values = list(values)
        self._validate_range(address, len(values))
        changes = dict(enumerate(values, address))
        for register, value in changes.items():
            if not isinstance(value, int) or not 0 <= value <= 65535:
                raise ModbusException(3, "register value must be a word")
            if register not in (*range(1000, 1004), 3002, 3003, 3004):
                raise ModbusException(2, "read-only, calibration, persistence, or unsupported feature")
            if register == 1000 and value & ~31:
                raise ModbusException(3, "unknown action bits")
            if register in (3002, 3003) and not 10 <= value <= 2000:
                raise ModbusException(3, "seek speed/gain must be in 0.1..20")
            if register == 3004 and not 10 <= value <= 500:
                raise ModbusException(3, "torque cap must be in 0.1..5 N m")
        # Calibration and protocol identity remain fixed. Operational tuning is
        # ephemeral and reverts when a new device/controller is constructed.
        config_names = {3002: "max_speed_rad_s", 3003: "seek_gain", 3004: "torque_cap_nm"}
        overrides = {config_names[a]: value / 100 for a, value in changes.items() if a in config_names}
        if overrides:
            self.controller.config = replace(self.controller.config, **overrides)
            self.controller._configure_actuator()
        for register in (1001, 1002, 1003):
            if register in changes:
                self.commands[register] = min(100, changes[register])
        if 1000 in changes:
            self.commands[1000] = changes[1000]
            self._action()
        elif not multiple and 1001 in changes and (self.status == 3 or self.commands[1000] & GTO):
            self._action()
        self.last_valid_frame = self.elapsed

    def _action(self) -> None:
        action = self.commands[1000]
        if action & CLR:
            self.fault_code = 0
            self.commands[1000] &= ~CLR
        if action & STOP:
            self.stop()
            return
        if not action & ACT:
            self.activating = self.releasing = False
            self.controller.set_enabled(False)
            self.controller.stop()
            return
        if self.fault_code:
            return
        if not self.controller.enabled:
            self.activating = True
            return  # GTO during first activation is not queued for later motion.
        if self.activating:
            return
        if action & ATR:
            self.releasing = True
            self.commands[1000] = 0
            self.controller.command(0, 50, 25)
        elif action & GTO:
            self.releasing = False
            self.controller.command(self.commands[1001], self.commands[1002], self.commands[1003])

    def handle_frame(self, frame: bytes) -> bytes | None:
        """Handle one complete RTU frame; invalid CRC/other slaves are ignored."""
        if len(frame) < 4 or frame[0] not in (0, self.slave_id) or crc16(frame[:-2]) != int.from_bytes(frame[-2:], "little"):
            return None
        broadcast, function = frame[0] == 0, frame[1]
        self.last_valid_frame = self.elapsed
        try:
            if function in (3, 4, 6):
                if len(frame) != 8:
                    raise ModbusException(3, "bad fixed request length")
                address, value = struct.unpack(">HH", frame[2:6])
                if function in (3, 4):
                    if broadcast:
                        return None
                    values = self.read_registers(address, value)
                    payload = bytes((self.slave_id, function, len(values) * 2)) + struct.pack(">" + "H" * len(values), *values)
                else:
                    self.write_registers(address, [value], multiple=False)
                    payload = frame[:6]
            elif function == 16:
                if len(frame) < 9:
                    raise ModbusException(3, "short multiple-write request")
                address, count, byte_count = struct.unpack(">HHB", frame[2:7])
                if not 1 <= count <= 60 or byte_count != count * 2 or len(frame) != byte_count + 9:
                    raise ModbusException(3, "invalid multiple-write request")
                values = list(struct.unpack(">" + "H" * count, frame[7:-2]))
                self.write_registers(address, values)
                payload = frame[:6]
            else:
                raise ModbusException(1, "unsupported function")
        except ModbusException as exc:
            payload = bytes((self.slave_id, function | 0x80, exc.code))
        return None if broadcast else with_crc(payload)

    def snapshot(self) -> dict:
        return {"status": self.status, "fault": self.fault_code,
                "protocol_version": "1.0", "calibration": "nominal_model_endpoints_no_homing",
                "temperature_available": False, "regrasp_supported": False,
                "watchdog_clock": "simulation_seconds", "watchdog_stops": self.watchdog_stops,
                "last_stop_reason": self.last_stop_reason, "elapsed_s": self.elapsed,
                "controller": self.controller.snapshot()}


class _TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        owner = self.server.owner
        self.request.settimeout(.2)
        with owner._clients_lock:
            owner._clients.add(self.request)
        buffer = bytearray()
        try:
            while not owner._closing.is_set():
                try:
                    incoming = self.request.recv(1024)
                except socket.timeout:
                    continue
                if not incoming:
                    return
                buffer.extend(incoming)
                if len(buffer) > 4096:
                    return
                # Requests have self-describing length. TCP packet boundaries
                # are irrelevant; both split and coalesced frames are accepted.
                while len(buffer) >= 2:
                    if buffer[1] == 16:
                        if len(buffer) < 7:
                            break
                        size = 9 + buffer[6]
                    else:
                        size = 8
                    if len(buffer) < size:
                        break
                    frame = bytes(buffer[:size])
                    del buffer[:size]
                    with owner.lock:
                        reply = owner.device.handle_frame(frame)
                    if reply:
                        self.request.sendall(reply)
        except (ConnectionError, OSError):
            pass
        finally:
            with owner._clients_lock:
                owner._clients.discard(self.request)


class ThreadedBridgeServer:
    """Local raw RTU-over-TCP endpoint; share ``lock`` with the physics loop."""

    def __init__(self, device: BridgeDevice, lock=None, *, host: str = "127.0.0.1", port: int = 54321):
        self.device = device
        self.lock = lock if lock is not None else threading.RLock()
        self._closing = threading.Event()
        self._clients_lock = threading.Lock()
        self._clients: set[socket.socket] = set()
        self._server = _TCPServer((host, port), _Handler)
        self._server.owner = self
        self.address = self._server.server_address
        self._thread = None

    def start(self):
        if self._closing.is_set():
            raise RuntimeError("server has been closed")
        if self._thread is None:
            self._thread = threading.Thread(target=self._server.serve_forever,
                                            kwargs={"poll_interval": .05}, daemon=True)
            self._thread.start()
        return self

    def close(self):
        if self._closing.is_set():
            return
        self._closing.set()
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=1)
        with self._clients_lock:
            for client in list(self._clients):
                try:
                    client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        self._server.server_close()
        with self.lock:
            self.device.stop("server_closed")

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()
