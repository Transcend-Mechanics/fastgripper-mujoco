"""Run genuine PolyScope gripper nodes against a local MuJoCo bridge.

Creates a fresh, labeled URSim Docker container with independent files and
loopback-only published ports. Never connects to a physical robot or reuses the
bench URSim configuration. Docker Desktop/Engine and a built FastGripper URCap
with the explicit mujoco: transport are required.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np

from .rtde import RTDEClient
from .ursim import URSimBridge, render_trace
from .ursim_scene import URSimScene

IMAGE = "universalrobots/ursim_e-series@sha256:a4c4365207d54d1a1a4ead87526ff3781e2e98ae703c72f362060a46688fa7a4"
LABEL = "fastgripper.simulation"
PORTS = {"ur5e": dict(dashboard=32999, secondary=33002, rtde=33004, vnc=6280, bridge=55431),
         "ur3e": dict(dashboard=33999, secondary=34002, rtde=34004, vnc=6380, bridge=55432)}


def docker(*args, timeout=30):
    result = subprocess.run(["docker", *map(str, args)], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"docker {args[0]}: {result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def verify_container(name):
    """Check explicit ownership and URControl simulation mode before any action."""
    info = json.loads(docker("inspect", name))[0]
    if info["Config"].get("Labels", {}).get(LABEL) != "mujoco":
        raise RuntimeError("Refusing container without the MuJoCo simulation label")
    if info.get("HostConfig", {}).get("Devices"):
        raise RuntimeError("Refusing a simulator container with host devices attached")
    commands = docker("exec", name, "ps", "-eo", "args")
    if not any("/ursim/URControl" in line and " -r" in line for line in commands.splitlines()):
        raise RuntimeError("URControl simulation flag -r is required")
    return info


def dashboard(port, command, timeout=8.):
    with socket.create_connection(("127.0.0.1", port), timeout) as connection:
        connection.recv(4096)
        connection.sendall((command + "\n").encode())
        return connection.recv(4096).decode().strip()


def rpc(container, method, *args):
    script = ("import json,sys,xmlrpc.client; "
              "p=xmlrpc.client.ServerProxy('http://127.0.0.1:40408/'); "
              "print(json.dumps(getattr(p,sys.argv[1])(*json.loads(sys.argv[2]))))")
    return json.loads(docker("exec", container, "python3", "-c", script, method, json.dumps(args), timeout=15))


def wait_for(predicate, *, timeout=120., description="condition"):
    deadline, last = time.monotonic()+timeout, None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, RuntimeError, ValueError) as error:
            last = str(error)
        time.sleep(.25)
    raise TimeoutError(f"Timed out waiting for {description}: {last}")


def make_program(directory: Path, arm: str, high, low) -> Path:
    """Ordinary Script/File arm moves around genuine URCap contributed nodes."""
    root = ET.Element("URProgram", name="fg_mujoco", installation="default",
                      installationRelativePath="default", directory="/ursim/programs",
                      createdIn="5.25.2", lastSavedIn="5.25.2", robotSerialNumber="20195599999",
                      createdInPolyscopeProgramVersion="2", lastSavedInPolycopeProgramVersion="2",
                      crcValue="633835311")
    kine = ET.SubElement(root, "kinematics", status="NOT_INITIALIZED", validChecksum="false")
    a, d = (([0,-.24355,-.2132,0,0,0],[.15185,0,0,.13105,.08535,.0921]) if arm == "ur3e"
            else ([0,-.425,-.3922,0,0,0],[.1625,0,0,.1333,.0997,.0996]))
    for name, values in (("deltaTheta", [0]*6), ("a", a), ("d",d),
                         ("alpha", [np.pi/2,0,0,np.pi/2,-np.pi/2,0]), ("jointChecksum", [-1]*6)):
        ET.SubElement(kine, name, value=", ".join(map(str,values)))
    children = ET.SubElement(root, "children")
    ET.SubElement(children, "InitVariablesNode")
    main = ET.SubElement(children,"MainProgram",runOnlyOnce="true",InitVariablesNode="true")
    nodes = ET.SubElement(main,"children")
    directory.mkdir(parents=True, exist_ok=True)

    def script(phase, body=""):
        contents = f"write_output_integer_register(24, {phase})\n" + body + "\n"
        path = directory / f"fg_phase_{phase}.script"
        path.write_text(contents)
        node = ET.SubElement(nodes,"Script",type="File")
        ET.SubElement(node,"cachedContents").text = contents
        ET.SubElement(node,"file").text = "/ursim/programs/" + path.name

    def gripper(action=None, check=False):
        node = ET.SubElement(nodes,"Contributed", strategyClass="com.transcend.fastgripper.urcap." + ("FastGripperCheck" if check else "FastGripperNode"),
                             strategyProgramNodeType="Grip Check" if check else "Gripper",
                             strategyURCapDeveloper="Transcend Mechanics",strategyURCapName="fastgripper-urcap")
        model = ET.SubElement(node,"dataModel")
        if action:
            ET.SubElement(model,"data",key="action",value=action)

    def move(q):
        return "movej([" + ",".join(f"{x:.12g}" for x in q) + "], a=0.3, v=0.15)\nsleep(0.5)"

    script(1); gripper("open")
    script(2, move(low))
    script(3); gripper("close")
    script(4); gripper(check=True)
    script(5, move(high))
    script(6, "sleep(2.0)"); gripper(check=True)
    script(7, move(low))
    script(8); gripper("open")
    script(9, move(high))
    script(10, "sleep(0.5)")
    result = directory/"fg_mujoco.urp"
    result.write_bytes(gzip.compress(ET.tostring(root), mtime=0))
    return result


def acceptance(report):
    rows = report["samples"]
    held = [r for r in rows if r["phase_code"] == 6]
    final = rows[-1] if rows else None
    retained = len(held) >= 20 and all(r["object_rise_m"] > .08 and
            min(r["contact_normal_force_n_by_side"].values()) > .05 for r in held)
    released = final is not None and final["phase_code"] == 10 and abs(final["object_rise_m"]) < .002 and max(final["contact_normal_force_n_by_side"].values()) < .05
    unwanted = [pair for pair in report["contact_pairs"] if not pair["task_contact"]]
    events = [event for event in report["events"] if event["event"] in ("rtde_reference_lost","simulation_clock_overrun","rtde_controller_clock_reset")]
    return {"success": bool(retained and released and not unwanted and not events and not report["error"]
                            and report["warning_count"] == 0 and report["overrun_count"] == 0),
            "retained": bool(retained), "released": bool(released),
            "minimum_raised_hold_rise_m": min((r["object_rise_m"] for r in held),default=None),
            "unexpected_contact_pairs": unwanted, "stream_or_clock_events": events}


def exercise_controls(bridge, container, ports, programs):
    """Exercise the real daemon watcher, empty grasp, and stream outage.

    Runs after the physical task, at the high pose with the object on the
    table. All program and transport actions stay in the owned URSim container.
    """
    root = ET.fromstring(gzip.decompress((programs/"fg_mujoco.urp").read_bytes()))
    root.set("name", "fg_controls")
    nodes = root.find("./children/MainProgram/children")
    original = list(nodes)
    nodes.clear()
    # Reuse the real Close URCap node, followed by a long ordinary wait so
    # the program stays PLAYING while testing its stop watcher.
    nodes.append(original[4])
    script = ET.SubElement(nodes, "Script", type="File")
    contents = "write_output_integer_register(24, 12)\nsleep(60.0)\n"
    (programs/"fg_controls_wait.script").write_text(contents)
    ET.SubElement(script, "cachedContents").text = contents
    ET.SubElement(script, "file").text = "/ursim/programs/fg_controls_wait.script"
    (programs/"fg_controls.urp").write_bytes(gzip.compress(ET.tostring(root), mtime=0))

    def status():
        result = rpc(container, "status")
        if not result.get("ok"):
            raise RuntimeError(f"Simulation daemon status failed: {result}")
        return result

    def local_state():
        with bridge.lock:
            return bridge.scene.controller.snapshot()

    def stopped_in(state):
        sample = status()
        return sample.get("program_state") == state and not sample.get("moving")

    def phase_is(phase):
        sample = bridge.rtde.latest()
        return sample is not None and sample.task_phase == phase

    watcher_before = rpc(container, "daemon_info")["watcher_stops"]
    if "Loading program" not in dashboard(ports["dashboard"], "load fg_controls.urp", timeout=20):
        raise RuntimeError("Could not load the simulator control-check program")
    if "Starting program" not in dashboard(ports["dashboard"], "play"):
        raise RuntimeError("Could not start the simulator control-check program")
    def closing_in_program():
        sample = status()
        return sample.get("moving") and sample.get("program_state") == "PLAYING" and 5 < sample["pos"] < 80
    wait_for(closing_in_program, timeout=20, description="URCap empty close motion")
    dashboard(ports["dashboard"], "pause")
    wait_for(lambda: stopped_in("PAUSED"),
             timeout=15, description="URCap pause stops the jaw")
    paused = local_state()
    time.sleep(.35)
    pause_drift = abs(local_state()["aperture_m"]-paused["aperture_m"])
    dashboard(ports["dashboard"], "play")
    wait_for(lambda: phase_is(12),
             timeout=30, description="URCap paused close resumes and completes")
    empty = status()
    empty_ok = empty["result"] == "at_position" and not empty["object_detected"]
    # Open while the program's wait is active, then stop the entire program.
    if not rpc(container, "goto", 0, 50, 25).get("ok"):
        raise RuntimeError("Simulator open command failed")
    wait_for(lambda: status().get("moving") and local_state()["position_pct"] < 95,
             timeout=10, description="jaw moving before program stop")
    dashboard(ports["dashboard"], "stop")
    wait_for(lambda: stopped_in("STOPPED"),
             timeout=15, description="program stop halts jaw motion")
    stopped = local_state()
    time.sleep(.35)
    stop_drift = abs(local_state()["aperture_m"]-stopped["aperture_m"])
    # RTDE loss is guarded by the physics service independently of the URCap.
    if not rpc(container, "goto", 0, 50, 25).get("ok"):
        raise RuntimeError("Simulator stream-loss setup command failed")
    bridge.rtde.stop()
    wait_for(lambda: bridge.device.last_stop_reason == "rtde_reference_lost", timeout=5,
             description="reference-loss stop")
    before = local_state()
    if not rpc(container, "goto", 100, 50, 25).get("ok"):
        raise RuntimeError("Outage probe must reach the physics command gate")
    time.sleep(.35)
    after = local_state()
    outage_drift = abs(after["aperture_m"]-before["aperture_m"])
    bridge.rtde.start()
    wait_for(lambda: bridge.rtde.latest() is not None, timeout=10, description="RTDE reconnect")
    time.sleep(.25)
    recovered = status()
    watcher_stops = rpc(container, "daemon_info")["watcher_stops"] - watcher_before
    diagnostics = bridge.report()
    healthy = (not diagnostics["error"] and not diagnostics["warning_count"] and
               not diagnostics["overrun_count"] and not any(
                   event["event"] == "rtde_controller_clock_reset" for event in diagnostics["events"]))
    checks = {"pause_stopped_motion": pause_drift < .001 and 1 < paused["position_pct"] < 95,
              "pause_resume_empty_grasp": empty_ok,
              "program_stop_halted_motion": stop_drift < .001 and 5 < stopped["position_pct"] < 99,
              "reference_loss_blocks_new_motion": outage_drift < .001,
              "reconnect_did_not_resume_motion": not recovered["moving"],
              "pause_drift_m": pause_drift, "stop_drift_m": stop_drift,
              "outage_drift_m": outage_drift, "empty_grasp_result": empty["result"],
              "watcher_stops_observed": watcher_stops, "watcher_handled_interruptions": watcher_stops >= 2,
              "physics_healthy": healthy}
    checks["success"] = all(checks[key] for key in (
        "pause_stopped_motion", "pause_resume_empty_grasp", "program_stop_halted_motion",
        "reference_loss_blocks_new_motion", "reconnect_did_not_resume_motion",
        "watcher_handled_interruptions", "physics_healthy"))
    return checks


def run_demo(*, arm, urcap: Path, output: Path, keep_container=False, render=False, control_checks=False):
    if not urcap.is_file():
        raise ValueError("A built FastGripper .jar/.urcap with mujoco: transport is required")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory")
    name = "fg-mujoco-"+arm
    if docker("ps","-a","--filter",f"name=^/{name}$","--format","{{.Names}}"):
        raise ValueError(f"Container {name} already exists; preserve it or explicitly remove that simulator first")
    output = output.resolve()
    programs, caps = output/"container/programs", output/"container/urcaps"
    programs.mkdir(parents=True,exist_ok=True); caps.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(urcap,caps/"fastgripper.jar")
    scene = URSimScene(arm=arm)
    high, low = scene.trajectory.target(1.), scene.trajectory.target(0.)
    make_program(programs,arm,high,low)
    ports = PORTS[arm]
    command = ["run","-d","--name",name,"--platform","linux/amd64","--label",LABEL+"=mujoco",
               "--add-host","host.docker.internal:host-gateway",
               "-e","ROBOT_MODEL="+("UR3" if arm=="ur3e" else "UR5"),
               "-v",str(caps)+":/urcaps:ro","-v",str(programs)+":/ursim/programs"]
    for field,inside in (("dashboard",29999),("secondary",30002),("rtde",30004),("vnc",6080)):
        command += ["-p",f"127.0.0.1:{ports[field]}:{inside}"]
    command.append(IMAGE)
    bridge = None
    created = False
    result = None
    try:
        docker(*command,timeout=60); created=True
        print(f"Started {name}; pendant http://localhost:{ports['vnc']}/vnc.html",flush=True)
        wait_for(lambda: "Robotmode" in dashboard(ports["dashboard"],"robotmode"),timeout=300,description="URSim dashboard")
        verify_container(name)
        wait_for(lambda: rpc(name,"ping").get("ok"),timeout=120,description="FastGripper URCap daemon")
        dashboard(ports["dashboard"],"close safety popup")
        dashboard(ports["dashboard"],"power on")
        time.sleep(2)
        dashboard(ports["dashboard"],"brake release")
        wait_for(lambda: "RUNNING" in dashboard(ports["dashboard"],"robotmode"),description="URSim brakes released")
        bootstrap = "def fg_sim_initial_pose():\n  movej("+str(high.tolist())+", a=0.5, v=0.5)\nend\n"
        with socket.create_connection(("127.0.0.1",ports["secondary"]),5) as connection:
            connection.sendall(bootstrap.encode())
        with RTDEClient("127.0.0.1",ports["rtde"]) as monitor:
            def settled():
                ref=monitor.latest()
                return ref is not None and np.max(np.abs(np.asarray(ref.actual_q)-high)) < .001 and np.max(np.abs(ref.actual_qd)) < .001
            wait_for(settled,timeout=120,description="initial high arm pose")
        print("URSim reached the initial pose; starting physics bridge",flush=True)
        bridge = URSimBridge(arm=arm,rtde_port=ports["rtde"],listen="0.0.0.0",port=ports["bridge"]).start()
        transport=f"mujoco:host.docker.internal:{ports['bridge']}"
        configured=rpc(name,"config_set",transport,9,115200)
        if not configured.get("ok"):
            raise RuntimeError(f"Explicit simulator transport rejected: {configured}")
        status=rpc(name,"status")
        if not status.get("simulated") or status.get("simulation_backend") != "mujoco":
            raise RuntimeError("Daemon did not identify the explicit MuJoCo backend")
        if not rpc(name,"activate").get("ok"):
            raise RuntimeError("Nominal simulated activation failed")
        wait_for(lambda: rpc(name,"status").get("state")=="ready",description="simulated gripper READY")
        loaded=dashboard(ports["dashboard"],"load fg_mujoco.urp",timeout=20)
        if "Loading program" not in loaded:
            raise RuntimeError(loaded)
        played=dashboard(ports["dashboard"],"play")
        if "Starting program" not in played:
            raise RuntimeError(played)
        print("PolyScope is executing genuine FastGripper URCap nodes",flush=True)
        deadline=time.monotonic()+120
        saw_playing=False
        while time.monotonic()<deadline:
            state=dashboard(ports["dashboard"],"programState")
            saw_playing |= "PLAYING" in state
            if bridge.error:
                raise RuntimeError(bridge.error)
            if saw_playing and "STOPPED" in state:
                break
            time.sleep(.25)
        else:
            raise TimeoutError("PolyScope program did not finish; inspect its pendant")
        time.sleep(.3)
        report=bridge.report()
        report["acceptance"]=acceptance(report)
        report["urcap_sha256"]=hashlib.sha256(urcap.read_bytes()).hexdigest()
        report["ursim_image"]=IMAGE
        report["daemon_status"]=rpc(name,"status")
        report["daemon_info"]=rpc(name,"daemon_info")
        # PolyScope compiles contributed nodes when Play is pressed; it does
        # not necessarily save a companion .script file. Record the actual
        # loaded program and daemon's completed-command history instead.
        report["program_sha256"]=hashlib.sha256((programs/"fg_mujoco.urp").read_bytes()).hexdigest()
        report["program_nodes"]={"gripper_open": 2, "gripper_close": 1, "grip_check": 2}
        report["service_history"]=rpc(name,"service_info")
        (output/"session.json").write_text(json.dumps(report,indent=2)+"\n")
        if control_checks and report["acceptance"]["success"]:
            print("Checking URCap pause/resume, empty grasp, stop and RTDE reconnection", flush=True)
            checks = exercise_controls(bridge, name, ports, programs)
            control_report = bridge.report()
            control_report["control_checks"] = checks
            (output/"control-checks.json").write_text(json.dumps(control_report, indent=2)+"\n")
            report["control_checks"] = checks
            (output/"session.json").write_text(json.dumps(report, indent=2)+"\n")
            if not checks["success"]:
                raise RuntimeError(f"URCap control checks failed: {checks}")
        if render:
            render_trace(report,output/"demo")
        result=report["acceptance"]
        print(json.dumps(result,indent=2),flush=True)
        if not result["success"]:
            raise RuntimeError("Physical URCap grasp/lift/release acceptance failed")
        return report
    except BaseException:
        if created:
            # Retain startup/program diagnostics before removing our container.
            for filename, command in (
                ("docker.log", ("logs", name)),
                ("urcontrol.log", ("exec", name, "tail", "-n", "150", "/ursim/URControl.log")),
                ("polyscope.log", ("exec", name, "tail", "-n", "150", "/ursim/polyscope.log")),
            ):
                try:
                    (output/filename).write_text(docker(*command)+"\n")
                except (RuntimeError, subprocess.TimeoutExpired):
                    pass
        raise
    finally:
        if bridge is not None:
            if result is None:
                (output/"failed-session.json").write_text(json.dumps(bridge.report(),indent=2)+"\n")
            bridge.close()
        if created:
            # Only this freshly-created, labeled container may be stopped.
            info=json.loads(docker("inspect",name))[0]
            if info["Config"].get("Labels",{}).get(LABEL)=="mujoco":
                try:
                    dashboard(ports["dashboard"],"stop")
                except OSError:
                    pass
                if not keep_container:
                    docker("rm","-f",name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",choices=tuple(PORTS),default="ur5e")
    parser.add_argument("--urcap",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--keep-container",action="store_true")
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--control-checks",action="store_true",help="Also exercise pause/resume, empty grasp, stop and RTDE reconnection")
    args=parser.parse_args()
    run_demo(arm=args.arm,urcap=args.urcap,output=args.output,
             keep_container=args.keep_container,render=args.render,control_checks=args.control_checks)


if __name__=="__main__":
    main()
