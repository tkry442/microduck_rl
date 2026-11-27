#!/usr/bin/env python3
"""Record pendulum logs from one Dynamixel XC330 for BAM identification.

Runs on the computer the servo is plugged into (a Raspberry Pi with a U2D2 is
the intended setup). Needs only ``numpy`` and ``rustypot`` — not BAM, torch or
mjlab — so it installs in a plain venv on a Pi 3B:

    python3 -m venv ~/sysid && ~/sysid/bin/pip install numpy rustypot

Workflow (one servo, ID 1 unless ``--id``):

    setup   find the servo at 57600 or 1 Mbps, switch it to 1 Mbps, return
            delay 0, position mode. Writes EEPROM; run once per servo.
    zero    torque off, let the arm hang, store that angle as 0 rad
            (BAM's pendulum model defines 0 as "arm straight down").
    check   print the angle for a few seconds so you can see 0 and ±90°.
    record  for one weight/arm combination, play BAM's 4 trajectories at each
            P gain and write one BAM-format JSON per run into --logdir.

The JSON format is BAM's raw format (``bam.process`` input). Differences from
``bam.dynamixel.record``, which this mirrors:

* the arm mass is stored as ``arm_mass`` (BAM's own recorder writes
  ``arm-mass``, which ``bam.logs`` does not read, so the arm silently counts
  as massless);
* all present-state registers are read in ONE bus transaction (PWM, current,
  velocity, position, voltage, temperature at addresses 124–146), so the
  sample rate is set by one round trip instead of five;
* the angle is relative to the stored zero rather than raw count 2048;
* it stops, torque off, on Ctrl-C, over-temperature or a bus error.

``--fake PARAMS.json`` replaces the servo with BAM's simulator (needs BAM, so
use it on the PC) to rehearse the whole pipeline without hardware.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- X330 control table (XL330 and XC330 share it) ---------------------------
ADDR_STATE = 124  # Present PWM .. Present Temperature
LEN_STATE = 23  # 124 PWM(2) 126 current(2) 128 vel(4) 132 pos(4) 136/140 traj(8) 144 vin(2) 146 temp(1)
COUNTS_PER_REV = 4096
RAD_PER_COUNT = 2 * math.pi / COUNTS_PER_REV
VEL_UNIT_RAD_S = 0.229 * 2 * math.pi / 60  # 0.229 rpm per count
PWM_LIMIT = 885  # Present PWM at 100 % duty
POSITION_MODE = 3

DEFAULT_KPS = (50, 100, 200, 300, 400)  # BAM's XL330 sweep; same firmware gain scale
TRAJECTORIES = ("sin_sin", "lift_and_drop", "up_and_down", "sin_time_square")
TRAJ_MAX_ABS_RAD = 2.1  # sin_sin peaks at ~2.07 rad; the goal must stay in 0..4095
TEMP_ABORT_C = 70
TEMP_RESUME_C = 55
ZERO_FILE = "zero_id{id}.txt"  # not .json: bam.process reads every *.json in the folder


# --- Trajectories: identical to bam/trajectory.py (6 s each) -----------------
def _cubic(keyframes, t):
    if t <= keyframes[0][0]:
        return keyframes[0][1]
    if t >= keyframes[-1][0]:
        return keyframes[-1][1]
    for (t0, x0, v0), (t1, x1, v1) in itertools.pairwise(keyframes):
        if t0 <= t <= t1:
            a = [[1, t0, t0**2, t0**3], [0, 1, 2 * t0, 3 * t0**2],
                 [1, t1, t1**2, t1**3], [0, 1, 2 * t1, 3 * t1**2]]
            w = np.linalg.solve(a, [x0, v0, x1, v1])
            return float(w[0] + w[1] * t + w[2] * t**2 + w[3] * t**3)
    raise AssertionError("unreachable")


def trajectory(name: str, t: float) -> tuple[float, bool]:
    """(goal angle [rad], torque enabled) at time t, as BAM defines them."""
    if name == "lift_and_drop":
        return _cubic([[0.0, 0.0, 0.0], [2.0, -math.pi / 2, 0.0]], t), t < 2.0
    if name == "sin_time_square":
        return math.sin(t**2), True
    if name == "up_and_down":
        kf = [[0.0, 0.0, 0.0], [3.0, math.pi / 2, 0.0], [6.0, 0.8 * math.pi / 2, 0.0]]
        return _cubic(kf, t), True
    if name == "sin_sin":
        return math.sin(t) * math.pi / 2 + math.sin(5.0 * t) * 0.5 * math.sin(2.0 * t), True
    raise ValueError(f"unknown trajectory {name!r}")


TRAJ_DURATION = 6.0


# --- Register decoding --------------------------------------------------------
def _s16(b: bytes, i: int) -> int:
    return int.from_bytes(b[i:i + 2], "little", signed=True)


def _s32(b: bytes, i: int) -> int:
    return int.from_bytes(b[i:i + 4], "little", signed=True)


def decode_state(block: bytes) -> dict:
    """Decode the 23-byte block read at address 124."""
    if len(block) != LEN_STATE:
        raise ValueError(f"expected {LEN_STATE} bytes, got {len(block)}")
    return {
        "pwm_raw": _s16(block, 0),
        "current_raw": _s16(block, 2),  # mA
        "vel_raw": _s32(block, 4),
        "pos_raw": _s32(block, 8),
        "vin_raw": int.from_bytes(block[20:22], "little"),  # 0.1 V
        "temp": block[22],
    }


def _as_bytes(x) -> bytes:
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple, bytes, bytearray)):
        x = x[0]  # some rustypot builds wrap single-id reads in a list
    return bytes(bytearray(int(v) for v in x))


def _first(x):
    return x[0] if isinstance(x, (list, tuple)) else x


# --- Buses --------------------------------------------------------------------
class RealBus:
    """One XC330 over rustypot (Protocol 2.0)."""

    def __init__(self, port: str, servo_id: int, baud: int):
        from rustypot import Xl330PyController  # XC330 = same control table

        self.id = servo_id
        self.c = Xl330PyController(port, baudrate=baud, timeout=0.02)

    @staticmethod
    def now() -> float:
        return time.perf_counter()

    @staticmethod
    def sleep(dt: float) -> None:
        time.sleep(dt)

    def read_state(self) -> dict:
        return decode_state(_as_bytes(self.c.read_raw_data(self.id, ADDR_STATE, LEN_STATE)))

    def write_goal_raw(self, raw: int) -> None:
        self.c.write_raw_goal_position(self.id, int(raw))

    def torque(self, on: bool) -> None:
        self.c.write_torque_enable(self.id, bool(on))

    def p_gain(self) -> int:
        return int(_first(self.c.read_position_p_gain(self.id)))

    def set_p_gain(self, kp: int) -> None:
        self.c.write_position_p_gain(self.id, int(kp))


class FakeBus:
    """BAM's simulator standing in for the servo (PC rehearsal / tests).

    Uses a virtual clock: each bus transaction advances it by ``io_dt``.
    """

    def __init__(self, params_json: str, mass: float, arm_mass: float, length: float,
                 vin: float = 11.1, io_dt: float = 0.002, zero_raw: int = 2048):
        here = Path(__file__).resolve().parent
        sys.path.insert(0, str(here))
        from bam_xc330 import register_xc330

        register_xc330()
        from bam.model import load_model
        from bam.simulate import Simulator

        self.model = load_model(params_json)
        self.model.actuator.vin = vin
        self.model.actuator.load_log({"mass": mass, "arm_mass": arm_mass, "length": length,
                                      "kp": 400, "vin": vin})
        self.sim = Simulator(self.model)
        self.sim.reset(0.0, 0.0)
        self.vin, self.io_dt, self.zero_raw = vin, io_dt, zero_raw
        self.t = 0.0
        self.goal = 0.0
        self.enabled = False
        self.kp = 400

    def _advance(self) -> None:
        self.model.actuator.kp = self.kp
        steps = max(1, round(self.io_dt / 0.001))
        for _ in range(steps):
            u = self.model.actuator.compute_control(self.goal, self.sim.q, self.sim.dq, 0.001)
            self.sim.step(u, self.enabled, 0.001)
        self.t += steps * 0.001

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        while dt > 0:
            self._advance()
            dt -= self.io_dt

    def read_state(self) -> dict:
        self._advance()
        q, dq = float(self.sim.q), float(self.sim.dq)
        duty = 0.0
        if self.enabled:
            duty = float(np.clip((self.goal - q) * self.kp * self.model.actuator.error_gain, -1, 1))
        return {"pwm_raw": round(duty * PWM_LIMIT), "current_raw": 0,
                "vel_raw": round(dq / VEL_UNIT_RAD_S),
                "pos_raw": round(self.zero_raw + q / RAD_PER_COUNT),
                "vin_raw": round(self.vin * 10), "temp": 35}

    def write_goal_raw(self, raw: int) -> None:
        self._advance()
        self.goal = (raw - self.zero_raw) * RAD_PER_COUNT

    def torque(self, on: bool) -> None:
        self._advance()
        self.enabled = bool(on)

    def p_gain(self) -> int:
        return self.kp

    def set_p_gain(self, kp: int) -> None:
        self.kp = int(kp)


# --- Helpers ------------------------------------------------------------------
def check_latency_timer(port: str) -> None:
    """FTDI adapters (U2D2) batch replies for 16 ms by default; 1 ms is needed."""
    path = Path("/sys/bus/usb-serial/devices") / Path(os.path.realpath(port)).name / "latency_timer"
    try:
        value = int(path.read_text().strip())
    except OSError:
        return
    if value > 1:
        print(f"!! {path} = {value} ms. Samples will be ~{value} ms apart. Fix with:\n"
              f"   echo 1 | sudo tee {path}")


def load_zero(logdir_or_dir: Path, servo_id: int) -> int:
    f = logdir_or_dir / ZERO_FILE.format(id=servo_id)
    if not f.exists():
        raise SystemExit(f"No {f.name} in {logdir_or_dir}. Run the 'zero' step first.")
    return int(json.loads(f.read_text())["zero_raw"])


def entry_from_state(s: dict, zero_raw: int) -> dict:
    return {
        "position": (s["pos_raw"] - zero_raw) * RAD_PER_COUNT,
        "speed": s["vel_raw"] * VEL_UNIT_RAD_S,
        "load": float(np.clip(s["pwm_raw"] / PWM_LIMIT, -1.0, 1.0)),
        "current": s["current_raw"] / 1000.0,
        "input_volts": s["vin_raw"] / 10.0,
        "temp": s["temp"],
    }


def wait_cool(bus, limit_c: int = TEMP_RESUME_C) -> None:
    t = bus.read_state()["temp"]
    while t > limit_c:
        print(f"   cooling: {t} °C > {limit_c} °C, waiting...")
        bus.sleep(10.0)
        t = bus.read_state()["temp"]


def hold_here(bus) -> None:
    """Set the goal to the present position (call before enabling torque)."""
    bus.write_goal_raw(bus.read_state()["pos_raw"])


def move_to(bus, zero_raw: int, goal: float, speed: float = 1.0) -> None:
    """Ramp the goal from the present angle to `goal` [rad] at `speed` rad/s."""
    q = (bus.read_state()["pos_raw"] - zero_raw) * RAD_PER_COUNT
    n = max(1, int(abs(goal - q) / (speed * 0.01)))
    for k in range(1, n + 1):
        bus.write_goal_raw(zero_raw + round((q + (goal - q) * k / n) / RAD_PER_COUNT))
        bus.sleep(0.01)


def record_one(bus, zero_raw: int, kp: int, traj: str, meta: dict) -> dict:
    """Play one trajectory at one P gain; return the BAM raw log."""
    bus.torque(False)
    bus.set_p_gain(kp)
    goal0, _ = trajectory(traj, 0.0)
    hold_here(bus)
    bus.torque(True)
    move_to(bus, zero_raw, goal0)
    bus.sleep(1.0)  # settle at the start pose, as bam.dynamixel.record does

    entries = []
    enabled = True
    t_start = bus.now()
    while (t := bus.now() - t_start) < TRAJ_DURATION:
        goal, want = trajectory(traj, t)
        if want != enabled:
            bus.torque(want)
            enabled = want
        if enabled:
            bus.write_goal_raw(zero_raw + round(goal / RAD_PER_COUNT))
        t0 = bus.now() - t_start
        s = bus.read_state()
        t1 = bus.now() - t_start
        if s["temp"] >= TEMP_ABORT_C:
            raise RuntimeError(f"servo at {s['temp']} °C (limit {TEMP_ABORT_C})")
        e = entry_from_state(s, zero_raw)
        e.update(timestamp=(t0 + t1) / 2, goal_position=goal, torque_enable=enabled)
        entries.append(e)

    # back to 0 gently, then release. After lift_and_drop the goal register
    # still says -90° while the arm hangs at 0: re-aim it first or it snaps.
    hold_here(bus)
    bus.torque(True)
    move_to(bus, zero_raw, 0.0)
    bus.torque(False)

    vin = float(np.mean([e["input_volts"] for e in entries]))
    return {**meta, "kp": kp, "vin": vin, "trajectory": traj, "entries": entries}


# --- Commands -----------------------------------------------------------------
def cmd_setup(a) -> None:
    from rustypot import Xl330PyController

    found = None
    for baud in (1_000_000, 57_600):
        c = Xl330PyController(a.port, baudrate=baud, timeout=0.05)
        try:
            if c.ping(a.id):
                found = (baud, c)
                break
        except Exception:  # noqa: BLE001, S110 — rustypot raises on timeout; try the next baud
            pass
    if found is None:
        raise SystemExit(f"No servo with ID {a.id} at 1 Mbps or 57600 on {a.port}. "
                         "Check power (the U2D2 does not power the servo) and the cable.")
    baud, c = found
    sid = a.id
    print(f"found ID {sid} at {baud} baud, model number {_first(c.read_model_number(sid))}, "
          f"firmware {_first(c.read_firmware_version(sid))}")
    c.write_torque_enable(sid, False)
    if _first(c.read_operating_mode(sid)) != POSITION_MODE:
        c.write_operating_mode(sid, POSITION_MODE)
        print("operating mode -> 3 (position)")
    c.write_return_delay_time(sid, 0)
    print("return delay time -> 0")
    print(f"Position P gain now {_first(c.read_position_p_gain(sid))} "
          f"(record.py sets it per run and restores it after)")
    if baud != 1_000_000:
        c.write_baud_rate(sid, 3)  # 3 = 1 Mbps
        time.sleep(0.1)
        c2 = Xl330PyController(a.port, baudrate=1_000_000, timeout=0.05)
        if not c2.ping(sid):
            raise SystemExit("baud change written but the servo does not answer at 1 Mbps")
        print("baud rate -> 1 Mbps (answers at the new speed)")
    print("setup OK")


def cmd_zero(a) -> None:
    bus = RealBus(a.port, a.id, a.baud)
    bus.torque(False)
    print("Torque off. Let the arm hang still, straight down. Measuring for 3 s...")
    time.sleep(1.0)
    raws = []
    t0 = time.time()
    while time.time() - t0 < 2.0:
        raws.append(bus.read_state()["pos_raw"])
        time.sleep(0.01)
    zero = round(float(np.median(raws)))
    margin = TRAJ_MAX_ABS_RAD / RAD_PER_COUNT
    if not (margin <= zero <= COUNTS_PER_REV - 1 - margin):
        raise SystemExit(
            f"Arm-down reads raw {zero}: the trajectories need {zero}±{margin:.0f} to stay inside "
            "0..4095. Take the arm off, turn it on the horn so that 'hanging down' is near the "
            "middle of the range (raw ≈ 2048, the horn's alignment mark), and run 'zero' again.")
    out = Path(a.logdir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / ZERO_FILE.format(id=a.id)
    f.write_text(json.dumps({"zero_raw": zero, "id": a.id,
                             "spread_counts": int(np.ptp(raws))}))
    print(f"zero = raw {zero} (spread {np.ptp(raws)} counts) -> {f}")


def cmd_check(a) -> None:
    bus = RealBus(a.port, a.id, a.baud)
    check_latency_timer(a.port)
    zero = load_zero(Path(a.logdir), a.id)
    bus.torque(False)
    print("Torque off. Move the arm by hand: down = 0°, horizontal = ±90°. Ctrl-C to stop.")
    n, t0 = 0, time.perf_counter()
    try:
        while True:
            s = bus.read_state()
            n += 1
            e = entry_from_state(s, zero)
            rate = n / (time.perf_counter() - t0)
            print(f"\r angle {math.degrees(e['position']):7.1f}°  {e['input_volts']:.1f} V  "
                  f"{e['temp']} °C  {rate:5.0f} reads/s ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()


def cmd_record(a) -> None:
    logdir = Path(a.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    if a.fake:
        bus = FakeBus(a.fake, a.mass, a.arm_mass, a.length)
        zero = bus.zero_raw
    else:
        check_latency_timer(a.port)
        bus = RealBus(a.port, a.id, a.baud)
        zero = load_zero(logdir, a.id)
    kps = [int(k) for k in a.kps.split(",")]
    trajs = a.trajectories.split(",")
    torque_hold = a.mass * 9.81 * a.length + a.arm_mass * 9.81 * a.length / 2
    print(f"load: {a.mass} kg at {a.length} m (+ arm {a.arm_mass} kg) "
          f"= {torque_hold:.2f} N·m held horizontally")
    if torque_hold > 0.75 and not a.force:
        raise SystemExit("That is over ~80 % of the XC330's 0.92 N·m stall torque: the arm may "
                         "not reach 90° and the servo will heat. Use a lighter weight or a "
                         "shorter arm (or --force).")
    meta = {"mass": a.mass, "arm_mass": a.arm_mass, "length": a.length, "motor": "xc330",
            "servo_id": a.id, "recorder": "microduck_rl/scripts/sysid/record_xc330.py"}
    kp_before = bus.p_gain()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"m{round(a.mass * 1000)}g_L{round(a.length * 1000)}mm"
    done = 0
    try:
        for kp in kps:
            for traj in trajs:
                if not a.fake:
                    wait_cool(bus)
                print(f"[{done + 1}/{len(kps) * len(trajs)}] kp {kp:3d}  {traj:16s}", end="", flush=True)
                log = record_one(bus, zero, kp, traj, meta)
                ts = np.array([e["timestamp"] for e in log["entries"]])
                dts = np.diff(ts) * 1000
                f = logdir / f"{tag}_kp{kp}_{traj}_{stamp}.json"
                f.write_text(json.dumps(log))
                print(f"  {len(ts)} samples, step {np.median(dts):.1f} ms "
                      f"(max {dts.max():.1f}), {log['vin']:.2f} V -> {f.name}")
                done += 1
                if traj == "sin_time_square" and not a.fake:
                    bus.sleep(3.0)
    finally:
        try:
            bus.torque(False)
            bus.set_p_gain(kp_before)
        except Exception as e:  # noqa: BLE001 — best effort on the way out
            print(f"\n!! could not release the servo cleanly: {e}")
    print(f"{done} runs written to {logdir}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("setup", "zero", "check", "record"))
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--id", type=int, default=1)
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--logdir", default="sysid_raw")
    p.add_argument("--mass", type=float, help="weight at the arm tip [kg]")
    p.add_argument("--arm-mass", type=float, help="the arm alone [kg]")
    p.add_argument("--length", type=float, help="axis to weight centre [m]")
    p.add_argument("--kps", default=",".join(map(str, DEFAULT_KPS)))
    p.add_argument("--trajectories", default=",".join(TRAJECTORIES))
    p.add_argument("--force", action="store_true", help="allow loads near stall")
    p.add_argument("--fake", metavar="PARAMS_JSON", help="simulate the servo with BAM (PC only)")
    a = p.parse_args(argv)
    if a.command == "record" and None in (a.mass, a.arm_mass, a.length):
        p.error("record needs --mass, --arm-mass and --length")
    {"setup": cmd_setup, "zero": cmd_zero, "check": cmd_check, "record": cmd_record}[a.command](a)


if __name__ == "__main__":
    main()
