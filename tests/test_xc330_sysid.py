"""XC330 identification pipeline: actuator registration + pendulum recorder.

The recorder (``scripts/sysid/record_xc330.py``) runs on a Pi without BAM, so
it carries its own copy of BAM's trajectories and register decoding. These
tests lock it to BAM:

* ``"xc330"`` is registered, uses the X330 firmware gain, and its fit ranges
  contain the catalogue estimate (BAM's XL330 range ``R ∈ [2, 5]`` does not);
* the recorder's trajectories equal ``bam.trajectory``;
* the 23-byte state block decodes signed values correctly;
* a log recorded from the simulated servo (``--fake``) is reproduced by BAM's
  simulator with the same parameters — i.e. the file is a valid BAM log with
  the right angle zero, sign, units and the ``arm_mass`` key BAM reads.
"""

import importlib.util
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SYSID = REPO / "scripts" / "sysid"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def xc330():
    return _load("_xc330", REPO / "src" / "mjlab_microduck" / "actuator" / "xc330.py")


@pytest.fixture(scope="module")
def rec():
    return _load("_record_xc330", SYSID / "record_xc330.py")


@pytest.fixture()
def truth_params(tmp_path, xc330):
    from bam.model import _resolve_json_path

    data = json.loads(Path(_resolve_json_path(None, "xl330", "m6")).read_text())
    data.update(kt=1.40, R=17.0, armature=0.004, q_offset=0.0, actuator="xc330")
    p = tmp_path / "truth.json"
    p.write_text(json.dumps(data))
    return p


def test_xc330_registered_with_x330_firmware_gain(xc330):
    import bam.actuators

    act = bam.actuators.actuators["xc330"]()
    ref = bam.actuators.actuators["xl330"]()
    assert act.error_gain == pytest.approx(ref.error_gain)
    assert act.vin == pytest.approx(11.1)


def test_xc330_fit_ranges_contain_catalogue_estimate(xc330):
    from bam.model import models

    model = models["m6"]()
    model.set_actuator(xc330.XC330Actuator(xc330.Pendulum))
    p = model.get_parameters()
    # catalogue: 0.92 N·m stall, 6.8 rad/s no-load at 11.1 V -> kt 1.63, R 19.7
    for name, value in (("kt", 1.63), ("R", 19.7)):
        assert p[name].min < value < p[name].max, name
        # and not so close to a bound that the optimizer is likely to pin there
        assert p[name].min < 0.5 * value and p[name].max > 2 * value, name


def test_fitted_json_loads_through_training_package():
    """A fit writes "actuator": "xc330"; training loads it via mjlab_microduck."""
    pytest.importorskip("mjlab")
    import bam.actuators

    import mjlab_microduck.actuator  # noqa: F401 — registers xc330

    assert "xc330" in bam.actuators.actuators


@pytest.mark.parametrize("name", ["sin_sin", "lift_and_drop", "up_and_down", "sin_time_square"])
def test_trajectories_match_bam(rec, name):
    from bam.trajectory import trajectories

    for t in np.linspace(0.0, 6.0, 97):
        goal, enable = rec.trajectory(name, float(t))
        ref_goal, ref_enable = trajectories[name](float(t))
        assert goal == pytest.approx(float(ref_goal), abs=1e-9)
        assert enable == ref_enable


def test_trajectories_stay_inside_goal_margin(rec):
    for name in rec.TRAJECTORIES:
        peak = max(abs(rec.trajectory(name, t)[0]) for t in np.linspace(0, 6, 2001))
        assert peak <= rec.TRAJ_MAX_ABS_RAD


def test_decode_state_signed_fields(rec):
    block = (
        struct.pack("<hhii", -442, -120, -37, 1024)
        + b"\x00" * 8  # velocity / position trajectory (unused)
        + struct.pack("<H", 117)
        + bytes([41])
    )
    s = rec.decode_state(block)
    assert s == {"pwm_raw": -442, "current_raw": -120, "vel_raw": -37,
                 "pos_raw": 1024, "vin_raw": 117, "temp": 41}
    e = rec.entry_from_state(s, zero_raw=2048)
    assert e["position"] == pytest.approx(-math.pi / 2)
    assert e["load"] == pytest.approx(-442 / 885)
    assert e["input_volts"] == pytest.approx(11.7)


def test_fake_recording_is_reproduced_by_bam(rec, truth_params, tmp_path, xc330):
    from bam.model import load_model
    from bam.simulate import Simulator

    logdir = tmp_path / "raw"
    rec.main(["record", "--fake", str(truth_params), "--mass", "0.3", "--arm-mass", "0.015",
              "--length", "0.15", "--kps", "200", "--trajectories", "sin_sin",
              "--logdir", str(logdir)])
    (f,) = logdir.glob("*.json")
    log = json.loads(f.read_text())
    assert log["arm_mass"] == 0.015 and "arm-mass" not in log
    assert log["kp"] == 200 and log["motor"] == "xc330"
    positions = np.array([e["position"] for e in log["entries"]])
    goals = np.array([e["goal_position"] for e in log["entries"]])
    assert np.abs(positions).max() > 0.5  # the arm actually swung
    assert np.corrcoef(positions, goals)[0, 1] > 0.8  # and followed the goal

    # Resample like bam.process does, then roll out BAM with the true params.
    dt = 0.005
    ts = np.array([e["timestamp"] for e in log["entries"]])
    grid = np.arange(0.0, ts[-1], dt)
    entries = [{"position": float(np.interp(t, ts, positions)),
                "speed": float(np.interp(t, ts, [e["speed"] for e in log["entries"]])),
                "goal_position": float(np.interp(t, ts, goals)),
                "torque_enable": True} for t in grid]
    processed = {**log, "dt": dt, "entries": entries}
    sim_pos, _, _ = Simulator(load_model(str(truth_params))).rollout_log(
        processed, simulate_control=True)
    mae = np.mean(np.abs(np.array(sim_pos) - np.array([e["position"] for e in entries])))
    assert mae < 0.02, mae  # ~1°; a sign or zero error gives > 0.5 rad
