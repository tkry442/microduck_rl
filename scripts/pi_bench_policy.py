#!/usr/bin/env python3
"""Measure walking-policy inference time on a small ARM board (e.g. Raspberry Pi 3B).

The deployed walking policies (velocity / VelStand) are an MLP actor
61 -> 512 -> 256 -> 128 -> 14 with ELU, with the observation normalizer baked
in by scripts/export.py. Inference cost depends only on that shape, not on the
trained weights, so by default this builds a same-shaped ONNX with random
weights. Pass --model path/to/policy.onnx to time a real export instead.

What it reports:
  1. per-call latency (mean / p50 / p99 / max) for 1 thread and N threads
  2. a real 50 Hz loop (sleep until each tick) for --loop-s seconds:
     inference time, wake-up lateness, and deadline misses
  3. a width sweep (x1, x2, x4, x8 hidden width) to show how much bigger a
     network could get before it eats the 20 ms tick

Only needs: numpy, onnx, onnxruntime.  Run:  python3 pi_bench_policy.py
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

OBS = 61
ACT = 14
HIDDEN = (512, 256, 128)


def build_mlp(path: str, hidden: tuple[int, ...], seed: int = 0) -> None:
    """Same graph shape as the exported actor: normalizer + Gemm/Elu stack."""
    rng = np.random.default_rng(seed)
    inits = [
        numpy_helper.from_array(rng.normal(0, 1, (1, OBS)).astype(np.float32), "obs_mean"),
        numpy_helper.from_array(rng.uniform(0.5, 2, (1, OBS)).astype(np.float32), "obs_std"),
    ]
    nodes = [
        helper.make_node("Sub", ["obs", "obs_mean"], ["n0"]),
        helper.make_node("Div", ["n0", "obs_std"], ["x0"]),
    ]
    prev, dims = "x0", (OBS, *hidden, ACT)
    for i in range(len(dims) - 1):
        w = rng.normal(0, 1 / np.sqrt(dims[i]), (dims[i + 1], dims[i])).astype(np.float32)
        b = np.zeros(dims[i + 1], np.float32)
        inits += [numpy_helper.from_array(w, f"W{i}"), numpy_helper.from_array(b, f"B{i}")]
        out = "actions" if i == len(dims) - 2 else f"g{i}"
        nodes.append(helper.make_node("Gemm", [prev, f"W{i}", f"B{i}"], [out], transB=1))
        if out != "actions":
            nodes.append(helper.make_node("Elu", [out], [f"x{i + 1}"]))
            prev = f"x{i + 1}"
    graph = helper.make_graph(
        nodes,
        "actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, OBS])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, ACT])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


def session(path: str, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def time_calls(sess: ort.InferenceSession, n: int, warmup: int = 200) -> np.ndarray:
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    x = np.random.default_rng(1).normal(0, 1, shape).astype(np.float32)
    feed = {inp.name: x}
    for _ in range(warmup):
        sess.run(None, feed)
    t = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns()
        sess.run(None, feed)
        t[i] = time.perf_counter_ns() - t0
    return t / 1e6  # ms


def fmt(t: np.ndarray) -> str:
    return (
        f"mean {t.mean():6.3f}  p50 {np.percentile(t, 50):6.3f}  "
        f"p99 {np.percentile(t, 99):6.3f}  max {t.max():6.3f}  [ms]"
    )


def board_state() -> str:
    out = []
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            out.append(f"temp {int(f.read()) / 1000:.1f}C")
    except OSError:
        pass
    try:
        thr = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3)
        out.append(thr.stdout.strip() or "vcgencmd: n/a")
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            out.append(f"cpu0 {int(f.read()) // 1000} MHz")
    except OSError:
        pass
    return ", ".join(out) or "n/a"


def run_loop(sess: ort.InferenceSession, hz: float, seconds: float) -> None:
    inp = sess.get_inputs()[0]
    feed = {inp.name: np.zeros((1, OBS), np.float32)}
    period = 1.0 / hz
    n = int(seconds * hz)
    infer, late = np.empty(n), np.empty(n)
    misses = 0
    nxt = time.perf_counter() + period
    for i in range(n):
        now = time.perf_counter()
        if nxt > now:
            time.sleep(nxt - now)
        woke = time.perf_counter()
        late[i] = (woke - nxt) * 1e3
        sess.run(None, feed)
        done = time.perf_counter()
        infer[i] = (done - woke) * 1e3
        if done > nxt + period:
            misses += 1
        nxt += period
        if nxt < done:  # fell behind: resync instead of bursting
            nxt = done + period
    print(f"  inference   : {fmt(infer)}")
    print(f"  wake-up late: {fmt(late)}")
    print(f"  deadline misses: {misses}/{n}  (tick = {period * 1e3:.1f} ms)")
    budget = period * 1e3
    worst = np.percentile(infer + late, 99.9)
    print(f"  p99.9 (late + inference) = {worst:.3f} ms -> {100 * worst / budget:.1f}% of the tick")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="real policy.onnx (default: same-shape dummy)")
    ap.add_argument("--n", type=int, default=5000, help="timed calls per config")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--loop-s", type=float, default=30.0)
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    print(f"host {platform.node()}  {platform.machine()}  cores {os.cpu_count()}")
    print(f"python {platform.python_version()}  onnxruntime {ort.__version__}  numpy {np.__version__}")
    print(f"board before: {board_state()}")

    path = args.model
    if path is None:
        path = "/tmp/microduck_actor_dummy.onnx"
        build_mlp(path, HIDDEN)
        print(f"model: dummy MLP {OBS}->{'->'.join(map(str, HIDDEN))}->{ACT} (ELU, normalizer baked)")
    else:
        print(f"model: {path}")

    print("\n[1] per-call latency")
    best = None
    for th in sorted({1, 2, os.cpu_count() or 1}):
        t = time_calls(session(path, th), args.n)
        print(f"  threads={th}: {fmt(t)}")
        if best is None or np.percentile(t, 99) < best[1]:
            best = (th, np.percentile(t, 99))
    th, p99 = best
    print(f"  best = threads={th}, p99 {p99:.3f} ms -> max rate ~{1000 / p99:.0f} Hz (inference only)")

    print(f"\n[2] real {args.hz:.0f} Hz loop for {args.loop_s:.0f} s (threads={th})")
    run_loop(session(path, th), args.hz, args.loop_s)

    if not args.no_sweep and args.model is None:
        print("\n[3] width sweep (how big could the net get?)")
        tick = 1000 / args.hz
        for k in (1, 2, 4, 8):
            hid = tuple(h * k for h in HIDDEN)
            p = f"/tmp/microduck_actor_x{k}.onnx"
            build_mlp(p, hid)
            t = time_calls(session(p, th), max(500, args.n // 5), warmup=50)
            p99k = np.percentile(t, 99)
            print(f"  x{k} {hid}: p99 {p99k:7.3f} ms = {100 * p99k / tick:5.1f}% of {tick:.0f} ms tick")

    print(f"\nboard after: {board_state()}")


if __name__ == "__main__":
    main()
