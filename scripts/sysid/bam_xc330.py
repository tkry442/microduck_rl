#!/usr/bin/env python3
"""Run BAM's identification tools with the XC330-T288 actuator registered.

BAM's ``process`` / ``fit`` / ``plot`` / ``mae`` look the actuator name up in
``bam.actuators.actuators``, which has no ``"xc330"``. This wrapper registers
it (``src/mjlab_microduck/actuator/xc330.py``) and then runs the BAM module
exactly as ``python -m bam.<tool>`` would, with the remaining arguments.

The XC330 module is loaded by file path so the wrapper does not import
``mjlab_microduck`` (and with it torch / mjlab / warp) just to fit a model.

Examples (run on the PC, from the microduck_rl checkout)::

    # 1) resample raw logs (from record_xc330.py) to a fixed 5 ms step
    uv run python scripts/sysid/bam_xc330.py process --raw sysid_data/raw --logdir sysid_data/processed --dt 0.005

    # 2) fit model m6, holding out the kp=300 logs for validation
    uv run --with optuna --with cmaes --with wandb \\
        python scripts/sysid/bam_xc330.py fit --actuator xc330 --model m6 \\
        --logdir sysid_data/processed --validation_kp 300 --trials 20000 \\
        --output sysid_data/xc330_m6.json

    # 3) compare real vs simulated traces
    uv run python scripts/sysid/bam_xc330.py plot --actuator xc330 \\
        --logdir sysid_data/processed --params sysid_data/xc330_m6.json --sim
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

TOOLS = ("process", "fit", "plot", "mae", "jitter")
XC330_MODULE = (
    Path(__file__).resolve().parents[2] / "src" / "mjlab_microduck" / "actuator" / "xc330.py"
)


def register_xc330() -> None:
    spec = importlib.util.spec_from_file_location("_microduck_bam_xc330", XC330_MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # registers "xc330" as an import side effect


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in TOOLS:
        print(__doc__)
        print(f"usage: bam_xc330.py {{{','.join(TOOLS)}}} [BAM arguments...]")
        raise SystemExit(2)
    tool, rest = argv[0], argv[1:]
    register_xc330()
    # bam.process / bam.fit write into these but do not create them.
    for flag in ("--logdir", "--output"):
        if tool in ("process", "fit") and flag in rest:
            target = Path(rest[rest.index(flag) + 1])
            (target if flag == "--logdir" and tool == "process" else target.parent).mkdir(
                parents=True, exist_ok=True
            )
    sys.argv = [f"bam.{tool}", *rest]
    runpy.run_module(f"bam.{tool}", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
