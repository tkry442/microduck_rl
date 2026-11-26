"""BAM actuator definition for the Dynamixel XC330-T288 (registered as ``"xc330"``).

BAM ships identified models for the XL330-M288 but not for the XC330. The XC330
uses the same X330 firmware and control table as the XL330 (same 4096-count
encoder, same ``Present PWM`` limit of 885, same Position P gain register), so
the firmware control law — ``duty = error_gain * kp * Δq`` — is the XL330's.
What differs is the motor and gearbox: the T288 runs at 11.1 V with roughly
4–5x the winding resistance and ~4x the back-EMF constant of the XL330 as BAM
fits it. BAM's ``XL330Actuator`` bounds ``R`` to [2, 5] Ω and ``kt`` to
[0.1, 3], so fitting XC330 logs as ``"xl330"`` would pin ``R`` at its upper
bound and push the error into the friction terms. This class only widens
those search ranges and sets the nominal voltage.

Importing this module registers ``"xc330"`` in ``bam.actuators.actuators`` —
the table ``bam.model.load_model`` and ``bam.fit`` look the JSON's
``"actuator"`` key up in. ``mjlab_microduck.actuator`` imports it, so any env
cfg that uses this package can load a fitted ``xc330`` JSON via
``FrictionDRBamActuatorCfg(json_path=...)``; ``scripts/sysid/bam_xc330.py``
imports it before handing over to BAM's own ``fit`` / ``process`` / ``plot``.

Catalogue reference (ROBOTIS e-manual, XC330-T288-T at 11.1 V): stall torque
0.92 N·m, no-load speed 65 rpm (6.8 rad/s). In BAM's DC-motor model
(output-side units) that is ``kt ≈ V / ω0 ≈ 1.63`` and
``R ≈ kt · V / τ_stall ≈ 19.7`` — the values the hand-made ``xc330_est.json``
used. The fitted values replace them.
"""

from __future__ import annotations

import bam.actuators
from bam.actuator import VoltageControlledActuator
from bam.dynamixel.actuator import (
    XL330_ENCODER_COUNTS_PER_REV,
    XL330_KP_DIVISOR,
    XL330_PWM_LIMIT,
)
from bam.parameter import Parameter
from bam.testbench import Pendulum

XC330_NOMINAL_VIN = 11.1
XC330_DEFAULT_KP = 400


class XC330Actuator(VoltageControlledActuator):
    """Dynamixel XC330-T288, voltage-controlled (position mode, X330 firmware)."""

    def __init__(self, testbench_class):
        super().__init__(
            testbench_class,
            vin=XC330_NOMINAL_VIN,
            kp=XC330_DEFAULT_KP,
            error_gain=(XL330_ENCODER_COUNTS_PER_REV / (2 * 3.141592653589793))
            / (XL330_KP_DIVISOR * XL330_PWM_LIMIT),
            max_pwm=1.0,
            # No firmware current limit modelled: at R ≈ 20 Ω the stall current
            # at 12.4 V is ~0.6 A, well under the XC330's current limit, so the
            # limiter never binds. Revisit if a fit lands R far below 10 Ω.
            max_current=None,
        )

    def initialize(self):
        # Start at the catalogue estimate, with ranges wide enough that neither
        # bound is what the optimizer ends up reporting.
        self.model.kt = Parameter(1.63, 0.3, 5.0)
        self.model.R = Parameter(19.7, 3.0, 60.0)
        self.model.armature = Parameter(0.005, 0.0001, 0.05)

    def get_extra_inertia(self) -> float:
        return self.model.armature.value


def register() -> None:
    """Add ``"xc330"`` to BAM's actuator table (idempotent)."""
    bam.actuators.actuators.setdefault("xc330", lambda: XC330Actuator(Pendulum))


register()
