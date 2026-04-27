"""passive_axial_controller.py
==================================================================
A variant controller that adds the Dionne, Giardina & Mahadevan
(2026) passive axial musculature model to our centipede.

This layers an **inter-segmental flexural spring + rotational damper**
on the body yaw joints, exactly as in Eqs. 6-8 of the paper:

    S_i = -k (θ_i - θ_{i-1}) - k (θ_i - θ_{i+1})       [flexural spring]
    D_i = -η (θ̇_i - θ̇_{i-1}) - η (θ̇_i - θ̇_{i+1})     [rotational damper]

The MuJoCo hinge joint angle `σ_j = θ_{j+1} - θ_j` is the *relative*
angle across the joint.  Applying torque τ_j = -k σ_j - η σ̇_j at each
actuator, combined with MuJoCo's automatic action-reaction between the
two segments the joint connects, reproduces S_i + D_i on every segment
exactly (with the free-boundary condition at the head and tail
automatic — there are no joint actuators past the end segments).

The **leg controllers are inherited unchanged** from the parent — hip
yaw and hip pitch continue to follow the CPG-driven sine waves and the
tibia / tarsus remain at their fixed posture targets.  What changes is
only how the body yaw joints are torqued.

Three operating modes selected via the `passive_axial:` YAML section:

  mode: "mixed"     — active CPG-driven body-yaw impedance PLUS passive
                      inter-segmental spring + damper.  This is the
                      paper's "full model" (active + passive bending).

  mode: "passive"   — passive inter-segmental spring + damper ONLY.
                      The CPG-driven body yaw drive is suppressed
                      (ctrl zeroed) except for the head-yaw servo if
                      `keep_head_servo: true`.  This is the paper's
                      ablation test — "does coordination emerge purely
                      from body mechanics?"

  mode: "active"    — equivalent to the parent controller (k = η = 0,
                      passive term not applied). Here for completeness.

Usage
-----
    from passive_axial_controller import PassiveAxialController
    ctrl = PassiveAxialController(model, "configs/farms_controller_passive.yaml")

And in your run script, swap the import from
`ImpedanceTravelingWaveController` to `PassiveAxialController`.
"""

import numpy as np
import yaml

from impedance_controller import ImpedanceTravelingWaveController
from kinematics import N_BODY_JOINTS


class PassiveAxialController(ImpedanceTravelingWaveController):
    """Layers a passive inter-segmental flexural spring + damper on the
    body yaw joints of the parent impedance controller.

    Reads an optional `passive_axial:` YAML section:

        passive_axial:
          mode:             "mixed"    # "mixed" | "passive" | "active"
          k:                0.5        # flexural stiffness  (Nm/rad)
          eta:              0.05       # rotational damping  (Nm·s/rad)
          keep_head_servo:  true       # keep active yaw-servo on joint 0
                                       # when mode == "passive"
    """

    def __init__(self, model, config_path, **kwargs):
        super().__init__(model, config_path, **kwargs)

        # utf-8 is explicit so Windows non-UTF-8 locales (e.g. gbk) don't
        # choke on the Greek letters / unicode arrows in the YAML comments.
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        pa = cfg.get("passive_axial", {}) or {}

        self.axial_mode     = str(pa.get("mode", "mixed")).lower()
        self.axial_k        = float(pa.get("k",   0.0))
        self.axial_eta      = float(pa.get("eta", 0.0))
        self.axial_keep_head= bool(pa.get("keep_head_servo", True))

        if self.axial_mode not in ("mixed", "passive", "active"):
            raise ValueError(
                f"passive_axial.mode must be 'mixed' | 'passive' | 'active', "
                f"got '{self.axial_mode}'")

        # Cache joint addresses as numpy arrays for vectorised read
        self._body_qpos_adr = np.asarray(self.body_jnt_qpos_adr, dtype=np.int64)
        self._body_dof_adr  = np.asarray(self.body_jnt_dof_adr,  dtype=np.int64)
        self._body_act_ids  = np.asarray(self.idx.body_act_ids,  dtype=np.int64)

        print(f"[PassiveAxial] mode={self.axial_mode}  "
              f"k={self.axial_k:.4f}  eta={self.axial_eta:.6f}  "
              f"keep_head_servo={self.axial_keep_head}")

    # ------------------------------------------------------------------ #
    # step                                                               #
    # ------------------------------------------------------------------ #
    def step(self, model, data, t=None):
        # Parent step writes the full active CPG-driven body-yaw torques
        # into data.ctrl[body_act_ids], plus legs / pitch / roll.
        super().step(model, data, t)

        if self.axial_mode == "active":
            return                      # nothing to add

        # Read body yaw angles + velocities from MuJoCo state
        q  = data.qpos[self._body_qpos_adr]   # (N_BODY_JOINTS,) — σ_j
        qd = data.qvel[self._body_dof_adr]    # (N_BODY_JOINTS,) — σ̇_j

        # σ_j = θ_{j+1} - θ_j  (MuJoCo hinge angle is relative).
        # Applying τ_j = -k σ_j at each joint reproduces the paper's
        # S_i on every segment via MuJoCo action-reaction.  Same for
        # damping.  Free boundary is automatic — segments at the head
        # and tail have only ONE adjacent joint.
        tau_passive = -self.axial_k * q - self.axial_eta * qd

        if self.axial_mode == "passive":
            # Replace the CPG-driven active torques entirely; optionally
            # keep the head-yaw servo (joint 0) for trajectory control.
            if self.axial_keep_head:
                # Keep joint 0's active torque, overwrite joints 1..N-1
                tau_passive_masked = tau_passive.copy()
                tau_passive_masked[0] = data.ctrl[self._body_act_ids[0]]
                data.ctrl[self._body_act_ids] = tau_passive_masked
            else:
                data.ctrl[self._body_act_ids] = tau_passive

        else:   # "mixed"
            # Add the passive torque on top of the active CPG-driven drive.
            data.ctrl[self._body_act_ids] += tau_passive
