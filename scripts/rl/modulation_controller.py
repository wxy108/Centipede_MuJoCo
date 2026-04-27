"""
modulation_controller.py
========================
Subclass of ImpedanceTravelingWaveController that accepts an external
"modulation action" — a per-segment phase nudge and amplitude scale on
the body yaw joints.  The CPG keeps providing the underlying rhythm;
the RL policy nudges it locally.

Action layout (numpy array, shape (n_action,)):
    Indices 0..n_seg-1:        phase nudge in [-1, 1]    -> Δφ_i ∈ [-π/4, π/4]
    Indices n_seg..2*n_seg-1:  amp scale in [-1, 1]      -> ε_i ∈ [0.5, 1.5]

Where n_seg is the number of MODULATED body yaw segments (default 18 —
all body yaw joints except joint_body_1, which stays under the head
heading servo).

Modulated body yaw target on segment i (i ∈ {1, ..., 18}):
    target_i(t) = blend_i · ε_i · A_envelope_i · A_base · sin(φ_i + Δφ_i)

Legs and pitch/roll/CPG-leg remain inherited unchanged from the parent.
"""

import math
import numpy as np

from impedance_controller import ImpedanceTravelingWaveController
from kinematics import N_BODY_JOINTS


# Action scaling (matches what the env exposes to the policy)
PHASE_NUDGE_MAX_RAD = math.pi / 4.0       # ±45°
AMP_SCALE_LO        = 0.5
AMP_SCALE_HI        = 1.5

# Segment 0 (joint_body_1) is the heading servo — never modulated
MODULATED_START = 1
MODULATED_END   = N_BODY_JOINTS           # exclusive
N_MOD_SEG       = MODULATED_END - MODULATED_START   # 18 segments
ACTION_DIM      = 2 * N_MOD_SEG                     # 36


class ModulationController(ImpedanceTravelingWaveController):
    """ImpedanceTravelingWaveController + per-segment CPG modulation."""

    ACTION_DIM = ACTION_DIM   # exposed for env construction
    N_MOD_SEG  = N_MOD_SEG

    def __init__(self, model, config_path, **kwargs):
        super().__init__(model, config_path, **kwargs)
        self._phase_nudge = np.zeros(N_BODY_JOINTS, dtype=float)
        self._amp_scale   = np.ones(N_BODY_JOINTS, dtype=float)

    # ------------------------------------------------------------------ #
    # External action interface — env calls set_action() before step()   #
    # ------------------------------------------------------------------ #
    def set_action(self, action):
        """`action`: numpy array of shape (ACTION_DIM,) with values in [-1, 1]."""
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size != ACTION_DIM:
            raise ValueError(f"action size {a.size} != expected {ACTION_DIM}")

        # Saturate
        a = np.clip(a, -1.0, 1.0)
        # Phase nudge
        self._phase_nudge[:] = 0.0
        self._phase_nudge[MODULATED_START:MODULATED_END] = (
            a[:N_MOD_SEG] * PHASE_NUDGE_MAX_RAD)
        # Amplitude scale (linearly mapped from [-1, 1] -> [LO, HI])
        self._amp_scale[:] = 1.0
        self._amp_scale[MODULATED_START:MODULATED_END] = (
            0.5 * (AMP_SCALE_LO + AMP_SCALE_HI)
            + 0.5 * (AMP_SCALE_HI - AMP_SCALE_LO) * a[N_MOD_SEG:])

    def reset_action(self):
        """Restore unmodulated CPG behaviour (used at episode start)."""
        self._phase_nudge[:] = 0.0
        self._amp_scale[:]   = 1.0

    # ------------------------------------------------------------------ #
    # Step — let parent compute everything, then re-write body yaw torques
    # ------------------------------------------------------------------ #
    def step(self, model, data, t=None):
        super().step(model, data, t)         # populates ctrl for legs/pitch/roll/yaw

        if t is None:
            t = data.time

        # Re-write body yaw ctrl with modulated targets, but only for
        # segments [MODULATED_START, MODULATED_END).  Segment 0 (head
        # heading servo) keeps the parent's value.
        for i in range(MODULATED_START, MODULATED_END):
            blend_i = self._seg_blend(t, i)
            if blend_i <= 0:
                target = 0.0
            else:
                if self.use_cpg and self._cpg_initialized:
                    phase = self.body_phases[i]
                else:
                    phase = self.omega * t - self._spatial_phase(i)
                amp_modded = (self._amp_scale[i]
                              * self.body_amp_scale[i]
                              * self.body_amp)
                target = blend_i * amp_modded * math.sin(
                    phase + self._phase_nudge[i])

            q    = data.qpos[self.body_jnt_qpos_adr[i]]
            qdot = data.qvel[self.body_jnt_dof_adr[i]]
            torque = (self.body_kp_vec[i] * (target - q)
                      - self.body_kv_vec[i] * qdot)
            data.ctrl[self.idx.body_act_ids[i]] = torque
            self.last_body_yaw_targets[i] = target
