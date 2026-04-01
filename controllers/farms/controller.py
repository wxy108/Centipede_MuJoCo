"""
farms_controller.py — Traveling wave controller for the FARMS centipede model
==============================================================================
Implements the same feedforward sinusoidal traveling wave used in the FARMS
Base_exp animat.yaml, adapted for MuJoCo position actuators.

Wave equations (from FARMS compute_body_wave / compute_leg_wave):

  Body joint i  :  θ = A_body · sin(ω·t − φ_s(i))
  Leg n, DOF d  :  θ = A[d] · sin(ω·t − φ_s(n) + offset[d]) + dc[d]
  Right side    :  θ = −A[d] · sin(ω·t − φ_s(n) + offset[d]) + dc[d]

  φ_s(i) = 2π · n_wave · speed · i / (N_BODY_JOINTS − 1)

All parameters are read from farms_config.yaml at construction time.
"""

import math
import numpy as np
import yaml

from kinematics import (
    FARMSModelIndex,
    N_BODY_JOINTS, N_LEGS, N_LEG_DOF, ACTIVE_DOFS,
)


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(path="farms_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Controller ────────────────────────────────────────────────────────────────

class FARMSTravelingWaveController:
    """
    Feedforward sinusoidal traveling wave controller for the FARMS centipede.

    Matches the FARMS Base_exp open-loop wave structure exactly:
      - Body: yaw traveling wave
      - Legs: per-DOF sinusoidal wave with phase offsets and L/R sign flip
      - Passive pitch joints: not actuated (spring-damper in XML handles them)

    Parameters are loaded from farms_config.yaml.
    """

    def __init__(self, model, config_path="farms_config.yaml"):
        cfg = load_config(config_path)

        # ── body wave params ──────────────────────────────────────────────
        bw = cfg["body_wave"]
        self.body_amp  = float(bw["amplitude"])
        self.freq      = float(bw["frequency"])
        self.n_wave    = float(bw["wave_number"])
        self.speed     = float(bw["speed"])
        self.omega     = 2.0 * math.pi * self.freq   # angular frequency rad/s

        # ── leg wave params ───────────────────────────────────────────────
        lw = cfg["leg_wave"]
        self.leg_amps         = np.array(lw["amplitudes"],    dtype=float)  # (4,)
        self.leg_phase_offsets= np.array(lw["phase_offsets"], dtype=float)  # (4,)
        self.leg_dc_offsets   = np.array(lw["dc_offsets"],    dtype=float)  # (4,)
        self.active_dofs      = set(lw["active_dofs"])

        # ── index resolution ──────────────────────────────────────────────
        self.idx = FARMSModelIndex(model)

        print(f"[FARMSTravelingWaveController] "
              f"A_body={self.body_amp:.3f} rad  "
              f"f={self.freq:.2f} Hz  "
              f"n_wave={self.n_wave:.1f}  "
              f"active_dofs={sorted(self.active_dofs)}")

    # ── internal ──────────────────────────────────────────────────────────────

    def _spatial_phase(self, i):
        """
        φ_s(i) = 2π · n_wave · speed · i / (N_BODY_JOINTS − 1)
        i is 0-indexed (0 = most anterior active joint).
        """
        return 2.0 * math.pi * self.n_wave * self.speed * i / max(N_BODY_JOINTS - 1, 1)

    # ── main step ─────────────────────────────────────────────────────────────

    def step(self, model, data, t=None):
        """
        Compute and write target positions to data.ctrl.

        Call once per simulation step (or at your desired control frequency).
        t defaults to data.time if not supplied.
        """
        if t is None:
            t = data.time

        # ── body yaw joints (joint_body_1 .. joint_body_19) ──────────────
        # Index i in 0..18 maps to joint_body_{i+1}
        # φ_s uses the same i so the wave propagates anterior→posterior
        for i in range(N_BODY_JOINTS):
            phase = self.omega * t - self._spatial_phase(i)
            target = self.body_amp * math.sin(phase)
            data.ctrl[self.idx.body_act_ids[i]] = target

        # ── leg joints ────────────────────────────────────────────────────
        # Leg n=0 is attached to link_body_1 (most anterior leg).
        # Use the same spatial phase index as the corresponding body joint.
        for n in range(N_LEGS):
            phi_s = self._spatial_phase(n)

            for si, side in enumerate(('L', 'R')):
                for dof in range(N_LEG_DOF):
                    act_id = self.idx.leg_act_ids[n, si, dof]

                    if dof not in self.active_dofs:
                        # Inactive DOF: hold at DC offset (zero by default)
                        data.ctrl[act_id] = self.leg_dc_offsets[dof]
                        continue

                    # Base phase: body phase + per-DOF offset
                    phase = self.omega * t - phi_s + self.leg_phase_offsets[dof]
                    wave  = math.sin(phase)

                    # Left: +amplitude, Right: −amplitude (FARMS convention)
                    sign   = 1.0 if si == 0 else -1.0
                    target = sign * self.leg_amps[dof] * wave + self.leg_dc_offsets[dof]
                    data.ctrl[act_id] = target

        # ── foot joints ───────────────────────────────────────────────────
        # Foot joints are actuated in the XML but FARMS has no explicit foot
        # control — hold at zero (passive contact geometry handles ground contact)
        # If foot actuators are present, they were given leg_dof3 gains which
        # are very small — zero target is fine.
