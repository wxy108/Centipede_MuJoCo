"""
impedance_controller.py — Impedance-based traveling wave controller
====================================================================
Torque-based impedance control for body yaw AND pitch joints:

    Body yaw:   tau = kp * (q_target - q) - kv * q_dot
    Body pitch: tau = pitch_kp * (0 - q) - pitch_kv * q_dot + gravity_comp

The yaw impedance makes the body compliant laterally: it follows the
traveling wave command but yields to external forces.

The pitch impedance with gravity compensation keeps the body straight
against gravity while remaining soft enough to conform to terrain.
Gravity compensation is computed per-joint based on subtree mass and
geometry — it exactly counteracts the static gravitational torque,
allowing very low pitch_kp for terrain compliance.

Leg joints remain position-actuated (their gains are already low).

Usage: drop-in replacement for FARMSTravelingWaveController after
the XML has been patched with add_compliance.py --add-pitch-actuators.
"""

import math
import numpy as np
import yaml
import mujoco

from kinematics import (
    FARMSModelIndex,
    N_BODY_JOINTS, N_LEGS, N_LEG_DOF, ACTIVE_DOFS,
)


def load_config(path="farms_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ImpedanceTravelingWaveController:
    """
    Impedance-based traveling wave controller.

    Body yaw joints:   torque = kp*(target - q) - kv*qdot
    Body pitch joints: torque = pitch_kp*(0 - q) - pitch_kv*qdot + grav_comp
    Body roll joints:  torque = roll_kp*(0 - q) - roll_kv*qdot + grav_comp
    Leg joints:        position control (unchanged)
    """

    def __init__(self, model, config_path="farms_config.yaml",
                 body_kp=None, body_kv=None,
                 pitch_kp=None, pitch_kv=None,
                 roll_kp=None, roll_kv=None):
        cfg = load_config(config_path)

        # ── body wave params ──
        bw = cfg["body_wave"]
        self.body_amp = float(bw["amplitude"])
        self.freq     = float(bw["frequency"])
        self.n_wave   = float(bw["wave_number"])
        self.speed    = float(bw["speed"])
        self.omega    = 2.0 * math.pi * self.freq

        # ── yaw impedance gains ──
        imp = cfg.get("impedance", {})
        self.body_kp = body_kp if body_kp is not None else float(imp.get("body_kp", 5.0))
        self.body_kv = body_kv if body_kv is not None else float(imp.get("body_kv", 0.05))

        # ── pitch impedance gains ──
        self.pitch_kp = pitch_kp if pitch_kp is not None else float(imp.get("pitch_kp", 0.005))
        self.pitch_kv = pitch_kv if pitch_kv is not None else float(imp.get("pitch_kv", 0.002))

        # ── roll impedance gains ──
        self.roll_kp = roll_kp if roll_kp is not None else float(imp.get("roll_kp", 0.005))
        self.roll_kv = roll_kv if roll_kv is not None else float(imp.get("roll_kv", 0.002))

        # ── leg wave params ──
        lw = cfg["leg_wave"]
        self.leg_amps          = np.array(lw["amplitudes"], dtype=float)
        self.leg_phase_offsets = np.array(lw["phase_offsets"], dtype=float)
        self.leg_dc_offsets    = np.array(lw["dc_offsets"], dtype=float)
        self.active_dofs       = set(lw["active_dofs"])

        # ── index resolution ──
        self.idx = FARMSModelIndex(model)

        # ── resolve body yaw joint IDs for reading q and qdot ──
        self.body_jnt_qpos_adr = []
        self.body_jnt_dof_adr  = []
        for i in range(N_BODY_JOINTS):
            jname = f"joint_body_{i+1}"
            jid = None
            for j in range(model.njnt):
                nm = model.joint(j).name
                if nm == jname:
                    jid = j
                    break
            if jid is None:
                raise ValueError(f"Joint {jname} not found in model")
            self.body_jnt_qpos_adr.append(model.jnt_qposadr[jid])
            self.body_jnt_dof_adr.append(model.jnt_dofadr[jid])

        # ── pitch joint addresses and gravity compensation ──
        self.has_pitch = self.idx.has_pitch_actuators
        if self.has_pitch:
            n_pitch = len(self.idx.pitch_jnt_ids)
            self.pitch_qpos_adr = np.array(
                [model.jnt_qposadr[j] for j in self.idx.pitch_jnt_ids], dtype=int)
            self.pitch_dof_adr = np.array(
                [model.jnt_dofadr[j] for j in self.idx.pitch_jnt_ids], dtype=int)

            # Gravity compensation is computed ONLINE each step using
            # data.qfrc_bias (gravity + Coriolis at current configuration).
            # This handles the fact that gravity torques on pitch joints
            # depend on the current body pose (not just q=0).

            print(f"[ImpedanceController] pitch_kp={self.pitch_kp:.4f} "
                  f"pitch_kv={self.pitch_kv:.4f}  "
                  f"({n_pitch} pitch joints, online grav_comp)")
        else:
            print(f"[ImpedanceController] No pitch actuators found — pitch is passive")

        # ── roll joint addresses and gravity compensation ──
        self.has_roll = self.idx.has_roll_actuators
        if self.has_roll:
            n_roll = len(self.idx.roll_jnt_ids)
            self.roll_qpos_adr = np.array(
                [model.jnt_qposadr[j] for j in self.idx.roll_jnt_ids], dtype=int)
            self.roll_dof_adr = np.array(
                [model.jnt_dofadr[j] for j in self.idx.roll_jnt_ids], dtype=int)

            print(f"[ImpedanceController] roll_kp={self.roll_kp:.4f} "
                  f"roll_kv={self.roll_kv:.4f}  "
                  f"({n_roll} roll joints, online grav_comp)")
        else:
            print(f"[ImpedanceController] No roll actuators found — roll is passive")

        print(f"[ImpedanceController] body_kp={self.body_kp:.2f} "
              f"body_kv={self.body_kv:.4f}  "
              f"A={self.body_amp:.3f} f={self.freq:.2f}Hz")

    def _spatial_phase(self, i):
        return 2.0 * math.pi * self.n_wave * self.speed * i / max(N_BODY_JOINTS - 1, 1)

    def step(self, model, data, t=None):
        if t is None:
            t = data.time

        # ── body yaw: impedance control ──
        for i in range(N_BODY_JOINTS):
            phase  = self.omega * t - self._spatial_phase(i)
            target = self.body_amp * math.sin(phase)

            q    = data.qpos[self.body_jnt_qpos_adr[i]]
            qdot = data.qvel[self.body_jnt_dof_adr[i]]

            # Impedance torque: spring toward target + viscous damping
            torque = self.body_kp * (target - q) - self.body_kv * qdot

            data.ctrl[self.idx.body_act_ids[i]] = torque

        # ── body pitch: impedance + online gravity compensation ──
        if self.has_pitch:
            for i in range(len(self.idx.pitch_act_ids)):
                q    = data.qpos[self.pitch_qpos_adr[i]]
                qdot = data.qvel[self.pitch_dof_adr[i]]

                # Online gravity compensation: qfrc_bias contains gravity +
                # Coriolis forces at the current configuration. This exactly
                # counteracts gravitational sag, allowing very soft pitch_kp.
                grav_bias = data.qfrc_bias[self.pitch_dof_adr[i]]

                # Impedance: soft spring toward 0 (straight) + damping + grav comp
                torque = (self.pitch_kp * (0.0 - q)
                          - self.pitch_kv * qdot
                          + grav_bias)

                data.ctrl[self.idx.pitch_act_ids[i]] = torque

        # ── body roll: impedance + online gravity compensation ──
        if self.has_roll:
            for i in range(len(self.idx.roll_act_ids)):
                q    = data.qpos[self.roll_qpos_adr[i]]
                qdot = data.qvel[self.roll_dof_adr[i]]

                # Online gravity compensation (same approach as pitch)
                grav_bias = data.qfrc_bias[self.roll_dof_adr[i]]

                # Impedance: soft spring toward 0 (no roll) + damping + grav comp
                torque = (self.roll_kp * (0.0 - q)
                          - self.roll_kv * qdot
                          + grav_bias)

                data.ctrl[self.idx.roll_act_ids[i]] = torque

        # ── legs: position control (unchanged) ──
        for n in range(N_LEGS):
            phi_s = self._spatial_phase(n)
            for si, side in enumerate(('L', 'R')):
                for dof in range(N_LEG_DOF):
                    act_id = self.idx.leg_act_ids[n, si, dof]
                    if dof not in self.active_dofs:
                        data.ctrl[act_id] = self.leg_dc_offsets[dof]
                        continue
                    phase  = self.omega * t - phi_s + self.leg_phase_offsets[dof]
                    wave   = math.sin(phase)
                    sign   = 1.0 if si == 0 else -1.0
                    target = sign * self.leg_amps[dof] * wave + self.leg_dc_offsets[dof]
                    data.ctrl[act_id] = target
