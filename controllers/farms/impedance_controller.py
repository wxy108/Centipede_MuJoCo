"""
impedance_controller.py — Impedance-based traveling wave controller
====================================================================
ALL joints use torque-based impedance control:

    Body yaw:   tau = body_kp * (target - q) - body_kv * qdot
    Body pitch: tau = pitch_kp * (0 - q) - pitch_kv * qdot
    Body roll:  tau = roll_kp  * (0 - q) - roll_kv  * qdot
    Leg joints: tau = leg_kp[dof] * (target - q) - leg_kv[dof] * qdot

All actuators must be <general> type in the XML (see add_compliance.py).
Usage: drop-in replacement for FARMSTravelingWaveController.
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

    Body yaw joints:   torque = body_kp*(target - q) - body_kv*qdot
    Body pitch joints: torque = pitch_kp*(0 - q) - pitch_kv*qdot
    Body roll joints:  torque = roll_kp*(0 - q) - roll_kv*qdot
    Leg joints:        torque = leg_kp[dof]*(target - q) - leg_kv[dof]*qdot
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

        # ── settling ──
        self.settle_time = float(imp.get("settle_time", 1.0))  # seconds
        self.ramp_time   = float(imp.get("ramp_time", 1.0))    # seconds to ramp gait in

        # ── leg impedance gains (per-DOF arrays) ──
        leg_imp = imp.get("leg", {})
        self.leg_kp = np.array(leg_imp.get("kp", [0.25, 0.25, 0.25, 0.25]), dtype=float)
        self.leg_kv = np.array(leg_imp.get("kv", [0.05, 0.05, 0.05, 0.05]), dtype=float)

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

        # ── leg joint addresses (for reading q and qdot) ──
        self.leg_jnt_qpos_adr = np.zeros((N_LEGS, 2, N_LEG_DOF), dtype=int)
        self.leg_jnt_dof_adr  = np.zeros((N_LEGS, 2, N_LEG_DOF), dtype=int)
        for n in range(N_LEGS):
            for si in range(2):
                for dof in range(N_LEG_DOF):
                    jid = self.idx.leg_jnt_ids[n, si, dof]
                    self.leg_jnt_qpos_adr[n, si, dof] = model.jnt_qposadr[jid]
                    self.leg_jnt_dof_adr[n, si, dof]  = model.jnt_dofadr[jid]

        print(f"[ImpedanceController] body_kp={self.body_kp:.4f} "
              f"body_kv={self.body_kv:.4f}  "
              f"A={self.body_amp:.3f} f={self.freq:.2f}Hz")
        print(f"[ImpedanceController] leg_kp={self.leg_kp.tolist()} "
              f"leg_kv={self.leg_kv.tolist()}")
        print(f"[ImpedanceController] settle={self.settle_time:.1f}s  "
              f"ramp={self.ramp_time:.1f}s  "
              f"(head→tail sequential: per-seg={self.ramp_time/2:.2f}s, "
              f"stagger={self.ramp_time/(2*max(N_BODY_JOINTS-1,1)):.3f}s)")

    def _spatial_phase(self, i):
        return 2.0 * math.pi * self.n_wave * self.speed * i / max(N_BODY_JOINTS - 1, 1)

    def _seg_blend(self, t, i, n_seg=None):
        """Per-segment gait blend for head→tail sequential activation.

        Segment i ramps from 0 to 1 over a window of length ``per_seg``, with
        each segment's start staggered by ``stagger``:

            per_seg = ramp_time / 2
            stagger = per_seg / (N - 1)
            start_i = settle_time + i * stagger
            end_i   = start_i + per_seg

        Segment 0 (head) starts at t=settle_time and finishes at
        t=settle_time+per_seg.  Segment N-1 (tail) starts at
        t=settle_time+per_seg and finishes exactly at t=settle_time+ramp_time.
        A smooth cosine is used for each segment's individual ramp.
        """
        if n_seg is None:
            n_seg = N_BODY_JOINTS
        if t < self.settle_time:
            return 0.0

        # Degenerate: zero ramp → step activation for everyone at settle_time
        if self.ramp_time <= 0.0:
            return 1.0

        per_seg = self.ramp_time * 0.5
        stagger = per_seg / max(n_seg - 1, 1)

        start_i = self.settle_time + i * stagger
        if t <= start_i:
            return 0.0
        elapsed = t - start_i
        if elapsed >= per_seg:
            return 1.0
        return 0.5 * (1.0 - math.cos(math.pi * elapsed / per_seg))

    def step(self, model, data, t=None):
        if t is None:
            t = data.time

        # ── body yaw: impedance control, head→tail sequential activation ──
        for i in range(N_BODY_JOINTS):
            blend_i = self._seg_blend(t, i)
            if blend_i > 0:
                phase  = self.omega * t - self._spatial_phase(i)
                target = blend_i * self.body_amp * math.sin(phase)
            else:
                target = 0.0

            q    = data.qpos[self.body_jnt_qpos_adr[i]]
            qdot = data.qvel[self.body_jnt_dof_adr[i]]

            torque = self.body_kp * (target - q) - self.body_kv * qdot

            data.ctrl[self.idx.body_act_ids[i]] = torque

        # ── body pitch: pure impedance (no gravity compensation) ──
        # At 2.5g total mass, gravitational pitch torques (~0.001 Nm) are
        # negligible vs ground reaction forces.  The impedance spring alone
        # holds neutral; adding qfrc_bias (either sign) only creates a
        # static pitch offset that fights the controller.
        if self.has_pitch:
            for i in range(len(self.idx.pitch_act_ids)):
                q    = data.qpos[self.pitch_qpos_adr[i]]
                qdot = data.qvel[self.pitch_dof_adr[i]]

                torque = (self.pitch_kp * (0.0 - q)
                          - self.pitch_kv * qdot)

                data.ctrl[self.idx.pitch_act_ids[i]] = torque

        # ── body roll: impedance + online gravity compensation ──
        if self.has_roll:
            for i in range(len(self.idx.roll_act_ids)):
                q    = data.qpos[self.roll_qpos_adr[i]]
                qdot = data.qvel[self.roll_dof_adr[i]]

                # Pure impedance (no gravity compensation, same reasoning as pitch)
                torque = (self.roll_kp * (0.0 - q)
                          - self.roll_kv * qdot)

                data.ctrl[self.idx.roll_act_ids[i]] = torque

        # ── legs: impedance control, ramp follows body segment index n ──
        for n in range(N_LEGS):
            phi_s   = self._spatial_phase(n)
            blend_n = self._seg_blend(t, n, n_seg=N_LEGS)
            for si, side in enumerate(('L', 'R')):
                for dof in range(N_LEG_DOF):
                    act_id = self.idx.leg_act_ids[n, si, dof]
                    if blend_n <= 0 or dof not in self.active_dofs:
                        target = self.leg_dc_offsets[dof]
                    else:
                        phase  = self.omega * t - phi_s + self.leg_phase_offsets[dof]
                        wave   = math.sin(phase)
                        sign   = 1.0 if si == 0 else -1.0
                        target = (blend_n * sign * self.leg_amps[dof] * wave
                                  + self.leg_dc_offsets[dof])

                    q    = data.qpos[self.leg_jnt_qpos_adr[n, si, dof]]
                    qdot = data.qvel[self.leg_jnt_dof_adr[n, si, dof]]
                    torque = self.leg_kp[dof] * (target - q) - self.leg_kv[dof] * qdot
                    data.ctrl[act_id] = torque
