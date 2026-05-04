"""
sensor_recorder.py — Per-timestep sensor capture for centipede trials
======================================================================
Drop-in recorder that captures the full per-frame state of the impedance-
controlled centipede and writes a compressed .npz file for downstream
analysis.

Captures (at a configurable decimation rate):

  Time
  ----
    time                 (T,)               simulation time in seconds

  COM & root
  ----------
    com_pos              (T, 3)             world-frame subtree COM
    com_vel              (T, 3)             world-frame subtree linear vel
    root_pos             (T, 3)             world-frame link_body_0 position
    root_quat            (T, 4)             world-frame link_body_0 quat [w,x,y,z]
    heading_rad          (T,)               world yaw of link_body_0 (rad)

  Per-segment (all 21 link_body_N)
  --------------------------------
    seg_pos              (T, 21, 3)         world-frame segment positions
    seg_quat             (T, 21, 4)         world-frame segment quats [w,x,y,z]

  Body yaw joints (N_BODY_JOINTS=19, joint_body_1..19)
  ----------------------------------------------------
    body_yaw_q           (T, 19)            measured joint angle (rad)
    body_yaw_qdot        (T, 19)            measured joint velocity (rad/s)
    body_yaw_target      (T, 19)            commanded q_target from controller
    body_yaw_cmd_torque  (T, 19)            ctrl signal (impedance output)
    body_yaw_act_torque  (T, 19)            actuator_force (what MuJoCo applied)

  Pitch joints (variable count n_pitch)
  -------------------------------------
    pitch_q              (T, n_pitch)
    pitch_qdot           (T, n_pitch)
    pitch_target         (T, n_pitch)
    pitch_cmd_torque     (T, n_pitch)
    pitch_act_torque     (T, n_pitch)
    pitch_jnt_ids        (n_pitch,)         global joint IDs for reference

  Roll joints (variable count n_roll)
  -----------------------------------
    roll_q               (T, n_roll)
    roll_qdot            (T, n_roll)
    roll_target          (T, n_roll)
    roll_cmd_torque      (T, n_roll)
    roll_act_torque      (T, n_roll)
    roll_jnt_ids         (n_roll,)

  Leg joints (19 legs × 2 sides × 4 DOF)
  --------------------------------------
    leg_q                (T, 19, 2, 4)
    leg_qdot             (T, 19, 2, 4)
    leg_target           (T, 19, 2, 4)
    leg_cmd_torque       (T, 19, 2, 4)
    leg_act_torque       (T, 19, 2, 4)

  Foot contact forces (19 × 2)
  ----------------------------
    foot_contact_force   (T, 19, 2, 3)      world-frame linear external force
                                            on each foot body (data.cfrc_ext
                                            rows 3..6 — linear component)
    foot_contact_torque  (T, 19, 2, 3)      world-frame external torque
                                            (data.cfrc_ext rows 0..3 — angular)
    foot_contact_mag     (T, 19, 2)         ‖foot_contact_force‖₂
    foot_in_contact      (T, 19, 2)         bool: mag > contact_threshold

  Terrain
  -------
    terrain_slope        (T,)               slope along heading at COM (rad)

  Metadata (scalars)
  ------------------
    dt                   simulation timestep
    dt_record            recorder output timestep
    contact_threshold    force magnitude treated as "in contact"
    total_mass_kg
    gravity_z
    n_legs, n_body_joints, n_leg_dof

Each recorder holds data in Python lists during the run and stacks them into
contiguous arrays at save() time, so memory growth is linear in frame count.
At 200 Hz × 10 s × 19 legs × 2 sides × 4 DOF the heaviest array (leg_q) is
only ~305 kB uncompressed — fine for sweeps with hundreds of trials.
"""

import math
import os
import numpy as np
import mujoco

from kinematics import (
    FARMSModelIndex,
    N_BODY_JOINTS, N_LEGS, N_LEG_DOF,
)


N_BODY_SEGS = 21  # link_body_0 .. link_body_20


class SensorRecorder:
    """Rich per-timestep sensor recorder for a centipede trial.

    Usage
    -----
        model = mujoco.MjModel.from_xml_path(xml_path)
        data  = mujoco.MjData(model)
        ctrl  = ImpedanceTravelingWaveController(model, cfg_path)

        rec = SensorRecorder(
            model, data, ctrl,
            dt_record=0.005,              # 200 Hz capture
            terrain_sampler=terrain,      # optional — TerrainSampler from sweep
            contact_threshold=0.001,      # 1 mN — anything above is "in contact"
        )

        for step_i in range(n_steps):
            ctrl.step(model, data)
            mujoco.mj_step(model, data)
            rec.maybe_record(model, data, ctrl)

        rec.save("trial.npz")
    """

    def __init__(self, model, data, ctrl,
                 dt_record=0.005,
                 terrain_sampler=None,
                 contact_threshold=1e-3,
                 settle_time=0.0):
        self.model    = model
        self.idx      = ctrl.idx if hasattr(ctrl, "idx") else FARMSModelIndex(model)
        self.dt       = float(model.opt.timestep)
        self.dt_record = float(dt_record)
        self.terrain_sampler   = terrain_sampler
        self.contact_threshold = float(contact_threshold)
        self.settle_time = float(settle_time)

        self._last_t = -np.inf

        # ── resolve body segment body ids (link_body_0 .. link_body_20) ──
        self.seg_body_ids = np.zeros(N_BODY_SEGS, dtype=int)
        for i in range(N_BODY_SEGS):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_body_{i}")
            if bid < 0:
                raise ValueError(f"Body 'link_body_{i}' not found")
            self.seg_body_ids[i] = bid

        # ── resolve foot body ids (foot_{n}_{side}) ──
        self.foot_body_ids = np.full((N_LEGS, 2), -1, dtype=int)
        missing = []
        for n in range(N_LEGS):
            for si in range(2):
                name = f"foot_{n}_{si}"
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid < 0:
                    missing.append(name)
                self.foot_body_ids[n, si] = bid
        if missing:
            print(f"[SensorRecorder] WARNING: {len(missing)} foot bodies missing "
                  f"(first 3: {missing[:3]}). Their contact data will be zeros.")

        # ── resolve body yaw joint addresses ──
        self.body_yaw_qposadr = np.zeros(N_BODY_JOINTS, dtype=int)
        self.body_yaw_dofadr  = np.zeros(N_BODY_JOINTS, dtype=int)
        for i in range(N_BODY_JOINTS):
            jid = int(self.idx.body_jnt_ids[i])
            self.body_yaw_qposadr[i] = model.jnt_qposadr[jid]
            self.body_yaw_dofadr[i]  = model.jnt_dofadr[jid]

        # ── pitch joint addresses ──
        self.has_pitch = getattr(self.idx, "has_pitch_actuators", False)
        if self.has_pitch:
            self.pitch_jnt_ids = np.array(self.idx.pitch_jnt_ids, dtype=int)
            self.pitch_qposadr = np.array(
                [model.jnt_qposadr[j] for j in self.pitch_jnt_ids], dtype=int)
            self.pitch_dofadr = np.array(
                [model.jnt_dofadr[j] for j in self.pitch_jnt_ids], dtype=int)
            self.pitch_act_ids = np.array(self.idx.pitch_act_ids, dtype=int)
            self.n_pitch = len(self.pitch_jnt_ids)
        else:
            self.pitch_jnt_ids = np.zeros(0, dtype=int)
            self.pitch_qposadr = np.zeros(0, dtype=int)
            self.pitch_dofadr  = np.zeros(0, dtype=int)
            self.pitch_act_ids = np.zeros(0, dtype=int)
            self.n_pitch = 0

        # ── roll joint addresses ──
        self.has_roll = getattr(self.idx, "has_roll_actuators", False)
        if self.has_roll:
            self.roll_jnt_ids = np.array(self.idx.roll_jnt_ids, dtype=int)
            self.roll_qposadr = np.array(
                [model.jnt_qposadr[j] for j in self.roll_jnt_ids], dtype=int)
            self.roll_dofadr = np.array(
                [model.jnt_dofadr[j] for j in self.roll_jnt_ids], dtype=int)
            self.roll_act_ids = np.array(self.idx.roll_act_ids, dtype=int)
            self.n_roll = len(self.roll_jnt_ids)
        else:
            self.roll_jnt_ids = np.zeros(0, dtype=int)
            self.roll_qposadr = np.zeros(0, dtype=int)
            self.roll_dofadr  = np.zeros(0, dtype=int)
            self.roll_act_ids = np.zeros(0, dtype=int)
            self.n_roll = 0

        # ── total mass / gravity (for downstream CoT computation) ──
        self.total_mass = float(sum(model.body_mass[i] for i in range(model.nbody)))
        self.gravity_z  = float(abs(model.opt.gravity[2]))

        # ── root body id (first free joint body) ──
        self.root_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")

        # ── storage lists (converted to arrays in save()) ──
        self._reset_buffers()

    def _reset_buffers(self):
        self.times = []
        # com / root
        self.com_pos    = []
        self.com_vel    = []
        self.root_pos   = []
        self.root_quat  = []
        self.heading    = []
        # segments
        self.seg_pos    = []
        self.seg_quat   = []
        # body yaw
        self.body_yaw_q   = []
        self.body_yaw_qd  = []
        self.body_yaw_tgt = []
        self.body_yaw_cmd = []
        self.body_yaw_act = []
        # pitch
        self.pitch_q   = []
        self.pitch_qd  = []
        self.pitch_tgt = []
        self.pitch_cmd = []
        self.pitch_act = []
        # roll
        self.roll_q   = []
        self.roll_qd  = []
        self.roll_tgt = []
        self.roll_cmd = []
        self.roll_act = []
        # legs
        self.leg_q   = []
        self.leg_qd  = []
        self.leg_tgt = []
        self.leg_cmd = []
        self.leg_act = []
        # foot contact
        self.foot_force  = []
        self.foot_torque = []
        # terrain
        self.slope = []

    # ──────────────────────────────────────────────────────────────────────
    # Per-step capture
    # ──────────────────────────────────────────────────────────────────────

    def maybe_record(self, model, data, ctrl):
        """Record this frame only if enough simulation time has passed."""
        if data.time - self._last_t < self.dt_record - 1e-10:
            return
        self._last_t = data.time
        self._record(model, data, ctrl)

    def _record(self, model, data, ctrl):
        t = data.time
        self.times.append(t)

        # ── COM / root ──
        com_p = data.subtree_com[self.root_body_id].copy()
        com_v = data.subtree_linvel[self.root_body_id].copy()
        self.com_pos.append(com_p)
        self.com_vel.append(com_v)

        root_p = data.xpos[self.root_body_id].copy()
        root_q = data.xquat[self.root_body_id].copy()   # [w, x, y, z]
        self.root_pos.append(root_p)
        self.root_quat.append(root_q)
        # Heading = world yaw of root (from rotation matrix)
        R = data.xmat[self.root_body_id].reshape(3, 3)
        heading = math.atan2(R[1, 0], R[0, 0])
        self.heading.append(heading)

        # ── Per-segment pos + quat ──
        sp = np.zeros((N_BODY_SEGS, 3))
        sq = np.zeros((N_BODY_SEGS, 4))
        for i, bid in enumerate(self.seg_body_ids):
            sp[i] = data.xpos[bid]
            sq[i] = data.xquat[bid]
        self.seg_pos.append(sp)
        self.seg_quat.append(sq)

        # ── Body yaw joints ──
        by_q   = data.qpos[self.body_yaw_qposadr].copy()
        by_qd  = data.qvel[self.body_yaw_dofadr].copy()
        by_tgt = ctrl.last_body_yaw_targets.copy()
        by_cmd = np.array([data.ctrl[self.idx.body_act_ids[i]]
                           for i in range(N_BODY_JOINTS)])
        by_act = np.array([data.actuator_force[self.idx.body_act_ids[i]]
                           for i in range(N_BODY_JOINTS)])
        self.body_yaw_q.append(by_q)
        self.body_yaw_qd.append(by_qd)
        self.body_yaw_tgt.append(by_tgt)
        self.body_yaw_cmd.append(by_cmd)
        self.body_yaw_act.append(by_act)

        # ── Pitch joints ──
        if self.n_pitch > 0:
            p_q   = data.qpos[self.pitch_qposadr].copy()
            p_qd  = data.qvel[self.pitch_dofadr].copy()
            p_tgt = ctrl.pitch_targets.copy() if hasattr(ctrl, "pitch_targets") \
                    else np.zeros(self.n_pitch)
            p_cmd = data.ctrl[self.pitch_act_ids].copy()
            p_act = data.actuator_force[self.pitch_act_ids].copy()
        else:
            p_q = p_qd = p_tgt = p_cmd = p_act = np.zeros(0)
        self.pitch_q.append(p_q)
        self.pitch_qd.append(p_qd)
        self.pitch_tgt.append(p_tgt)
        self.pitch_cmd.append(p_cmd)
        self.pitch_act.append(p_act)

        # ── Roll joints ──
        if self.n_roll > 0:
            r_q   = data.qpos[self.roll_qposadr].copy()
            r_qd  = data.qvel[self.roll_dofadr].copy()
            r_tgt = ctrl.last_roll_targets.copy() if hasattr(ctrl, "last_roll_targets") \
                    else np.zeros(self.n_roll)
            r_cmd = data.ctrl[self.roll_act_ids].copy()
            r_act = data.actuator_force[self.roll_act_ids].copy()
        else:
            r_q = r_qd = r_tgt = r_cmd = r_act = np.zeros(0)
        self.roll_q.append(r_q)
        self.roll_qd.append(r_qd)
        self.roll_tgt.append(r_tgt)
        self.roll_cmd.append(r_cmd)
        self.roll_act.append(r_act)

        # ── Leg joints ──
        lq  = np.zeros((N_LEGS, 2, N_LEG_DOF))
        lqd = np.zeros((N_LEGS, 2, N_LEG_DOF))
        lt  = ctrl.last_leg_targets.copy() if hasattr(ctrl, "last_leg_targets") \
              else np.zeros((N_LEGS, 2, N_LEG_DOF))
        lc  = np.zeros((N_LEGS, 2, N_LEG_DOF))
        la  = np.zeros((N_LEGS, 2, N_LEG_DOF))
        for n in range(N_LEGS):
            for si in range(2):
                for dof in range(N_LEG_DOF):
                    jid = int(self.idx.leg_jnt_ids[n, si, dof])
                    aid = int(self.idx.leg_act_ids[n, si, dof])
                    lq[n, si, dof]  = data.qpos[model.jnt_qposadr[jid]]
                    lqd[n, si, dof] = data.qvel[model.jnt_dofadr[jid]]
                    lc[n, si, dof]  = data.ctrl[aid]
                    la[n, si, dof]  = data.actuator_force[aid]
        self.leg_q.append(lq)
        self.leg_qd.append(lqd)
        self.leg_tgt.append(lt)
        self.leg_cmd.append(lc)
        self.leg_act.append(la)

        # ── Foot contact forces ──
        # data.cfrc_ext[body_id] is a 6D vector [torque_x, torque_y, torque_z,
        # force_x, force_y, force_z] of net external spatial force acting on
        # the body, expressed at the COM in world frame.  cfrc_ext is NOT
        # populated by mj_step — we have to call mj_rnePostConstraint first
        # so that contact & constraint forces are accumulated into it.
        mujoco.mj_rnePostConstraint(model, data)
        ff = np.zeros((N_LEGS, 2, 3))
        ft = np.zeros((N_LEGS, 2, 3))
        for n in range(N_LEGS):
            for si in range(2):
                bid = int(self.foot_body_ids[n, si])
                if bid >= 0:
                    c = data.cfrc_ext[bid]
                    ft[n, si] = c[0:3]    # torque
                    ff[n, si] = c[3:6]    # linear force
        self.foot_force.append(ff)
        self.foot_torque.append(ft)

        # ── Terrain slope ──
        if self.terrain_sampler is not None:
            try:
                slope = self.terrain_sampler.get_slope_along(
                    com_p[0], com_p[1], heading)
            except Exception:
                slope = float("nan")
        else:
            slope = float("nan")
        self.slope.append(slope)

    # ──────────────────────────────────────────────────────────────────────
    # Persist to disk
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path):
        """Stack lists into arrays and write a compressed .npz."""
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        # Stack (handles the T=0 case gracefully)
        def stack(lst, shape_if_empty):
            if len(lst) == 0:
                return np.zeros(shape_if_empty, dtype=float)
            return np.array(lst)

        T = len(self.times)
        times = np.array(self.times, dtype=float)

        foot_force  = stack(self.foot_force,  (0, N_LEGS, 2, 3))
        foot_mag    = np.linalg.norm(foot_force, axis=-1) if T > 0 else np.zeros((0, N_LEGS, 2))
        foot_in_contact = (foot_mag > self.contact_threshold) if T > 0 else np.zeros((0, N_LEGS, 2), dtype=bool)

        np.savez_compressed(
            path,
            # Meta
            dt                = np.array([self.dt]),
            dt_record         = np.array([self.dt_record]),
            contact_threshold = np.array([self.contact_threshold]),
            settle_time       = np.array([self.settle_time]),
            total_mass_kg     = np.array([self.total_mass]),
            gravity_z         = np.array([self.gravity_z]),
            n_body_joints     = np.array([N_BODY_JOINTS]),
            n_body_segments   = np.array([N_BODY_SEGS]),
            n_legs            = np.array([N_LEGS]),
            n_leg_dof         = np.array([N_LEG_DOF]),
            # Time
            time              = times,
            # COM / root
            com_pos           = stack(self.com_pos,  (0, 3)),
            com_vel           = stack(self.com_vel,  (0, 3)),
            root_pos          = stack(self.root_pos, (0, 3)),
            root_quat         = stack(self.root_quat,(0, 4)),
            heading_rad       = np.array(self.heading, dtype=float),
            # Segments
            seg_pos           = stack(self.seg_pos,  (0, N_BODY_SEGS, 3)),
            seg_quat          = stack(self.seg_quat, (0, N_BODY_SEGS, 4)),
            # Body yaw
            body_yaw_q        = stack(self.body_yaw_q,   (0, N_BODY_JOINTS)),
            body_yaw_qdot     = stack(self.body_yaw_qd,  (0, N_BODY_JOINTS)),
            body_yaw_target   = stack(self.body_yaw_tgt, (0, N_BODY_JOINTS)),
            body_yaw_cmd_torque = stack(self.body_yaw_cmd, (0, N_BODY_JOINTS)),
            body_yaw_act_torque = stack(self.body_yaw_act, (0, N_BODY_JOINTS)),
            # Pitch
            pitch_q             = stack(self.pitch_q,   (0, self.n_pitch)),
            pitch_qdot          = stack(self.pitch_qd,  (0, self.n_pitch)),
            pitch_target        = stack(self.pitch_tgt, (0, self.n_pitch)),
            pitch_cmd_torque    = stack(self.pitch_cmd, (0, self.n_pitch)),
            pitch_act_torque    = stack(self.pitch_act, (0, self.n_pitch)),
            pitch_jnt_ids       = self.pitch_jnt_ids,
            # Roll
            roll_q              = stack(self.roll_q,   (0, self.n_roll)),
            roll_qdot           = stack(self.roll_qd,  (0, self.n_roll)),
            roll_target         = stack(self.roll_tgt, (0, self.n_roll)),
            roll_cmd_torque     = stack(self.roll_cmd, (0, self.n_roll)),
            roll_act_torque     = stack(self.roll_act, (0, self.n_roll)),
            roll_jnt_ids        = self.roll_jnt_ids,
            # Legs
            leg_q               = stack(self.leg_q,   (0, N_LEGS, 2, N_LEG_DOF)),
            leg_qdot            = stack(self.leg_qd,  (0, N_LEGS, 2, N_LEG_DOF)),
            leg_target          = stack(self.leg_tgt, (0, N_LEGS, 2, N_LEG_DOF)),
            leg_cmd_torque      = stack(self.leg_cmd, (0, N_LEGS, 2, N_LEG_DOF)),
            leg_act_torque      = stack(self.leg_act, (0, N_LEGS, 2, N_LEG_DOF)),
            # Foot contact
            foot_contact_force  = foot_force,
            foot_contact_torque = stack(self.foot_torque, (0, N_LEGS, 2, 3)),
            foot_contact_mag    = foot_mag,
            foot_in_contact     = foot_in_contact,
            # Terrain
            terrain_slope       = np.array(self.slope, dtype=float),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Post-hoc helpers (can be imported by analysis scripts or used interactively)
# ══════════════════════════════════════════════════════════════════════════════

def quat_to_euler(q):
    """Convert a batch of [w, x, y, z] quaternions to (roll, pitch, yaw).

    Accepts array shape (..., 4) and returns (..., 3) in radians, using the
    intrinsic Z-Y-X (yaw-pitch-roll) convention MuJoCo's scene uses.
    """
    q = np.asarray(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # roll (x-axis)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    # pitch (y-axis)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    # yaw (z-axis)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.stack([roll, pitch, yaw], axis=-1)
