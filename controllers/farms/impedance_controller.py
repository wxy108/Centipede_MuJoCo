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

        # ── head-end spatial amplitude envelope (body yaw only) ──
        # Shape: [hold zone] → [cosine taper] → [full wave]
        #   indices 0 .. head_hold_joints-1              → amplitude 0 (rigid)
        #   next head_ramp_joints joints                 → cosine from
        #                                                  head_ramp_min up to 1
        #   remaining joints                             → amplitude 1
        # Example with head_hold_joints=2, head_ramp_joints=2, head_ramp_min=0:
        #     i=0,1 → 0.00   (fully held straight)
        #     i=2   → 0.25   (taper)
        #     i=3   → 0.75   (taper)
        #     i≥4   → 1.00   (full wave)
        self.head_hold_joints = int(bw.get("head_hold_joints", 0))
        self.head_ramp_joints = int(bw.get("head_ramp_joints", 2))
        self.head_ramp_min    = float(bw.get("head_ramp_min", 0.0))
        self._build_body_amp_scale()

        # ── CPG oscillator network (Kuramoto-style chain) ───────────────────
        # Each body-yaw joint owns a phase φ_i integrated locally with
        # nearest-neighbor coupling.  The target becomes sin(φ_i) rather than
        # sin(ω·t − k·s_i), so a perturbation at the head propagates down the
        # chain one segment at a time instead of being absorbed by a global
        # clock.  Each leg has its own phase coupled to the body segment above
        # it (segment n → leg n), so legs "follow" their body segment.
        cpg_cfg = cfg.get("cpg", {})
        self.use_cpg = bool(cpg_cfg.get("enabled", True))
        # Negative value → restoring coupling that pulls φ_i toward its
        # preferred neighbor relation.  |Ω| small → slow relaxation, visible
        # follow-the-leader behavior.  FARMS uses −20.
        self.cpg_omega        = float(cpg_cfg.get("coupling_omega", -5.0))
        self.cpg_leg_omega    = float(cpg_cfg.get("leg_coupling_omega",
                                                  self.cpg_omega))
        # Desired inter-segment phase lag: 2π · wave_number / (N − 1)
        self.cpg_delta = (2.0 * math.pi * self.n_wave * self.speed
                          / max(N_BODY_JOINTS - 1, 1))
        # Phase states (initialized lazily on first step so they match the
        # traveling wave at t=settle_time).
        self.body_phases = None    # shape (N_BODY_JOINTS,)
        self.leg_phases  = None    # shape (N_LEGS,)
        self._cpg_initialized = False

        # ── yaw impedance gains ──
        imp = cfg.get("impedance", {})
        self.body_kp = body_kp if body_kp is not None else float(imp.get("body_kp", 5.0))
        self.body_kv = body_kv if body_kv is not None else float(imp.get("body_kv", 0.05))

        # Per-joint kp: held head joints get a stiffer kp so they behave like a
        # rigid nose cone pulled by the rest of the body's momentum, rather
        # than a compliant link that wags under the traveling wave.  The
        # number of stiffened joints matches head_hold_joints; joints beyond
        # that use the standard body_kp.
        self.body_kp_head = float(imp.get("body_kp_head", self.body_kp))
        self.body_kv_head = float(imp.get("body_kv_head", self.body_kv))
        self._build_body_kp_vec()

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

        # ── head heading servo (world-space yaw tracking) ──
        # Instead of driving joint_body_1 with the travelling-wave sin, we use
        # it as a heading servo: drive its impedance target to counter-rotate
        # link_body_1 so the reaction torque on the head (link_body_0) steers
        # the head back toward its INITIAL world yaw (the "forward" direction,
        # whatever it is in the XML — the FARMS MJCF uses a non-identity
        # initial rotation).  The rest of the chain entrains to the head
        # through the CPG.
        self.head_yaw_gain = float(imp.get("head_yaw_gain", 1.0))
        # Clamp on the servo's q_target so it never asks joint_body_1 to go
        # past its hinge range (±0.628 rad in the FARMS model); without a
        # clamp, a large transient error would saturate against the joint
        # limit, wasting servo authority into the constraint solver.
        self.head_yaw_target_clip = float(imp.get("head_yaw_target_clip", 0.5))
        # Reference yaw rate (rad/s): if nonzero, head_yaw_ref advances with
        # time after settle.  This converts the "straight-line" heading servo
        # into a "circle" trajectory — the robot steers along a constantly
        # rotating heading.  omega = v/R for radius R at forward speed v.
        # Negative → clockwise (yaw decreasing), positive → counter-clockwise.
        self.head_yaw_rate = float(imp.get("head_yaw_rate", 0.0))
        self.head_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")
        if self.head_body_id < 0:
            raise ValueError("Body 'link_body_0' not found — heading servo disabled")
        # Reference yaw latched on the first call to step() (not at __init__
        # time — the freejoint's xmat is only valid after mj_forward/mj_step).
        # When head_yaw_rate != 0, this reference is advanced each step after
        # the gait has switched on (t >= settle_time).
        self.head_yaw_ref = None

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

            # ── Head-end pitch offset (keep the nose up off the terrain) ──
            # Parse the segment index from each pitch joint's name
            # ("joint_pitch_body_N") so we know which joints are head-end.
            # The first `head_pitch_joints` (in segment order) get a
            # non-zero target of `head_pitch_offset` (rad); the rest stay
            # at 0.  Positive offset here pitches the nose up in the model
            # convention used by the FARMS MJCF (hinge axis pointing so that
            # +q rotates the nose toward +z).  Flip the sign via the config
            # if your terrain collision suggests otherwise.
            self.head_pitch_offset = float(imp.get("head_pitch_offset", 0.0))
            self.head_pitch_joints = int(imp.get("head_pitch_joints", 1))
            # Stiffer gains for the head pitch joint(s) so they actually hold
            # the offset against gravity and contact loads, instead of
            # sagging back toward 0 like the soft global pitch_kp would.
            self.head_pitch_kp = float(imp.get("head_pitch_kp", self.pitch_kp * 20.0))
            self.head_pitch_kv = float(imp.get("head_pitch_kv", self.pitch_kv * 20.0))

            pitch_seg_order = []
            for j in self.idx.pitch_jnt_ids:
                nm = model.joint(int(j)).name
                try:
                    seg = int(nm.rsplit("_", 1)[-1])
                except ValueError:
                    seg = 10**9
                pitch_seg_order.append(seg)
            # Per-joint targets and gains, indexed same as pitch_jnt_ids.
            self.pitch_targets = np.zeros(n_pitch, dtype=float)
            self.pitch_kp_vec  = np.full(n_pitch, self.pitch_kp, dtype=float)
            self.pitch_kv_vec  = np.full(n_pitch, self.pitch_kv, dtype=float)
            # Sort pitch_ids by segment index to find the N head-end joints
            order = np.argsort(pitch_seg_order)
            for rank, k in enumerate(order):
                if rank < max(self.head_pitch_joints, 0):
                    self.pitch_targets[k] = self.head_pitch_offset
                    self.pitch_kp_vec[k]  = self.head_pitch_kp
                    self.pitch_kv_vec[k]  = self.head_pitch_kv

            # Gravity compensation is computed ONLINE each step using
            # data.qfrc_bias (gravity + Coriolis at current configuration).
            # This handles the fact that gravity torques on pitch joints
            # depend on the current body pose (not just q=0).

            print(f"[ImpedanceController] pitch_kp={self.pitch_kp:.4f} "
                  f"pitch_kv={self.pitch_kv:.4f}  "
                  f"({n_pitch} pitch joints, online grav_comp)")
            if abs(self.head_pitch_offset) > 1e-9 and self.head_pitch_joints > 0:
                print(f"[ImpedanceController] head_pitch_offset="
                      f"{self.head_pitch_offset:+.3f} rad "
                      f"({math.degrees(self.head_pitch_offset):+.1f} deg)  "
                      f"on first {self.head_pitch_joints} pitch joint(s)")
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
              f"body_kv={self.body_kv:.4f}  (head kp={self.body_kp_head:.4f}, "
              f"kv={self.body_kv_head:.4f} on first {self.head_hold_joints} joints)  "
              f"A={self.body_amp:.3f} f={self.freq:.2f}Hz")
        n_show = max(self.head_hold_joints + self.head_ramp_joints + 1, 4)
        head_scales = ", ".join(f"{s:.2f}" for s in self.body_amp_scale[:n_show])
        print(f"[ImpedanceController] body yaw envelope: hold={self.head_hold_joints} "
              f"taper={self.head_ramp_joints}  min={self.head_ramp_min:.2f}  "
              f"first scales=[{head_scales}...]")
        print(f"[ImpedanceController] leg_kp={self.leg_kp.tolist()} "
              f"leg_kv={self.leg_kv.tolist()}")
        print(f"[ImpedanceController] CPG={'ON' if self.use_cpg else 'OFF'}  "
              f"Ω_body={self.cpg_omega:.2f}  Ω_leg={self.cpg_leg_omega:.2f}  "
              f"Δ={self.cpg_delta:.3f} rad/seg")
        print(f"[ImpedanceController] settle={self.settle_time:.1f}s  "
              f"ramp={self.ramp_time:.1f}s  "
              f"(head→tail sequential: per-seg={self.ramp_time/2:.2f}s, "
              f"stagger={self.ramp_time/(2*max(N_BODY_JOINTS-1,1)):.3f}s)")

    def _build_body_amp_scale(self):
        """Precompute per-joint amplitude scale for the body yaw only.

        Shape: hold zone → cosine taper → full.
          indices 0..H-1               → 0                (H = head_hold_joints)
          indices H..H+R-1             → cosine amin → 1  (R = head_ramp_joints)
          remaining                    → 1
        """
        n = N_BODY_JOINTS
        scale = np.ones(n, dtype=float)

        H = max(int(self.head_hold_joints), 0)
        R = max(int(self.head_ramp_joints), 0)
        amin = float(self.head_ramp_min)

        # Hold zone: zero amplitude
        for i in range(min(H, n)):
            scale[i] = 0.0

        # Taper zone: cosine evaluated at the R interior points
        #     k/(R+1) * pi   for k = 1..R
        # so that R=1 -> [0.5], R=2 -> [0.25, 0.75], R=3 -> [~0.146, 0.5, ~0.854],
        # R=4 -> [~0.095, ~0.345, ~0.655, ~0.905].  The endpoints (full hold and
        # full wave) are handled by the surrounding regions, so we never emit
        # exactly 0 or 1 inside the taper band.  amin linearly biases the floor.
        if R > 0:
            for k in range(1, R + 1):
                i = H + k - 1
                if i >= n:
                    break
                s = amin + (1.0 - amin) * 0.5 * (
                    1.0 - math.cos(math.pi * k / (R + 1))
                )
                scale[i] = s

        self.body_amp_scale = scale

    def _build_body_kp_vec(self):
        """Per-joint body-yaw kp/kv vector.

        First `head_hold_joints` entries use (body_kp_head, body_kv_head);
        the rest use (body_kp, body_kv).  This stiffens the rigid head zone
        so it actually tracks q_target=0 instead of being dragged by the wave.
        """
        n = N_BODY_JOINTS
        kp_vec = np.full(n, self.body_kp, dtype=float)
        kv_vec = np.full(n, self.body_kv, dtype=float)
        H = max(int(self.head_hold_joints), 0)
        for i in range(min(H, n)):
            kp_vec[i] = self.body_kp_head
            kv_vec[i] = self.body_kv_head
        self.body_kp_vec = kp_vec
        self.body_kv_vec = kv_vec

    def _spatial_phase(self, i):
        return 2.0 * math.pi * self.n_wave * self.speed * i / max(N_BODY_JOINTS - 1, 1)

    # ──────────────────────────────────────────────────────────────────────
    # CPG (Kuramoto chain) helpers
    # ──────────────────────────────────────────────────────────────────────

    def _init_cpg_phases(self, t0):
        """Seed each oscillator to match the open-loop traveling wave at t0.

        Using φ_i(0) = ω·t0 − k·s_i makes the CPG start in its locked
        steady-state, so the first gait cycle is indistinguishable from the
        original global-clock controller.  Any subsequent perturbation
        (heading bias at the head, contact disturbance, etc.) is what the
        coupling then redistributes down the chain.
        """
        body = np.zeros(N_BODY_JOINTS, dtype=float)
        for i in range(N_BODY_JOINTS):
            body[i] = self.omega * t0 - self._spatial_phase(i)
        legs = np.zeros(N_LEGS, dtype=float)
        for n in range(N_LEGS):
            # Legs share their segment's body phase; DOF-specific offsets
            # are applied at target-computation time, not stored in the phase.
            legs[n] = body[n]
        self.body_phases = body
        self.leg_phases  = legs
        self._cpg_initialized = True

    def _integrate_cpg_phases(self, dt):
        """One forward-Euler step of the Kuramoto chain.

        Body chain:
            φ_dot_0   = ω + A·Ω·sin(φ_0   − (φ_1 + Δ))                   # head
            φ_dot_i   = ω + A·Ω·[sin(φ_i − (φ_{i-1} − Δ))
                                + sin(φ_i − (φ_{i+1} + Δ))]              # interior
            φ_dot_N-1 = ω + A·Ω·sin(φ_{N-1} − (φ_{N-2} − Δ))             # tail

        Legs (one phase per segment; L/R and per-DOF offsets added at output):
            φ_leg_dot_n = ω + A_leg·Ω_leg·sin(φ_leg_n − φ_body_n)
        """
        phi = self.body_phases
        d   = self.cpg_delta
        N   = N_BODY_JOINTS
        gain_body = self.body_amp * self.cpg_omega
        new_phi = np.empty_like(phi)
        for i in range(N):
            if N == 1:
                fdot = self.omega
            elif i == 0:
                fdot = self.omega + gain_body * math.sin(phi[0] - (phi[1] + d))
            elif i == N - 1:
                fdot = self.omega + gain_body * math.sin(phi[i] - (phi[i - 1] - d))
            else:
                fdot = self.omega + gain_body * (
                    math.sin(phi[i] - (phi[i - 1] - d)) +
                    math.sin(phi[i] - (phi[i + 1] + d))
                )
            new_phi[i] = phi[i] + fdot * dt
        self.body_phases = new_phi

        # Leg phases: each leg is pulled toward its body segment's phase.
        # Amplitude scale uses hip-yaw (DOF 0) amplitude as a representative
        # gain magnitude.
        leg_amp_ref = float(self.leg_amps[0]) if len(self.leg_amps) > 0 else 1.0
        gain_leg = leg_amp_ref * self.cpg_leg_omega
        lphi = self.leg_phases
        new_lphi = np.empty_like(lphi)
        for n in range(N_LEGS):
            fdot = self.omega + gain_leg * math.sin(lphi[n] - new_phi[n])
            new_lphi[n] = lphi[n] + fdot * dt
        self.leg_phases = new_lphi

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

        # ── CPG phase integration (once per step, before any targets) ──
        # Defer init + integration until the gait actually turns on.  This
        # removes a subtle timing bug: seeding at t0=settle_time while the
        # integrator runs from t=0 previously caused phases to over-advance
        # by omega*settle_time before the first gait step.  Now we wait and
        # seed right at t=settle_time so phase(settle_time)=omega*settle_time
        # - d*i, matching the open-loop formula exactly.
        if self.use_cpg and t >= self.settle_time:
            if not self._cpg_initialized:
                self._init_cpg_phases(t)
            dt_int = model.opt.timestep
            self._integrate_cpg_phases(dt_int)

        # ── body yaw: impedance control, head→tail sequential activation ──
        for i in range(N_BODY_JOINTS):
            if i == 0:
                # Heading servo on the head: always on (no blend/ramp), driven
                # by world-yaw error of link_body_0 relative to its initial
                # yaw (latched on the first step).  q_target = K · err pulls
                # joint_body_1 to counter-rotate link_body_1; the reaction
                # torque on the head steers ψ → ψ_ref.
                R_head = data.xmat[self.head_body_id].reshape(3, 3)
                yaw_world = math.atan2(R_head[1, 0], R_head[0, 0])
                if self.head_yaw_ref is None:
                    self.head_yaw_ref = yaw_world
                # Advance reference yaw (circle trajectory) once the gait has
                # switched on. Keeping ref fixed during settle avoids steering
                # against an un-activated gait.
                if self.head_yaw_rate != 0.0 and t >= self.settle_time:
                    self.head_yaw_ref += self.head_yaw_rate * model.opt.timestep
                err = yaw_world - self.head_yaw_ref
                # Wrap to [-π, π]
                if err >  math.pi: err -= 2.0 * math.pi
                if err < -math.pi: err += 2.0 * math.pi
                target = self.head_yaw_gain * err
                # Clamp to joint-range-safe interval
                c = self.head_yaw_target_clip
                if target >  c: target =  c
                if target < -c: target = -c
            else:
                blend_i = self._seg_blend(t, i)
                if blend_i > 0:
                    if self.use_cpg and self._cpg_initialized:
                        phase = self.body_phases[i]
                    else:
                        phase = self.omega * t - self._spatial_phase(i)
                    target = (blend_i
                              * self.body_amp_scale[i]
                              * self.body_amp
                              * math.sin(phase))
                else:
                    target = 0.0

            q    = data.qpos[self.body_jnt_qpos_adr[i]]
            qdot = data.qvel[self.body_jnt_dof_adr[i]]

            torque = self.body_kp_vec[i] * (target - q) - self.body_kv_vec[i] * qdot

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

                # Head-end joints use a non-zero target (nose pitched up) and
                # stiffer per-joint gains so the nose actually holds that
                # offset against gravity + ground reaction, instead of
                # sagging back to 0 like the soft global pitch_kp would.
                tgt = self.pitch_targets[i]
                torque = (self.pitch_kp_vec[i] * (tgt - q)
                          - self.pitch_kv_vec[i] * qdot)

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
            blend_n = self._seg_blend(t, n, n_seg=N_LEGS)
            if self.use_cpg and self._cpg_initialized:
                # Leg oscillator already tracks its body segment.
                leg_base_phase = self.leg_phases[n]
            else:
                leg_base_phase = self.omega * t - self._spatial_phase(n)
            for si, side in enumerate(('L', 'R')):
                for dof in range(N_LEG_DOF):
                    act_id = self.idx.leg_act_ids[n, si, dof]
                    if blend_n <= 0 or dof not in self.active_dofs:
                        target = self.leg_dc_offsets[dof]
                    else:
                        phase  = leg_base_phase + self.leg_phase_offsets[dof]
                        wave   = math.sin(phase)
                        sign   = 1.0 if si == 0 else -1.0
                        target = (blend_n * sign * self.leg_amps[dof] * wave
                                  + self.leg_dc_offsets[dof])

                    q    = data.qpos[self.leg_jnt_qpos_adr[n, si, dof]]
                    qdot = data.qvel[self.leg_jnt_dof_adr[n, si, dof]]
                    torque = self.leg_kp[dof] * (target - q) - self.leg_kv[dof] * qdot
                    data.ctrl[act_id] = torque
