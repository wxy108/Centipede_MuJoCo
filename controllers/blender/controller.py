"""
controller.py — Traveling-wave locomotion controller
=====================================================

Architecture (CPG-ready PD servo)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The controller is split into two independent layers:

  Layer 1 — TrajectoryGenerator (abstract interface)
      Produces (q_target, q_dot_target) pairs for every joint every timestep.

      Current implementation:  SinusoidalTrajectoryGenerator
          Analytical derivatives — no finite differences, no lag.
          Body:  q = A·sin(ωt−φ),   q̇ = A·ω·cos(ωt−φ)
          Leg:   q = ±A·f(θ)+dc,    q̇ = ±A·f'(θ)

      Future replacement:  HopfCPGTrajectoryGenerator  (drop-in swap)
          Reads (x, ẋ) directly from the Hopf oscillator state.
          x  ≈ A·sin(ωt−φ)   → q_target
          ẋ  ≈ A·ω·cos(ωt−φ) → q_dot_target
          No finite differences needed — velocity is a first-class
          output of the oscillator ODE, not a derived quantity.
          This is identical in spirit to what the sinusoidal generator
          already provides analytically, so the servo layer never changes.

  Layer 2 — TravelingWaveController
      Consumes (q, q_dot) from the generator.
      Writes q_target  → position actuators  (kp × position error)
             q_dot_target → velocity actuators  (kv × velocity error)
      Combined MuJoCo force per joint:
          F = kp·(q_target − q) + kv·(q_dot_target − q̇)
      Velocity actuators are optional: if not present in the XML
      (id = −1) they are silently skipped and the controller
      degrades gracefully to pure P control.

To swap the generator for a CPG, change ONE line in __init__:
    self.traj = SinusoidalTrajectoryGenerator(config)
    →
    self.traj = HopfCPGTrajectoryGenerator(config)
Nothing else in this file or in run.py needs to change.
"""

from abc import ABC, abstractmethod
import numpy as np
import yaml
import os
import mujoco
from kinematics import (
    ModelIndex, N_BODY_JOINTS, N_LEGS_PER_SIDE, N_LEG_DOF,
    body_joint_name, leg_joint_name,
)


def load_config(path=None):
    """Load controller configuration from YAML."""
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(script_dir, "..", "..")
        path = os.path.join(project_root, "configs", "blender_controller.yaml")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════
# Abstract trajectory generator interface
# ══════════════════════════════════════════════════════════════════════

class TrajectoryGenerator(ABC):
    """
    Abstract base: produces (q_target, q_dot_target) for every joint.

    The contract is intentionally minimal so any upstream motor program
    — sinusoidal wave, CPG, replay, RL policy — can be dropped in
    without touching TravelingWaveController or the optimizer.

    Design note on velocity:
        For a Hopf oscillator with state (x, y):
            q_target     ∝ x
            q_dot_target ∝ ẋ  (computed by the ODE itself)
        This is structurally identical to what SinusoidalTrajectoryGenerator
        already provides analytically. The servo layer therefore never needs
        to know which generator is upstream.
    """

    @abstractmethod
    def body_target(self, seg_i: int, t: float):
        """
        Returns (q_target [rad], q_dot_target [rad/s]) for body joint
        at 0-based segment index seg_i at time t.
        """

    @abstractmethod
    def leg_target(self, seg_i: int, side_i: int, dof: int, t: float):
        """
        Returns (q_target [rad], q_dot_target [rad/s]) for a leg joint.
            seg_i  : 0-based segment index (head→tail)
            side_i : 0 = left, 1 = right
            dof    : 0 = yaw, 1 = upper pitch, 2 = lower pitch
        """

    @abstractmethod
    def reset(self):
        """Reset any internal state (called on simulation reset)."""


# ══════════════════════════════════════════════════════════════════════
# Sinusoidal implementation — current
# ══════════════════════════════════════════════════════════════════════

class SinusoidalTrajectoryGenerator(TrajectoryGenerator):
    """
    Pure traveling-wave generator with fully analytical velocity output.

    Why analytical and not finite-difference:
        d/dt [A·sin(ωt − φ)]  =  A·ω·cos(ωt − φ)
    Both are O(1), zero lag, zero noise amplification.  This exactly
    mirrors what a settled Hopf oscillator provides from its (x, ẋ)
    state, making the CPG swap a true drop-in replacement.

    Wave equations (from FARMS feedforward):
        Body joint i:
            q     = A_body · sin(ωt − φᵢ)
            q̇     = A_body · ω · cos(ωt − φᵢ)

        Leg DOF d at segment i, left side:
            q     = +A[d] · f (ωt − φᵢ + δ[d]) + dc[d]
            q̇     = +A[d] · f'(ωt − φᵢ + δ[d])
        Right side: negate both q and q̇ (dc offset unchanged).

        f  = sin  when duty_factor == 0  (default)
        f' = ω·cos
        For duty_factor > 0: piecewise stance/swing — see _duty_waveform_vel.
    """

    def __init__(self, config: dict):
        bw = config['body_wave']
        lw = config['leg_wave']

        self.body_amp    = float(bw['amplitude'])
        self.frequency   = float(bw['frequency'])
        self.n_wave      = float(bw['wave_number'])
        self.speed       = float(bw['speed'])
        self.omega       = 2.0 * np.pi * self.frequency
        self.wave_offset = 2.0 * np.pi * self.n_wave
        self.n_bj_wave   = int(config.get('n_body_joints_wave', 19))

        # Per-DOF leg parameters — pad to N_LEG_DOF
        def _pad(src, length):
            arr = np.zeros(length)
            src = np.asarray(src, dtype=float)
            arr[:min(len(src), length)] = src[:length]
            return arr

        self.leg_amps          = _pad(lw['amplitudes'],                N_LEG_DOF)
        self.leg_phase_offsets = _pad(lw['phase_offsets'],             N_LEG_DOF)
        self.leg_dc_offsets    = _pad(lw.get('dc_offsets', [0.0]*3),   N_LEG_DOF)
        self.active_dofs       = set(lw.get('active_dofs', [0, 1]))
        self.duty_factor       = float(config.get('duty_factor', 0.0))

    # ── Spatial phase ──────────────────────────────────────────────────

    def spatial_phase(self, seg_i: int) -> float:
        """φᵢ = wave_offset · (i / (N−1)) · speed"""
        N = max(self.n_bj_wave - 1, 1)
        return self.wave_offset * (seg_i / N) * self.speed

    # ── Body ───────────────────────────────────────────────────────────

    def body_target(self, seg_i: int, t: float):
        if seg_i >= self.n_bj_wave:
            return 0.0, 0.0
        theta = self.omega * t - self.spatial_phase(seg_i)
        return (self.body_amp * np.sin(theta),
                self.body_amp * self.omega * np.cos(theta))

    # ── Leg ────────────────────────────────────────────────────────────

    def leg_target(self, seg_i: int, side_i: int, dof: int, t: float):
        sign = 1.0 if side_i == 0 else -1.0

        # Passive joint: hold at dc_offset, zero velocity command
        if dof not in self.active_dofs:
            return self.leg_dc_offsets[dof], 0.0

        theta = self.omega * t - self.spatial_phase(seg_i) + self.leg_phase_offsets[dof]

        if self.duty_factor > 0.0 and dof <= 1:
            wave, wave_dot = self._duty_waveform_vel(theta, dof)
        else:
            wave     = np.sin(theta)
            wave_dot = self.omega * np.cos(theta)

        A = self.leg_amps[dof]
        dc = self.leg_dc_offsets[dof]
        # dc offset is constant → its time derivative is zero
        return (sign * A * wave + dc,
                sign * A * wave_dot)

    # ── Duty-factor waveform with analytical velocity ──────────────────

    def _duty_waveform_vel(self, theta: float, dof: int):
        """
        Piecewise stance/swing waveform and its exact time derivative.

        Stance phase (0 ≤ p < φ_stance):
            yaw:   sin(π·s),    ṡ = ω/φ_s  →  vel = (π·ω/φ_s)·cos(π·s)
            pitch: +1,           vel = 0

        Swing phase (φ_stance ≤ p < 2π):
            yaw:  −sin(π·s),    ṡ = ω/(2π−φ_s)  →  vel = −(π·ω/(2π−φ_s))·cos(π·s)
            pitch: −1,           vel = 0

        Note: discontinuity in vel at phase transitions is physically real
        (instantaneous direction reversal). When CPG arrives this issue
        disappears — the oscillator produces smooth trajectories.
        """
        cycle = 2.0 * np.pi
        phi_s = cycle * self.duty_factor
        p = theta % cycle
        if p < 0.0:
            p += cycle

        if p < phi_s:
            s = (p / phi_s) if phi_s > 1e-12 else 0.0
            if dof == 0:
                vel = (np.pi * self.omega / phi_s) * np.cos(np.pi * s)
                return np.sin(np.pi * s), vel
            else:
                return 1.0, 0.0
        else:
            span = cycle - phi_s
            s = ((p - phi_s) / span) if span > 1e-12 else 0.0
            if dof == 0:
                vel = -(np.pi * self.omega / span) * np.cos(np.pi * s)
                return -np.sin(np.pi * s), vel
            else:
                return -1.0, 0.0

    def reset(self):
        pass  # Stateless — nothing to reset


# ══════════════════════════════════════════════════════════════════════
# Future CPG generator stub (not yet implemented)
# ══════════════════════════════════════════════════════════════════════

class HopfCPGTrajectoryGenerator(TrajectoryGenerator):
    """
    Placeholder for the future FARMS-style Hopf CPG generator.

    When implemented, this class will:
      - Maintain per-segment oscillator states (xᵢ, yᵢ) for body and legs
      - Integrate the Hopf ODE each timestep:
            ẋ = (μ − r²)·x − ω·y  +  coupling + descending drive
            ẏ = (μ − r²)·y + ω·x
      - Return position and velocity directly from oscillator state:
            q         ∝ xᵢ         (no finite difference)
            q_dot     ∝ ẋᵢ         (from the ODE — exact, zero lag)

    The interface (body_target / leg_target) is identical to
    SinusoidalTrajectoryGenerator, so TravelingWaveController
    requires zero changes when this swap happens.
    """

    def __init__(self, config: dict):
        raise NotImplementedError(
            "HopfCPGTrajectoryGenerator is not yet implemented. "
            "Use SinusoidalTrajectoryGenerator for now."
        )

    def body_target(self, seg_i, t):
        raise NotImplementedError

    def leg_target(self, seg_i, side_i, dof, t):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════
# Main controller
# ══════════════════════════════════════════════════════════════════════

class TravelingWaveController:
    """
    PD servo controller driven by a TrajectoryGenerator.

    Writes:
        data.ctrl[pos_actuator] = q_target          → kp × (q_target − q)
        data.ctrl[vel_actuator] = q_dot_target       → kv × (q_dot_target − q̇)

    Velocity actuators are optional. If not present in the XML
    (mj_name2id returns −1), they are silently skipped and the
    controller operates as a pure P servo.

    To swap the trajectory generator (sinusoidal → CPG):
        Change one line in __init__:
            self.traj = SinusoidalTrajectoryGenerator(config)
        to:
            self.traj = HopfCPGTrajectoryGenerator(config)
        Nothing else changes.
    """

    def __init__(self, model, config=None, config_path=None):
        if config is None:
            config = load_config(config_path)
        self.config = config
        self.model  = model
        self.idx    = ModelIndex(model)

        # ── Trajectory generator — swap here for CPG ──
        self.traj = SinusoidalTrajectoryGenerator(config)

        # ── Mirror key params at top level for run.py logging ──
        bw = config['body_wave']
        self.frequency          = bw['frequency']
        self.n_wave             = bw['wave_number']
        self.body_amp           = bw['amplitude']
        self.speed              = bw['speed']
        self.n_body_joints_wave = config.get('n_body_joints_wave', 19)

        # ── Appendage config ──
        ap = config.get('appendages', {})
        self.ant_cfg  = ap.get('antenna',   {})
        self.frc_cfg  = ap.get('forcipule', {})
        self.mnd_cfg  = ap.get('mandible',  {})
        self._ou_states: dict = {}

        self._build_index_arrays()

    # ── Actuator index caching ────────────────────────────────────────

    def _build_index_arrays(self):
        """
        Resolve actuator names → integer ids at load time.
        Velocity actuator ids default to −1 if not present in the XML,
        allowing graceful fallback to pure P control.
        """
        OBJ_ACT = mujoco.mjtObj.mjOBJ_ACTUATOR

        def _act_id(name: str) -> int:
            return mujoco.mj_name2id(self.model, OBJ_ACT, name)

        # ── Body ──
        self.body_pos_act = np.zeros(N_BODY_JOINTS, dtype=int)
        self.body_vel_act = np.full(N_BODY_JOINTS, -1, dtype=int)
        for i in range(N_BODY_JOINTS):
            jn = body_joint_name(i + 1)
            self.body_pos_act[i] = self.idx.pos_act[jn]
            self.body_vel_act[i] = _act_id(f"v_{jn}")

        # ── Legs ──
        self.leg_pos_act = np.zeros((N_LEGS_PER_SIDE, 2, N_LEG_DOF), dtype=int)
        self.leg_vel_act = np.full((N_LEGS_PER_SIDE, 2, N_LEG_DOF), -1, dtype=int)
        for n in range(N_LEGS_PER_SIDE):
            for si, side in enumerate(('L', 'R')):
                for dof in range(N_LEG_DOF):
                    jn = leg_joint_name(n + 1, side, dof)
                    self.leg_pos_act[n, si, dof] = self.idx.pos_act[jn]
                    self.leg_vel_act[n, si, dof] = _act_id(f"v_{jn}")

        # ── Appendages ──
        self.ant_act = {
            'Ly': _act_id('p_jantLy'), 'Lp': _act_id('p_jantLp'),
            'Ry': _act_id('p_jantRy'), 'Rp': _act_id('p_jantRp'),
        }
        self.frc_act = {'L': _act_id('p_jfrcL'), 'R': _act_id('p_jfrcR')}
        self.mnd_act = {'L': _act_id('p_jmndL'), 'R': _act_id('p_jmndR')}

    # ── Ornstein-Uhlenbeck noise ──────────────────────────────────────

    def _ou_step(self, key: str, amp: float, freq: float, dt: float) -> float:
        """Smooth random wandering: std ≈ amp, correlation time ≈ 1/freq."""
        x = self._ou_states.get(key, 0.0)
        x += -freq * x * dt + amp * np.sqrt(2.0 * freq * dt) * np.random.randn()
        self._ou_states[key] = x
        return x

    # ── Main step ─────────────────────────────────────────────────────

    def step(self, data, t: float = None):
        """
        Compute and apply PD control signals for one timestep.

        For each joint:
            data.ctrl[pos_act] = q_target
            data.ctrl[vel_act] = q_dot_target   (skipped if vel_act == −1)
        """
        if t is None:
            t = data.time
        dt = self.model.opt.timestep

        # ── Body joints ──
        for i in range(N_BODY_JOINTS):
            q, q_dot = self.traj.body_target(i, t)
            data.ctrl[self.body_pos_act[i]] = q
            if self.body_vel_act[i] >= 0:
                data.ctrl[self.body_vel_act[i]] = q_dot

        # ── Leg joints ──
        for n in range(N_LEGS_PER_SIDE):
            for si in range(2):
                for dof in range(N_LEG_DOF):
                    q, q_dot = self.traj.leg_target(n, si, dof, t)
                    data.ctrl[self.leg_pos_act[n, si, dof]] = q
                    if self.leg_vel_act[n, si, dof] >= 0:
                        data.ctrl[self.leg_vel_act[n, si, dof]] = q_dot

        # ── Antennae ──
        ac     = self.ant_cfg
        a_amp  = ac.get('noise_amp',  0.3)
        a_freq = ac.get('noise_freq', 2.0)
        for key, rest_key, act_key in [
            ('ant_Ly', 'rest_yaw_L',   'Ly'),
            ('ant_Lp', 'rest_pitch_L', 'Lp'),
            ('ant_Ry', 'rest_yaw_R',   'Ry'),
            ('ant_Rp', 'rest_pitch_R', 'Rp'),
        ]:
            aid = self.ant_act[act_key]
            if aid >= 0:
                data.ctrl[aid] = (ac.get(rest_key, 0.0)
                                  + self._ou_step(key, a_amp, a_freq, dt))

        # ── Forcipules ──
        fc     = self.frc_cfg
        f_amp  = fc.get('noise_amp',  0.05)
        f_freq = fc.get('noise_freq', 1.5)
        for side in ('L', 'R'):
            aid = self.frc_act[side]
            if aid >= 0:
                data.ctrl[aid] = (fc.get(f'rest_{side}', 0.0)
                                  + self._ou_step(f'frc_{side}', f_amp, f_freq, dt))

        # ── Mandibles ──
        mc     = self.mnd_cfg
        m_amp  = mc.get('noise_amp',  0.03)
        m_freq = mc.get('noise_freq', 1.0)
        for side in ('L', 'R'):
            aid = self.mnd_act[side]
            if aid >= 0:
                data.ctrl[aid] = (mc.get(f'rest_{side}', 0.0)
                                  + self._ou_step(f'mnd_{side}', m_amp, m_freq, dt))
