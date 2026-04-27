"""
centipede_env.py — Gymnasium environment for centipede locomotion RL
=====================================================================

Pipeline (one episode)
----------------------
  reset()
    1. Sample a (terrain_wavelength, terrain_amplitude, terrain_seed) from
       the configured ranges.  Generate a heightfield + patched XML ONCE
       per environment process and reuse it for several episodes (configurable).
    2. Sample a velocity command — target forward speed in body-frame x —
       from [v_lo, v_hi] m/s.
    3. mj_resetData().  Re-init the CPG controller.

  step(action)
    4. Pass action ∈ [-1, 1]^36 to the modulation controller (per-segment
       Δφ_i ∈ [-π/4, π/4], ε_i ∈ [0.5, 1.5]).
    5. Run `n_substeps` MuJoCo timesteps with that action held constant.
    6. Compute observation, reward, termination from the new state.

Observation (proprioception only, ~280 floats)
----------------------------------------------
  - joint q + qdot for body yaw (19 + 19), body pitch (20 + 20)
  - joint q + qdot for active leg DOFs hip_yaw + hip_pitch (76 + 76)
  - root state: height z (1), orientation R as 6 numbers (xx,xy,xz,yx,yy,yz),
    body-frame linear velocity (3), body-frame angular velocity (3)
  - foot contact flags: 19 × 2 = 38 (binary)
  - CPG phase representation: sin(phase_body_0), cos(phase_body_0) → 2
  - velocity command: target body-x velocity (m/s) → 1

Reward (per RL step)
--------------------
  r = w_speed_match  * tracking term (saturating)
    + w_alive
    - w_action_l2    * ‖a‖² / ACTION_DIM
    - w_force        * max(0, peak_F/W − force_limit)
    - w_buckle       * 1{buckled}                      (terminal)

Termination
-----------
  - root pitch or roll exceed ±45°  -> buckle, terminate, big -ve reward
  - max episode steps reached       -> truncate
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

import gymnasium as gym
from gymnasium import spaces

import mujoco

# ── Make project imports importable from inside this file ──────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "sweep"))
sys.path.insert(0, SCRIPT_DIR)                   # for modulation_controller

from kinematics import (FARMSModelIndex, N_BODY_JOINTS, N_LEGS, N_LEG_DOF,
                        ACTIVE_DOFS)
from modulation_controller import ModulationController, ACTION_DIM
from wavelength_sweep import (generate_single_wavelength_terrain,
                              save_wavelength_terrain)


# ── Worker-safe terrain XML patching ──────────────────────────────────────
# The reference patch_xml_terrain() in wavelength_sweep.py always writes to
# `<base_xml>.sweep_tmp.xml` (a SHARED path), which races catastrophically
# when multiple SubprocVecEnv workers patch in parallel.  We re-implement
# the same logic here but write to a caller-supplied unique path AND set
# meshdir to an absolute path so the patched XML is relocatable.

def _patch_terrain_to_unique_xml(base_xml: str, png_path: str,
                                 z_max: float, output_xml: str) -> str:
    """Like wavelength_sweep.patch_xml_terrain() but writes to `output_xml`
    (worker-unique) instead of a shared `<base>.sweep_tmp.xml` location, and
    rewrites <compiler meshdir=...> to an absolute path so the patched XML
    is portable to a different directory."""
    from lxml import etree
    from PIL import Image

    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(base_xml, parser)
    root = tree.getroot()

    abs_png = os.path.abspath(png_path).replace("\\", "/")
    base_xml_dir = os.path.dirname(os.path.abspath(base_xml))

    # Resolve relative meshdir / texturedir / assetdir to absolute (so the
    # patched XML works when written outside the original model dir)
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            v = compiler.get(key)
            if v is not None and not os.path.isabs(v):
                compiler.set(key, os.path.abspath(
                    os.path.join(base_xml_dir, v)) + os.sep)

    # ── hfield asset ──────────────────────────────────────────────────────
    asset = root.find("asset")
    hfield = asset.find("hfield[@name='terrain']")
    if hfield is not None:
        hfield.set("file", abs_png)
        hfield.set("size", f"0.500 0.500 {z_max:.4f} 0.001")
    else:
        etree.SubElement(asset, "hfield", {
            "name": "terrain",
            "file": abs_png,
            "size": f"0.500 0.500 {z_max:.4f} 0.001",
            "nrow": "1024", "ncol": "1024",
        })

    # ── hfield geom in worldbody ──────────────────────────────────────────
    worldbody = root.find("worldbody")
    terrain_geom = worldbody.find("geom[@name='terrain_geom']")
    if terrain_geom is None:
        etree.SubElement(worldbody, "geom", {
            "type": "hfield", "name": "terrain_geom",
            "hfield": "terrain", "pos": "0 0 0",
            "conaffinity": "1", "condim": "3",
            "friction": "1.6 0.005 0.0001",
        })

    # ── spawn height ──────────────────────────────────────────────────────
    arr = np.array(Image.open(png_path).convert("L"), dtype=np.float32)
    nrow, ncol = arr.shape
    cy, cx = nrow // 2, ncol // 2
    r = 8
    patch = arr[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
    terrain_h = (float(patch.max()) / 255.0) * z_max
    spawn_z = terrain_h + 0.015

    for body in root.iter("body"):
        if body.find("freejoint") is not None:
            body.set("pos", f"0 0 {spawn_z:.4f}")
            break

    os.makedirs(os.path.dirname(os.path.abspath(output_xml)), exist_ok=True)
    tree.write(output_xml, xml_declaration=True, encoding="utf-8",
               pretty_print=False)
    return output_xml


# ════════════════════════════════════════════════════════════════════════════
# Config dataclass
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CentipedeEnvConfig:
    # Model + controller
    xml_path:    str = os.path.join(PROJECT_ROOT, "models", "farms",
                                    "centipede.xml")
    config_path: str = os.path.join(PROJECT_ROOT, "configs",
                                    "farms_controller.yaml")

    # Episode timing
    rl_step_dt: float  = 0.02            # 20 ms between policy decisions
    episode_seconds: float = 10.0        # 500 RL steps / episode
    settle_seconds: float  = 2.0         # 1 s settle + 1 s ramp (matches
                                         # ctrl YAML).  Don't shorten; NaN
                                         # divergence during the initial
                                         # impact onto rough terrain otherwise.
    # control_skip kept at 1 for stability — frame-skipping caused diverging
    # accelerations during impact and didn't materially improve throughput
    # (the parent controller's pure-Python loop is the bottleneck regardless).
    control_skip: int = 1

    # Velocity command
    v_cmd_lo:  float = 0.005             # 5 mm/s
    v_cmd_hi:  float = 0.040             # 40 mm/s
    v_cmd_sigma: float = 0.012           # ±12 mm/s tolerance for tracking

    # Terrain randomization (per env worker; pool resampled every N resets)
    terrain_wavelength_lo: float = 10.0  # mm
    terrain_wavelength_hi: float = 30.0  # mm
    terrain_amplitude_lo:  float = 0.005 # m
    terrain_amplitude_hi:  float = 0.012 # m
    terrain_pool_size:     int   = 8     # patched XMLs cached per env
    terrain_pool_resample_episodes: int = 200  # regenerate pool every N episodes

    # Reward weights
    w_speed_match: float = 5.0
    w_alive:       float = 0.05
    w_action_l2:   float = 0.05
    w_force:       float = 0.5
    force_limit:   float = 4.0           # F/W ratio
    w_buckle:      float = 50.0

    # Buckle thresholds
    max_root_pitch_deg: float = 45.0
    max_root_roll_deg:  float = 45.0

    # Logging / video
    enable_video: bool = False           # if True, render frames every RL step
    video_width:  int  = 1280
    video_height: int  = 720


# ════════════════════════════════════════════════════════════════════════════
# Environment
# ════════════════════════════════════════════════════════════════════════════

class CentipedeEnv(gym.Env):
    """Gymnasium env for centipede CPG-modulation RL."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: Optional[CentipedeEnvConfig] = None,
                 worker_id: Optional[int] = None):
        super().__init__()
        self.cfg = config or CentipedeEnvConfig()
        self.worker_id = worker_id if worker_id is not None else os.getpid()

        # Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

        # Observation dims (computed once we have a model loaded)
        self._obs_dim = self._compute_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

        # Terrain pool (per worker, lives in tmp dir)
        self._terrain_dir = tempfile.mkdtemp(
            prefix=f"centipede_rl_w{self.worker_id}_")
        self._terrain_pool = []   # list of patched XML paths
        self._n_resets = 0
        self._regenerate_terrain_pool(seed=self.worker_id * 1000)

        # Pre-load first terrain so spaces / index are available
        self._cur_terrain_idx = 0
        self._load_model(self._terrain_pool[self._cur_terrain_idx])

        # Step bookkeeping
        self._n_substeps   = max(1, int(round(self.cfg.rl_step_dt
                                              / self.model.opt.timestep)))
        self._max_steps    = max(1, int(round(self.cfg.episode_seconds
                                              / self.cfg.rl_step_dt)))
        self._cur_step     = 0
        self._v_cmd        = 0.02

        # Optional renderer (only if enable_video)
        self._renderer = None
        self._frames   = []
        if self.cfg.enable_video:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.cfg.video_height, width=self.cfg.video_width)

    # ------------------------------------------------------------------ #
    # Setup helpers                                                      #
    # ------------------------------------------------------------------ #

    def _compute_obs_dim(self):
        # body yaw  q + qdot  : 19 + 19 = 38
        # body pitch q + qdot : 20 + 20 = 40
        # leg active q + qdot : 19*2*2*2 = 152
        # root: z (1), R-flat (6), v_body (3), w_body (3) = 13
        # foot contact flags  : 19*2 = 38
        # CPG phase repr      : 2
        # velocity command    : 1
        return 38 + 40 + 152 + 13 + 38 + 2 + 1   # = 284

    def _regenerate_terrain_pool(self, seed: int):
        """Generate K patched XML files, one per pool slot."""
        # Clear existing pool (best effort)
        for p in self._terrain_pool:
            try: os.remove(p)
            except Exception: pass
        self._terrain_pool = []

        rng = np.random.default_rng(seed)
        for k in range(self.cfg.terrain_pool_size):
            wl_mm = float(rng.uniform(
                self.cfg.terrain_wavelength_lo,
                self.cfg.terrain_wavelength_hi))
            amp_m = float(rng.uniform(
                self.cfg.terrain_amplitude_lo,
                self.cfg.terrain_amplitude_hi))
            t_seed = int(rng.integers(0, 1_000_000))

            h, _, _ = generate_single_wavelength_terrain(
                wavelength_m=wl_mm * 1e-3,
                amplitude_m=amp_m,
                seed=t_seed)
            png_path = save_wavelength_terrain(
                h, wl_mm * 1e-3, t_seed, self._terrain_dir)
            z_max = max(2.0 * amp_m, 1e-3)

            # Worker-unique output path — no race with other workers
            unique = os.path.join(
                self._terrain_dir,
                f"patched_w{self.worker_id}_{k:03d}.xml")
            _patch_terrain_to_unique_xml(
                self.cfg.xml_path, png_path,
                z_max=z_max, output_xml=unique)
            self._terrain_pool.append(unique)

    def _load_model(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.idx   = FARMSModelIndex(self.model)
        self.ctrl  = ModulationController(self.model, self.cfg.config_path)

        self._root_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")

        # Cache index arrays
        self._body_yaw_qpos_adr = np.asarray(
            self.ctrl.body_jnt_qpos_adr, dtype=np.int64)
        self._body_yaw_dof_adr  = np.asarray(
            self.ctrl.body_jnt_dof_adr,  dtype=np.int64)

        self._pitch_qpos_adr = np.asarray(
            self.ctrl.idx.pitch_jnt_ids
            if hasattr(self.ctrl.idx, "pitch_jnt_ids") else [], dtype=np.int64)
        if self._pitch_qpos_adr.size:
            self._pitch_qpos_adr = np.asarray(
                [self.model.jnt_qposadr[j] for j in self._pitch_qpos_adr],
                dtype=np.int64)
            self._pitch_dof_adr = np.asarray(
                [self.model.jnt_dofadr[j] for j in self.ctrl.idx.pitch_jnt_ids],
                dtype=np.int64)
        else:
            self._pitch_dof_adr = np.zeros(0, dtype=np.int64)

        # Active leg joints (DOFs 0,1 = hip_yaw, hip_pitch) for both sides
        self._leg_qpos_adr = []
        self._leg_dof_adr  = []
        for n in range(N_LEGS):
            for si in range(2):
                for dof in (0, 1):       # hip_yaw, hip_pitch only
                    jid = self.idx.leg_jnt_ids[n, si, dof]
                    self._leg_qpos_adr.append(self.model.jnt_qposadr[jid])
                    self._leg_dof_adr.append(self.model.jnt_dofadr[jid])
        self._leg_qpos_adr = np.asarray(self._leg_qpos_adr, dtype=np.int64)
        self._leg_dof_adr  = np.asarray(self._leg_dof_adr,  dtype=np.int64)

        # Foot body IDs for contact detection
        self._foot_body_ids = []
        for n in range(N_LEGS):
            for side in ("L", "R"):
                # foot is the most distal body — name pattern from URDF
                name = f"link_leg_{n}_{side}_foot"
                bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid < 0:
                    # Fallback to tarsus if foot doesn't exist
                    name = f"link_leg_{n}_{side}_tarsus"
                    bid = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                self._foot_body_ids.append(bid)
        self._foot_body_ids = np.asarray(self._foot_body_ids, dtype=np.int64)

        self._body_weight = float(np.sum(self.model.body_mass) * 9.81)

    # ------------------------------------------------------------------ #
    # Standard Gym API                                                   #
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None):
        super().reset(seed=seed)
        rng = self.np_random

        # Maybe regenerate pool
        self._n_resets += 1
        if (self._n_resets % self.cfg.terrain_pool_resample_episodes) == 0:
            self._regenerate_terrain_pool(
                seed=int(rng.integers(0, 1 << 30)))

        # Pick a terrain — reload model only if it's a different file
        new_idx = int(rng.integers(0, len(self._terrain_pool)))
        if new_idx != self._cur_terrain_idx:
            self._load_model(self._terrain_pool[new_idx])
            self._cur_terrain_idx = new_idx
            if self.cfg.enable_video:
                self._renderer = mujoco.Renderer(
                    self.model,
                    height=self.cfg.video_height,
                    width=self.cfg.video_width)
        else:
            mujoco.mj_resetData(self.model, self.data)

        self.ctrl.reset_action()
        # Re-init parent CPG state (force re-seeding on next step)
        self.ctrl._cpg_initialized = False
        self.ctrl.head_yaw_ref = None

        # Sample velocity command for this episode
        self._v_cmd = float(rng.uniform(self.cfg.v_cmd_lo, self.cfg.v_cmd_hi))
        self._cur_step = 0
        self._frames = []

        # Run settle phase: hold action zero for cfg.settle_seconds so the
        # body relaxes onto the heightfield before the policy takes over.
        # FULL control rate here — frame-skipping during impact-heavy settle
        # leads to inadequate damping and diverging joint accelerations (NaN
        # in qvel/qacc at the freejoint).  Settle is rare (once per episode)
        # so the cost is acceptable.
        n_settle_substeps = int(self.cfg.settle_seconds / self.model.opt.timestep)
        for _ in range(n_settle_substeps):
            self.ctrl.step(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = {"v_cmd": self._v_cmd,
                "terrain_idx": self._cur_terrain_idx}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self.ctrl.set_action(action)

        # Run all MuJoCo substeps with this action held constant.
        # Controller is called every substep — frame-skipping the controller
        # caused divergent simulations on impact-heavy terrain and didn't
        # measurably improve throughput.
        for _ in range(self._n_substeps):
            self.ctrl.step(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

        # Compute contact forces once at the end of the RL step
        mujoco.mj_rnePostConstraint(self.model, self.data)
        peak_fw = float(np.max(np.linalg.norm(
            self.data.cfrc_ext[self._foot_body_ids, 3:6], axis=1))
            ) / max(self._body_weight, 1e-9)

        if self.cfg.enable_video and self._renderer is not None:
            self._renderer.update_scene(self.data)
            self._frames.append(self._renderer.render().copy())

        self._cur_step += 1
        obs = self._get_obs()

        # ── Reward components ─────────────────────────────────────────
        v_body_x = self._root_body_lin_vel()[0]   # m/s, body-frame fwd
        # Saturating tracking reward: 1.0 at command, 0.0 when |err| > 3σ
        err = (v_body_x - self._v_cmd) / max(self.cfg.v_cmd_sigma, 1e-9)
        speed_match = max(0.0, 1.0 - (err * err) ** 0.5)   # |err| / σ saturating

        action_l2 = float(np.mean(action * action))
        force_excess = max(0.0, peak_fw - self.cfg.force_limit)

        # Buckle check
        R = self.data.xmat[self._root_body_id].reshape(3, 3)
        root_pitch_deg = math.degrees(math.asin(-max(-1.0, min(1.0, R[2, 0]))))
        root_roll_deg  = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        buckled = (abs(root_pitch_deg) > self.cfg.max_root_pitch_deg or
                   abs(root_roll_deg)  > self.cfg.max_root_roll_deg)

        reward = (self.cfg.w_speed_match * speed_match
                  + self.cfg.w_alive
                  - self.cfg.w_action_l2 * action_l2
                  - self.cfg.w_force     * force_excess)
        terminated = False
        if buckled:
            reward -= self.cfg.w_buckle
            terminated = True

        truncated = (self._cur_step >= self._max_steps)

        info = {
            "v_body_x":    v_body_x,
            "v_cmd":       self._v_cmd,
            "speed_err":   v_body_x - self._v_cmd,
            "peak_fw":     peak_fw,
            "buckled":     buckled,
            "action_l2":   action_l2,
            "root_pitch":  root_pitch_deg,
            "root_roll":   root_roll_deg,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.cfg.video_height, width=self.cfg.video_width)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
        # Best-effort cleanup of terrain pool dir
        try:
            for p in self._terrain_pool:
                try: os.remove(p)
                except Exception: pass
            os.rmdir(self._terrain_dir)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Observation construction                                           #
    # ------------------------------------------------------------------ #

    def _root_body_lin_vel(self):
        """Linear velocity of root in BODY frame."""
        # data.cvel is in world frame: [ang(3), lin(3)]
        v_world = self.data.cvel[self._root_body_id, 3:6]
        R = self.data.xmat[self._root_body_id].reshape(3, 3)
        return R.T @ v_world

    def _root_body_ang_vel(self):
        w_world = self.data.cvel[self._root_body_id, 0:3]
        R = self.data.xmat[self._root_body_id].reshape(3, 3)
        return R.T @ w_world

    def _get_obs(self):
        # Joint q + qdot
        body_yaw_q   = self.data.qpos[self._body_yaw_qpos_adr]
        body_yaw_qd  = self.data.qvel[self._body_yaw_dof_adr]

        if self._pitch_qpos_adr.size:
            pitch_q  = self.data.qpos[self._pitch_qpos_adr]
            pitch_qd = self.data.qvel[self._pitch_dof_adr]
        else:
            pitch_q  = np.zeros(20, dtype=float)
            pitch_qd = np.zeros(20, dtype=float)

        leg_q  = self.data.qpos[self._leg_qpos_adr]
        leg_qd = self.data.qvel[self._leg_dof_adr]

        # Root state
        root_z = float(self.data.xpos[self._root_body_id, 2])
        R = self.data.xmat[self._root_body_id].reshape(3, 3)
        R_flat6 = R[:2, :].flatten()                 # 6 numbers (rows 0,1)
        v_body = self._root_body_lin_vel()
        w_body = self._root_body_ang_vel()

        # Foot contact (binary): use cfrc_ext magnitude > eps
        # rnePostConstraint was run inside step() — values still valid here
        ext = self.data.cfrc_ext[self._foot_body_ids, 3:6]
        contact = (np.linalg.norm(ext, axis=1) > 1e-3).astype(np.float32)

        # CPG phase representation
        if (hasattr(self.ctrl, "_cpg_initialized")
                and self.ctrl._cpg_initialized
                and self.ctrl.body_phases is not None):
            phi0 = float(self.ctrl.body_phases[0])
        else:
            phi0 = 0.0
        cpg_repr = np.array([math.sin(phi0), math.cos(phi0)], dtype=np.float32)

        # Velocity command
        v_cmd_obs = np.array([self._v_cmd], dtype=np.float32)

        obs = np.concatenate([
            body_yaw_q.astype(np.float32),
            body_yaw_qd.astype(np.float32),
            pitch_q.astype(np.float32),
            pitch_qd.astype(np.float32),
            leg_q.astype(np.float32),
            leg_qd.astype(np.float32),
            np.array([root_z], dtype=np.float32),
            R_flat6.astype(np.float32),
            v_body.astype(np.float32),
            w_body.astype(np.float32),
            contact,
            cpg_repr,
            v_cmd_obs,
        ])
        return obs.astype(np.float32)

    def get_video_frames(self):
        """Return frames recorded since last reset (only if enable_video)."""
        return list(self._frames)


# ════════════════════════════════════════════════════════════════════════════
# Convenience factory
# ════════════════════════════════════════════════════════════════════════════

def make_env(rank: int = 0, config: Optional[CentipedeEnvConfig] = None,
             seed: int = 0):
    """Returns a thunk that builds a CentipedeEnv.  Used with VecEnv."""
    def _init():
        cfg = config or CentipedeEnvConfig()
        env = CentipedeEnv(cfg, worker_id=rank)
        env.reset(seed=seed + rank)
        return env
    return _init
