"""
run.py — Simulation runner for the FARMS centipede model
=========================================================
Loads configs/farms_controller.yaml, creates the controller, runs in viewer or headless mode.
Auto-creates timestamped output folders:
    outputs/data/run_MM_DD_YYYY/results_FARMS.npz
    outputs/videos/run_MM_DD_YYYY/output_FARMS.mp4   (if --video flag used)

Usage:
    python run.py                                    # viewer, default config
    python farms_run.py --config my_config.yaml      # custom config
    python farms_run.py --headless --duration 10     # headless + record data
    python farms_run.py --video out.mp4              # offscreen video
    python farms_run.py --headless --video out.mp4 --duration 10
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import mujoco
import mujoco.viewer

from kinematics import FARMSModelIndex, N_BODY_JOINTS, N_LEGS, N_LEG_DOF
from controller import FARMSTravelingWaveController, load_config

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.join(SCRIPT_DIR, "..", "..")   # project root
DEFAULT_XML = os.path.join(BASE_DIR, "models", "farms", "centipede.xml")
DEFAULT_CFG  = os.path.join(BASE_DIR, "configs", "farms_controller.yaml")
DATA_DIR     = os.path.join(BASE_DIR, "outputs", "data")
VIDEO_DIR    = os.path.join(BASE_DIR, "outputs", "videos")


def make_run_tag():
    return f"run_{datetime.now().strftime('%m_%d_%Y_%H%M%S')}"


# ── Recorder ──────────────────────────────────────────────────────────────────

class FARMSRecorder:
    """Records COM trajectory, body joint angles, and leg joint angles."""

    def __init__(self, model, idx, dt_record=0.01):
        self.model     = model
        self.idx       = idx
        self.dt_record = dt_record
        self._last_t   = -np.inf

        self.times        = []
        self.com_pos      = []   # (T, 3)
        self.com_vel      = []   # (T, 3)
        self.body_jnt_pos = []   # (T, N_BODY_JOINTS)
        self.leg_jnt_pos  = []   # (T, N_LEGS, 2, N_LEG_DOF)

    def maybe_record(self, data):
        if data.time - self._last_t < self.dt_record - 1e-10:
            return
        self._last_t = data.time
        self.times.append(data.time)
        self.com_pos.append(self.idx.com_pos(data))
        self.com_vel.append(self.idx.com_vel(data))

        bj = np.array([self.idx.body_joint_pos(data, i + 1)
                       for i in range(N_BODY_JOINTS)])
        self.body_jnt_pos.append(bj)

        lj = np.zeros((N_LEGS, 2, N_LEG_DOF))
        for n in range(N_LEGS):
            for si, side in enumerate(('L', 'R')):
                for dof in range(N_LEG_DOF):
                    lj[n, si, dof] = self.idx.leg_joint_pos(data, n, side, dof)
        self.leg_jnt_pos.append(lj)

    def save(self, path):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.savez_compressed(
            path,
            time          = np.array(self.times),
            com_pos       = np.array(self.com_pos),
            com_vel       = np.array(self.com_vel),
            body_jnt_pos  = np.array(self.body_jnt_pos),
            leg_jnt_pos   = np.array(self.leg_jnt_pos),
        )
        print(f"  Saved {len(self.times)} frames → {path}")


# ── Camera settings ───────────────────────────────────────────────────────────

CAM_DISTANCE  = 0.2    # metres — tight framing for ~10cm centipede
CAM_AZIMUTH   = 60      # degrees
CAM_ELEVATION = -35     # degrees


def _update_camera(model, data, viewer, idx):
    """Keep camera lookat on root body COM."""
    com = idx.com_pos(data)
    viewer.cam.lookat[:] = com
    viewer.cam.distance  = CAM_DISTANCE
    viewer.cam.azimuth   = CAM_AZIMUTH
    viewer.cam.elevation = CAM_ELEVATION


def _make_tracking_camera(idx, data):
    """Create a MuJoCo camera struct that tracks the centipede COM."""
    cam = mujoco.MjvCamera()
    com = idx.com_pos(data)
    cam.lookat[:] = com
    cam.distance  = CAM_DISTANCE
    cam.azimuth   = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    return cam


def _update_tracking_camera(cam, idx, data):
    """Update a tracking camera's lookat to follow COM."""
    cam.lookat[:] = idx.com_pos(data)


# ── Viewer mode ───────────────────────────────────────────────────────────────

def run_viewer(model, data, ctrl, cfg, recorder=None):
    duration = cfg["simulation"]["duration"]
    dt       = model.opt.timestep
    rt_ratio = 1.0   # real-time ratio

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.time()
        while viewer.is_running() and data.time < duration:
            step_start = time.time()

            ctrl.step(model, data)
            mujoco.mj_step(model, data)
            if recorder:
                recorder.maybe_record(data)

            _update_camera(model, data, viewer, ctrl.idx)
            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - step_start
            remaining = dt / rt_ratio - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print(f"  Simulation finished at t={data.time:.3f}s")


# ── Headless mode ─────────────────────────────────────────────────────────────

def run_headless(model, data, ctrl, cfg, recorder=None, video_path=None):
    duration = cfg["simulation"]["duration"]
    dt       = model.opt.timestep
    n_steps  = int(duration / dt)

    # Optional video renderer
    renderer = None
    frames   = []
    cam      = None
    video_dt = cfg.get("recording", {}).get("dt_record", 0.01)
    last_frame_t = -np.inf

    if video_path:
        try:
            import mediapy
            vid_w = min(1280, model.vis.global_.offwidth)
            vid_h = min(720,  model.vis.global_.offheight)
            renderer = mujoco.Renderer(model, height=vid_h, width=vid_w)
            cam = _make_tracking_camera(ctrl.idx, data)
            print(f"  Video → {video_path}  (tracking camera, 1280×720)")
        except ImportError:
            print("  WARNING: mediapy not installed — skipping video.")
            video_path = None

    print(f"  Running {n_steps} steps ({duration:.1f}s) headless...")
    t0 = time.time()
    for _ in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)
        if recorder:
            recorder.maybe_record(data)
        if renderer and video_path:
            # Only render at video framerate, not every physics step
            if data.time - last_frame_t >= video_dt - 1e-10:
                _update_tracking_camera(cam, ctrl.idx, data)
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())
                last_frame_t = data.time

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s wall ({elapsed/duration:.2f}× real-time)")

    if video_path and frames:
        import mediapy
        dirname = os.path.dirname(video_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        fps = int(1.0 / video_dt)
        mediapy.write_video(video_path, frames, fps=fps)
        print(f"  Video saved → {video_path}  ({len(frames)} frames, {fps} fps)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FARMS centipede simulation runner")
    parser.add_argument("--model",    default=DEFAULT_XML,  help="Path to centipede.xml")
    parser.add_argument("--config",   default=DEFAULT_CFG,  help="Path to farms_config.yaml")
    parser.add_argument("--headless", action="store_true",  help="Run without viewer")
    parser.add_argument("--duration", type=float,           help="Override simulation duration (s)")
    parser.add_argument("--video",    default=None,         help="Save video to this path")
    parser.add_argument("--output",   default=None,         help="Override .npz output path")
    args = parser.parse_args()

    # ── Load model and config ──────────────────────────────────────────────
    print(f"[farms_run] Loading model: {args.model}")
    model = mujoco.MjModel.from_xml_path(args.model)
    data  = mujoco.MjData(model)

    cfg = load_config(args.config)
    if args.duration:
        cfg["simulation"]["duration"] = args.duration

    # ── Controller ────────────────────────────────────────────────────────
    ctrl = FARMSTravelingWaveController(model, config_path=args.config)

    # ── Recorder ──────────────────────────────────────────────────────────
    tag      = make_run_tag()
    npz_path = args.output or os.path.join(DATA_DIR, tag, "results_FARMS.npz")
    recorder = FARMSRecorder(model, ctrl.idx,
                             dt_record=cfg["recording"]["dt_record"])

    video_path = args.video
    if video_path is None and args.headless:
        pass  # no video unless explicitly requested
    elif video_path is not None and not os.path.isabs(video_path):
        video_path = os.path.join(VIDEO_DIR, tag, video_path)

    # ── Run ───────────────────────────────────────────────────────────────
    print(f"[farms_run] duration={cfg['simulation']['duration']}s  "
          f"mode={'headless' if args.headless else 'viewer'}")

    if args.headless:
        run_headless(model, data, ctrl, cfg,
                     recorder=recorder, video_path=video_path)
    else:
        run_viewer(model, data, ctrl, cfg, recorder=recorder)

    # ── Save data ─────────────────────────────────────────────────────────
    recorder.save(npz_path)
    print("[farms_run] Done.")


if __name__ == "__main__":
    main()
