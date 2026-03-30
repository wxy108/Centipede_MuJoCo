"""
run.py — Main simulation runner
================================
Loads config.yaml, creates the controller, runs the simulation.
Auto-creates timestamped output folders:
    4-Data/run_MM_DD_YYYY/results.npz
    5-Video/run_MM_DD_YYYY/output.mp4

Usage:
    python run.py                           # viewer mode, stops at config duration
    python run.py --config custom.yaml      # custom config
    python run.py --headless --duration 10  # headless + record
    python run.py --video out.mp4 --duration 10 --fps 60  # offscreen video
"""

import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np
import mujoco
import mujoco.viewer

from kinematics import ModelIndex, N_BODY_JOINTS, N_LEGS_PER_SIDE, body_joint_name, leg_joint_name, touch_sensor_name
from controller import TravelingWaveController, load_config

# ── Paths (relative to this script) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..")      # 3-MUJOCO/
DEFAULT_MODEL = os.path.join(BASE_DIR, "3-Model", "1-Centipede", "centipede.xml")
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "config.yaml")
DATA_DIR = os.path.join(BASE_DIR, "4-Data")
VIDEO_DIR = os.path.join(BASE_DIR, "5-Video")


def make_run_tag():
    """Generate a timestamped run folder name, e.g. run_03_23_2026"""
    return f"run_{datetime.now().strftime('%m_%d_%Y')}"


class Recorder:
    """
    Records simulation data into numpy arrays.

    Saved arrays (in results.npz):
        time              (T,)              simulation time
        com_pos           (T, 3)            b0 subtree COM position
        com_vel           (T, 3)            b0 subtree COM linear velocity
        body_joint_pos    (T, 20)           actual body joint positions   [rad]
        body_joint_vel    (T, 20)           actual body joint velocities  [rad/s]
        body_joint_pos_cmd (T, 20)          commanded body positions      [rad]
        body_joint_vel_cmd (T, 20)          commanded body velocities     [rad/s]
        leg_joint_pos     (T, 19, 2, 3)     actual leg joint positions    [rad]
        leg_joint_vel     (T, 19, 2, 3)     actual leg joint velocities   [rad/s]
        leg_joint_pos_cmd  (T, 19, 2, 3)    commanded leg positions       [rad]
        leg_joint_vel_cmd  (T, 19, 2, 3)    commanded leg velocities      [rad/s]
        touch_forces      (T, 19, 2)        foot touch sensor values      [N]

    Having both actual and commanded velocities lets analyze_tracking.py
    plot and quantify PD velocity tracking without reconstructing anything.
    """

    def __init__(self, model, idx, ctrl, dt_record=0.01):
        self.model     = model
        self.idx       = idx
        self.ctrl      = ctrl      # needed to read commanded signals each frame
        self.dt_record = dt_record
        self.last_t    = -np.inf

        self.times   = []
        self.com_pos = []
        self.com_vel = []

        # Actual
        self.body_jp  = []   # positions
        self.body_jv  = []   # velocities
        self.leg_jp   = []
        self.leg_jv   = []
        self.touch    = []

        # Commanded
        self.body_jp_cmd = []
        self.body_jv_cmd = []
        self.leg_jp_cmd  = []
        self.leg_jv_cmd  = []

    def maybe_record(self, data):
        if data.time - self.last_t < self.dt_record - 1e-10:
            return
        self.last_t = data.time
        t = data.time
        self.times.append(t)

        # ── COM ──
        b0 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "b0")
        self.com_pos.append(data.subtree_com[b0].copy())
        self.com_vel.append(data.subtree_linvel[b0].copy())

        # ── Body joints: actual pos + vel ──
        bj_pos = np.zeros(N_BODY_JOINTS)
        bj_vel = np.zeros(N_BODY_JOINTS)
        for i in range(N_BODY_JOINTS):
            jn = body_joint_name(i + 1)
            bj_pos[i] = self.idx.get_joint_pos(data, jn)
            bj_vel[i] = self.idx.get_joint_vel(data, jn)
        self.body_jp.append(bj_pos)
        self.body_jv.append(bj_vel)

        # ── Body joints: commanded pos + vel ──
        bj_cmd_pos = np.zeros(N_BODY_JOINTS)
        bj_cmd_vel = np.zeros(N_BODY_JOINTS)
        for i in range(N_BODY_JOINTS):
            q, q_dot = self.ctrl.traj.body_target(i, t)
            bj_cmd_pos[i] = q
            bj_cmd_vel[i] = q_dot
        self.body_jp_cmd.append(bj_cmd_pos)
        self.body_jv_cmd.append(bj_cmd_vel)

        # ── Leg joints: actual pos + vel ──
        lj_pos = np.zeros((N_LEGS_PER_SIDE, 2, 3))
        lj_vel = np.zeros((N_LEGS_PER_SIDE, 2, 3))
        tf     = np.zeros((N_LEGS_PER_SIDE, 2))
        for n in range(N_LEGS_PER_SIDE):
            for si, s in enumerate(('L', 'R')):
                for d in range(3):
                    jn = leg_joint_name(n + 1, s, d)
                    lj_pos[n, si, d] = self.idx.get_joint_pos(data, jn)
                    lj_vel[n, si, d] = self.idx.get_joint_vel(data, jn)
                tf[n, si] = self.idx.get_touch_force(data, n + 1, s)
        self.leg_jp.append(lj_pos)
        self.leg_jv.append(lj_vel)
        self.touch.append(tf)

        # ── Leg joints: commanded pos + vel ──
        lj_cmd_pos = np.zeros((N_LEGS_PER_SIDE, 2, 3))
        lj_cmd_vel = np.zeros((N_LEGS_PER_SIDE, 2, 3))
        for n in range(N_LEGS_PER_SIDE):
            for si in range(2):
                for d in range(3):
                    q, q_dot = self.ctrl.traj.leg_target(n, si, d, t)
                    lj_cmd_pos[n, si, d] = q
                    lj_cmd_vel[n, si, d] = q_dot
        self.leg_jp_cmd.append(lj_cmd_pos)
        self.leg_jv_cmd.append(lj_cmd_vel)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            time               = np.array(self.times),
            com_pos            = np.array(self.com_pos),
            com_vel            = np.array(self.com_vel),
            # Actual
            body_joint_pos     = np.array(self.body_jp),
            body_joint_vel     = np.array(self.body_jv),
            leg_joint_pos      = np.array(self.leg_jp),
            leg_joint_vel      = np.array(self.leg_jv),
            touch_forces       = np.array(self.touch),
            # Commanded
            body_joint_pos_cmd = np.array(self.body_jp_cmd),
            body_joint_vel_cmd = np.array(self.body_jv_cmd),
            leg_joint_pos_cmd  = np.array(self.leg_jp_cmd),
            leg_joint_vel_cmd  = np.array(self.leg_jv_cmd),
        )
        print(f"Saved {len(self.times)} frames -> {path}")


def run_viewer(model, data, ctrl, duration, recorder=None):
    """Interactive viewer with real-time pacing. Stops at duration."""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.3
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -50
        b0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b0")
        viewer.cam.lookat[:] = data.subtree_com[b0]

        print(f"Running: n={ctrl.n_wave}, A_body={ctrl.body_amp:.2f}, "
              f"f={ctrl.frequency:.1f}Hz, duration={duration:.1f}s. "
              f"Close window or wait to stop.")

        t_print = 0
        while viewer.is_running() and data.time < duration:
            t0 = time.time()
            ctrl.step(data)
            mujoco.mj_step(model, data)
            if recorder:
                recorder.maybe_record(data)

            if data.time - t_print > 2.0:
                cx = data.subtree_com[b0][0]
                print(f"  t={data.time:6.2f}s  COM_x={cx:+.4f}m")
                t_print = data.time

            viewer.cam.lookat[:] = data.subtree_com[b0]
            viewer.sync()

            dt = time.time() - t0
            if dt < model.opt.timestep:
                time.sleep(model.opt.timestep - dt)

        print(f"Viewer finished at t={data.time:.2f}s")


def run_headless(model, data, ctrl, duration, recorder):
    """Headless simulation for fixed duration."""
    n_steps = int(duration / model.opt.timestep)
    b0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b0")
    t0_wall = time.time()
    t_print = 0

    print(f"Headless: {duration}s, {n_steps} steps...")
    for _ in range(n_steps):
        ctrl.step(data)
        mujoco.mj_step(model, data)
        recorder.maybe_record(data)

        if data.time - t_print > 1.0:
            elapsed = time.time() - t0_wall
            rtf = data.time / elapsed if elapsed > 0 else 0
            print(f"  t={data.time:6.2f}s  COM_x={data.subtree_com[b0][0]:+.4f}m  RTF={rtf:.1f}x")
            t_print = data.time

    wall = time.time() - t0_wall
    print(f"Done: {duration}s in {wall:.1f}s (RTF={duration / wall:.1f}x)")


def run_video(model, data, ctrl, duration, recorder, video_path, fps=60):
    """Render offscreen and write video file."""
    import mediapy as media

    width, height = 1920, 1080
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, height, width)

    # Camera setup (matches viewer defaults)
    cam = mujoco.MjvCamera()
    cam.distance = 0.3
    cam.azimuth = 45
    cam.elevation = -50
    b0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b0")
    cam.lookat[:] = data.subtree_com[b0]

    # Scene options: only show visual geoms (group 1)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[0] = True   # group 0 (unused)
    scene_option.geomgroup[1] = True    # group 1 (visual meshes)
    scene_option.geomgroup[2] = False   # group 2 (collision)
    scene_option.geomgroup[3] = False   # group 3 (foot spheres)

    n_steps = int(duration / model.opt.timestep)
    frame_interval = max(1, int(1.0 / (fps * model.opt.timestep)))
    frames = []
    t_print = 0

    print(f"Recording: {duration}s @ {fps}fps, {width}x{height} -> {video_path}")
    t0 = time.time()

    for step in range(n_steps):
        ctrl.step(data)
        mujoco.mj_step(model, data)
        if recorder:
            recorder.maybe_record(data)

        # Capture frame at the desired FPS
        if step % frame_interval == 0:
            cam.lookat[:] = data.subtree_com[b0]
            renderer.update_scene(data, cam, scene_option=scene_option)
            frames.append(renderer.render().copy())

        if data.time - t_print > 1.0:
            elapsed = time.time() - t0
            rtf = data.time / elapsed if elapsed > 0 else 0
            n_frames = len(frames)
            print(f"  t={data.time:6.2f}s  frames={n_frames}  "
                  f"[{100 * step / n_steps:.0f}%]  RTF={rtf:.1f}x")
            t_print = data.time

    renderer.close()

    # Write video
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    media.write_video(video_path, frames, fps=fps)
    wall = time.time() - t0
    print(f"Saved {len(frames)} frames -> {video_path}  "
          f"({wall:.1f}s wall time)")


def main():
    p = argparse.ArgumentParser(description="Centipede MuJoCo simulation")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--video", default=None, help="Output video filename (e.g. test.mp4)")
    p.add_argument("--fps", type=int, default=60, help="Video FPS (default 60)")
    p.add_argument("--duration", type=float, default=None,
                   help="Simulation duration (overrides config)")
    p.add_argument("--output", default="results.npz",
                   help="Data output filename (e.g. results.npz)")
    p.add_argument("--tag", default=None,
                   help="Run folder name (default: run_MM_DD_YYYY)")
    args = p.parse_args()

    # Load
    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found")
        sys.exit(1)

    config = load_config(args.config)
    print(f"Model:  {model_path}")
    print(f"Config: {os.path.abspath(args.config)}")

    # Duration: CLI overrides config
    duration = args.duration
    if duration is None:
        duration = config.get('simulation', {}).get('duration', 10.0)

    # ── Output paths ──
    run_tag = args.tag if args.tag else make_run_tag()

    data_dir = os.path.join(DATA_DIR, run_tag)
    video_dir = os.path.join(VIDEO_DIR, run_tag)
    os.makedirs(data_dir, exist_ok=True)

    # Resolve output filenames into timestamped folders
    if os.path.dirname(args.output) == "":
        args.output = os.path.join(data_dir, args.output)
    if args.video and os.path.dirname(args.video) == "":
        os.makedirs(video_dir, exist_ok=True)
        args.video = os.path.join(video_dir, args.video)

    print(f"  Data  -> {data_dir}")
    if args.video:
        print(f"  Video -> {video_dir}")

    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    print(f"  {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators, {model.nv} DOF")
    print(f"  Duration: {duration}s")

    # Controller
    ctrl = TravelingWaveController(model, config=config)
    bw = config['body_wave']
    print(f"  wave: n={bw['wave_number']}, A_body={bw['amplitude']}, "
          f"f={bw['frequency']}Hz, speed={bw['speed']}")
    lw = config['leg_wave']
    print(f"  legs: A={lw['amplitudes']}, "
          f"phase={[f'{x:.2f}' for x in lw['phase_offsets']]}, "
          f"active={lw.get('active_dofs', [0, 1])}")

    # Also copy config into data folder for reproducibility
    import shutil
    config_copy = os.path.join(data_dir, "config.yaml")
    shutil.copy2(os.path.abspath(args.config), config_copy)
    print(f"  Config copied -> {config_copy}")

    # Init
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    idx = ModelIndex(model)
    dt_rec = config.get('simulation', {}).get('record_interval', 0.01)
    recorder = Recorder(model, idx, ctrl, dt_record=dt_rec)

    # Run
    if args.video:
        run_video(model, data, ctrl, duration, recorder,
                  args.video, args.fps)
        if recorder.times:
            recorder.save(args.output)
    elif args.headless:
        run_headless(model, data, ctrl, duration, recorder)
        recorder.save(args.output)
    else:
        try:
            run_viewer(model, data, ctrl, duration, recorder)
        except KeyboardInterrupt:
            print("\nStopped.")
        if recorder.times:
            recorder.save(args.output)

    print(f"\nRun complete: {run_tag}")


if __name__ == "__main__":
    main()