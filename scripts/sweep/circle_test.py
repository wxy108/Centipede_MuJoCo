#!/usr/bin/env python3
"""
circle_test.py — make the centipede follow a circular trajectory on flat ground.

The heading servo on joint_body_0 is augmented with a `head_yaw_rate` (rad/s)
that rotates the reference yaw after the gait switches on.  When the head
tracks a slowly rotating heading and the body walks forward at speed v, the
COM traces a circle of radius R = v / |yaw_rate|.

Usage
-----
  # direct: yaw rate in rad/s (negative = clockwise = yaw decreasing)
  python scripts/sweep/circle_test.py --yaw-rate -0.1 --duration 30 --video

  # geometric: target circle radius (m) + nominal forward speed (m/s)
  python scripts/sweep/circle_test.py --radius 0.15 --speed 0.02 --duration 30 --video

Writes an MP4 and a CSV log of the COM trajectory.  Also draws a top-down
PNG of the (x, y) path for quick inspection.
"""

import argparse
import math
import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import yaml

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))

import mujoco  # noqa: E402

from impedance_controller import ImpedanceTravelingWaveController  # noqa: E402
from kinematics import FARMSModelIndex  # noqa: E402

from flat_test import patch_xml_plane_only  # reuse hfield-stripping patch
from wavelength_sweep import (  # noqa: E402
    XML_PATH, CONFIG_PATH,
    VID_W, VID_H, VID_FPS,
    CAM_DISTANCE, CAM_ELEVATION,
)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "circle_test")


def make_temp_config(yaw_rate):
    """Write a temp YAML with head_yaw_rate patched in."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("impedance", {})["head_yaw_rate"] = float(yaw_rate)
    fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="_circle_cfg_",
                                    dir=os.path.dirname(CONFIG_PATH))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp_path


def try_plot(xy, out_png, radius_target=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    xy = np.asarray(xy)
    if xy.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xy[:, 0] * 1000, xy[:, 1] * 1000, '-', lw=1.2, color='C0',
            label='COM')
    ax.plot(xy[0, 0] * 1000, xy[0, 1] * 1000, 'go', label='start')
    ax.plot(xy[-1, 0] * 1000, xy[-1, 1] * 1000, 'rs', label='end')
    if radius_target:
        th = np.linspace(0, 2 * math.pi, 200)
        # Show the target circle centred at (0, R*sign) — tangent at origin
        # assuming the robot starts heading along +x; sign follows yaw_rate.
        r_mm = radius_target * 1000
        # Draw both candidate centres (±y) since sign depends on yaw_rate sign
        for cy in (r_mm, -r_mm):
            ax.plot(r_mm * np.cos(th), cy + r_mm * np.sin(th), '--',
                    color='gray', lw=0.7, alpha=0.7)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(os.path.basename(out_png))
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--yaw-rate", type=float, default=None,
                   help="Target yaw rate (rad/s). Negative → clockwise.")
    p.add_argument("--radius",   type=float, default=None,
                   help="Target circle radius (m). Use with --speed.")
    p.add_argument("--speed",    type=float, default=0.02,
                   help="Nominal forward speed (m/s), only used with --radius. "
                        "Default 0.02 m/s ~ 20 mm/s.")
    p.add_argument("--sign",     choices=("cw", "ccw"), default="cw",
                   help="Direction when using --radius (cw = yaw decreasing).")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Total sim duration (s). Default 30 s.")
    p.add_argument("--fps",      type=int,   default=VID_FPS)
    p.add_argument("--cam-distance", type=float, default=0.40,
                   help="Camera orbit distance (m). Default 0.40 (top-down view).")
    p.add_argument("--cam-elevation", type=float, default=-80.0,
                   help="Camera elevation (deg). -90=top-down. Default -80.")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    # ── Compute yaw_rate ──────────────────────────────────────────────────────
    if args.yaw_rate is None and args.radius is None:
        args.yaw_rate = -0.05  # gentle clockwise default
    if args.yaw_rate is None:
        mag = args.speed / max(args.radius, 1e-6)
        args.yaw_rate = -mag if args.sign == "cw" else mag

    radius_target = (args.speed / abs(args.yaw_rate)
                     if abs(args.yaw_rate) > 1e-9 else float('inf'))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT_DIR, exist_ok=True)
    run_tag = f"yaw{args.yaw_rate:+.4f}_d{args.duration:.0f}s"
    out_mp4 = os.path.join(OUT_DIR, f"circle_{ts}_{run_tag}.mp4")
    out_csv = os.path.join(OUT_DIR, f"circle_{ts}_{run_tag}.csv")
    out_png = os.path.join(OUT_DIR, f"circle_{ts}_{run_tag}.png")

    # ── Video availability ────────────────────────────────────────────────────
    can_video = False
    if not args.no_video:
        try:
            import mediapy  # noqa
            can_video = True
        except ImportError:
            print("  WARNING: mediapy not installed — skipping video.")

    # ── Build model on flat ground ────────────────────────────────────────────
    tmp_xml    = patch_xml_plane_only(XML_PATH, spawn_z=0.015)
    tmp_config = make_temp_config(args.yaw_rate)

    print("=" * 70)
    print("Circle Trajectory Test")
    print("=" * 70)
    print(f"  yaw_rate = {args.yaw_rate:+.4f} rad/s "
          f"({math.degrees(args.yaw_rate):+.2f} deg/s)")
    print(f"  target radius ≈ {radius_target*1000:.1f} mm  "
          f"(at nominal v = {args.speed*1000:.1f} mm/s)")
    print(f"  duration = {args.duration:.1f} s")
    print(f"  video   = {out_mp4 if can_video else 'OFF'}")
    print(f"  csv     = {out_csv}")
    print()

    model = mujoco.MjModel.from_xml_path(tmp_xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ctrl = ImpedanceTravelingWaveController(model, tmp_config)
    idx  = FARMSModelIndex(model)

    dt = model.opt.timestep
    n_steps = int(args.duration / dt)

    renderer = None
    cam      = None
    if can_video:
        renderer = mujoco.Renderer(model, height=VID_H, width=VID_W)
        cam = mujoco.MjvCamera()
        cam.lookat[:]   = idx.com_pos(data)
        cam.distance    = args.cam_distance
        cam.elevation   = args.cam_elevation
        cam.azimuth     = 90.0

    frames = []
    frame_dt = 1.0 / args.fps
    last_frame_t = -1.0

    # Log the COM every ~10 ms
    log_interval = max(int(0.01 / dt), 1)
    traj = []   # (t, x, y, z, yaw_ref, yaw_world)
    head_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        if step_i % log_interval == 0:
            com = idx.com_pos(data)
            Rh = data.xmat[head_body_id].reshape(3, 3)
            yaw_world = math.atan2(Rh[1, 0], Rh[0, 0])
            yaw_ref = ctrl.head_yaw_ref if ctrl.head_yaw_ref is not None else 0.0
            traj.append((data.time, com[0], com[1], com[2], yaw_ref, yaw_world))

        if renderer is not None and data.time - last_frame_t >= frame_dt - 1e-6:
            cam.lookat[:] = idx.com_pos(data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
            last_frame_t = data.time

    # ── Save outputs ──────────────────────────────────────────────────────────
    traj_arr = np.array(traj)
    with open(out_csv, "w") as f:
        f.write("t,x,y,z,yaw_ref,yaw_world\n")
        for row in traj_arr:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")

    if renderer is not None and frames:
        import mediapy
        mediapy.write_video(out_mp4, frames, fps=args.fps)

    # Top-down plot
    xy = traj_arr[:, 1:3] if traj_arr.size else np.zeros((0, 2))
    plotted = try_plot(xy, out_png, radius_target=radius_target)

    # Clean up
    for path in (tmp_xml, tmp_config):
        if os.path.exists(path):
            os.remove(path)

    # ── Summary ───────────────────────────────────────────────────────────────
    if traj_arr.size:
        start_xy = traj_arr[0, 1:3]
        end_xy   = traj_arr[-1, 1:3]
        path_len = float(np.sum(np.linalg.norm(
            np.diff(traj_arr[:, 1:3], axis=0), axis=1)))
        yaw_drift = float(traj_arr[-1, 5] - traj_arr[0, 5])
    else:
        start_xy = end_xy = (0.0, 0.0)
        path_len = 0.0
        yaw_drift = 0.0

    print("\n" + "=" * 70)
    print(f"DONE  (t = {data.time:.2f} s)")
    print("=" * 70)
    print(f"  start (mm): ({start_xy[0]*1000:+6.1f}, {start_xy[1]*1000:+6.1f})")
    print(f"  end   (mm): ({end_xy[0]*1000:+6.1f}, {end_xy[1]*1000:+6.1f})")
    print(f"  path length: {path_len*1000:.1f} mm")
    print(f"  yaw drift:   {math.degrees(yaw_drift):+7.1f} deg "
          f"(target {math.degrees(args.yaw_rate * args.duration):+.1f} deg)")
    print(f"\n  CSV:   {out_csv}")
    if plotted:
        print(f"  plot:  {out_png}")
    if renderer is not None and frames:
        print(f"  video: {out_mp4}")


if __name__ == "__main__":
    main()
