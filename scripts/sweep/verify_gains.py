#!/usr/bin/env python3
"""
verify_gains.py — Visual A/B test: old vs new pitch gains on same terrain.

Runs two simulations on identical rough terrain, recording:
  A) Old gains: pitch_kp=0.0087, pitch_kv=0.0023 (v2)
  B) New gains: pitch_kp=0.0706, pitch_kv=0.0482 (v2.1)

For each, prints body conformity metrics and optionally saves video.
This helps answer: does the body actually conform to terrain, or stay rigid?

Usage
-----
  python scripts/sweep/verify_gains.py
  python scripts/sweep/verify_gains.py --video
  python scripts/sweep/verify_gains.py --duration 8 --video
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import mujoco

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "terrain", "generator"))

from impedance_controller import ImpedanceTravelingWaveController, load_config
from kinematics import FARMSModelIndex

XML_PATH    = os.path.join(PROJECT_ROOT, "models", "farms", "centipede.xml")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "farms_controller.yaml")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "outputs", "verify_gains")

# Video settings
VID_W, VID_H = 1280, 720
VID_FPS      = 30
CAM_DISTANCE = 0.20
CAM_AZIMUTH  = 60
CAM_ELEVATION = -35

# Two gain sets to compare
GAIN_SETS = {
    "v2_soft": {
        "label": "v2 (soft): kp=0.0087, kv=0.0023",
        "pitch_kp": 0.008692,
        "pitch_kv": 0.002285,
    },
    "v2.1_stiff": {
        "label": "v2.1 (stiff): kp=0.0706, kv=0.0482",
        "pitch_kp": 0.070565,
        "pitch_kv": 0.048195,
    },
}


class TerrainSampler:
    def __init__(self, model):
        hf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        self.nrow = model.hfield_nrow[hf_id]
        self.ncol = model.hfield_ncol[hf_id]
        self.x_half = model.hfield_size[hf_id, 0]
        self.y_half = model.hfield_size[hf_id, 1]
        self.z_top  = model.hfield_size[hf_id, 2]
        n_data = self.nrow * self.ncol
        start = model.hfield_adr[hf_id]
        self.data = model.hfield_data[start:start + n_data].reshape(
            self.nrow, self.ncol).copy()
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_geom")
        self.offset = model.geom_pos[geom_id].copy() if geom_id >= 0 else np.zeros(3)

    def get_height(self, x, y):
        lx = x - self.offset[0]
        ly = y - self.offset[1]
        u = np.clip((lx + self.x_half) / (2.0 * self.x_half), 0, 1 - 1e-10)
        v = np.clip((ly + self.y_half) / (2.0 * self.y_half), 0, 1 - 1e-10)
        col = u * (self.ncol - 1)
        row = v * (self.nrow - 1)
        c0, r0 = int(col), int(row)
        c1, r1 = min(c0 + 1, self.ncol - 1), min(r0 + 1, self.nrow - 1)
        fc, fr = col - c0, row - r0
        h = (self.data[r0, c0] * (1 - fc) * (1 - fr) +
             self.data[r0, c1] * fc * (1 - fr) +
             self.data[r1, c0] * (1 - fc) * fr +
             self.data[r1, c1] * fc * fr)
        return self.offset[2] + h * self.z_top


def run_trial(gains, duration, save_video=None):
    """
    Run one simulation with the given pitch gains.
    Returns a dict of detailed metrics about body conformity.
    """
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    ctrl = ImpedanceTravelingWaveController(
        model, CONFIG_PATH,
        pitch_kp=gains["pitch_kp"],
        pitch_kv=gains["pitch_kv"],
    )
    idx = FARMSModelIndex(model)
    terrain = TerrainSampler(model)

    # Find pitch joint IDs and body IDs for per-segment terrain height
    pitch_jnt_ids = []
    pitch_body_ids = []
    for j in range(model.njnt):
        nm = model.joint(j).name
        if nm and 'joint_pitch_body' in nm:
            pitch_jnt_ids.append(j)
            pitch_body_ids.append(model.jnt_bodyid[j])

    n_steps = int(duration / model.opt.timestep)
    dt = model.opt.timestep
    SETTLE = 500

    # Video setup
    renderer = None
    frames = []
    vid_cam = None
    last_frame_t = -1.0

    if save_video:
        try:
            import mediapy  # noqa
            renderer = mujoco.Renderer(model, height=VID_H, width=VID_W)
            vid_cam = mujoco.MjvCamera()
            vid_cam.distance = CAM_DISTANCE
            vid_cam.azimuth = CAM_AZIMUTH
            vid_cam.elevation = CAM_ELEVATION
        except (ImportError, Exception):
            save_video = None

    # Time-series storage
    pitch_angles_ts = []       # per-timestep mean |pitch| (deg)
    terrain_height_ts = []     # terrain height under COM
    body_height_ts = []        # actual body COM z
    conformity_errors = []     # per-segment: |body_z - terrain_z - nominal|
    energy_sum = 0.0

    # Get nominal clearance (body height on flat ground)
    NOMINAL_CLEARANCE = 0.0258  # from CLAUDE.md

    total_mass = sum(model.body_mass[i] for i in range(model.nbody))
    gravity = abs(model.opt.gravity[2])
    start_pos = None

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        if step_i == SETTLE:
            com = idx.com_pos(data)
            start_pos = com.copy()

        # Sample every 50 steps after settling
        if step_i > SETTLE and step_i % 50 == 0:
            com = idx.com_pos(data)

            # Mean pitch angle
            pitches = [data.qpos[model.jnt_qposadr[j]] for j in pitch_jnt_ids]
            mean_pitch_deg = float(np.mean(np.abs(pitches))) * 180.0 / math.pi
            pitch_angles_ts.append(mean_pitch_deg)

            # COM height vs terrain
            t_h = terrain.get_height(com[0], com[1])
            terrain_height_ts.append(t_h)
            body_height_ts.append(com[2])

            # Per-segment conformity: for each pitch body, check how close
            # body_z is to terrain_z + nominal
            seg_errors = []
            for bid in pitch_body_ids:
                bx, by, bz = data.xpos[bid]
                th = terrain.get_height(bx, by)
                expected_z = th + NOMINAL_CLEARANCE
                seg_errors.append(abs(bz - expected_z))
            conformity_errors.append(np.mean(seg_errors))

            # Energy
            for a in range(model.nu):
                tau = abs(data.actuator_force[a])
                jnt_id = model.actuator_trnid[a, 0]
                if 0 <= jnt_id < model.njnt:
                    dof_adr = model.jnt_dofadr[jnt_id]
                    omega = abs(data.qvel[dof_adr])
                    energy_sum += tau * omega * dt * 50  # ×50 because sampling every 50 steps

        # Video
        if renderer and save_video:
            vid_dt = 1.0 / VID_FPS
            if data.time - last_frame_t >= vid_dt - 1e-6:
                vid_cam.lookat[:] = idx.com_pos(data)
                renderer.update_scene(data, camera=vid_cam)
                frames.append(renderer.render().copy())
                last_frame_t = data.time

    # Save video
    if save_video and frames:
        import mediapy
        os.makedirs(os.path.dirname(save_video), exist_ok=True)
        mediapy.write_video(save_video, frames, fps=VID_FPS)

    # Compute final metrics
    end_pos = idx.com_pos(data)
    if start_pos is not None:
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
    else:
        distance = 0.0

    eff_time = max(data.time - SETTLE * dt, 0.01)
    speed = distance / eff_time

    if distance > 0.001:
        cot = energy_sum / (total_mass * gravity * distance)
    else:
        cot = float('inf')

    # Body height variation vs terrain height variation
    body_z = np.array(body_height_ts)
    terr_z = np.array(terrain_height_ts)
    clearance = body_z - terr_z

    return {
        "mean_pitch_deg":      float(np.mean(pitch_angles_ts)) if pitch_angles_ts else 0,
        "max_pitch_deg":       float(np.max(pitch_angles_ts)) if pitch_angles_ts else 0,
        "mean_conformity_mm":  float(np.mean(conformity_errors) * 1000) if conformity_errors else 0,
        "p90_conformity_mm":   float(np.percentile(conformity_errors, 90) * 1000) if conformity_errors else 0,
        "clearance_mean_mm":   float(np.mean(clearance) * 1000) if len(clearance) else 0,
        "clearance_std_mm":    float(np.std(clearance) * 1000) if len(clearance) else 0,
        "terrain_std_mm":      float(np.std(terr_z) * 1000) if len(terr_z) else 0,
        "body_z_std_mm":       float(np.std(body_z) * 1000) if len(body_z) else 0,
        "speed_mm_s":          float(speed * 1000),
        "distance_mm":         float(distance * 1000),
        "cot":                 float(min(cot, 1e6)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--video", action="store_true",
                        help="Save comparison videos")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Pitch Gain Verification — A/B Comparison")
    print("=" * 70)
    print(f"  Terrain: current model default (from centipede.xml)")
    print(f"  Duration: {args.duration}s")
    print()

    all_results = {}

    for name, gains in GAIN_SETS.items():
        print(f"--- {gains['label']} ---")

        vid_path = None
        if args.video:
            vid_path = os.path.join(OUTPUT_DIR, f"{name}.mp4")

        t0 = time.time()
        metrics = run_trial(gains, args.duration, save_video=vid_path)
        elapsed = time.time() - t0

        all_results[name] = metrics

        print(f"  Speed:             {metrics['speed_mm_s']:.1f} mm/s  "
              f"(distance: {metrics['distance_mm']:.0f} mm)")
        print(f"  CoT:               {metrics['cot']:.1f}")
        print(f"  Mean |pitch|:      {metrics['mean_pitch_deg']:.2f} deg")
        print(f"  Max |pitch|:       {metrics['max_pitch_deg']:.2f} deg")
        print(f"  Conformity error:  mean={metrics['mean_conformity_mm']:.2f} mm  "
              f"p90={metrics['p90_conformity_mm']:.2f} mm")
        print(f"  Clearance:         {metrics['clearance_mean_mm']:.2f} +/- "
              f"{metrics['clearance_std_mm']:.2f} mm")
        print(f"  Terrain z std:     {metrics['terrain_std_mm']:.2f} mm")
        print(f"  Body z std:        {metrics['body_z_std_mm']:.2f} mm")

        # Key ratio: does body z vary as much as terrain z?
        if metrics['terrain_std_mm'] > 0.01:
            tracking_ratio = metrics['body_z_std_mm'] / metrics['terrain_std_mm']
            print(f"  Tracking ratio:    {tracking_ratio:.2f}  "
                  f"(1.0 = perfect conform, 0.0 = rigid flat)")
        if vid_path:
            print(f"  Video: {vid_path}")
        print(f"  ({elapsed:.0f}s wall time)")
        print()

    # Side-by-side comparison
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25s}  {'v2 (soft)':<18s}  {'v2.1 (stiff)':<18s}  {'Verdict'}")
    print("-" * 80)

    a = all_results["v2_soft"]
    b = all_results["v2.1_stiff"]

    def compare(metric, unit, lower_better=True):
        va, vb = a[metric], b[metric]
        if lower_better:
            winner = "v2 soft" if va < vb else "v2.1 stiff" if vb < va else "tie"
        else:
            winner = "v2 soft" if va > vb else "v2.1 stiff" if vb > va else "tie"
        print(f"  {metric:<23s}  {va:>12.2f} {unit:<4s}  {vb:>12.2f} {unit:<4s}  {winner}")

    compare("speed_mm_s", "mm/s", lower_better=False)
    compare("cot", "", lower_better=True)
    compare("mean_pitch_deg", "deg", lower_better=False)  # more pitch = more conforming
    compare("mean_conformity_mm", "mm", lower_better=True)
    compare("clearance_std_mm", "mm", lower_better=False)  # more variation = more conforming

    if a["terrain_std_mm"] > 0.01:
        ratio_a = a["body_z_std_mm"] / a["terrain_std_mm"]
        ratio_b = b["body_z_std_mm"] / b["terrain_std_mm"]
        winner = "v2 soft" if ratio_a > ratio_b else "v2.1 stiff"
        print(f"  {'tracking_ratio':<23s}  {ratio_a:>12.2f} {'':4s}  {ratio_b:>12.2f} {'':4s}  {winner}")

    print()
    print("  Tracking ratio: body_z_std / terrain_z_std")
    print("    1.0 = body perfectly follows terrain (conforming)")
    print("    0.0 = body stays flat regardless of terrain (rigid)")
    print()
    print("  If v2.1 has LOWER tracking ratio + LOWER pitch angle,")
    print("  it's staying rigid — NOT conforming. The optimizer cheated.")


if __name__ == "__main__":
    main()
