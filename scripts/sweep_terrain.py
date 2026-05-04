#!/usr/bin/env python3
"""
sweep_terrain.py — Sweep across terrain roughness levels for ML data collection
================================================================================
Generates a dataset of centipede locomotion across a *ladder* of terrain
roughness levels. Each run produces a .npz file containing time-series sensor
data (body yaw joints, pitch joints, leg joints, COM position/velocity,
contact forces) labelled with terrain roughness parameters.

The collected data is intended for training an ML model to sense/classify
terrain roughness from proprioceptive and exteroceptive signals.

Terrain roughness ladder:
  - 3 terrain heightmap directories (mild / medium / aggressive spatial freq)
  - × N z_max values (physical height scaling from near-flat to very rough)
  - = a fine-grained roughness ladder

Output structure:
  outputs/terrain_sweep/
    sweep_manifest.json          — index of all runs with labels
    flat_zmax0.0010/run.npz      — baseline flat terrain
    mild_zmax0.0050/run.npz      — mild terrain, low amplitude
    mild_zmax0.0100/run.npz      — mild terrain, medium amplitude
    ...
    aggressive_zmax0.0800/run.npz — aggressive terrain, high amplitude

Each .npz contains:
  time             (T,)         — timestamps
  com_pos          (T, 3)       — center-of-mass xyz
  com_vel          (T, 3)       — center-of-mass velocity
  body_jnt_pos     (T, 19)      — body yaw joint positions
  body_jnt_cmd     (T, 19)      — body yaw commanded positions
  pitch_jnt_pos    (T, N_pitch) — passive pitch joint positions
  pitch_jnt_vel    (T, N_pitch) — passive pitch joint velocities
  leg_jnt_pos      (T, 19, 2, 4)— leg joint positions
  contact_forces   (T, 3)       — net ground reaction force (xyz)
  segment_heights  (T, 21)      — z-height of each body segment

Usage (from project root):
    python scripts/sweep_terrain.py
    python scripts/sweep_terrain.py --duration 10 --dt 0.005
    python scripts/sweep_terrain.py --video          # also render videos
    python scripts/sweep_terrain.py --jobs 1         # sequential (safe)

The script patches centipede.xml in-place for each run, then restores it.
"""

import os
import sys
import re
import json
import time
import shutil
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(BASE, "controllers", "farms"))

import mujoco

from kinematics import FARMSModelIndex, N_BODY_JOINTS, N_LEGS, N_LEG_DOF
from controller import FARMSTravelingWaveController, load_config
from impedance_controller import ImpedanceTravelingWaveController

XML_PATH    = os.path.join(BASE, "models", "farms", "centipede.xml")
XML_BACKUP  = XML_PATH + ".terrain_sweep_backup"
CONFIG_PATH = os.path.join(BASE, "configs", "farms_controller.yaml")
TERRAIN_DIR = os.path.join(BASE, "terrain", "output")
OUTPUT_DIR  = os.path.join(BASE, "outputs", "terrain_sweep")

# ═══════════════════════════════════════════════════════════════════════
#  Terrain roughness ladder
# ═══════════════════════════════════════════════════════════════════════

# Terrain heightmap directories (spatial frequency increases)
TERRAIN_MAPS = {
    "flat": {
        "png": os.path.join(TERRAIN_DIR, "flat_terrain.png"),
        "label": "flat",
        "spatial_freq": 0.0,
    },
    "mild": {
        "png": os.path.join(TERRAIN_DIR, "low0.0040_mid0.0020_high0.0010_s0", "1.png"),
        "label": "mild",
        "spatial_freq": 0.004,
    },
    "medium": {
        "png": os.path.join(TERRAIN_DIR, "low0.0060_mid0.0030_high0.0020_s0", "1.png"),
        "label": "medium",
        "spatial_freq": 0.006,
    },
    "aggressive": {
        "png": os.path.join(TERRAIN_DIR, "low0.0100_mid0.0050_high0.0030_s0", "1.png"),
        "label": "aggressive",
        "spatial_freq": 0.010,
    },
}

# z_max values: physical height of terrain features (metres)
# For flat, only use a tiny z_max. For others, sweep a range.
ZMAX_FLAT = [0.001]
ZMAX_LADDER = [0.005, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.080]


def build_run_configs():
    """Build the full list of (terrain_key, z_max, run_label) configs."""
    configs = []

    # Flat baseline
    for z in ZMAX_FLAT:
        configs.append({
            "terrain_key": "flat",
            "z_max": z,
            "run_label": f"flat_zmax{z:.4f}",
            "roughness_index": 0.0,  # scalar label for ML
        })

    # Rough terrains × z_max ladder
    for tkey in ["mild", "medium", "aggressive"]:
        tinfo = TERRAIN_MAPS[tkey]
        for z in ZMAX_LADDER:
            # Roughness index = spatial_freq × z_max (a combined scalar)
            roughness = tinfo["spatial_freq"] * z * 1e3  # scale for readability
            configs.append({
                "terrain_key": tkey,
                "z_max": z,
                "run_label": f"{tkey}_zmax{z:.4f}",
                "roughness_index": round(roughness, 6),
            })

    # Sort by roughness_index for a clean ladder
    configs.sort(key=lambda c: c["roughness_index"])
    return configs


# ═══════════════════════════════════════════════════════════════════════
#  XML patching
# ═══════════════════════════════════════════════════════════════════════

def patch_xml(base_xml, terrain_png, z_max):
    """Patch hfield path and z_max in the XML string."""
    x = base_xml

    # Terrain file path — use string find/replace instead of regex
    # to avoid Windows backslash escaping issues
    import re as _re
    m = _re.search(r'<hfield\s+name="terrain"\s+file="([^"]*)"', x)
    if m:
        old_path = m.group(1)
        x = x.replace(f'file="{old_path}"', f'file="{terrain_png}"')

    # z_max in size attribute
    def fix_size(m):
        parts = m.group(2).split()
        if len(parts) >= 3:
            parts[2] = f"{z_max:.6g}"
        return f'{m.group(1)}{" ".join(parts)}"'
    x = _re.sub(r'(<hfield[^>]*\bsize=")([^"]*)"', fix_size, x)
    return x


# ═══════════════════════════════════════════════════════════════════════
#  Simulation + data recording
# ═══════════════════════════════════════════════════════════════════════

def run_single(xml_text, duration, dt_record, record_video=False):
    """
    Run one simulation and collect rich sensor data.

    Returns:
        data_dict: dict of numpy arrays (the .npz contents)
        status: "ok" | "diverged" | "buckled" | error string
        frames: list of RGB arrays if record_video, else []
    """
    # Write patched XML and load
    with open(XML_PATH, 'w') as f:
        f.write(xml_text)

    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        return None, f"xml_error: {e}", []

    data = mujoco.MjData(model)
    cfg  = load_config(CONFIG_PATH)

    # Build controller — use impedance controller for compliant body yaw
    ctrl = ImpedanceTravelingWaveController(model, CONFIG_PATH)
    idx  = ctrl.idx

    # Find pitch joint IDs
    pitch_ids = []
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm and ('joint_pitch_body' in nm):
            pitch_ids.append(i)

    # Find all body IDs for segment heights
    body_ids = []
    for i in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if nm and 'link_body_' in nm:
            body_ids.append(i)
    body_ids.sort(key=lambda bid: int(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid).split('_')[-1]))

    # Video setup
    frames = []
    renderer = None
    cam = None
    if record_video:
        try:
            # Use model's offscreen buffer size (set via <visual><global offwidth/offheight>)
            vid_w = model.vis.global_.offwidth
            vid_h = model.vis.global_.offheight
            renderer = mujoco.Renderer(model, height=vid_h, width=vid_w)
            cam = mujoco.MjvCamera()
            cam.distance  = 0.2
            cam.azimuth   = 60
            cam.elevation = -35
        except Exception as e:
            print(f"    WARNING: Renderer init failed: {e}")
            renderer = None

    # Recording buffers
    n_steps  = int(duration / model.opt.timestep)
    last_rec = -np.inf
    last_vid = -np.inf
    video_dt = 1.0 / 30  # 30 fps for video

    rec = {
        'time': [], 'com_pos': [], 'com_vel': [],
        'body_jnt_pos': [], 'body_jnt_cmd': [],
        'pitch_jnt_pos': [], 'pitch_jnt_vel': [],
        'leg_jnt_pos': [], 'contact_forces': [],
        'segment_heights': [],
    }

    buckled = False
    bw_amp = bw['amplitude']
    bw_om  = 2 * np.pi * bw['frequency']
    bw_nw  = bw['wave_number']
    bw_sp  = bw['speed']

    for s in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # Stability check
        if s % 200 == 0:
            if np.any(np.isnan(data.qpos[:10])) or np.any(np.abs(data.qpos[:10]) > 50):
                return None, "diverged", []
            for jid in pitch_ids:
                if abs(data.qpos[model.jnt_qposadr[jid]]) > np.radians(55):
                    buckled = True

        # Record sensor data
        if data.time - last_rec >= dt_record - 1e-10:
            last_rec = data.time
            t = data.time
            rec['time'].append(t)
            rec['com_pos'].append(idx.com_pos(data).copy())
            rec['com_vel'].append(idx.com_vel(data).copy())

            # Body yaw joints (actual)
            bj = np.array([idx.body_joint_pos(data, i+1) for i in range(N_BODY_JOINTS)])
            rec['body_jnt_pos'].append(bj)

            # Body yaw joints (commanded)
            cmd = np.array([bw_amp * np.sin(bw_om * t - 2*np.pi*bw_nw*bw_sp*i/18)
                            for i in range(N_BODY_JOINTS)])
            rec['body_jnt_cmd'].append(cmd)

            # Pitch joints
            ppos = np.array([data.qpos[model.jnt_qposadr[j]] for j in pitch_ids])
            pvel = np.array([data.qvel[model.jnt_dofadr[j]]  for j in pitch_ids])
            rec['pitch_jnt_pos'].append(ppos)
            rec['pitch_jnt_vel'].append(pvel)

            # Leg joints
            lj = np.zeros((N_LEGS, 2, N_LEG_DOF))
            for n in range(N_LEGS):
                for si, side in enumerate(('L', 'R')):
                    for dof in range(N_LEG_DOF):
                        lj[n, si, dof] = idx.leg_joint_pos(data, n, side, dof)
            rec['leg_jnt_pos'].append(lj)

            # Net contact force (sum of all contact forces in world frame)
            cf = np.zeros(3)
            for ci in range(data.ncon):
                c = data.contact[ci]
                # Get 6D wrench for this contact
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, wrench)
                # Transform from contact frame to world frame
                frame = c.frame.reshape(3, 3)
                cf += frame.T @ wrench[:3]
            rec['contact_forces'].append(cf)

            # Per-segment z heights
            seg_z = np.array([data.xpos[bid, 2] for bid in body_ids])
            rec['segment_heights'].append(seg_z)

        # Video frame
        if renderer and data.time - last_vid >= video_dt - 1e-10:
            last_vid = data.time
            cam.lookat[:] = idx.com_pos(data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    # Clean up renderer to avoid __del__ errors on Windows
    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass

    # Pack into dict
    out = {}
    for k, v in rec.items():
        out[k] = np.array(v)

    status = "buckled" if buckled else "ok"
    return out, status, frames


# ═══════════════════════════════════════════════════════════════════════
#  Main sweep
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sweep terrain roughness ladder for ML data collection")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration per run (default: 10s)")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Recording timestep (default: 0.01s = 100Hz)")
    parser.add_argument("--video", action="store_true",
                        help="Also save .mp4 video for each run")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only matching labels (e.g. 'aggressive' or 'flat')")
    parser.add_argument("--zmax-only", type=float, default=None,
                        help="Run only this z_max value")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Backup XML
    if not os.path.exists(XML_BACKUP):
        shutil.copy2(XML_PATH, XML_BACKUP)
    with open(XML_BACKUP, 'r') as f:
        base_xml = f.read()

    configs = build_run_configs()

    # Filter if requested
    if args.only:
        configs = [c for c in configs if args.only in c['run_label']]
    if args.zmax_only is not None:
        configs = [c for c in configs if abs(c['z_max'] - args.zmax_only) < 1e-6]

    print(f"\n{'='*72}")
    print(f"  TERRAIN ROUGHNESS SWEEP — {len(configs)} configurations")
    print(f"  Duration: {args.duration}s | Recording dt: {args.dt}s | Video: {args.video}")
    print(f"{'='*72}")

    # Print the roughness ladder
    print(f"\n  {'#':>3}  {'Label':<30s} {'z_max':>8} {'Roughness':>10}")
    print(f"  {'-'*55}")
    for i, c in enumerate(configs):
        print(f"  {i+1:3d}  {c['run_label']:<30s} {c['z_max']:8.4f} {c['roughness_index']:10.4f}")

    manifest = []
    total_t0 = time.time()

    for i, c in enumerate(configs):
        tkey  = c['terrain_key']
        tinfo = TERRAIN_MAPS[tkey]
        z_max = c['z_max']
        label = c['run_label']
        run_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(run_dir, exist_ok=True)

        npz_path = os.path.join(run_dir, "run.npz")
        if os.path.exists(npz_path):
            print(f"\n  [{i+1:2d}/{len(configs)}] {label} — SKIP (already exists)")
            # Load existing manifest entry
            meta_path = os.path.join(run_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    manifest.append(json.load(f))
            continue

        print(f"\n  [{i+1:2d}/{len(configs)}] {label}  (z_max={z_max}, roughness={c['roughness_index']:.4f})")

        # Patch XML
        xml = patch_xml(base_xml, tinfo['png'], z_max)

        # Run simulation
        t0 = time.time()
        data_dict, status, frames = run_single(
            xml, args.duration, args.dt, record_video=args.video)
        wall = time.time() - t0

        if data_dict is None:
            print(f"    FAILED: {status} ({wall:.0f}s)")
            manifest.append({
                'label': label, 'terrain_key': tkey, 'z_max': z_max,
                'roughness_index': c['roughness_index'],
                'status': status, 'wall_time': wall,
            })
            continue

        # Compute summary stats
        t_mask = data_dict['time'] > 1.0  # skip first 1s warmup
        if t_mask.sum() > 0:
            trk_err = np.sqrt(np.mean(
                (data_dict['body_jnt_pos'][t_mask] - data_dict['body_jnt_cmd'][t_mask])**2))
            pitch_std = np.std(data_dict['pitch_jnt_pos'][t_mask])
            pitch_max = np.max(np.abs(data_dict['pitch_jnt_pos'][t_mask]))
            com_z_std = np.std(data_dict['com_pos'][t_mask, 2])
            com_z_min = np.min(data_dict['com_pos'][t_mask, 2])
            seg_z_std = np.std(data_dict['segment_heights'][t_mask])
        else:
            trk_err = pitch_std = pitch_max = com_z_std = com_z_min = seg_z_std = 0

        # Save .npz
        np.savez_compressed(npz_path, **data_dict)

        # Save metadata
        meta = {
            'label': label,
            'terrain_key': tkey,
            'terrain_png': tinfo['png'],
            'spatial_freq': tinfo['spatial_freq'],
            'z_max': z_max,
            'roughness_index': c['roughness_index'],
            'duration': args.duration,
            'dt_record': args.dt,
            'status': status,
            'wall_time': round(wall, 1),
            'n_frames': len(data_dict['time']),
            'stats': {
                'tracking_rms_deg': round(float(np.degrees(trk_err)), 4),
                'pitch_std_deg':    round(float(np.degrees(pitch_std)), 4),
                'pitch_max_deg':    round(float(np.degrees(pitch_max)), 2),
                'com_z_std_mm':     round(float(com_z_std * 1000), 3),
                'com_z_min_mm':     round(float(com_z_min * 1000), 2),
                'seg_z_std_mm':     round(float(seg_z_std * 1000), 3),
            }
        }
        with open(os.path.join(run_dir, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        manifest.append(meta)

        print(f"    {status:8s} | trk={np.degrees(trk_err):.3f}° "
              f"pitch_max={np.degrees(pitch_max):.1f}° "
              f"z_std={com_z_std*1000:.2f}mm "
              f"seg_z_std={seg_z_std*1000:.2f}mm | {wall:.0f}s")

        # Save video
        if args.video and frames:
            vid_path = os.path.join(run_dir, "video.mp4")
            saved = False
            # Try mediapy first
            try:
                import mediapy
                mediapy.write_video(vid_path, frames, fps=30)
                saved = True
            except ImportError:
                pass
            except Exception as e:
                print(f"    mediapy failed: {e}")
            # Fallback to imageio
            if not saved:
                try:
                    import imageio
                    writer = imageio.get_writer(vid_path, fps=30)
                    for frame in frames:
                        writer.append_data(frame)
                    writer.close()
                    saved = True
                except ImportError:
                    pass
                except Exception as e:
                    print(f"    imageio failed: {e}")
            # Fallback to raw frames as .npy
            if not saved:
                npy_path = os.path.join(run_dir, "frames.npy")
                np.save(npy_path, np.array(frames))
                print(f"    No video encoder found. Raw frames saved: {npy_path}")
                print(f"    Install: pip install mediapy  OR  pip install imageio imageio-ffmpeg")
            else:
                print(f"    Video: {vid_path} ({len(frames)} frames)")
        elif args.video and not frames:
            print(f"    WARNING: No video frames captured (renderer may have failed)")

    # Restore original XML
    shutil.copy2(XML_BACKUP, XML_PATH)
    print(f"\n  Restored original centipede.xml")

    # Save manifest
    manifest_path = os.path.join(OUTPUT_DIR, "sweep_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump({
            'total_configs': len(configs),
            'duration_per_run': args.duration,
            'dt_record': args.dt,
            'terrain_maps': {k: {'label': v['label'], 'spatial_freq': v['spatial_freq']}
                             for k, v in TERRAIN_MAPS.items()},
            'z_max_values': {'flat': ZMAX_FLAT, 'rough': ZMAX_LADDER},
            'runs': manifest,
        }, f, indent=2)
    print(f"  Manifest: {manifest_path}")

    # Print summary table
    total_wall = time.time() - total_t0
    print(f"\n{'='*72}")
    print(f"  SWEEP COMPLETE — {len(manifest)} runs in {total_wall:.0f}s")
    print(f"{'='*72}")
    print(f"\n  {'Label':<30s} {'Rough':>7} {'Status':>8} {'Trk°':>6} "
          f"{'PMax°':>6} {'Zseg':>7}")
    print(f"  {'-'*72}")
    for m in manifest:
        s = m.get('stats', {})
        print(f"  {m['label']:<30s} {m['roughness_index']:7.4f} {m['status']:>8s} "
              f"{s.get('tracking_rms_deg', -1):6.3f} "
              f"{s.get('pitch_max_deg', -1):6.1f} "
              f"{s.get('seg_z_std_mm', -1):7.2f}")

    print(f"\n  Data ready for ML in: {OUTPUT_DIR}/")
    print(f"  Each run.npz contains: time, com_pos, com_vel, body_jnt_pos,")
    print(f"    body_jnt_cmd, pitch_jnt_pos, pitch_jnt_vel, leg_jnt_pos,")
    print(f"    contact_forces, segment_heights")
    print(f"\n  Label columns for ML:")
    print(f"    roughness_index  — combined scalar (spatial_freq × z_max × 1000)")
    print(f"    terrain_key      — categorical (flat/mild/medium/aggressive)")
    print(f"    z_max            — terrain height in metres")


if __name__ == "__main__":
    main()
