#!/usr/bin/env python3
"""
test_terrain_levels.py  (Option A, v2)
=======================================
Generate terrain PNGs at increasing roughness, smooth them,
patch the XML, run headless sim, save video per level.

Key physics:
  - hfield z_max=0.05 → full pixel range maps to ±50mm
  - Centipede leg length ~7mm, COM height ~5mm
  - Amplitudes chosen so physical features span 1-8mm
  - Spawn height computed per level: terrain_peak + 5mm clearance
  - Gaussian smooth removes sub-pixel sharp edges

Usage:
    python test_terrain_levels.py
    python test_terrain_levels.py --duration 5
"""

import argparse
import os
import re
import sys
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
XML_PATH    = os.path.join(BASE_DIR, "models", "farms", "centipede.xml")
CONTROL_DIR = os.path.join(BASE_DIR, "controllers", "farms")
TERRAIN_DIR = os.path.join(BASE_DIR, "terrain", "generator")
VIDEO_DIR   = os.path.join(BASE_DIR, "outputs", "videos", "terrain_test")

# Add project dirs to path for local imports (farms_controller, generate_terrain_multifreq)
for _d in (CONTROL_DIR, TERRAIN_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ── Fixed hfield geometry ──
HFIELD_XY   = 0.5    # terrain half-extent (metres)
HFIELD_ZMAX = 0.05   # max height scale (metres) — ±50mm full range
HFIELD_ZMIN = 0.005  # sub-ground margin
HFIELD_NROW = 256
HFIELD_NCOL = 256
SMOOTH_SIGMA = 8     # Gaussian blur in pixels

CLEARANCE = 0.005    # 5mm above terrain peak for spawn

# ── Roughness ladder ──
# Amplitudes scaled so physical features are 1-8mm with z_max=0.05
# Physical height ≈ (pixel_deviation / 128) × z_max
LEVELS = [
    {"label": "flat",        "low": 0.0,   "high": 0.0,   "hf": 12},
    {"label": "very_gentle", "low": 0.002, "high": 0.001, "hf": 12},
    {"label": "gentle",      "low": 0.005, "high": 0.003, "hf": 12},
    {"label": "moderate",    "low": 0.01,  "high": 0.005, "hf": 15},
    {"label": "challenging", "low": 0.015, "high": 0.008, "hf": 18},
    {"label": "rough",       "low": 0.02,  "high": 0.01,  "hf": 20},
]


def generate_terrain_png(level: dict, output_dir: str) -> tuple:
    """
    Generate + smooth terrain PNG.
    Returns (png_path, physical_peak_height_m).
    """
    from generate_terrain_multifreq import load_config, generate_terrain
    from PIL import Image
    from scipy.ndimage import gaussian_filter

    cfg = load_config(os.path.join(TERRAIN_DIR, "terrain_config.yaml"))
    hmap, meta = generate_terrain(
        cfg,
        low_amplitude=level["low"],
        high_amplitude=level["high"],
        high_freq_centre=level["hf"],
        seed=42,
    )

    # Smooth
    hmap_f = gaussian_filter(hmap.astype(np.float32), sigma=SMOOTH_SIGMA)
    hmap_out = np.clip(hmap_f, 0, 255).astype(np.uint8)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "1.png")
    Image.fromarray(hmap_out, mode="L").save(png_path)

    # Compute physical peak height
    # MuJoCo: height = (2 * pixel / 255 - 1) * z_max
    max_pix = float(hmap_out.max())
    min_pix = float(hmap_out.min())
    peak_h  = (2.0 * max_pix / 255.0 - 1.0) * HFIELD_ZMAX  # highest point
    low_h   = (2.0 * min_pix / 255.0 - 1.0) * HFIELD_ZMAX   # lowest point
    range_h = peak_h - low_h

    print(f"  PNG: pixels [{int(min_pix)}, {int(max_pix)}]  "
          f"std={hmap_out.std():.1f}  "
          f"physical: peak={peak_h*1000:.1f}mm  "
          f"range={range_h*1000:.1f}mm")

    return png_path, peak_h


def patch_xml(level: dict, png_path: str = None, spawn_z: float = 0.01):
    """Patch XML for this terrain level."""
    with open(XML_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    if level["low"] == 0 and level["high"] == 0:
        # Flat plane
        content = re.sub(
            r'(<geom\b[^>]*\bname="ground"[^>]*\b)type="hfield"',
            r'\1type="plane"', content)
        content = re.sub(r'\s*hfield="terrain"', '', content)
        if not re.search(r'name="ground"[^/]*\bsize=', content):
            content = re.sub(
                r'(<geom\b[^>]*\bname="ground")',
                r'\1 size="2 2 0.01"', content)
        print(f"  Ground -> flat plane, spawn_z={spawn_z*1000:.1f}mm")
    else:
        # Ensure hfield type
        if re.search(r'name="ground"[^/]*type="plane"', content):
            content = re.sub(
                r'(<geom\b[^>]*\bname="ground"[^>]*\b)type="plane"',
                r'\1type="hfield"', content)
            content = re.sub(
                r'(<geom\b[^>]*\bname="ground"[^>]*)\s+size="[^"]*"',
                r'\1', content)
            if 'hfield="terrain"' not in content.split('name="ground"')[1][:200]:
                content = re.sub(
                    r'(<geom\b[^>]*\bname="ground")',
                    r'\1 hfield="terrain"', content)

        # PNG path (absolute)
        content = re.sub(
            r'(<hfield\b[^>]*\bname="terrain"[^>]*\bfile=")[^"]*(")',
            rf'\g<1>{png_path}\g<2>', content)

        # hfield size
        sz = f"{HFIELD_XY} {HFIELD_XY} {HFIELD_ZMAX} {HFIELD_ZMIN}"
        content = re.sub(
            r'(<hfield\b[^>]*\bname="terrain"[^>]*\bsize=")[^"]*(")',
            rf'\g<1>{sz}\g<2>', content)

        # nrow / ncol
        content = re.sub(
            r'(<hfield\b[^>]*\bname="terrain"[^>]*\bnrow=")[^"]*(")',
            rf'\g<1>{HFIELD_NROW}\g<2>', content)
        content = re.sub(
            r'(<hfield\b[^>]*\bname="terrain"[^>]*\bncol=")[^"]*(")',
            rf'\g<1>{HFIELD_NCOL}\g<2>', content)

        print(f"  hfield z_max={HFIELD_ZMAX*1000:.0f}mm, "
              f"spawn_z={spawn_z*1000:.1f}mm")

    # Spawn height
    content = re.sub(
        r'(<body\s+name="link_body_0"\s+pos="0 0 )[0-9.]+(")',
        rf'\g<1>{spawn_z:.4f}\g<2>', content)

    with open(XML_PATH, "w", encoding="utf-8") as f:
        f.write(content)


def run_trial(duration: float, video_path: str) -> dict:
    """Run headless sim, save video, return stats."""
    import mujoco
    import mediapy as media
    from farms_controller import FARMSTravelingWaveController

    config_path = os.path.join(CONTROL_DIR, "farms_config.yaml")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    ctrl  = FARMSTravelingWaveController(model, config_path)

    n_steps     = int(duration / model.opt.timestep)
    fps         = 30
    frame_every = max(1, int(1.0 / (fps * model.opt.timestep)))

    w, h = 640, 480
    model.vis.global_.offwidth  = w
    model.vis.global_.offheight = h
    renderer  = mujoco.Renderer(model, h, w)
    cam       = mujoco.MjvCamera()
    cam.distance  = 0.2
    cam.azimuth   = 60
    cam.elevation = -35
    scene_opt = mujoco.MjvOption()
    scene_opt.geomgroup[0] = True
    scene_opt.geomgroup[1] = False
    scene_opt.geomgroup[2] = True
    scene_opt.geomgroup[3] = False

    b0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")

    # Find touch sensors for contact fraction
    touch_idx = []
    adr = 0
    for i in range(model.nsensor):
        dim = model.sensor_dim[i]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or ""
        if "touch" in name.lower():
            touch_idx.extend(range(adr, adr + dim))
        adr += dim

    frames = []
    com_start = None
    contact_sum = 0
    contact_count = 0

    for step in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        if com_start is None:
            com_start = data.subtree_com[b0_id].copy()

        # Track contact
        if touch_idx and step % 10 == 0:
            t_vals = data.sensordata[touch_idx]
            frac = np.sum(np.abs(t_vals) > 0.01) / len(touch_idx)
            contact_sum += frac
            contact_count += 1

        # Video frame
        if step % frame_every == 0:
            cam.lookat[:] = data.subtree_com[b0_id]
            renderer.update_scene(data, cam, scene_option=scene_opt)
            frames.append(renderer.render().copy())

    renderer.close()
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    media.write_video(video_path, frames, fps=fps)

    com_end = data.subtree_com[b0_id].copy()
    disp = np.linalg.norm(com_end[:2] - com_start[:2])
    avg_contact = contact_sum / max(contact_count, 1)

    return {
        "frames": len(frames),
        "displacement_cm": disp * 100,
        "contact_frac": avg_contact,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    os.makedirs(VIDEO_DIR, exist_ok=True)

    # Save original XML
    with open(XML_PATH, "r", encoding="utf-8") as f:
        original_xml = f.read()

    print("=" * 60)
    print("TERRAIN ROUGHNESS TEST (Option A, v2)")
    print(f"  Levels   : {len(LEVELS)}")
    print(f"  Duration : {args.duration}s")
    print(f"  hfield   : xy={HFIELD_XY}m  z_max={HFIELD_ZMAX*1000:.0f}mm")
    print(f"  Smooth   : sigma={SMOOTH_SIGMA}px")
    print(f"  Videos   : {VIDEO_DIR}")
    print("=" * 60)

    results = []

    for level in LEVELS:
        label = level["label"]
        print(f"\n{'─'*50}")
        print(f"{label.upper()}  (low={level['low']:.4f}  high={level['high']:.4f})")
        print(f"{'─'*50}")

        if level["low"] == 0 and level["high"] == 0:
            spawn_z = CLEARANCE + 0.005  # just above flat ground
            patch_xml(level, png_path=None, spawn_z=spawn_z)
        else:
            out_dir = os.path.join(TERRAIN_DIR, "terrain_output", label)
            png_path, peak_h = generate_terrain_png(level, out_dir)
            # Spawn above highest terrain point + clearance
            spawn_z = max(peak_h, 0) + CLEARANCE
            patch_xml(level, png_path=png_path, spawn_z=spawn_z)

        video_path = os.path.join(VIDEO_DIR, f"{label}.mp4")
        try:
            stats = run_trial(args.duration, video_path)
            status = "STABLE" if stats["contact_frac"] > 0.15 else "UNSTABLE"
            print(f"  {stats['frames']}fr | "
                  f"disp={stats['displacement_cm']:.1f}cm | "
                  f"contact={stats['contact_frac']:.0%} | "
                  f"{status}")
            print(f"  Video -> {video_path}")
            results.append({"label": label, **stats, "status": status})
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"label": label, "status": "ERROR"})

    # Restore XML
    with open(XML_PATH, "w", encoding="utf-8") as f:
        f.write(original_xml)

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Level':<14} {'Disp(cm)':>9} {'Contact':>9} {'Status':<10}")
    print(f"{'─'*60}")
    for r in results:
        if "displacement_cm" in r:
            print(f"{r['label']:<14} {r['displacement_cm']:>8.1f}  "
                  f"{r['contact_frac']:>8.0%}  {r['status']:<10}")
        else:
            print(f"{r['label']:<14} {'':>8}  {'':>8}  {r['status']:<10}")
    print(f"{'='*60}")
    print(f"XML restored. Videos in: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
