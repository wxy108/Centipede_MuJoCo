#!/usr/bin/env python3
"""
flat_test.py — simple flat-ground locomotion test (video only).

Uses the existing XML (models/farms/centipede.xml) + current farms_controller.yaml.
Strips the heightfield (geom + asset) out of the model and lets the centipede
walk on the default `type="plane"` floor, then runs for --duration seconds with
the tracking camera and writes an MP4.

The soft-start profile is inherited from the config:
    settle_time = 1.0 s   (neutral hold)
    ramp_time   = 1.0 s   (head→tail cosine activation)
→ active gait from t=2.0s onward.

Usage
-----
    python scripts/sweep/flat_test.py                  # 10 s video, config defaults
    python scripts/sweep/flat_test.py --duration 10 --out mytest.mp4
"""

import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))

import mujoco  # noqa: E402
from impedance_controller import ImpedanceTravelingWaveController  # noqa: E402
from kinematics import FARMSModelIndex  # noqa: E402

from wavelength_sweep import (  # noqa: E402
    XML_PATH, CONFIG_PATH,
    VID_W, VID_H, VID_FPS,
    CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION,
)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "flat_test")


def patch_xml_plane_only(xml_path, spawn_z=0.015):
    """Return a temp XML where the hfield is removed and only the floor plane
    carries contact. The centipede body's freejoint position is set to
    (0, 0, spawn_z) so it starts just above the plane at z=0.
    """
    from lxml import etree
    tree = etree.parse(xml_path)
    root = tree.getroot()

    # Remove any hfield geoms from the worldbody (or anywhere they might sit)
    for geom in list(root.iter("geom")):
        if geom.get("type") == "hfield":
            geom.getparent().remove(geom)

    # Remove the hfield asset entry too
    asset = root.find("asset")
    if asset is not None:
        for hf in list(asset.findall("hfield")):
            asset.remove(hf)

    # Place the centipede's root body at a sensible height above z=0
    for body in root.iter("body"):
        if body.find("freejoint") is not None:
            body.set("pos", f"0 0 {spawn_z:.4f}")
            break

    tmp_xml = xml_path + ".plane_only.xml"
    tree.write(tmp_xml, xml_declaration=True, encoding="utf-8", pretty_print=False)
    return tmp_xml


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=float, default=10.0,
                   help="Total seconds (default 10). Includes settle+ramp from config.")
    p.add_argument("--out", type=str, default=None,
                   help="Output MP4 path. Default: outputs/flat_test/flat_<ts>.mp4")
    p.add_argument("--fps", type=int, default=VID_FPS)
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(OUT_DIR, f"flat_{ts}.mp4")

    # 1. Strip hfield → plane-only ground
    tmp_xml = patch_xml_plane_only(XML_PATH, spawn_z=0.015)

    # 2. Build model + controller
    try:
        import mediapy  # noqa: F401
    except ImportError:
        print("ERROR: mediapy is required for video. Install with:")
        print("  pip install mediapy --break-system-packages")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(tmp_xml)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ctrl = ImpedanceTravelingWaveController(model, CONFIG_PATH)
    idx  = FARMSModelIndex(model)

    dt = model.opt.timestep
    n_steps = int(args.duration / dt)

    renderer = mujoco.Renderer(model, height=VID_H, width=VID_W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = idx.com_pos(data)
    cam.distance  = CAM_DISTANCE
    cam.azimuth   = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION

    frames = []
    frame_dt = 1.0 / args.fps
    last_frame_t = -1.0

    print(f"Running {args.duration:.1f}s on flat ground (dt={dt}s, {n_steps} steps)")
    print(f"  settle_time = {ctrl.settle_time:.1f}s  ramp_time = {ctrl.ramp_time:.1f}s")
    print(f"  output      = {out_path}")

    for step_i in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        if data.time - last_frame_t >= frame_dt - 1e-6:
            cam.lookat[:] = idx.com_pos(data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
            last_frame_t = data.time

    import mediapy
    mediapy.write_video(out_path, frames, fps=args.fps)

    # Clean up temp artefacts
    if os.path.exists(tmp_xml):
        os.remove(tmp_xml)

    com = idx.com_pos(data)
    print(f"\nDone.  final t={data.time:.2f}s  COM=({com[0]*1000:.1f},"
          f"{com[1]*1000:.1f},{com[2]*1000:.1f}) mm")
    print(f"Video: {out_path}")


if __name__ == "__main__":
    main()
