#!/usr/bin/env python3
"""
Patch centipede.xml to use a generated terrain PNG heightfield.

Usage
-----
Apply terrain (PNG) + fix spawn height:
  python patch_xml.py --terrain "...\\terrain\\output\\low0.0300_high0.0120_hf18.0_s0\\1.png"

Swap terrain PNG only (fast, no full repatch):
  python patch_farms_xml.py --terrain "...\\1.png" --terrain-only

Add/update velocity sensor only:
  python patch_farms_xml.py --sensors-only

Notes
-----
- Uses lxml in-place editing to preserve original formatting.
- The PNG path in the XML is written relative to the XML file location
  (models/farms/). Do NOT copy PNGs elsewhere.
- Spawn z convention:
    pixel 128  →  MuJoCo height ≈ TERRAIN_WORLD_HEIGHT * 0.5
    spawn z    =  TERRAIN_WORLD_HEIGHT * 0.5 + SPAWN_CLEARANCE
  The flat centering pad guarantees height=0 at world (0,0) → pixel 128.
"""

import argparse
import os
import re
import numpy as np
from PIL import Image
from lxml import etree

# ── fixed paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
XML_PATH      = os.path.join(PROJECT_ROOT, "models", "farms", "centipede.xml")

# ── terrain constants ────────────────────────────────────────────────────────
TERRAIN_WORLD_HEIGHT = 0.05   # default z_max (metres); overridden per level via --z-max
TERRAIN_WORLD_SIZE   = 0.5    # half-extent (metres) — terrain is 1m × 1m
SPAWN_CLEARANCE      = 0.015  # 15 mm clearance above terrain peak

# SPAWN_Z is computed dynamically from the PNG at patch time — see spawn_z_from_png().


# ── helpers ───────────────────────────────────────────────────────────────────
def _parse_xml(path):
    parser = etree.XMLParser(remove_blank_text=False)
    tree   = etree.parse(path, parser)
    return tree, tree.getroot()


def _write_xml(tree, path):
    """Write back without reformatting — preserves original indentation."""
    tree.write(path, xml_declaration=True, encoding="utf-8",
               pretty_print=False)


def _rel_path(png_abs, xml_abs):
    """Return a dynamically computed absolute path to the PNG.
    MuJoCo resolves hfield paths relative to meshdir (not the XML dir),
    so relative paths break. Absolute paths work on any machine as long
    as they're computed at patch time rather than hardcoded."""
    _ = xml_abs  # unused — kept for call-site compatibility
    return os.path.abspath(png_abs)


# ── patch functions ───────────────────────────────────────────────────────────
def spawn_z_from_png(png_path, terrain_world_height, clearance, sample_radius=8):
    """
    Read the PNG and compute spawn z from the actual terrain height at world (0,0).
    World (0,0) = image centre pixel (nrow//2, ncol//2).
    Samples a +-sample_radius patch and takes the max so the centipede
    clears the highest point directly under its body at spawn.
    """
    arr = np.array(Image.open(png_path).convert("L"), dtype=np.float32)
    nrow, ncol = arr.shape
    cy, cx = nrow // 2, ncol // 2
    r = sample_radius
    patch = arr[max(0, cy-r):cy+r+1, max(0, cx-r):cx+r+1]
    terrain_h = (float(patch.max()) / 255.0) * terrain_world_height
    return terrain_h + clearance


def patch_terrain(root, png_path, xml_path, z_max=None):
    """
    Update (or create) the <hfield> asset and <geom type="hfield"> in worldbody.
    Also adjust spawn z of the freejoint body.

    Parameters
    ----------
    z_max : float or None
        Per-level hfield z_max (metres).  If None, uses TERRAIN_WORLD_HEIGHT.
        The sweep passes a level-specific value so that every PNG uses the
        full 0-255 range and the physical roughness comes from z_max.
    """
    z = z_max if z_max is not None else TERRAIN_WORLD_HEIGHT
    rel = _rel_path(png_path, xml_path).replace("\\", "/")  # forward slashes in XML

    # ── hfield asset ──────────────────────────────────────────────────────
    asset = root.find("asset")
    if asset is None:
        asset = etree.SubElement(root, "asset")

    hfield = asset.find("hfield[@name='terrain']")
    if hfield is None:
        hfield = etree.SubElement(asset, "hfield")
        hfield.set("name", "terrain")

    hfield.set("file",    rel)
    hfield.set("size",    f"{TERRAIN_WORLD_SIZE:.3f} {TERRAIN_WORLD_SIZE:.3f} "
                          f"{z:.4f} 0.001")
    hfield.set("nrow",    "1024")
    hfield.set("ncol",    "1024")

    # ── hfield geom in worldbody ──────────────────────────────────────────
    worldbody = root.find("worldbody")
    if worldbody is not None:
        geom = worldbody.find("geom[@type='hfield']")
        if geom is None:
            geom = etree.SubElement(worldbody, "geom")
            geom.set("type",  "hfield")
            geom.set("name",  "terrain_geom")
        geom.set("hfield",   "terrain")
        geom.set("pos",      "0 0 0")
        geom.set("conaffinity", "1")
        geom.set("condim",   "3")
        geom.set("friction", "1.0 0.005 0.0001")

    # ── spawn height — read actual terrain height at world (0,0) from PNG ──
    spawn_z = spawn_z_from_png(png_path, z, SPAWN_CLEARANCE)
    patch_spawn(root, spawn_z)

    print(f"[patch_farms_xml] terrain → {rel}")
    print(f"[patch_farms_xml] z_max = {z:.4f} m  nrow=1024 ncol=1024")
    print(f"[patch_farms_xml] spawn z → {spawn_z:.4f} m  "
          f"(terrain@origin {spawn_z - SPAWN_CLEARANCE:.4f} m + {SPAWN_CLEARANCE} clearance)")


def patch_spawn(root, spawn_z=None, rotation_deg=None):
    """Set freejoint body position and optional Z rotation."""
    if spawn_z is None:
        spawn_z = TERRAIN_WORLD_HEIGHT * 0.85 + SPAWN_CLEARANCE
    for body in root.iter("body"):
        if body.find("freejoint") is not None:
            body.set("pos", f"0 0 {spawn_z:.4f}")
            if rotation_deg is not None:
                body.set("euler", f"0 0 {rotation_deg:.2f}")
            elif body.get("euler") is not None:
                del body.attrib["euler"]  # clear previous rotation
            return
    print("[patch_farms_xml] WARNING: no freejoint body found — spawn z not updated")


def patch_flat_ground(root):
    """Remove hfield and restore a simple flat plane for L0 baseline."""
    # Remove hfield from asset
    asset = root.find("asset")
    if asset is not None:
        for hf in asset.findall("hfield"):
            asset.remove(hf)

    # Replace hfield geom with flat plane in worldbody
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for geom in worldbody.findall("geom[@type='hfield']"):
            worldbody.remove(geom)
        # Add flat plane if not already present
        if worldbody.find("geom[@name='floor']") is None:
            floor = etree.SubElement(worldbody, "geom")
            floor.set("name",        "floor")
            floor.set("type",        "plane")
            floor.set("size",        "0 0 0.01")
            floor.set("pos",         "0 0 0")
            floor.set("conaffinity", "1")
            floor.set("condim",      "3")
            floor.set("friction",    "1.0 0.005 0.0001")

    # Set fixed spawn z for flat ground
    patch_spawn(root, spawn_z=SPAWN_CLEARANCE)
    print("[patch_farms_xml] flat ground restored (plane geom, no hfield)")
    print(f"[patch_farms_xml] spawn z -> {SPAWN_CLEARANCE:.4f} m")


def patch_sensors(root):
    """
    Add (or update) a framelinvel sensor on the root body for velocity logging.
    Requires refname='world' (MuJoCo ≥ 3.x).
    """
    # Find root body (first body with freejoint)
    root_body_name = None
    for body in root.iter("body"):
        if body.find("freejoint") is not None:
            root_body_name = body.get("name")
            break

    if root_body_name is None:
        print("[patch_farms_xml] WARNING: no freejoint body found — sensor skipped")
        return

    sensor_el = root.find("sensor")
    if sensor_el is None:
        sensor_el = etree.SubElement(root, "sensor")

    # Remove old com_vel if present
    for old in sensor_el.findall("framelinvel[@name='com_vel']"):
        sensor_el.remove(old)

    etree.SubElement(sensor_el, "framelinvel",
                     name="com_vel",
                     objtype="body",
                     objname=root_body_name,
                     reftype="world",
                     refname="world")

    print(f"[patch_farms_xml] sensor com_vel → body '{root_body_name}'")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--terrain",      type=str, default=None,
                   help="Absolute path to terrain PNG (1.png from generate_terrain_multifreq.py)")
    p.add_argument("--terrain-only", action="store_true",
                   help="Only update terrain + spawn z, skip sensor patch")
    p.add_argument("--sensors-only", action="store_true",
                   help="Only add/update velocity sensor, skip terrain patch")
    p.add_argument("--model",        type=str, default=XML_PATH,
                   help=f"Path to centipede.xml (default: {XML_PATH})")
    p.add_argument("--rotation-deg", type=float, default=None,
                   help="Initial yaw rotation of centipede around Z axis in degrees")
    p.add_argument("--flat-ground",  action="store_true",
                   help="Remove hfield and use flat plane (true flat baseline)")
    p.add_argument("--z-max",       type=float, default=None,
                   help="Override hfield z_max (metres). Per-level scaling "
                        "so each PNG uses full 0-255 range.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"XML not found: {args.model}")

    tree, root = _parse_xml(args.model)

    did_something = False

    if args.flat_ground:
        patch_flat_ground(root)
        did_something = True

    elif not args.sensors_only and args.terrain is not None:
        if not os.path.isfile(args.terrain):
            raise FileNotFoundError(f"PNG not found: {args.terrain}")
        patch_terrain(root, args.terrain, args.model, z_max=args.z_max)
        did_something = True

    # rotation-only (no terrain change — preserve existing spawn z)
    if args.rotation_deg is not None and not did_something:
        # Read current spawn z from XML so we don't overwrite it
        current_z = None
        for body in root.iter("body"):
            if body.find("freejoint") is not None:
                pos = body.get("pos", "0 0 0").split()
                if len(pos) >= 3:
                    current_z = float(pos[2])
                break
        patch_spawn(root, spawn_z=current_z, rotation_deg=args.rotation_deg)
        did_something = True
    elif args.rotation_deg is not None:
        # Terrain was already patched above — just update rotation
        for body in root.iter("body"):
            if body.find("freejoint") is not None:
                pos = body.get("pos", "0 0 0").split()
                if len(pos) >= 3:
                    body.set("pos", f"0 0 {pos[2]}")  # preserve z
                body.set("euler", f"0 0 {args.rotation_deg:.2f}")
                break
        did_something = True

    if not args.terrain_only and not args.sensors_only:
        patch_sensors(root)
        did_something = True

    if args.sensors_only:
        patch_sensors(root)
        did_something = True

    if args.terrain_only and args.terrain is None and args.rotation_deg is None:
        print("[patch_farms_xml] --terrain-only requires --terrain <path> or --rotation-deg")
        return

    if did_something:
        _write_xml(tree, args.model)
        print(f"[patch_farms_xml] Wrote: {args.model}")
    else:
        print("[patch_farms_xml] Nothing to do. Provide --terrain or --sensors-only.")


if __name__ == "__main__":
    main()
