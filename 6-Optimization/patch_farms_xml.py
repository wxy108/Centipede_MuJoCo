#!/usr/bin/env python3
"""
patch_farms_xml.py
==================
Patches the FARMS centipede.xml with two additions:

  1. TERRAIN SUPPORT
     - Adds an <hfield> asset pointing to a heightmap PNG
     - Replaces the flat ground plane with an hfield geom
     - The terrain PNG path is configurable (or swapped at runtime)

  2. FULL SENSOR SUITE  (for terrain roughness recognition)
     - jointpos + jointvel for all 19 body yaw joints
     - jointpos + jointvel for all 16 pitch joints (injected + SDF passive)
     - jointpos + jointvel for all 19×2×4 = 152 leg joints
     - touch sensors for all 19×2 = 38 feet (requires foot sites, added here)
     - framepos sensor on root body (COM position)
     - framelinvel sensor on root body (COM velocity)

     Total new sensors: 19×2 + 20×2 + 152×2 + 38 + 2 = 420 sensors

Usage:
    # Patch with default flat heightmap (128 = flat grey)
    python patch_farms_xml.py

    # Patch with a specific terrain PNG
    python patch_farms_xml.py --terrain terrain_output/low0.0100_high0.0050_hf12.0_s0/1.png

    # Swap terrain only (sensors already added)
    python patch_farms_xml.py --terrain new_terrain/1.png --terrain-only

    # Restore from backup
    python patch_farms_xml.py --restore
"""

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ── default paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.join(SCRIPT_DIR, "..")
XML_PATH    = os.path.join(BASE_DIR, "3-Model", "2-Centipede_FARMS", "centipede.xml")
BACKUP_PATH = XML_PATH + ".sensors_backup"

# Terrain world dimensions — must match terrain_config.yaml
TERRAIN_WORLD_SIZE   = 2.0   # metres (terrain tile side length)
TERRAIN_WORLD_HEIGHT = 0.15  # metres (full height range)
HFIELD_NROW          = 512   # must match image size
HFIELD_NCOL          = 512


# ── helpers ───────────────────────────────────────────────────────────────────

def prettify(root: ET.Element) -> str:
    raw    = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    lines  = [l for l in pretty.splitlines() if not l.startswith("<?xml")]
    return "\n".join(lines)


def backup(xml_path: str):
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(xml_path, BACKUP_PATH)
        print(f"  Backup → {BACKUP_PATH}")
    else:
        print(f"  Backup already exists: {BACKUP_PATH}")


# ── terrain patching ──────────────────────────────────────────────────────────

def patch_terrain(root: ET.Element, terrain_png: str):
    """
    Add/update hfield asset and replace ground plane with hfield geom.

    MuJoCo hfield convention:
      <hfield name="terrain" file="1.png"
              nrow="512" ncol="512"
              size="Xhalf Yhalf Zscale BaseThickness"/>
      where size[0,1] = half the world size in X and Y (metres)
            size[2]   = full height range (metres)
            size[3]   = thickness of the solid below the surface

    The ground geom sits at z=0 in the XML.  The hfield's zero level
    corresponds to pixel value 0; pixel 255 → z = TERRAIN_WORLD_HEIGHT.
    Spawn the centipede at z = TERRAIN_WORLD_HEIGHT/2 + clearance so it
    starts above the mid-height level of the terrain.
    """
    # Relative path from XML directory to PNG
    xml_dir = os.path.dirname(os.path.abspath(
        XML_PATH))  # used for display only; MuJoCo resolves relative to XML
    rel_png = os.path.relpath(os.path.abspath(terrain_png), xml_dir) \
              if terrain_png else "terrain/1.png"
    rel_png = rel_png.replace("\\", "/")  # normalise on Windows

    half_xy  = TERRAIN_WORLD_SIZE / 2.0
    z_scale  = TERRAIN_WORLD_HEIGHT
    z_base   = 0.02  # solid thickness below surface (metres)

    # ── asset: add/update <hfield> ──
    asset_el = root.find("asset")
    if asset_el is None:
        asset_el = ET.SubElement(root, "asset")

    hfield_el = asset_el.find("hfield[@name='terrain']")
    if hfield_el is None:
        hfield_el = ET.Element("hfield", name="terrain")
        # Insert before first mesh
        meshes = asset_el.findall("mesh")
        if meshes:
            idx = list(asset_el).index(meshes[0])
            asset_el.insert(idx, hfield_el)
        else:
            asset_el.insert(0, hfield_el)

    hfield_el.set("file",  rel_png)
    hfield_el.set("nrow",  str(HFIELD_NROW))
    hfield_el.set("ncol",  str(HFIELD_NCOL))
    hfield_el.set("size",  f"{half_xy} {half_xy} {z_scale} {z_base}")

    # ── worldbody: replace plane geom with hfield geom ──
    worldbody = root.find("worldbody")
    plane = worldbody.find("geom[@name='ground']")
    if plane is not None:
        plane.set("type",    "hfield")
        plane.set("hfield",  "terrain")
        plane.set("pos",     f"0 0 0")
        # Remove plane-specific attributes
        for attr in ("size", "rgba"):
            if attr in plane.attrib:
                del plane.attrib[attr]
        # Keep material if present, or set a default
        if "material" not in plane.attrib:
            plane.set("rgba", "0.6 0.5 0.4 1")
        print(f"  Ground geom → hfield  (file: {rel_png})")
    else:
        # No existing ground geom — add one
        ET.SubElement(worldbody, "geom",
                      name="ground", type="hfield", hfield="terrain",
                      pos="0 0 0", contype="1", conaffinity="1",
                      condim="3", friction="0.8 0.005 0.0001",
                      rgba="0.6 0.5 0.4 1")
        print("  Added hfield ground geom")

    # ── update freejoint spawn height ──
    # Spawn centipede at mid-terrain height + clearance
    spawn_z = TERRAIN_WORLD_HEIGHT * 0.5 + 0.015
    for body in root.iter("body"):
        if body.find("freejoint") is not None:
            pos = body.get("pos", "0 0 0").split()
            pos[2] = f"{spawn_z:.4f}"
            body.set("pos", " ".join(pos))
            print(f"  Freejoint spawn height → z={spawn_z:.4f} m")
            break


def swap_terrain_only(xml_path: str, terrain_png: str):
    """Fast path: only update the hfield file attribute, don't rebuild sensors."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    rel_png = os.path.relpath(os.path.abspath(terrain_png), xml_dir).replace("\\", "/")

    hfield = root.find(".//hfield[@name='terrain']")
    if hfield is None:
        print("ERROR: No hfield 'terrain' found. Run without --terrain-only first.")
        return False

    hfield.set("file", rel_png)
    with open(xml_path, "w") as f:
        f.write(prettify(root))
    print(f"  Terrain → {rel_png}")
    return True


# ── sensor patching ───────────────────────────────────────────────────────────

def _all_joint_names(root: ET.Element):
    """Return categorised joint names from the XML."""
    joints = root.findall(".//joint")
    names  = [j.get("name") for j in joints if j.get("name")]

    body_yaw = sorted([n for n in names if n.startswith("joint_body_")],
                      key=lambda n: int(n.split("_")[-1]))
    pitch    = sorted([n for n in names if "pitch" in n or "passive" in n],
                      key=lambda n: n)
    leg      = sorted([n for n in names if n.startswith("joint_leg_")],
                      key=lambda n: n)
    foot     = sorted([n for n in names if n.startswith("joint_foot_")],
                      key=lambda n: n)

    return body_yaw, pitch, leg, foot


def patch_sensors(root: ET.Element):
    """
    Add/replace the <sensor> section with a comprehensive suite.
    Also adds foot sites (spheres at foot body origins) needed for touch sensors.
    """

    body_yaw, pitch_jnts, leg_jnts, foot_jnts = _all_joint_names(root)
    print(f"  Found: {len(body_yaw)} body-yaw  {len(pitch_jnts)} pitch  "
          f"{len(leg_jnts)} leg  {len(foot_jnts)} foot joints")

    # ── 1. Add foot sites for touch sensors ──────────────────────────────────
    # Touch sensors in MuJoCo require a <site> attached to the body.
    # Foot links are named link_leg_{n}_{L/R}_3 (tarsus) or foot_{n}_{0/1}.
    # We add a zero-size site at the origin of each foot body.
    feet_with_sites = set()
    for body in root.iter("body"):
        bname = body.get("name", "")
        # Identify foot bodies: link_leg_*_3 (tarsus, the last leg DOF)
        # and any body named foot_*
        is_foot = (
            (bname.startswith("link_leg_") and bname.endswith("_3")) or
            bname.startswith("foot_")
        )
        if not is_foot:
            continue
        site_name = f"site_{bname}"
        if body.find(f"site[@name='{site_name}']") is None:
            ET.SubElement(body, "site",
                          name=site_name, pos="0 0 0",
                          size="0.0003", group="4")
        feet_with_sites.add(bname)

    print(f"  Foot sites added/verified: {len(feet_with_sites)}")

    # ── 2. Build sensor block ─────────────────────────────────────────────────
    # Remove existing sensor block if present
    old = root.find("sensor")
    if old is not None:
        root.remove(old)

    sensor_el = ET.SubElement(root, "sensor")

    def add_joint_sensors(jname: str):
        ET.SubElement(sensor_el, "jointpos", name=f"sp_{jname}", joint=jname)
        ET.SubElement(sensor_el, "jointvel", name=f"sv_{jname}", joint=jname)

    # Body yaw joints
    for jn in body_yaw:
        add_joint_sensors(jn)

    # Pitch + passive joints
    for jn in pitch_jnts:
        add_joint_sensors(jn)

    # Leg joints (all 4 DOF per leg)
    for jn in leg_jnts:
        add_joint_sensors(jn)

    # Foot joints
    for jn in foot_jnts:
        add_joint_sensors(jn)

    # Touch sensors for each foot site
    n_touch = 0
    for body in root.iter("body"):
        bname = body.get("name", "")
        site_name = f"site_{bname}"
        if body.find(f"site[@name='{site_name}']") is not None:
            ET.SubElement(sensor_el, "touch",
                          name=f"touch_{bname}", site=site_name)
            n_touch += 1

    # COM position and velocity (root body = link_body_0)
    root_body_id = "link_body_0"
    ET.SubElement(sensor_el, "framepos",
                  name="com_pos", objtype="body", objname=root_body_id)
    ET.SubElement(sensor_el, "framelinvel",
                  name="com_vel", objtype="body", objname=root_body_id,
                  reftype="world", refname="world")

    n_sensors = len(list(sensor_el))
    print(f"  Total sensors added: {n_sensors}")
    print(f"    jointpos+vel: {len(body_yaw)+len(pitch_jnts)+len(leg_jnts)+len(foot_jnts)} × 2")
    print(f"    touch       : {n_touch}")
    print(f"    framepos+vel: 2")

    return n_sensors


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Patch FARMS centipede.xml: add terrain hfield + sensor suite")
    p.add_argument("--xml",           default=XML_PATH,
                   help="Path to centipede.xml")
    p.add_argument("--terrain",       default=None,
                   help="Path to terrain heightmap PNG (1.png from generate_terrain_multifreq.py)")
    p.add_argument("--terrain-only",  action="store_true",
                   help="Only swap terrain PNG, skip sensor patching")
    p.add_argument("--sensors-only",  action="store_true",
                   help="Only add sensors, keep existing ground geom")
    p.add_argument("--restore",       action="store_true",
                   help="Restore from backup")
    args = p.parse_args()

    xml_path = args.xml

    # ── restore ──
    if args.restore:
        backup_p = xml_path + ".sensors_backup"
        if os.path.exists(backup_p):
            shutil.copy2(backup_p, xml_path)
            print(f"Restored from {backup_p}")
        else:
            print(f"ERROR: No backup found at {backup_p}")
        return

    # ── terrain-only swap (fast path) ──
    if args.terrain_only:
        if args.terrain is None:
            print("ERROR: --terrain-only requires --terrain <path>")
            return
        print(f"[patch_farms_xml] Swapping terrain only...")
        swap_terrain_only(xml_path, args.terrain)
        return

    # ── full patch ──
    print(f"[patch_farms_xml] Patching: {xml_path}")
    backup(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    if not args.sensors_only:
        print("\n[1/2] Patching terrain...")
        terrain_png = args.terrain or "terrain/1.png"
        patch_terrain(root, terrain_png)

    print("\n[2/2] Adding sensors...")
    patch_sensors(root)

    # Write output
    with open(xml_path, "w") as f:
        f.write(prettify(root))
    print(f"\n  Written: {xml_path}")
    print("\n[patch_farms_xml] Done.")
    print("\nTo swap terrain later (fast, no rebuild needed):")
    print(f"  python patch_farms_xml.py --terrain <path/to/1.png> --terrain-only")


if __name__ == "__main__":
    main()
