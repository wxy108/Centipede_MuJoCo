"""
patch_forcelimits.py — Patch pitch joint stiffness and damping
==============================================================
Updates all pitch body joints and passive pitch joints with the
specified compliance parameters.

Default values (terrain-conforming, overdamped):
  stiffness = 1e-3  (10x softer than original 1e-2)
  damping   = 1e-4  (10x softer than original 1e-3)

Usage:
  python patch_forcelimits.py                           # apply defaults
  python patch_forcelimits.py --stiffness 5e-4 --damping 5e-5
"""
import re
import os
import argparse

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

def patch_pitch_compliance(stiffness="1e-3", damping="1e-4"):
    path = os.path.join(PROJECT_ROOT, "models", "farms", "centipede.xml")
    with open(path, 'r') as f:
        content = f.read()

    # Replace stiffness on all pitch joints
    content = re.sub(
        r'(name="joint_(?:pitch_body|passive)_\d+"[^/]*?)stiffness="[^"]*"',
        rf'\1stiffness="{stiffness}"',
        content
    )

    # Replace damping on all pitch joints
    content = re.sub(
        r'(name="joint_(?:pitch_body|passive)_\d+"[^/]*?)damping="[^"]*"',
        rf'\1damping="{damping}"',
        content
    )

    with open(path, 'w') as f:
        f.write(content)
    print(f"Patched pitch joints: stiffness={stiffness}, damping={damping}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stiffness", default="1e-3",
                   help="Pitch joint stiffness (default: 1e-3)")
    p.add_argument("--damping", default="1e-4",
                   help="Pitch joint damping (default: 1e-4)")
    args = p.parse_args()
    patch_pitch_compliance(args.stiffness, args.damping)