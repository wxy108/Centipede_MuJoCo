#!/usr/bin/env python3
"""
patch_passive_and_roll.py — Patch centipede.xml to:
  1. Add pitch actuators for the 4 joint_passive_N joints (fix 4-segment folding)
  2. Zero stiffness/damping on joint_passive_N (impedance controller takes over)
  3. Add roll actuators for all body roll joints (joint_roll_body_N + joint_roll_passive_N)
  4. Zero stiffness/damping on body roll joints (impedance controller takes over)

Run from repo root:
    python scripts/optimization/farms/patch_passive_and_roll.py
"""

import re
import os

XML_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "farms", "centipede.xml")
XML_PATH = os.path.normpath(XML_PATH)


def patch_xml():
    with open(XML_PATH, "r", encoding="utf-8") as f:
        xml = f.read()

    # ── 1. Zero stiffness/damping on joint_passive_N (pitch passive joints) ──
    for i in range(4):
        old = re.search(
            rf'<joint\s+name="joint_passive_{i}"[^/]*/>', xml
        )
        if old:
            tag = old.group(0)
            new_tag = re.sub(r'damping="[^"]*"', 'damping="0"', tag)
            new_tag = re.sub(r'stiffness="[^"]*"', 'stiffness="0"', new_tag)
            xml = xml.replace(tag, new_tag)
            print(f"  Zeroed springs on joint_passive_{i}")

    # ── 2. Zero stiffness/damping on joint_roll_body_N ──
    for i in range(20):
        old = re.search(
            rf'<joint\s+name="joint_roll_body_{i}"[^/]*/>', xml
        )
        if old:
            tag = old.group(0)
            new_tag = re.sub(r'damping="[^"]*"', 'damping="0"', tag)
            new_tag = re.sub(r'stiffness="[^"]*"', 'stiffness="0"', new_tag)
            xml = xml.replace(tag, new_tag)
            print(f"  Zeroed springs on joint_roll_body_{i}")

    # ── 3. Zero stiffness/damping on joint_roll_passive_N ──
    for i in range(4):
        old = re.search(
            rf'<joint\s+name="joint_roll_passive_{i}"[^/]*/>', xml
        )
        if old:
            tag = old.group(0)
            new_tag = re.sub(r'damping="[^"]*"', 'damping="0"', tag)
            new_tag = re.sub(r'stiffness="[^"]*"', 'stiffness="0"', new_tag)
            xml = xml.replace(tag, new_tag)
            print(f"  Zeroed springs on joint_roll_passive_{i}")

    # ── 4. Add actuators before </actuator> ──
    new_actuators = "\n    <!-- Pitch actuators for passive transition joints -->\n"
    for i in range(4):
        new_actuators += (
            f'    <general name="act_joint_passive_{i}" '
            f'joint="joint_passive_{i}" '
            f'gainprm="1 0 0" biasprm="0 0 0" ctrllimited="false"/>\n'
        )

    new_actuators += "    <!-- Roll actuators for body roll joints -->\n"
    for i in range(20):
        new_actuators += (
            f'    <general name="act_joint_roll_body_{i}" '
            f'joint="joint_roll_body_{i}" '
            f'gainprm="1 0 0" biasprm="0 0 0" ctrllimited="false"/>\n'
        )

    new_actuators += "    <!-- Roll actuators for passive transition roll joints -->\n"
    for i in range(4):
        new_actuators += (
            f'    <general name="act_joint_roll_passive_{i}" '
            f'joint="joint_roll_passive_{i}" '
            f'gainprm="1 0 0" biasprm="0 0 0" ctrllimited="false"/>\n'
        )

    # Insert before </actuator>
    xml = xml.replace("  </actuator>", new_actuators + "  </actuator>")

    # ── Write back ──
    with open(XML_PATH, "w", encoding="utf-8") as f:
        f.write(xml)

    print(f"\nPatched: {XML_PATH}")
    print(f"  Added 4 pitch actuators (joint_pitch_body_3..3)")
    print(f"  Added 20 body roll actuators (joint_roll_body_0..19)")
    print(f"  Added 4 passive roll actuators (joint_roll_body_3..3)")
    print(f"  Total new actuators: 28")


if __name__ == "__main__":
    print("Patching centipede.xml ...")
    patch_xml()
    print("Done!")
