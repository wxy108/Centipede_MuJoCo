#!/usr/bin/env python3
"""
rename_passive_to_pitch.py — Unify passive joint naming with pitch_body naming.

Renames across the ENTIRE repo:
  joint_pitch_body_3  → joint_pitch_body_3
  joint_pitch_body_7  → joint_pitch_body_7
  joint_pitch_body_11  → joint_pitch_body_11
  joint_pitch_body_15  → joint_pitch_body_15

And all derived names:
  link_passive_N         → link_pitch_body_M
  joint_passive_N        → joint_pitch_body_M
  joint_roll_passive_N   → joint_roll_body_M
  act_joint_passive_N    → act_joint_pitch_body_M
  act_joint_roll_passive_N → act_joint_roll_body_M
  sp_joint_passive_N     → sp_joint_pitch_body_M
  sv_joint_passive_N     → sv_joint_pitch_body_M

Also cleans up Python files:
  - Removes dual-pattern checks ('joint_passive' in nm or ...)
  - Simplifies regex patterns that handled both naming conventions

Run from repo root:
    python scripts/optimization/farms/rename_passive_to_pitch.py
"""

import os
import re
import glob

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Mapping: passive index → pitch_body index
INDEX_MAP = {0: 3, 1: 7, 2: 11, 3: 15}

# ═══════════════════════════════════════════════════════════════════
# Step 1: Exact string replacements (order matters — longer first)
# ═══════════════════════════════════════════════════════════════════

def build_replacements():
    """Build all exact string replacement pairs."""
    pairs = []
    for old_i, new_i in INDEX_MAP.items():
        # Actuator names (longest first to avoid partial matches)
        pairs.append((f"act_joint_roll_passive_{old_i}", f"act_joint_roll_body_{new_i}"))
        pairs.append((f"act_joint_passive_{old_i}", f"act_joint_pitch_body_{new_i}"))
        # Sensor names
        pairs.append((f"sp_joint_passive_{old_i}", f"sp_joint_pitch_body_{new_i}"))
        pairs.append((f"sv_joint_passive_{old_i}", f"sv_joint_pitch_body_{new_i}"))
        # Joint names (roll first — longer)
        pairs.append((f"joint_roll_passive_{old_i}", f"joint_roll_body_{new_i}"))
        pairs.append((f"joint_passive_{old_i}", f"joint_pitch_body_{new_i}"))
        # Body/link names
        pairs.append((f"link_passive_{old_i}", f"link_pitch_body_{new_i}"))
    return pairs


# ═══════════════════════════════════════════════════════════════════
# Step 2: Regex pattern replacements for Python files
# ═══════════════════════════════════════════════════════════════════

# These regex-based patterns appear in many optimization scripts
REGEX_REPLACEMENTS = [
    # Pattern: r'joint_pitch_body_\d+' → r'joint_pitch_body_\d+'
    # (but only when it's a standalone regex pattern, not combined)
    (r"r'joint_passive_\\d\+'", r"r'joint_pitch_body_\\d+'"),
    (r'r"joint_passive_\\d\+"', r'r"joint_pitch_body_\\d+"'),

    # Pattern: 'joint_passive' in nm → (remove, since joint_pitch_body already covers it)
    # These appear as: 'joint_pitch_body' in nm
    (r"'joint_pitch_body' in (\w+) or 'joint_passive' in \1",
     r"'joint_pitch_body' in \1"),
    (r"'joint_pitch_body' in (\w+) or 'joint_passive_' in \1",
     r"'joint_pitch_body' in \1"),

    # Pattern: joint_roll_passive_\d+ regex → joint_roll_body_\d+
    (r"r'joint_roll_passive_\\d\+'", r"r'joint_roll_body_\\d+'"),
    (r'r"joint_roll_passive_\\d\+"', r'r"joint_roll_body_\\d+"'),

    # Pattern: joint_pitch_body alternation → joint_pitch_body
    (r"\(\?:joint_pitch_body\|passive\)", r"joint_pitch_body"),

    # Pattern: joint_pitch_body_\d+ → joint_pitch_body_\d+
    (r"\(joint_pitch_body_\\d\+\|joint_passive_\\d\+\)", r"joint_pitch_body_\\d+"),

    # Comments referencing old naming
    (r"joint_pitch_body_3 \.\. joint_pitch_body_15\s+\(passive, no actuator\)",
     "  (indices 3,7,11,15 are transition joints between modules)"),
]


def apply_exact_replacements(text, pairs):
    """Apply all exact string replacements."""
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def apply_regex_replacements(text):
    """Apply regex-based pattern cleanups for Python files."""
    for pattern, replacement in REGEX_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    return text


def process_file(filepath, pairs, is_python=False):
    """Process a single file."""
    with open(filepath, "r", encoding="utf-8") as f:
        original = f.read()

    modified = apply_exact_replacements(original, pairs)
    if is_python:
        modified = apply_regex_replacements(modified)

    if modified != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(modified)
        return True
    return False


def fix_xml_before_rename(xml_path):
    """
    Fix issues in centipede.xml BEFORE the rename:
    1. Remove phantom act_joint_roll_body_{3,7,11,15} actuators
       (these target joints that don't exist yet — they'll be created by the rename
        of joint_roll_passive_N, but act_joint_roll_passive_N already handles them,
        so keeping both would create duplicates after rename)
    2. Clean up XML comments referencing "passive"
    """
    with open(xml_path, "r", encoding="utf-8") as f:
        xml = f.read()

    removed = 0
    for new_i in INDEX_MAP.values():  # 3, 7, 11, 15
        # Remove phantom roll actuator lines
        pattern = (
            rf'\s*<general name="act_joint_roll_body_{new_i}" '
            rf'joint="joint_roll_body_{new_i}" [^/]*/>\n'
        )
        xml, n = re.subn(pattern, '\n', xml)
        removed += n

    # Clean up XML comments
    xml = xml.replace(
        "<!-- Pitch actuators for passive transition joints -->",
        "<!-- Pitch actuators for transition joints (indices 3,7,11,15) -->"
    )
    xml = xml.replace(
        "<!-- Roll actuators for passive transition roll joints -->",
        "<!-- Roll actuators for transition roll joints (indices 3,7,11,15) -->"
    )

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)

    print(f"  Pre-rename cleanup: removed {removed} phantom roll actuators")


def main():
    pairs = build_replacements()

    print("Rename mapping:")
    for old_i, new_i in INDEX_MAP.items():
        print(f"  passive_{old_i} → pitch_body_{new_i}")
    print()

    # ── Pre-rename: fix phantom actuators in XML ──
    xml_path = os.path.join(REPO_ROOT, "models", "farms", "centipede.xml")
    fix_xml_before_rename(xml_path)

    # ── Process XML ──
    if process_file(xml_path, pairs):
        print(f"  [XML] Updated: {os.path.relpath(xml_path, REPO_ROOT)}")
    else:
        print(f"  [XML] No changes: {os.path.relpath(xml_path, REPO_ROOT)}")

    # ── Process all Python files ──
    py_files = glob.glob(os.path.join(REPO_ROOT, "**", "*.py"), recursive=True)
    updated = []
    for fp in sorted(py_files):
        if process_file(fp, pairs, is_python=True):
            updated.append(fp)
            print(f"  [PY]  Updated: {os.path.relpath(fp, REPO_ROOT)}")

    # ── Process YAML files ──
    yaml_files = glob.glob(os.path.join(REPO_ROOT, "**", "*.yaml"), recursive=True)
    yaml_files += glob.glob(os.path.join(REPO_ROOT, "**", "*.yml"), recursive=True)
    for fp in sorted(yaml_files):
        if process_file(fp, pairs):
            updated.append(fp)
            print(f"  [YAML] Updated: {os.path.relpath(fp, REPO_ROOT)}")

    print(f"\nDone! Updated {1 + len(updated)} files total.")
    print(f"\nAfter running this script, verify with:")
    print(f"  grep -rn 'passive' models/farms/centipede.xml")
    print(f"  (should return 0 results)")


if __name__ == "__main__":
    main()
