#!/usr/bin/env python3
"""
add_compliance.py — Add roll joints + convert body yaw to impedance control
============================================================================

1. BODY YAW → TORQUE-BASED IMPEDANCE:
   Change <position kp=64.945> to <general gainprm="1"> (direct torque).
   Controller computes: tau = kp*(target - q) - kv*qdot with tunable gains.

2. PASSIVE ROLL JOINTS ON BODY:
   Add joint_roll_body_N (X-axis) as second joint on link_pitch_body_N.
   MuJoCo composes multiple joints on one body → pitch+roll compound joint.

3. PASSIVE ROLL JOINTS ON LEGS:
   Add joint_roll_leg_N_L/R (X-axis) as second joint on link_leg_N_L/R_0.
   Gives the hip a roll DOF for terrain compliance.

Usage:
    python add_compliance.py                          # patch with defaults
    python add_compliance.py --rollk 5e-3 --rolld 2e-3
    python add_compliance.py --restore                # undo all changes
"""
import os, re, sys, shutil, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
XML_PATH = os.path.join(BASE, "models", "farms", "centipede.xml")
XML_BACKUP = XML_PATH + ".pre_compliance_backup"


def convert_body_actuators_to_general(xml):
    """Convert <position name="act_joint_body_N" ...> to <general ...>"""
    count = 0
    def repl(m):
        nonlocal count; count += 1
        name, joint = m.group(1), m.group(2)
        return (f'<general name="{name}" joint="{joint}" '
                f'gainprm="1 0 0" biasprm="0 0 0" ctrllimited="false"/>')

    xml = re.sub(
        r'<position name="(act_joint_body_\d+)" joint="(joint_body_\d+)" '
        r'kp="[^"]*" kv="[^"]*" forcelimited="[^"]*" forcerange="[^"]*"/>',
        repl, xml)
    return xml, count


def add_body_roll_joints(xml, roll_k, roll_d):
    """
    Add a roll joint (X-axis) right after each pitch joint on link_pitch_body_N.
    MuJoCo allows multiple joints on the same body → compound pitch+roll.
    """
    count = 0
    def repl(m):
        nonlocal count; count += 1
        pitch_line = m.group(0)
        # Extract the index N from joint_pitch_body_N
        idx = re.search(r'joint_pitch_body_(\d+)', pitch_line).group(1)
        roll_joint = (
            f'\n        <joint name="joint_roll_body_{idx}" type="hinge" '
            f'pos="0 0 0" axis="1 0 0" range="-0.5236 0.5236" '
            f'stiffness="{roll_k:.6e}" damping="{roll_d:.6e}"/>'
        )
        return pitch_line + roll_joint

    # Match both joint_pitch_body_N and joint_passive_N (both are body pitch joints)
    xml = re.sub(
        r'<joint name="joint_pitch_body_\d+" type="hinge" [^/]*/>', repl, xml)
    # Also handle joint_passive_N naming variant
    def repl_passive(m):
        nonlocal count; count += 1
        pitch_line = m.group(0)
        idx = re.search(r'joint_passive_(\d+)', pitch_line).group(1)
        roll_joint = (
            f'\n        <joint name="joint_roll_passive_{idx}" type="hinge" '
            f'pos="0 0 0" axis="1 0 0" range="-0.5236 0.5236" '
            f'stiffness="{roll_k:.6e}" damping="{roll_d:.6e}"/>'
        )
        return pitch_line + roll_joint
    xml = re.sub(
        r'<joint name="joint_passive_\d+" type="hinge" [^/]*/>', repl_passive, xml)
    return xml, count


def add_leg_roll_joints(xml, leg_roll_k, leg_roll_d):
    """
    Add a passive roll joint (X-axis) as a second joint on link_leg_N_L/R_0.
    Currently that body has joint_leg_N_L/R_0 (hip yaw, Z-axis).
    Adding a roll gives the hip coxa a twist compliance.
    """
    count = 0
    def repl(m):
        nonlocal count; count += 1
        yaw_line = m.group(0)
        # Extract leg index, side from joint_leg_N_S_0
        parts = re.search(r'joint_leg_(\d+)_(L|R)_0', yaw_line)
        idx, side = parts.group(1), parts.group(2)
        roll_joint = (
            f'\n            <joint name="joint_roll_leg_{idx}_{side}" type="hinge" '
            f'pos="0 0 0" axis="1 0 0" range="-0.3491 0.3491" '
            f'stiffness="{leg_roll_k:.6e}" damping="{leg_roll_d:.6e}"/>'
        )
        return yaw_line + roll_joint

    xml = re.sub(
        r'<joint name="joint_leg_\d+_[LR]_0" type="hinge" [^/]*/>', repl, xml)
    return xml, count


def add_pitch_actuators(xml):
    """
    Add <general> actuators for each pitch joint (body + passive).
    These allow the impedance controller to apply torque for gravity
    compensation + soft compliance, rather than relying on passive springs.

    Also sets pitch joint stiffness/damping to 0 (controller handles everything).
    """
    # 1. Zero out passive spring on pitch joints (controller replaces it)
    def zero_pitch_spring(m):
        return m.group(0).replace(
            f'stiffness="{m.group("k")}"', 'stiffness="0"'
        ).replace(
            f'damping="{m.group("d")}"', 'damping="0"'
        )

    xml = re.sub(
        r'<joint name="(?:joint_pitch_body_\d+|joint_passive_\d+)" '
        r'type="hinge" [^/]*stiffness="(?P<k>[^"]*)"[^/]*damping="(?P<d>[^"]*)"[^/]*/>',
        zero_pitch_spring, xml)

    # 2. Add general actuators for pitch joints in the <actuator> section
    # Find all pitch joint names
    pitch_names = re.findall(
        r'<joint name="joint_pitch_body_\d+" type="hinge"', xml)

    # Build actuator lines
    act_lines = []
    for jname in pitch_names:
        act_name = "act_" + jname
        act_lines.append(
            f'    <general name="{act_name}" joint="{jname}" '
            f'gainprm="1 0 0" biasprm="0 0 0" ctrllimited="false"/>')

    # Insert before </actuator>
    act_block = "\n".join(act_lines)
    xml = xml.replace("</actuator>", act_block + "\n  </actuator>")

    return xml, len(pitch_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-rollk", type=float, default=5e-3,
                        help="Body roll stiffness (default: 5e-3)")
    parser.add_argument("--body-rolld", type=float, default=2e-3,
                        help="Body roll damping (default: 2e-3)")
    parser.add_argument("--leg-rollk", type=float, default=1e-2,
                        help="Leg roll stiffness (default: 1e-2, stiffer than body)")
    parser.add_argument("--leg-rolld", type=float, default=5e-3,
                        help="Leg roll damping (default: 5e-3)")
    parser.add_argument("--add-pitch-actuators", action="store_true",
                        help="Add general actuators for pitch joints (impedance control)")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    if args.restore:
        if os.path.exists(XML_BACKUP):
            shutil.copy2(XML_BACKUP, XML_PATH)
            print(f"Restored from {XML_BACKUP}")
        else:
            print("No backup found!")
        return

    # Backup
    if not os.path.exists(XML_BACKUP):
        shutil.copy2(XML_PATH, XML_BACKUP)
        print(f"Backup saved: {XML_BACKUP}")

    with open(XML_PATH, 'r') as f:
        xml = f.read()

    # Verify file is complete
    if not xml.strip().endswith('</mujoco>'):
        print("ERROR: XML file appears truncated!")
        return

    # 1. Convert body yaw actuators to general (torque mode)
    xml, n_act = convert_body_actuators_to_general(xml)
    print(f"[1] Converted {n_act} body actuators → general (torque mode)")

    # 2. Add body roll joints
    xml, n_broll = add_body_roll_joints(xml, args.body_rollk, args.body_rolld)
    print(f"[2] Added {n_broll} body roll joints (k={args.body_rollk}, d={args.body_rolld})")

    # 3. Add leg roll joints
    xml, n_lroll = add_leg_roll_joints(xml, args.leg_rollk, args.leg_rolld)
    print(f"[3] Added {n_lroll} leg roll joints (k={args.leg_rollk}, d={args.leg_rolld})")

    # 4. Optionally add pitch actuators for impedance control
    n_pitch = 0
    if args.add_pitch_actuators:
        xml, n_pitch = add_pitch_actuators(xml)
        print(f"[4] Added {n_pitch} pitch actuators (general, torque mode)")
        print(f"    Pitch joint springs zeroed — controller handles gravity comp")

    with open(XML_PATH, 'w') as f:
        f.write(xml)
    print(f"\nSaved: {XML_PATH}")
    print(f"Total new joints: {n_broll + n_lroll}")
    if n_pitch:
        print(f"Total new actuators: {n_pitch} (pitch)")
    print(f"Body yaw actuators now in torque mode — update controller!")


if __name__ == "__main__":
    main()
