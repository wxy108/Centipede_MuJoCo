"""
apply_optimal_gains.py
Restores the FARMS XML from backup and applies the optimal gains
found by the Bayesian optimizer (iteration 192, cost=0.00528).
"""
import re
import os
import shutil

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
XML_PATH    = os.path.join(PROJECT_ROOT, "models", "farms", "centipede.xml")
BACKUP_PATH = XML_PATH + ".sensors_backup"

# ── Restore backup ────────────────────────────────────────────────────
if not os.path.exists(BACKUP_PATH):
    print(f"ERROR: backup not found: {BACKUP_PATH}")
    print("Cannot restore — please regenerate centipede.xml from sdf_to_mjcf.py")
    exit(1)

shutil.copy2(BACKUP_PATH, XML_PATH)
print(f"Restored from backup: {BACKUP_PATH}")

# ── Apply optimal gains ───────────────────────────────────────────────
with open(XML_PATH, "r", encoding="utf-8") as f:
    content = f.read()

n_total = 0

def sub(pattern, replacement, text):
    new, n = re.subn(pattern, replacement, text)
    return new, n

# Body joints: kp=64.945043, kv=0.055213531
content, n = sub(
    r'(name="act_joint_body_\d+"[^/]*?)kp="[^"]+" kv="[^"]+"',
    r'\1kp="64.945043" kv="0.055213531"',
    content)
print(f"  Body joints patched   : {n}")
n_total += n

# Leg DOF0: kp=1.2695223, kv=0.0055957565
content, n = sub(
    r'(name="act_joint_leg_\d+_[LR]_0"[^/]*?)kp="[^"]+" kv="[^"]+"',
    r'\1kp="1.2695223" kv="0.0055957565"',
    content)
print(f"  Leg DOF0 patched      : {n}")
n_total += n

# Leg DOF1: kp=0.14754977, kv=0.0011443435
content, n = sub(
    r'(name="act_joint_leg_\d+_[LR]_1"[^/]*?)kp="[^"]+" kv="[^"]+"',
    r'\1kp="0.14754977" kv="0.0011443435"',
    content)
print(f"  Leg DOF1 patched      : {n}")
n_total += n

# Leg DOF2: kp=1.2695223, kv=0.00043637087
content, n = sub(
    r'(name="act_joint_leg_\d+_[LR]_2"[^/]*?)kp="[^"]+" kv="[^"]+"',
    r'\1kp="1.2695223" kv="0.00043637087"',
    content)
print(f"  Leg DOF2 patched      : {n}")
n_total += n

# Leg DOF3: kp=1.2695223, kv=0.0090957829
content, n = sub(
    r'(name="act_joint_leg_\d+_[LR]_3"[^/]*?)kp="[^"]+" kv="[^"]+"',
    r'\1kp="1.2695223" kv="0.0090957829"',
    content)
print(f"  Leg DOF3 patched      : {n}")
n_total += n

# Foot joints: kp=1.2695223, kv=0.0090957829
content, n = sub(
    r'(name="act_joint_foot_\d+_\d"[^/]*?)kp="[^"]+" kv="[^"]+"',
    r'\1kp="1.2695223" kv="0.0090957829"',
    content)
print(f"  Foot joints patched   : {n}")
n_total += n

with open(XML_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nTotal actuators patched: {n_total}")
print(f"Written: {XML_PATH}")
print("Done.")
