"""
farms_kinematics.py — Naming conventions and index lookups for the FARMS centipede model
=========================================================================================
Single source of truth for joint/actuator names in centipede.xml generated from
the FARMS SDF. All other FARMS-model scripts import from here.

Joint naming convention (from sdf_to_mjcf.py output):
  Body yaw joints  : joint_body_1  .. joint_body_19   (joint_body_0 is welded)
  Body pitch joints: joint_pitch_body_0 .. joint_pitch_body_19  (impedance + grav comp)
                     indices 3,7,11,15 are transition joints between modules
  Leg joints       : joint_leg_{n}_L_{d}  /  joint_leg_{n}_R_{d}
                     n = 0..18 (19 legs per side), d = 0..3 (4 DOF)
  Foot joints      : joint_foot_{n}_0  /  joint_foot_{n}_1  (n = 0..18)
  Freejoint        : root

Actuator naming:
  act_joint_body_{i}        i = 1..19
  act_joint_leg_{n}_{L|R}_{d}  n = 0..18, d = 0..3
  act_joint_foot_{n}_{0|1}

Notes:
  - joint_body_0 is welded (no joint element, no actuator)
  - Pitch joints: impedance-controlled with gravity compensation
  - DOF 2 and 3 of leg have near-zero amplitude in FARMS params
  - Right leg = negated left leg waveform
"""

import numpy as np
import mujoco

# ═══════════════════════════════════════════════════════════
# Model constants
# ═══════════════════════════════════════════════════════════

N_BODY_SEGS     = 21    # link_body_0 .. link_body_20
N_BODY_JOINTS   = 19    # joint_body_1 .. joint_body_19 (body_0 is welded)
N_LEGS          = 19    # leg index n = 0..18 per side
N_LEG_DOF       = 4     # DOF 0 (hip yaw), 1 (hip pitch), 2 (tibia), 3 (tarsus)
ACTIVE_DOFS     = (0, 1)  # only DOF 0 and 1 driven in FARMS model

LINK_LENGTH     = 0.005   # metres (5 mm inter-segment spacing)


# ═══════════════════════════════════════════════════════════
# Naming functions
# ═══════════════════════════════════════════════════════════

def body_joint_name(i):
    """Yaw joint connecting link_body_{i-1} to link_body_{i}. i = 1..19."""
    assert 1 <= i <= N_BODY_JOINTS, f"body joint index {i} out of range 1..{N_BODY_JOINTS}"
    return f"joint_body_{i}"

def pitch_joint_name(i):
    """Passive pitch joint before body segment i. i = 1..19."""
    return f"joint_pitch_body_{i}"

def leg_joint_name(n, side, dof):
    """
    Leg joint name.
    n    : 0..18 (leg number, 0 = most anterior)
    side : 'L' or 'R'
    dof  : 0..3
    """
    assert 0 <= n < N_LEGS,    f"leg index {n} out of range 0..{N_LEGS-1}"
    assert side in ('L', 'R'), f"side must be 'L' or 'R', got {side!r}"
    assert 0 <= dof < N_LEG_DOF, f"dof {dof} out of range 0..{N_LEG_DOF-1}"
    return f"joint_leg_{n}_{side}_{dof}"

def foot_joint_name(n, side_idx):
    """Foot joint. n = 0..18, side_idx = 0 (left) or 1 (right)."""
    return f"joint_foot_{n}_{side_idx}"

def body_act_name(i):
    """Position actuator for body yaw joint i. i = 1..19."""
    return f"act_joint_body_{i}"

def pitch_act_name(i):
    """General actuator for body pitch joint. i = 0..19 (matches joint_pitch_body_N)."""
    return f"act_joint_pitch_body_{i}"

def leg_act_name(n, side, dof):
    """Position actuator for leg joint."""
    return f"act_joint_leg_{n}_{side}_{dof}"

def foot_act_name(n, side_idx):
    """Position actuator for foot joint."""
    return f"act_joint_foot_{n}_{side_idx}"


# ═══════════════════════════════════════════════════════════
# Index resolution
# ═══════════════════════════════════════════════════════════

class FARMSModelIndex:
    """
    Resolves all joint/actuator names to integer MuJoCo indices at load time.
    Raises a clear error if any expected element is missing from the model.

    Usage:
        idx = FARMSModelIndex(model)
        # body actuator ids: idx.body_act_ids[i]  i=0..18  (body_1..body_19)
        # leg actuator ids:  idx.leg_act_ids[n, si, d]  n=0..18, si=0(L)/1(R), d=0..3
    """

    def __init__(self, model):
        self.model = model

        # ── body actuators (indices 0..18 → joint_body_1..joint_body_19) ──
        self.body_act_ids = np.zeros(N_BODY_JOINTS, dtype=int)
        for i in range(N_BODY_JOINTS):
            name = body_act_name(i + 1)
            aid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"Actuator not found in model: {name!r}")
            self.body_act_ids[i] = aid

        # ── leg actuators (n, side, dof) ──
        self.leg_act_ids = np.zeros((N_LEGS, 2, N_LEG_DOF), dtype=int)
        for n in range(N_LEGS):
            for si, side in enumerate(('L', 'R')):
                for dof in range(N_LEG_DOF):
                    name = leg_act_name(n, side, dof)
                    aid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                    if aid < 0:
                        raise ValueError(f"Actuator not found in model: {name!r}")
                    self.leg_act_ids[n, si, dof] = aid

        # ── body joint ids (for reading positions) ──
        self.body_jnt_ids = np.zeros(N_BODY_JOINTS, dtype=int)
        for i in range(N_BODY_JOINTS):
            name = body_joint_name(i + 1)
            jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint not found in model: {name!r}")
            self.body_jnt_ids[i] = jid

        # ── leg joint ids ──
        self.leg_jnt_ids = np.zeros((N_LEGS, 2, N_LEG_DOF), dtype=int)
        for n in range(N_LEGS):
            for si, side in enumerate(('L', 'R')):
                for dof in range(N_LEG_DOF):
                    name = leg_joint_name(n, side, dof)
                    jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if jid < 0:
                        raise ValueError(f"Joint not found in model: {name!r}")
                    self.leg_jnt_ids[n, si, dof] = jid

        # ── pitch actuators (optional — may not exist) ──
        # All pitch joints are named joint_pitch_body_N (N=0..19, skipping none)
        # Indices 3,7,11,15 are transition joints between body modules.
        self.pitch_act_ids  = []   # actuator ids (parallel with pitch_jnt_ids)
        self.pitch_jnt_ids  = []   # joint ids
        self.pitch_body_ids = []   # body ids (for gravity comp)
        self.has_pitch_actuators = False
        for j in range(model.njnt):
            jname = model.joint(j).name
            if not jname.startswith("joint_pitch_body_"):
                continue
            aname = "act_" + jname
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid >= 0:
                self.pitch_act_ids.append(aid)
                self.pitch_jnt_ids.append(j)
                self.pitch_body_ids.append(model.jnt_bodyid[j])
                self.has_pitch_actuators = True
        self.pitch_act_ids  = np.array(self.pitch_act_ids,  dtype=int)
        self.pitch_jnt_ids  = np.array(self.pitch_jnt_ids,  dtype=int)
        self.pitch_body_ids = np.array(self.pitch_body_ids, dtype=int)

        # ── roll actuators (optional — may not exist) ──
        # All body roll joints are named joint_roll_body_N
        self.roll_act_ids  = []
        self.roll_jnt_ids  = []
        self.has_roll_actuators = False
        for j in range(model.njnt):
            jname = model.joint(j).name
            if not jname.startswith("joint_roll_body_"):
                continue
            aname = "act_" + jname
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid >= 0:
                self.roll_act_ids.append(aid)
                self.roll_jnt_ids.append(j)
                self.has_roll_actuators = True
        self.roll_act_ids = np.array(self.roll_act_ids, dtype=int)
        self.roll_jnt_ids = np.array(self.roll_jnt_ids, dtype=int)

        # ── root freejoint (for COM tracking) ──
        self.root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")
        if self.root_body_id < 0:
            raise ValueError("Body 'link_body_0' not found in model")

        n_pitch_str = f", {len(self.pitch_act_ids)} pitch actuators" if self.has_pitch_actuators else ""
        n_roll_str = f", {len(self.roll_act_ids)} roll actuators" if self.has_roll_actuators else ""
        print(f"[FARMSModelIndex] Loaded: "
              f"{N_BODY_JOINTS} body actuators, "
              f"{N_LEGS}×2×{N_LEG_DOF} leg actuators{n_pitch_str}{n_roll_str}")

    # ── convenience accessors ──────────────────────────────

    def body_joint_pos(self, data, i):
        """Body joint angle. i = 1..19."""
        return data.qpos[self.model.jnt_qposadr[self.body_jnt_ids[i - 1]]]

    def leg_joint_pos(self, data, n, side, dof):
        """Leg joint angle."""
        si = 0 if side == 'L' else 1
        return data.qpos[self.model.jnt_qposadr[self.leg_jnt_ids[n, si, dof]]]

    def com_pos(self, data):
        """Root body subtree COM position."""
        return data.subtree_com[self.root_body_id].copy()

    def com_vel(self, data):
        """Root body subtree linear velocity."""
        return data.subtree_linvel[self.root_body_id].copy()
