"""
kinematics.py — Naming conventions, index lookups, and wave functions
=====================================================================
Single source of truth for the centipede MuJoCo model's joint/actuator/sensor
naming scheme. All other scripts import from here.

Model reference:  centipede_model_reference.pdf
Convention:       Raw Blender world space, animal faces +X, dorsal +Z.
"""

import numpy as np
import mujoco

# ═══════════════════════════════════════════════════════════
# Model constants
# ═══════════════════════════════════════════════════════════

N_BODY_SEGMENTS = 21          # b0 (head) .. b20 (terminal)
N_BODY_JOINTS   = 20          # jb1 .. jb20
N_LEGS_PER_SIDE = 19          # leg 1..19
N_LEGS_TOTAL    = 38          # 19 left + 19 right
N_LEG_DOF       = 3           # DOF 0 (yaw), DOF 1 (upper pitch), DOF 2 (lower pitch)
N_WALKING_LINKS = 19          # segments that carry walking legs (b1..b19)

LINK_LENGTH     = 0.005       # meters (a = 5 mm)
BODY_LENGTH     = N_WALKING_LINKS * LINK_LENGTH   # L = 0.095 m


# ═══════════════════════════════════════════════════════════
# Naming functions
# ═══════════════════════════════════════════════════════════

def body_name(i):
    """Body name. i = 0 (head) .. 20 (terminal)."""
    return f"b{i}"

def body_joint_name(i):
    """Body joint name. i = 1..20. Connects b(i-1) to b(i)."""
    return f"jb{i}"

def leg_body_name(n, side):
    """Upper leg body. n = 1..19, side = 'L' or 'R'."""
    return f"l{n}{side}"

def leg_lower_name(n, side):
    """Lower leg body."""
    return f"l{n}{side}l"

def foot_name(n, side):
    """Foot body."""
    return f"f{n}{side}"

def leg_joint_name(n, side, dof):
    """Leg joint. n = 1..19, side = 'L'/'R', dof = 0/1/2."""
    return f"jl{n}{side}{dof}"

def pos_actuator_name(joint_name):
    """Position actuator for a joint."""
    return f"p_{joint_name}"

def vel_actuator_name(joint_name):
    """Velocity actuator for a joint."""
    return f"v_{joint_name}"

def jointpos_sensor_name(joint_name):
    """Joint position sensor."""
    return f"sp_{joint_name}"

def jointvel_sensor_name(joint_name):
    """Joint velocity sensor."""
    return f"sv_{joint_name}"

def touch_sensor_name(n, side):
    """Foot touch sensor. n = 1..19, side = 'L'/'R'."""
    return f"t{n}{side}"

def foot_site_name(n, side):
    """Foot site (used by touch sensor)."""
    return f"sf{n}{side}"


# ═══════════════════════════════════════════════════════════
# Index resolution
# ═══════════════════════════════════════════════════════════

class ModelIndex:
    """
    Resolves joint/actuator/sensor names to integer indices at load time.

    Both position and velocity actuator indices are cached.
    Velocity actuator ids are −1 when not present in the XML, so the
    controller can check `if vel_id >= 0` and skip gracefully.

    Usage:
        model = mujoco.MjModel.from_xml_path("centipede.xml")
        idx   = ModelIndex(model)

        # Position actuator
        data.ctrl[idx.pos_act["jb7"]] = 0.3

        # Velocity actuator (−1 if not in XML)
        if idx.vel_act["jb7"] >= 0:
            data.ctrl[idx.vel_act["jb7"]] = 0.05

        # Touch sensor
        force = data.sensordata[model.sensor_adr[idx.touch["t7L"]]]
    """

    def __init__(self, model):
        self.model = model

        OBJ_JNT = mujoco.mjtObj.mjOBJ_JOINT
        OBJ_ACT = mujoco.mjtObj.mjOBJ_ACTUATOR
        OBJ_SEN = mujoco.mjtObj.mjOBJ_SENSOR

        def _jid(name):
            return mujoco.mj_name2id(model, OBJ_JNT, name)

        def _aid(name):
            return mujoco.mj_name2id(model, OBJ_ACT, name)

        def _sid(name):
            return mujoco.mj_name2id(model, OBJ_SEN, name)

        # ── Body joint ids ──
        self.body_joint = {}
        for i in range(1, N_BODY_JOINTS + 1):
            name = body_joint_name(i)
            self.body_joint[name] = _jid(name)

        # ── Leg joint ids ──
        self.leg_joint = {}
        for n in range(1, N_LEGS_PER_SIDE + 1):
            for side in ('L', 'R'):
                for dof in range(N_LEG_DOF):
                    name = leg_joint_name(n, side, dof)
                    self.leg_joint[name] = _jid(name)

        # ── Position actuators ──
        self.pos_act = {}
        for i in range(1, N_BODY_JOINTS + 1):
            jn = body_joint_name(i)
            self.pos_act[jn] = _aid(pos_actuator_name(jn))
        for n in range(1, N_LEGS_PER_SIDE + 1):
            for side in ('L', 'R'):
                for dof in range(N_LEG_DOF):
                    jn = leg_joint_name(n, side, dof)
                    self.pos_act[jn] = _aid(pos_actuator_name(jn))

        # ── Velocity actuators (−1 when absent from XML) ──
        self.vel_act = {}
        for i in range(1, N_BODY_JOINTS + 1):
            jn = body_joint_name(i)
            self.vel_act[jn] = _aid(vel_actuator_name(jn))
        for n in range(1, N_LEGS_PER_SIDE + 1):
            for side in ('L', 'R'):
                for dof in range(N_LEG_DOF):
                    jn = leg_joint_name(n, side, dof)
                    self.vel_act[jn] = _aid(vel_actuator_name(jn))

        # ── Touch sensors ──
        self.touch = {}
        for n in range(1, N_LEGS_PER_SIDE + 1):
            for side in ('L', 'R'):
                name = touch_sensor_name(n, side)
                self.touch[name] = _sid(name)

        # ── Jointpos sensors ──
        self.jointpos = {}
        for i in range(1, N_BODY_JOINTS + 1):
            jn = body_joint_name(i)
            self.jointpos[jn] = _sid(jointpos_sensor_name(jn))
        for n in range(1, N_LEGS_PER_SIDE + 1):
            for side in ('L', 'R'):
                for dof in range(N_LEG_DOF):
                    jn = leg_joint_name(n, side, dof)
                    self.jointpos[jn] = _sid(jointpos_sensor_name(jn))

        # ── Jointvel sensors (sv_jb*, sv_jl*) ──
        # Already emitted by generate_mjcf.py. Authoritative velocity source
        # for PD analysis — avoids index arithmetic across the free joint.
        self.jointvel = {}
        for i in range(1, N_BODY_JOINTS + 1):
            jn = body_joint_name(i)
            self.jointvel[jn] = _sid(jointvel_sensor_name(jn))
        for n in range(1, N_LEGS_PER_SIDE + 1):
            for side in ('L', 'R'):
                for dof in range(N_LEG_DOF):
                    jn = leg_joint_name(n, side, dof)
                    self.jointvel[jn] = _sid(jointvel_sensor_name(jn))

    # ── Convenience readers ───────────────────────────────────────────

    def get_touch_force(self, data, n, side):
        """Read touch sensor value for foot (n, side)."""
        sen_id = self.touch[touch_sensor_name(n, side)]
        return data.sensordata[self.model.sensor_adr[sen_id]]

    def get_joint_pos(self, data, joint_name):
        """Read joint position from sensor."""
        sen_id = self.jointpos[joint_name]
        return data.sensordata[self.model.sensor_adr[sen_id]]

    def get_joint_vel(self, data, joint_name):
        """Read joint velocity from sensor."""
        sen_id = self.jointvel[joint_name]
        return data.sensordata[self.model.sensor_adr[sen_id]]
