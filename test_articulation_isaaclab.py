#!/usr/bin/env python3
"""
test_articulation.py — Load centipede as Articulation, print joint names
=========================================================================
Usage:
    python scripts/test_articulation.py
"""

import os
import sys

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": False})

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

def main():
    sim_cfg = SimulationCfg(dt=0.0005)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.3, 0.3, 0.15], target=[0.0, 0.0, 0.0])

    # ── Ground plane ──
    from pxr import UsdGeom, UsdPhysics, Gf
    stage = sim.stage
    ground_path = "/World/GroundPlane"
    UsdGeom.Xform.Define(stage, ground_path)
    plane_path = ground_path + "/CollisionPlane"
    plane_geom = UsdGeom.Plane.Define(stage, plane_path)
    plane_geom.GetAxisAttr().Set("Z")
    plane_geom.GetExtentAttr().Set([(-50, -50, 0), (50, 50, 0)])
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(plane_path))

    # ── Load centipede USD manually, then remove the extra worldBody root ──
    usd_path = os.path.join(BASE_DIR, "assets", "centipede.usd")

    # Step 1: Spawn the USD into the stage at /World/Centipede
    spawn_cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
    spawn_cfg.func("/World/Centipede", spawn_cfg, translation=(0.0, 0.0, 0.02))

    # Step 2: Remove the extra worldBody prim that the MJCF importer created
    # (it's a second articulation root that confuses Isaac Lab)
    world_body_prim = stage.GetPrimAtPath("/World/Centipede/worldBody")
    if world_body_prim.IsValid():
        stage.RemovePrim("/World/Centipede/worldBody")
        print("Removed /World/Centipede/worldBody (extra articulation root)")

    # Step 3: Create Articulation pointing to the real robot root
    # Use spawn=None since we already spawned it above
    robot_cfg = ArticulationCfg(
        prim_path="/World/Centipede/link_body_0",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.02),
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=10.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.reset()

    print(f"\n{'='*60}")
    print(f"Articulation loaded!")
    print(f"  num_joints: {robot.num_joints}")
    print(f"  num_bodies: {robot.num_bodies}")
    print(f"{'='*60}")

    # Print all joint names
    print(f"\n--- All joint names ({robot.num_joints}) ---")
    joint_names = robot.joint_names
    for i, name in enumerate(joint_names):
        print(f"  [{i:3d}] {name}")

    # Print all body names
    print(f"\n--- All body names ({robot.num_bodies}) ---")
    body_names = robot.body_names
    for i, name in enumerate(body_names):
        print(f"  [{i:3d}] {name}")

    # Test finding specific joints
    print(f"\n--- Joint lookup tests ---")
    test_names = [
        "joint_body_1", "joint_body_19",
        "link_pitch_body_0", "link_pitch_body_19",
        "link_leg_0_L_0", "joint_leg_0_L_1",
        "link_leg_0_R_0", "joint_leg_0_R_1",
    ]
    for name in test_names:
        try:
            ids = robot.find_joints(name)
            print(f"  {name:30s} → index {ids}")
        except Exception as e:
            print(f"  {name:30s} → ERROR: {e}")

    # Run a few steps to see if it stands
    print(f"\nRunning 2000 steps (1 second)...")
    for step in range(2000):
        robot.update(sim_cfg.dt)
        # Zero torques
        robot.set_joint_effort_target(
            torch.zeros(1, robot.num_joints, device=robot.device))
        robot.write_data_to_sim()
        sim.step()

    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    print(f"Root position after 1s: ({root_pos[0]:.4f}, {root_pos[1]:.4f}, {root_pos[2]:.4f})")

    # Keep window open for inspection
    print("\nPress Ctrl+C to exit...")
    while sim_app.is_running():
        sim.step()

    sim_app.close()

if __name__ == "__main__":
    main()
