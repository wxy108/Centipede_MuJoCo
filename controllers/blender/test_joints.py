"""
test_joints.py — Verify joint conventions by wiggling individual joints
=======================================================================
Sends a sinusoidal signal to one joint at a time while keeping all others
at zero. Use the MuJoCo viewer to visually confirm each joint moves in
the expected direction.

Usage:
    # Test body joint 7
    python test_joints.py --joint jb7

    # Test left leg 7 yaw
    python test_joints.py --joint jl7L0

    # Test all body joints one by one (5s each)
    python test_joints.py --sweep body

    # Test all leg DOFs for leg 7
    python test_joints.py --sweep leg7

    # Test all leg yaw joints
    python test_joints.py --sweep yaw
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

from kinematics import (
    ModelIndex, N_BODY_JOINTS, N_LEGS_PER_SIDE,
    body_joint_name, leg_joint_name, pos_actuator_name,
)

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "models", "blender", "centipede.xml"
)


def test_single_joint(model, data, idx, joint_name, amp=0.3, freq=0.5, duration=5.0):
    """Wiggle a single joint in the viewer."""
    
    act_name = pos_actuator_name(joint_name)
    try:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
    except Exception:
        print(f"ERROR: Actuator '{act_name}' not found.")
        return
    
    if act_id < 0:
        print(f"ERROR: Actuator '{act_name}' not found (id={act_id}).")
        return
    
    print(f"Testing joint: {joint_name}  (actuator: {act_name}, id={act_id})")
    print(f"  amp={amp:.3f} rad, freq={freq:.1f} Hz, duration={duration:.1f}s")
    print(f"  Watch the viewer. Close window or wait to finish.")
    
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.15
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        
        b0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b0")
        viewer.cam.lookat[:] = data.subtree_com[b0_id]
        
        start = data.time
        while viewer.is_running() and (data.time - start) < duration:
            t = data.time - start
            
            # Zero all controls
            data.ctrl[:] = 0
            
            # Drive the target joint
            data.ctrl[act_id] = amp * np.sin(2 * np.pi * freq * t)
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            dt_wall = model.opt.timestep
            time.sleep(max(0, dt_wall - 0.0001))
    
    print(f"  Done: {joint_name}")


def sweep_joints(model, data, idx, joint_names, amp=0.3, freq=0.5, per_joint=5.0):
    """Sweep through a list of joints, testing each one."""
    print(f"Sweeping {len(joint_names)} joints, {per_joint}s each.\n")
    
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 0.15
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        
        b0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b0")
        viewer.cam.lookat[:] = data.subtree_com[b0_id]
        
        for joint_name in joint_names:
            if not viewer.is_running():
                break
            
            act_name = pos_actuator_name(joint_name)
            act_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if act_id < 0:
                print(f"  SKIP: {joint_name} (no actuator)")
                continue
            
            print(f"  Testing: {joint_name} ...", end="", flush=True)
            
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            
            start = data.time
            while viewer.is_running() and (data.time - start) < per_joint:
                t = data.time - start
                data.ctrl[:] = 0
                data.ctrl[act_id] = amp * np.sin(2 * np.pi * freq * t)
                mujoco.mj_step(model, data)
                viewer.cam.lookat[:] = data.subtree_com[b0_id]
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - 0.0001))
            
            print(" done")
    
    print("\nSweep complete.")


def main():
    parser = argparse.ArgumentParser(description="Test centipede joints")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--joint", type=str, default=None,
                        help="Single joint name to test (e.g. jb7, jl7L0)")
    parser.add_argument("--sweep", type=str, default=None,
                        choices=["body", "yaw", "pitch1", "pitch2",
                                 "leg1", "leg7", "leg10", "leg19", "all_legs"],
                        help="Sweep a set of joints")
    parser.add_argument("--amp", type=float, default=0.3)
    parser.add_argument("--freq", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration per joint (seconds)")
    args = parser.parse_args()
    
    model_path = os.path.abspath(args.model)
    print(f"Loading: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    idx = ModelIndex(model)
    
    if args.joint:
        test_single_joint(model, data, idx, args.joint,
                          amp=args.amp, freq=args.freq, duration=args.duration)
    
    elif args.sweep:
        joints = []
        
        if args.sweep == "body":
            joints = [body_joint_name(i) for i in range(1, N_BODY_JOINTS + 1)]
        
        elif args.sweep == "yaw":
            for n in range(1, N_LEGS_PER_SIDE + 1):
                joints.append(leg_joint_name(n, 'L', 0))
                joints.append(leg_joint_name(n, 'R', 0))
        
        elif args.sweep == "pitch1":
            for n in range(1, N_LEGS_PER_SIDE + 1):
                joints.append(leg_joint_name(n, 'L', 1))
                joints.append(leg_joint_name(n, 'R', 1))
        
        elif args.sweep == "pitch2":
            for n in range(1, N_LEGS_PER_SIDE + 1):
                joints.append(leg_joint_name(n, 'L', 2))
                joints.append(leg_joint_name(n, 'R', 2))
        
        elif args.sweep.startswith("leg"):
            n = int(args.sweep[3:])
            for side in ('L', 'R'):
                for dof in range(3):
                    joints.append(leg_joint_name(n, side, dof))
        
        elif args.sweep == "all_legs":
            for n in [1, 5, 10, 15, 19]:
                for side in ('L', 'R'):
                    for dof in range(3):
                        joints.append(leg_joint_name(n, side, dof))
        
        sweep_joints(model, data, idx, joints,
                     amp=args.amp, freq=args.freq, per_joint=args.duration)
    
    else:
        print("Specify --joint or --sweep. Use --help for options.")


if __name__ == "__main__":
    main()
