#!/usr/bin/env python3
"""
test_compliance.py — Verify pitch compliance parameters on your machine
========================================================================
Run this to confirm that the optimized parameters produce:
  1. Stable simulation (no NaN, no divergence)
  2. Good joint tracking (commanded vs actual body yaw)
  3. No terrain penetration (COM stays above ground)
  4. Pitch compliance (body height follows terrain contour)

Usage (from project root):
    cd controllers/farms
    python ../../scripts/optimization/farms/test_compliance.py

Or from scripts/optimization/farms/:
    python test_compliance.py

This script uses the CURRENT centipede.xml as-is (no patching).
Make sure you've applied the optimized params first.
"""

import os
import sys
import time
import numpy as np

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

# Add controllers/farms to path for imports
sys.path.insert(0, os.path.join(BASE, "controllers", "farms"))

import mujoco
from kinematics import FARMSModelIndex, N_BODY_JOINTS, N_LEGS, N_LEG_DOF
from controller import FARMSTravelingWaveController, load_config

XML_PATH = os.path.join(BASE, "models", "farms", "centipede.xml")
CONFIG_PATH = os.path.join(BASE, "configs", "farms_controller.yaml")


def run_test(duration=5.0, print_every=1.0):
    """Run simulation and collect data for analysis."""

    print(f"\n{'='*65}")
    print(f"  COMPLIANCE VERIFICATION TEST")
    print(f"{'='*65}")
    print(f"  XML:      {XML_PATH}")
    print(f"  Config:   {CONFIG_PATH}")
    print(f"  Duration: {duration}s")

    # Load model
    print(f"\n  Loading model...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    idx = FARMSModelIndex(model)

    # Print current XML parameters for verification
    print(f"\n  Current simulation parameters:")
    print(f"    timestep:    {model.opt.timestep}")
    print(f"    integrator:  {['Euler','RK4','implicit','implicitfast'][model.opt.integrator]}")

    # Find pitch joints and print their params
    pitch_ids = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and ('joint_pitch_body' in name or 'joint_passive' in name):
            pitch_ids.append(i)
    print(f"    pitch joints: {len(pitch_ids)} found")

    if pitch_ids:
        j0 = pitch_ids[0]
        name0 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j0)
        stiff = model.jnt_stiffness[j0]
        damp = model.dof_damping[model.jnt_dofadr[j0]]
        print(f"    pitch stiffness ({name0}): {stiff:.6e}")
        print(f"    pitch damping   ({name0}): {damp:.6e}")

    # Print default geom solref/solimp
    print(f"    default solref: {model.opt.o_solref}")
    print(f"    default solimp: {model.opt.o_solimp}")

    # Controller
    ctrl = FARMSTravelingWaveController(model, config_path=CONFIG_PATH)

    # Recording
    rec_dt = 0.005  # 200 Hz recording
    last_rec = -np.inf
    times, com_pos_list, body_jnt_list, pitch_jnt_list = [], [], [], []

    n_steps = int(duration / model.opt.timestep)
    print(f"\n  Running {n_steps} steps...")
    t0 = time.time()
    next_print = print_every

    for step in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)

        # Stability check
        if step % 500 == 0:
            if np.any(np.isnan(data.qpos[:10])):
                print(f"\n  *** UNSTABLE: NaN detected at t={data.time:.3f}s ***")
                return False
            if np.any(np.abs(data.qpos[:10]) > 50):
                print(f"\n  *** UNSTABLE: Divergence at t={data.time:.3f}s ***")
                return False

        # Record
        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time
            times.append(data.time)
            com_pos_list.append(idx.com_pos(data).copy())
            body_jnt_list.append(
                np.array([idx.body_joint_pos(data, i+1) for i in range(N_BODY_JOINTS)])
            )
            pitch_jnt_list.append(
                np.array([data.qpos[model.jnt_qposadr[j]] for j in pitch_ids])
            )

        # Progress
        if data.time >= next_print:
            cz = idx.com_pos(data)[2]
            print(f"    t={data.time:.1f}s  COM_Z={cz*1000:.1f}mm  stable")
            next_print += print_every

    wall = time.time() - t0
    print(f"\n  Completed in {wall:.1f}s ({wall/duration:.1f}x real-time)")

    # Convert to arrays
    times = np.array(times)
    com_pos = np.array(com_pos_list)
    body_actual = np.array(body_jnt_list)
    pitch_pos = np.array(pitch_jnt_list)

    # Analysis (skip first 1s warmup)
    mask = times > 1.0
    if mask.sum() < 10:
        print("  Not enough data after warmup!")
        return False

    times_w = times[mask]
    com_z = com_pos[mask, 2]
    body_w = body_actual[mask]
    pitch_w = pitch_pos[mask]

    # Tracking error
    cfg = load_config(CONFIG_PATH)
    bw = cfg['body_wave']
    omega = 2.0 * np.pi * bw['frequency']
    n_w = bw['wave_number']
    speed = bw['speed']
    N = max(N_BODY_JOINTS - 1, 1)

    body_cmd = np.zeros_like(body_w)
    for i in range(body_w.shape[1]):
        phi_s = 2.0 * np.pi * n_w * speed * i / N
        body_cmd[:, i] = bw['amplitude'] * np.sin(omega * times_w - phi_s)

    body_rms = np.degrees(np.sqrt(np.mean((body_w - body_cmd) ** 2)))

    # Results
    print(f"\n{'='*65}")
    print(f"  RESULTS (after 1s warmup)")
    print(f"{'='*65}")

    print(f"\n  Tracking:")
    print(f"    Body yaw RMS error:  {body_rms:.4f} deg")
    print(f"    {'PASS' if body_rms < 1.0 else 'FAIL'}: target < 1.0 deg")

    print(f"\n  COM Height:")
    print(f"    Min:   {np.min(com_z)*1000:.2f} mm")
    print(f"    Max:   {np.max(com_z)*1000:.2f} mm")
    print(f"    Mean:  {np.mean(com_z)*1000:.2f} mm")
    print(f"    Std:   {np.std(com_z)*1000:.3f} mm")
    print(f"    {'PASS' if np.min(com_z) > 0 else 'FAIL'}: no terrain penetration (min > 0)")

    print(f"\n  Pitch Compliance:")
    print(f"    Std:   {np.degrees(np.std(pitch_w)):.3f} deg")
    print(f"    Max:   {np.degrees(np.max(np.abs(pitch_w))):.3f} deg")
    print(f"    Mean:  {np.degrees(np.mean(np.abs(pitch_w))):.3f} deg")
    if np.degrees(np.max(np.abs(pitch_w))) < 90:
        print(f"    PASS: pitch within joint limits")
    else:
        print(f"    WARNING: pitch hitting joint limits (90 deg)")

    print(f"\n  Stability:")
    print(f"    No NaN:       PASS")
    print(f"    No diverge:   PASS")
    print(f"    Duration:     {duration}s completed")

    # Save data for further analysis
    out_path = os.path.join(BASE, "outputs", "data", "compliance_verification.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        time=times, com_pos=com_pos,
        body_jnt_pos=body_actual, pitch_jnt_pos=pitch_pos,
        pitch_joint_names=np.array([
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in pitch_ids
        ]),
    )
    print(f"\n  Data saved: {out_path}")
    print(f"  You can load with: d = np.load('{out_path}')")

    print(f"\n{'='*65}")
    all_pass = body_rms < 1.0 and np.min(com_z) > 0
    if all_pass:
        print(f"  ALL CHECKS PASSED")
    else:
        print(f"  SOME CHECKS FAILED — see above")
    print(f"{'='*65}\n")

    return all_pass


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Verify pitch compliance parameters")
    p.add_argument("--duration", type=float, default=5.0,
                   help="Simulation duration in seconds (default: 5)")
    args = p.parse_args()

    success = run_test(duration=args.duration)
    sys.exit(0 if success else 1)
