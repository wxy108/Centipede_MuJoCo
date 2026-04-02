#!/usr/bin/env python3
"""
evaluate_compliance.py — Run simulation and extract detailed pitch compliance data
==================================================================================
After optimization, run this to evaluate the centipede on various terrain levels.
Records pitch joint angles, COM trajectory, tracking error, and terrain conformity.

Usage:
    python evaluate_compliance.py --terrain flat --duration 10
    python evaluate_compliance.py --terrain rough --duration 10
    python evaluate_compliance.py --terrain all --duration 10
"""

import argparse
import os
import sys
import re
import time
import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE       = os.path.join(SCRIPT_DIR, "..", "..", "..")
sys.path.insert(0, os.path.join(BASE, "controllers", "farms"))

import mujoco
from kinematics import FARMSModelIndex, N_BODY_JOINTS, N_LEGS, N_LEG_DOF
from controller import FARMSTravelingWaveController, load_config

XML_PATH     = os.path.join(BASE, "models", "farms", "centipede.xml")
CONFIG_PATH  = os.path.join(BASE, "configs", "farms_controller.yaml")
TERRAIN_DIR  = os.path.join(BASE, "terrain", "output")
OUTPUT_DIR   = os.path.join(BASE, "outputs", "data", "compliance_eval")
FLAT_TERRAIN = os.path.join(TERRAIN_DIR, "flat_terrain.png")


def patch_terrain_in_xml(xml_path, terrain_png, z_max=0.04):
    """Swap terrain heightfield in the XML."""
    with open(xml_path, 'r') as f:
        xml = f.read()
    xml = re.sub(r'(<hfield\s+name="terrain"\s+file=")[^"]*(")',
                 rf'\g<1>{terrain_png}\2', xml)
    # Patch z_max in hfield size
    def replacer(m):
        prefix = m.group(1)
        parts = m.group(2).split()
        parts[2] = f"{z_max:.6g}"
        return f'{prefix}{" ".join(parts)}{m.group(3)}'
    xml = re.sub(r'(<hfield\s+name="terrain"\s+file="[^"]*"\s+size=")([^"]*)(">)',
                 replacer, xml)
    with open(xml_path, 'w') as f:
        f.write(xml)


class ComplianceRecorder:
    """Records pitch joint angles, body yaw, leg angles, and COM."""

    def __init__(self, model, data, idx, dt_record=0.005):
        self.model     = model
        self.idx       = idx
        self.dt_record = dt_record
        self._last_t   = -np.inf

        # Find pitch joint IDs
        self.pitch_joint_ids = []
        self.pitch_joint_names = []
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and ('joint_pitch_body' in name or 'joint_passive' in name):
                self.pitch_joint_ids.append(i)
                self.pitch_joint_names.append(name)
        print(f"  Found {len(self.pitch_joint_ids)} pitch joints")

        self.times          = []
        self.com_pos        = []
        self.com_vel        = []
        self.body_jnt_pos   = []
        self.leg_jnt_pos    = []
        self.pitch_jnt_pos  = []
        self.pitch_jnt_vel  = []

    def maybe_record(self, data):
        if data.time - self._last_t < self.dt_record - 1e-10:
            return
        self._last_t = data.time
        self.times.append(data.time)
        self.com_pos.append(self.idx.com_pos(data).copy())
        self.com_vel.append(self.idx.com_vel(data).copy())

        # Body yaw joints
        bj = np.array([self.idx.body_joint_pos(data, i + 1)
                       for i in range(N_BODY_JOINTS)])
        self.body_jnt_pos.append(bj)

        # Leg joints
        lj = np.zeros((N_LEGS, 2, N_LEG_DOF))
        for n in range(N_LEGS):
            for si, side in enumerate(('L', 'R')):
                for dof in range(N_LEG_DOF):
                    lj[n, si, dof] = self.idx.leg_joint_pos(data, n, side, dof)
        self.leg_jnt_pos.append(lj)

        # Pitch joints (the critical compliance data)
        pitch_pos = np.array([data.qpos[self.model.jnt_qposadr[jid]]
                              for jid in self.pitch_joint_ids])
        pitch_vel = np.array([data.qvel[self.model.jnt_dofadr[jid]]
                              for jid in self.pitch_joint_ids])
        self.pitch_jnt_pos.append(pitch_pos)
        self.pitch_jnt_vel.append(pitch_vel)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            time            = np.array(self.times),
            com_pos         = np.array(self.com_pos),
            com_vel         = np.array(self.com_vel),
            body_jnt_pos    = np.array(self.body_jnt_pos),
            leg_jnt_pos     = np.array(self.leg_jnt_pos),
            pitch_jnt_pos   = np.array(self.pitch_jnt_pos),
            pitch_jnt_vel   = np.array(self.pitch_jnt_vel),
            pitch_joint_names = np.array(self.pitch_joint_names),
        )
        print(f"  Saved {len(self.times)} frames -> {path}")


def run_compliance_eval(terrain_label, terrain_png, z_max, duration):
    """Run one compliance evaluation on a given terrain."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {terrain_label}")
    print(f"  Terrain:    {terrain_png}")
    print(f"  Z max:      {z_max*1000:.1f} mm")
    print(f"{'='*60}")

    # Patch XML with terrain
    patch_terrain_in_xml(XML_PATH, terrain_png, z_max)

    # Load model
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # Controller
    ctrl = FARMSTravelingWaveController(model, config_path=CONFIG_PATH)

    # Recorder with high-freq pitch data
    recorder = ComplianceRecorder(model, data, ctrl.idx, dt_record=0.005)

    # Run
    n_steps = int(duration / model.opt.timestep)
    print(f"  Running {n_steps} steps ({duration:.1f}s)...")
    t0 = time.time()

    for step in range(n_steps):
        ctrl.step(model, data)
        mujoco.mj_step(model, data)
        recorder.maybe_record(data)

        # Check for instability
        if np.any(np.isnan(data.qpos)) or np.any(np.abs(data.qpos) > 100):
            print(f"  UNSTABLE at t={data.time:.3f}s (step {step})")
            break

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/duration:.1f}× real-time)")

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"{terrain_label}.npz")
    recorder.save(out_path)

    # Quick analysis
    times      = np.array(recorder.times)
    com_pos    = np.array(recorder.com_pos)
    pitch_pos  = np.array(recorder.pitch_jnt_pos)

    mask = times > 1.0  # skip warmup
    if mask.sum() > 10:
        print(f"\n  Analysis (after 1s warmup):")
        print(f"    COM Z:   mean={np.mean(com_pos[mask,2])*1000:.2f}mm  "
              f"std={np.std(com_pos[mask,2])*1000:.3f}mm  "
              f"min={np.min(com_pos[mask,2])*1000:.2f}mm  "
              f"max={np.max(com_pos[mask,2])*1000:.2f}mm")
        pitch_rms = np.degrees(np.sqrt(np.mean(pitch_pos[mask]**2)))
        pitch_max = np.degrees(np.max(np.abs(pitch_pos[mask])))
        pitch_std = np.degrees(np.std(pitch_pos[mask]))
        print(f"    Pitch:   RMS={pitch_rms:.3f}°  max={pitch_max:.3f}°  "
              f"std={pitch_std:.3f}°")

        # Tracking error
        body_actual = np.array(recorder.body_jnt_pos)[mask]
        cfg = load_config(CONFIG_PATH)
        bw = cfg['body_wave']
        omega = 2.0 * np.pi * bw['frequency']
        n_w = bw['wave_number']
        N = max(18, 1)
        times_w = times[mask]
        body_cmd = np.zeros_like(body_actual)
        for i in range(body_actual.shape[1]):
            phi_s = 2.0 * np.pi * n_w * bw['speed'] * i / N
            body_cmd[:, i] = bw['amplitude'] * np.sin(omega * times_w - phi_s)
        body_rms = np.degrees(np.sqrt(np.mean((body_actual - body_cmd)**2)))
        print(f"    Body tracking RMS: {body_rms:.3f}°")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compliance evaluation")
    parser.add_argument('--terrain', default='all',
                        choices=['flat', 'rough', 'all'],
                        help='Which terrain to test')
    parser.add_argument('--duration', type=float, default=10.0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define terrain levels
    terrain_configs = []

    if args.terrain in ('flat', 'all'):
        terrain_configs.append(('L0_flat', FLAT_TERRAIN, 0.001))

    if args.terrain in ('rough', 'all'):
        # Find available terrain dirs
        for d in sorted(os.listdir(TERRAIN_DIR)):
            dpath = os.path.join(TERRAIN_DIR, d)
            if os.path.isdir(dpath):
                png = os.path.join(dpath, "1.png")
                if os.path.exists(png):
                    # Parse z_max from dir name or use defaults
                    if 'low0.0040' in d:
                        terrain_configs.append((f'L1_mild_{d}', png, 0.010))
                    elif 'low0.0060' in d:
                        terrain_configs.append((f'L2_mod_{d}', png, 0.040))
                    elif 'low0.0100' in d:
                        terrain_configs.append((f'L3_rough_{d}', png, 0.070))
                    else:
                        terrain_configs.append((f'custom_{d}', png, 0.040))

    if not terrain_configs:
        terrain_configs = [('L0_flat', FLAT_TERRAIN, 0.001)]

    results = []
    for label, png, z_max in terrain_configs:
        out = run_compliance_eval(label, png, z_max, args.duration)
        results.append((label, out))

    print(f"\n\n{'='*60}")
    print(f"ALL EVALUATIONS COMPLETE")
    print(f"{'='*60}")
    for label, path in results:
        print(f"  {label:30s} -> {path}")
    print(f"\nResults in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
