#!/usr/bin/env python3
"""
comprehensive_tune.py — Multi-Objective Parameter Optimization for FARMS Centipede
====================================================================================
Tunes ALL relevant simulation parameters to achieve:
  1. Simulation stability (no NaN, no divergence)
  2. Accurate joint tracking (commanded vs actual)
  3. No terrain penetration (COM stays above ground)
  4. Pitch compliance (body conforms to terrain surface)

Extended parameter space (24 parameters):
  - 10 actuator gains (kp, kv for body + 4 leg DOFs)
  - 6 joint passive damping groups
  - 2 pitch spring parameters (stiffness + damping)
  - 2 solver contact parameters (solref timeconst + damping ratio)
  - 2 solimp parameters (dmin + dmax)
  - 1 geom friction (sliding)
  - 1 timestep

Two-phase optimization:
  Phase 1: Flat terrain → stability + tracking accuracy
  Phase 2: Rough terrain → compliance + no penetration (using Phase 1 best as seed)

Usage:
    python comprehensive_tune.py --phase 1 --n-calls 150 --n-initial 25
    python comprehensive_tune.py --phase 2 --n-calls 100 --n-initial 15
    python comprehensive_tune.py --phase both --n-calls 120
"""

import argparse
import json
import os
import sys
import re
import shutil
import subprocess
import pickle
import time
import numpy as np
import yaml

try:
    from skopt import gp_minimize
    from skopt.space import Real
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "scikit-optimize", "--break-system-packages"])
    from skopt import gp_minimize
    from skopt.space import Real

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =====================================================================
# PATHS
# =====================================================================

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE        = os.path.join(SCRIPT_DIR, "..", "..", "..")
XML_PATH    = os.path.join(BASE, "models", "farms", "centipede.xml")
XML_BACKUP  = XML_PATH + ".comprehensive_backup"
CONTROL_DIR = os.path.join(BASE, "controllers", "farms")
CONFIG_PATH = os.path.join(BASE, "configs", "farms_controller.yaml")
TUNE_DIR    = os.path.join(BASE, "outputs", "data", "comprehensive_tuning")
OPT_DIR     = os.path.join(BASE, "outputs", "optimization", "comprehensive")
TERRAIN_DIR = os.path.join(BASE, "terrain", "output")

# Terrain files
FLAT_TERRAIN = os.path.join(TERRAIN_DIR, "flat_terrain.png")
ROUGH_TERRAIN_DIRS = sorted([
    os.path.join(TERRAIN_DIR, d)
    for d in os.listdir(TERRAIN_DIR)
    if os.path.isdir(os.path.join(TERRAIN_DIR, d))
])


# =====================================================================
# XML patching helpers
# =====================================================================

def _patch_actuator_attr(xml, name_pattern, attr, value):
    """Set attr on all <position> actuators whose name matches regex."""
    def replacer(m):
        line = m.group(0)
        line = re.sub(rf'{attr}="[^"]*"', f'{attr}="{value:.8g}"', line)
        return line
    pattern = rf'<position\s[^>]*name="{name_pattern}"[^>]*/>'
    return re.sub(pattern, replacer, xml)


def _patch_joint_attr(xml, name_pattern, attr, value):
    """Set attr on all <joint> elements whose name matches regex."""
    def replacer(m):
        line = m.group(0)
        if f'{attr}="' in line:
            line = re.sub(rf'{attr}="[^"]*"', f'{attr}="{value:.8g}"', line)
        else:
            line = line.replace('/>', f' {attr}="{value:.8g}"/>')
        return line
    pattern = rf'<joint\s[^>]*name="{name_pattern}"[^>]*/>'
    return re.sub(pattern, replacer, xml)


def _patch_option_timestep(xml, timestep):
    """Patch <option timestep=...>."""
    return re.sub(r'(<option\s[^>]*timestep=")[^"]*(")', rf'\g<1>{timestep:.6g}\2', xml)


def _patch_default_solref(xml, timeconst, dampratio):
    """Patch solref in <default><geom .../>."""
    val = f"{timeconst:.6g} {dampratio:.6g}"
    return re.sub(r'(solref=")[^"]*(")', rf'\g<1>{val}\2', xml)


def _patch_default_solimp(xml, dmin, dmax, width=0.001):
    """Patch solimp in <default><geom .../>."""
    val = f"{dmin:.6g} {dmax:.6g} {width:.6g}"
    return re.sub(r'(solimp=")[^"]*(")', rf'\g<1>{val}\2', xml)


def _patch_default_friction(xml, sliding, torsional=0.02, rolling=0.001):
    """Patch friction in <default><geom .../>."""
    val = f"{sliding:.6g} {torsional:.6g} {rolling:.6g}"
    # Patch default geom friction
    return re.sub(r'(<default>\s*\n\s*<joint[^/]*/>\s*\n\s*<geom[^>]*friction=")[^"]*(")',
                  rf'\g<1>{val}\2', xml, flags=re.DOTALL)


def _patch_geom_friction_all(xml, sliding, torsional=0.02, rolling=0.001):
    """Patch friction on default geom and floor/terrain geoms."""
    # Default geom
    def patch_default(m):
        line = m.group(0)
        val = f"{sliding:.6g} {torsional:.6g} {rolling:.6g}"
        return re.sub(r'friction="[^"]*"', f'friction="{val}"', line)

    # Patch in <default> section
    xml = re.sub(r'<geom\s+condim="3"[^/]*/>', patch_default, xml, count=1)

    # Patch floor and terrain_geom friction
    floor_friction = f"{sliding:.6g} 0.005 0.0001"
    xml = re.sub(r'(<geom\s+name="floor"[^>]*friction=")[^"]*(")', rf'\g<1>{floor_friction}\2', xml)
    xml = re.sub(r'(<geom\s+type="hfield"[^>]*friction=")[^"]*(")', rf'\g<1>{floor_friction}\2', xml)

    return xml


def _patch_hfield_path(xml, terrain_png_path):
    """Swap the hfield file= to a different terrain PNG."""
    return re.sub(r'(<hfield\s+name="terrain"\s+file=")[^"]*(")',
                  rf'\g<1>{terrain_png_path}\2', xml)


def _patch_hfield_zmax(xml, z_max):
    """Patch the hfield size Z component (3rd number in size="x y z r")."""
    def replacer(m):
        parts = m.group(1).split()
        parts[2] = f"{z_max:.6g}"
        return f'size="{" ".join(parts)}"'
    return re.sub(r'size="([^"]*)"', replacer, xml, count=1)


# =====================================================================
# Generate flat terrain PNG if missing
# =====================================================================

def ensure_flat_terrain():
    if os.path.exists(FLAT_TERRAIN):
        return
    os.makedirs(os.path.dirname(FLAT_TERRAIN), exist_ok=True)
    try:
        from PIL import Image
        img = np.full((1024, 1024), 128, dtype=np.uint8)
        Image.fromarray(img).save(FLAT_TERRAIN)
    except ImportError:
        # Fallback: use raw bytes
        import struct
        import zlib
        width = height = 1024
        raw = b'\x00' + b'\x80' * width
        raw_data = raw * height
        def make_png(w, h, data):
            def chunk(ctype, body):
                return struct.pack('>I', len(body)) + ctype + body + struct.pack('>I', zlib.crc32(ctype + body) & 0xffffffff)
            header = b'\x89PNG\r\n\x1a\n'
            ihdr = struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)
            return header + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(data)) + chunk(b'IEND', b'')
        with open(FLAT_TERRAIN, 'wb') as f:
            f.write(make_png(width, height, raw_data))
    print(f"Created flat terrain: {FLAT_TERRAIN}")


# =====================================================================
# Comprehensive Optimizer
# =====================================================================

class ComprehensiveCentipedeOptimizer:
    """
    Multi-objective optimizer for FARMS centipede simulation parameters.

    Phase 1 (flat terrain): Optimize for stability + tracking accuracy
    Phase 2 (rough terrain): Optimize pitch compliance while maintaining stability
    """

    # Current best known values (from previous optimization + manual tuning)
    INITIAL_VALUES = {
        # Actuator kp (from apply_optimal_gains.py)
        'body_kp':           64.945,
        'leg_dof0_kp':       1.2695,
        'leg_dof1_kp':       0.1475,
        'leg_dof2_kp':       1.2695,
        'leg_dof3_kp':       1.2695,
        # Actuator kv
        'body_kv':           0.0552,
        'leg_dof0_kv':       0.00560,
        'leg_dof1_kv':       0.00114,
        'leg_dof2_kv':       0.000436,
        'leg_dof3_kv':       0.00910,
        # Joint passive damping
        'body_yaw_damping':  4.075e-06,
        'leg_dof01_damping': 3.0e-08,
        'leg_dof2_damping':  8.531e-05,
        'leg_dof3_damping':  6.117e-03,
        # Passive pitch springs
        'pitch_stiffness':   1.0e-03,
        'pitch_damping':     1.0e-04,
        # Solver / contact parameters
        'solref_timeconst':  0.005,     # contact time constant
        'solref_dampratio':  1.0,       # contact damping ratio
        'solimp_dmin':       0.9,       # min impedance
        'solimp_dmax':       0.95,      # max impedance
        'geom_friction':     0.8,       # sliding friction coefficient
        # Simulation timestep
        'timestep':          0.0005,    # seconds
    }

    FAILURE_PENALTY = 10.0

    def __init__(self, phase=1, prior_weight=0.3,
                 body_weight=0.5, leg_weight=0.5,
                 duration=5.0, terrain_path=None):
        self.phase        = phase
        self.prior_weight = prior_weight
        self.body_weight  = body_weight
        self.leg_weight   = leg_weight
        self.duration     = duration
        self.terrain_path = terrain_path

        os.makedirs(OPT_DIR,  exist_ok=True)
        os.makedirs(TUNE_DIR, exist_ok=True)

        if not os.path.exists(XML_BACKUP):
            shutil.copy2(XML_PATH, XML_BACKUP)
            print(f"Backed up: {XML_BACKUP}")

        self.define_parameter_space()

        self.iteration_count     = 0
        self.best_cost           = float('inf')
        self.best_params         = None
        self.best_metrics        = None
        self.history             = []

        print(f"\n{'='*65}")
        print(f"COMPREHENSIVE CENTIPEDE OPTIMIZER — Phase {phase}")
        print(f"{'='*65}")
        print(f"  Parameters:    {len(self.param_names)}")
        print(f"  Prior weight:  {prior_weight}")
        print(f"  Body weight:   {body_weight}  Leg weight: {leg_weight}")
        print(f"  Duration:      {duration}s")
        print(f"  Terrain:       {'FLAT' if phase == 1 else terrain_path or 'ROUGH'}")

    # =================================================================
    # Search space
    # =================================================================

    def define_parameter_space(self):
        scale = 2.0 * (1.0 - self.prior_weight) + 0.5 * self.prior_weight

        # Per-parameter clamping ranges
        CLAMPS = {
            'kp':            (1e-3,    200.0),
            'kv':            (1e-6,    1.0),
            'damping':       (1e-9,    0.1),
            'stiffness':     (1e-6,    0.1),
            'solref_tc':     (0.001,   0.05),    # contact time constant
            'solref_dr':     (0.5,     2.0),     # damping ratio (linear scale)
            'solimp_d':      (0.8,     0.999),   # impedance (linear scale)
            'friction':      (0.3,     2.0),     # friction coeff
            'timestep':      (0.0002,  0.002),   # seconds
        }

        def ptype(name):
            if name == 'timestep':          return 'timestep'
            if name == 'geom_friction':     return 'friction'
            if name == 'solref_timeconst':  return 'solref_tc'
            if name == 'solref_dampratio':  return 'solref_dr'
            if 'solimp' in name:            return 'solimp_d'
            if 'stiffness' in name:         return 'stiffness'
            for k in ['kv', 'kp', 'damping']:
                if k in name:
                    return k
            return 'kp'

        self.space       = []
        self.param_names = []

        for name, initial in self.INITIAL_VALUES.items():
            pt = ptype(name)
            lo_clamp, hi_clamp = CLAMPS[pt]

            # Use log-uniform for most parameters, linear for bounded ones
            if pt in ('solref_dr', 'solimp_d', 'friction', 'timestep'):
                lo = max(lo_clamp, initial * 0.5)
                hi = min(hi_clamp, initial * 2.0)
                prior = 'uniform'
            else:
                lo = max(lo_clamp, initial / (10 ** scale))
                hi = min(hi_clamp, initial * (10 ** scale))
                prior = 'log-uniform'

            self.space.append(Real(lo, hi, name=name, prior=prior))
            self.param_names.append(name)

    # =================================================================
    # Apply parameters -> patch XML
    # =================================================================

    def apply_params(self, params_dict):
        p = params_dict

        with open(XML_BACKUP, 'r', encoding='utf-8') as f:
            xml = f.read()

        # -- TIMESTEP --
        xml = _patch_option_timestep(xml, p['timestep'])

        # -- SOLVER / CONTACT --
        xml = _patch_default_solref(xml, p['solref_timeconst'], p['solref_dampratio'])
        xml = _patch_default_solimp(xml, p['solimp_dmin'], p['solimp_dmax'])
        xml = _patch_geom_friction_all(xml, p['geom_friction'])

        # -- ACTUATOR KP/KV --
        xml = _patch_actuator_attr(xml, r'act_joint_body_\d+', 'kp', p['body_kp'])
        xml = _patch_actuator_attr(xml, r'act_joint_body_\d+', 'kv', p['body_kv'])

        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_0', 'kp', p['leg_dof0_kp'])
        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_0', 'kv', p['leg_dof0_kv'])

        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_1', 'kp', p['leg_dof1_kp'])
        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_1', 'kv', p['leg_dof1_kv'])

        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_2', 'kp', p['leg_dof2_kp'])
        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_2', 'kv', p['leg_dof2_kv'])

        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_3', 'kp', p['leg_dof3_kp'])
        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_3', 'kv', p['leg_dof3_kv'])

        xml = _patch_actuator_attr(xml, r'act_joint_foot_\d+_[01]', 'kp', p['leg_dof3_kp'])
        xml = _patch_actuator_attr(xml, r'act_joint_foot_\d+_[01]', 'kv', p['leg_dof3_kv'])

        # -- JOINT PASSIVE DAMPING --
        xml = _patch_joint_attr(xml, r'joint_body_\d+',          'damping', p['body_yaw_damping'])
        xml = _patch_joint_attr(xml, r'joint_leg_\d+_[LR]_0',   'damping', p['leg_dof01_damping'])
        xml = _patch_joint_attr(xml, r'joint_leg_\d+_[LR]_1',   'damping', p['leg_dof01_damping'])
        xml = _patch_joint_attr(xml, r'joint_leg_\d+_[LR]_2',   'damping', p['leg_dof2_damping'])
        xml = _patch_joint_attr(xml, r'joint_leg_\d+_[LR]_3',   'damping', p['leg_dof3_damping'])
        xml = _patch_joint_attr(xml, r'joint_foot_\d+_[01]',    'damping', p['leg_dof3_damping'])

        # -- PASSIVE PITCH SPRINGS --
        xml = _patch_joint_attr(xml, r'joint_pitch_body_\d+', 'stiffness', p['pitch_stiffness'])
        xml = _patch_joint_attr(xml, r'joint_pitch_body_\d+', 'damping',   p['pitch_damping'])
        xml = _patch_joint_attr(xml, r'joint_pitch_body_\d+',    'stiffness', p['pitch_stiffness'])
        xml = _patch_joint_attr(xml, r'joint_pitch_body_\d+',    'damping',   p['pitch_damping'])

        # -- TERRAIN --
        if self.phase == 1:
            # Flat terrain for Phase 1
            xml = _patch_hfield_path(xml, FLAT_TERRAIN)
            xml = _patch_hfield_zmax(xml, 0.001)  # near-zero height
        elif self.terrain_path:
            xml = _patch_hfield_path(xml, self.terrain_path)

        with open(XML_PATH, 'w', encoding='utf-8') as f:
            f.write(xml)
        return True

    # =================================================================
    # Run simulation
    # =================================================================

    def run_simulation(self, tag, duration=None):
        if duration is None:
            duration = self.duration
        output_dir  = os.path.join(TUNE_DIR, f"phase{self.phase}", tag)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "results.npz")

        run_script = os.path.join(CONTROL_DIR, "run.py")

        try:
            result = subprocess.run(
                [sys.executable, run_script,
                 "--headless", "--duration", str(duration),
                 "--output", output_path, "--config", CONFIG_PATH],
                capture_output=True, text=True, cwd=CONTROL_DIR,
                timeout=max(300, int(duration * 50)))
        except subprocess.TimeoutExpired:
            return None, "timeout"

        stdout = result.stdout.lower()
        stderr = result.stderr.lower() if result.stderr else ""

        if "nan" in stdout or "nan" in stderr:
            return None, "nan"
        if result.returncode != 0:
            return None, f"exit_{result.returncode}"
        if not os.path.exists(output_path):
            return None, "no_output"

        return output_path, "ok"

    # =================================================================
    # Metrics computation
    # =================================================================

    def compute_metrics(self, data_path):
        """
        Compute comprehensive metrics from simulation data.

        Returns dict with:
          - body_rms: body yaw tracking error (rad)
          - leg_rms: leg tracking error (rad)
          - leg_rms_dof: per-DOF leg tracking error
          - com_z_min: minimum COM height (penetration check)
          - com_z_mean: mean COM height
          - com_z_std: COM height std (stability indicator)
          - pitch_deflection_std: std of pitch joint angles (compliance indicator)
          - pitch_deflection_max: max pitch deflection
          - stable: whether simulation appears stable
        """
        WARMUP_TIME = 0.5

        try:
            d           = np.load(data_path)
            times       = d['time']
            body_actual = d['body_jnt_pos']
            leg_actual  = d['leg_jnt_pos']
            com_pos     = d['com_pos']
        except Exception as e:
            print(f"    NPZ load error: {e}")
            return None

        # Check for NaN/Inf
        if (np.any(np.isnan(body_actual)) or np.any(np.isnan(leg_actual))
                or np.any(np.isnan(com_pos))):
            return None

        # Check for divergence: COM should stay in reasonable range
        if np.any(np.abs(com_pos) > 10.0):
            return None

        mask = times >= WARMUP_TIME
        if mask.sum() < 10:
            return None

        times_w       = times[mask]
        body_actual_w = body_actual[mask]
        leg_actual_w  = leg_actual[mask]
        com_pos_w     = com_pos[mask]

        # ── Tracking error ──
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        bw    = config['body_wave']
        lw    = config['leg_wave']
        omega = 2.0 * np.pi * bw['frequency']
        n_w   = bw['wave_number']
        speed = bw['speed']
        N     = max(18, 1)

        # Body command
        body_cmd = np.zeros_like(body_actual_w)
        for i in range(body_actual_w.shape[1]):
            phi_s = 2.0 * np.pi * n_w * speed * i / N
            body_cmd[:, i] = bw['amplitude'] * np.sin(omega * times_w - phi_s)
        body_rms = float(np.sqrt(np.mean((body_actual_w - body_cmd) ** 2)))

        # Leg command
        n_dof = leg_actual_w.shape[-1]
        def _pad(src, n):
            a = np.zeros(n)
            src = np.asarray(src, dtype=float)
            a[:min(len(src), n)] = src[:n]
            return a

        amps   = _pad(lw['amplitudes'],    n_dof)
        poff   = _pad(lw['phase_offsets'],  n_dof)
        dcoff  = _pad(lw.get('dc_offsets', [0.0]*n_dof), n_dof)
        active = set(lw.get('active_dofs', [0, 1]))

        n_legs  = leg_actual_w.shape[1]
        leg_cmd = np.zeros_like(leg_actual_w)
        for n in range(n_legs):
            phi_s = 2.0 * np.pi * n_w * speed * n / N
            for si in range(2):
                sign = 1.0 if si == 0 else -1.0
                for dof in range(n_dof):
                    if dof not in active:
                        leg_cmd[:, n, si, dof] = dcoff[dof]
                    else:
                        phase = omega * times_w - phi_s + poff[dof]
                        leg_cmd[:, n, si, dof] = (sign * amps[dof] *
                                                  np.sin(phase) + dcoff[dof])

        leg_rms = float(np.sqrt(np.mean((leg_actual_w - leg_cmd) ** 2)))
        leg_rms_dof = [
            float(np.sqrt(np.mean((leg_actual_w[:,:,:,d] - leg_cmd[:,:,:,d]) ** 2)))
            for d in range(n_dof)]

        # ── COM height metrics ──
        com_z     = com_pos_w[:, 2]
        com_z_min = float(np.min(com_z))
        com_z_mean = float(np.mean(com_z))
        com_z_std  = float(np.std(com_z))

        # ── Pitch compliance metrics ──
        # We don't have direct pitch joint data in the NPZ, but we can infer
        # from the body yaw actual vs commanded: the residual that ISN'T tracking
        # error could indicate pitch effects. Better: look at COM Z variation.
        #
        # For a more direct measure, we need to record pitch joint angles.
        # For now, use COM Z variation as a proxy for pitch compliance.
        pitch_deflection_std = com_z_std
        pitch_deflection_max = float(np.max(com_z) - np.min(com_z))

        return {
            'body_rms':             body_rms,
            'leg_rms':              leg_rms,
            'leg_rms_dof':          leg_rms_dof,
            'com_z_min':            com_z_min,
            'com_z_mean':           com_z_mean,
            'com_z_std':            com_z_std,
            'pitch_deflection_std': pitch_deflection_std,
            'pitch_deflection_max': pitch_deflection_max,
            'stable':               True,
        }

    # =================================================================
    # Objective function
    # =================================================================

    def objective(self, params_list):
        self.iteration_count += 1
        tag         = f"p{self.phase}_{self.iteration_count:04d}"
        params_dict = dict(zip(self.param_names, params_list))
        p           = params_dict

        if not self.apply_params(params_dict):
            self._log(tag, p, 'gen_fail', None)
            return self.FAILURE_PENALTY

        data_path, status = self.run_simulation(tag)
        if data_path is None:
            self._log(tag, p, status, None)
            print(f"  [{self.iteration_count:3d}] "
                  f"ts={p['timestep']:.4f} solref=({p['solref_timeconst']:.4f},{p['solref_dampratio']:.2f}) "
                  f"pitch_k={p['pitch_stiffness']:.2e} pitch_d={p['pitch_damping']:.2e} "
                  f"-> {status.upper()}")
            return self.FAILURE_PENALTY

        metrics = self.compute_metrics(data_path)
        if metrics is None:
            self._log(tag, p, 'metrics_fail', None)
            print(f"  [{self.iteration_count:3d}] -> METRICS_FAIL")
            return self.FAILURE_PENALTY

        # ── Cost function ──
        cost = self._compute_cost(metrics, params_dict)
        is_best = cost < self.best_cost

        if is_best:
            self.best_cost    = cost
            self.best_params  = params_list.copy()
            self.best_metrics = metrics.copy()

        self._log(tag, p, 'ok', metrics)

        print(f"  [{self.iteration_count:3d}] "
              f"ts={p['timestep']:.4f} "
              f"pitch_k={p['pitch_stiffness']:.2e} "
              f"body={np.degrees(metrics['body_rms']):.2f}° "
              f"leg={np.degrees(metrics['leg_rms']):.2f}° "
              f"Zmin={metrics['com_z_min']*1000:.1f}mm "
              f"cost={cost:.5f}{'  ★' if is_best else ''}")

        if self.iteration_count % 10 == 0:
            self.save_progress()

        return cost

    def _compute_cost(self, metrics, params):
        """
        Multi-objective cost function.

        Phase 1 (flat): Maximize tracking accuracy + stability
        Phase 2 (rough): Add compliance + no-penetration objectives
        """
        body_rms = metrics['body_rms']
        leg_rms  = metrics['leg_rms']

        # Base tracking cost
        tracking_cost = self.body_weight * body_rms + self.leg_weight * leg_rms

        if self.phase == 1:
            # Phase 1: Pure tracking + stability
            # Penalize if COM drops too low (possible penetration even on flat)
            penetration_penalty = 0.0
            if metrics['com_z_min'] < 0.005:   # less than 5mm above ground
                penetration_penalty = 2.0 * (0.005 - metrics['com_z_min'])

            # Penalize excessive COM height variation (instability)
            stability_penalty = 0.0
            if metrics['com_z_std'] > 0.005:
                stability_penalty = 1.0 * (metrics['com_z_std'] - 0.005)

            cost = tracking_cost + penetration_penalty + stability_penalty

        else:
            # Phase 2: Tracking + compliance + no-penetration
            # Penalize terrain penetration more heavily
            penetration_penalty = 0.0
            if metrics['com_z_min'] < 0.002:
                penetration_penalty = 5.0 * (0.002 - metrics['com_z_min'])

            # REWARD pitch compliance (more COM Z variation = more conforming)
            # But not too much (that would be instability)
            compliance_reward = 0.0
            z_range = metrics['pitch_deflection_max']
            if 0.002 < z_range < 0.02:
                # Sweet spot: some compliance but not crazy
                compliance_reward = -0.5 * z_range  # negative = reward
            elif z_range >= 0.02:
                # Too much = instability
                compliance_reward = 1.0 * (z_range - 0.02)

            cost = tracking_cost + penetration_penalty + compliance_reward

        return float(cost)

    def _log(self, tag, p, status, metrics):
        entry = {
            'iter':     self.iteration_count,
            'tag':      tag,
            'phase':    self.phase,
            'status':   status,
            **{k: float(v) for k, v in p.items()},
        }
        if metrics is not None:
            entry.update({
                'body_rms':     metrics['body_rms'],
                'leg_rms':      metrics['leg_rms'],
                'com_z_min':    metrics['com_z_min'],
                'com_z_mean':   metrics['com_z_mean'],
                'com_z_std':    metrics['com_z_std'],
                'pitch_max':    metrics['pitch_deflection_max'],
                'cost':         self._compute_cost(metrics, p),
            })
            for d, v in enumerate(metrics['leg_rms_dof']):
                entry[f'leg_rms_dof{d}'] = v
        else:
            entry['cost'] = self.FAILURE_PENALTY
        self.history.append(entry)

    # =================================================================
    # Optimization runner
    # =================================================================

    def optimize(self, n_calls=150, n_initial_points=25):
        print(f"\n  Total evaluations:  {n_calls}")
        print(f"  Initial random:     {n_initial_points}")
        print(f"  Acquisition:        EI")
        print()

        x0 = None
        if self.prior_weight > 0.2:
            x0 = [[self.INITIAL_VALUES[n] for n in self.param_names]]
            print("  Seeding with initial values as first evaluation")

        try:
            result = gp_minimize(
                func=self.objective,
                dimensions=self.space,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                x0=x0,
                acq_func='EI',
                acq_optimizer='lbfgs',
                random_state=42,
                verbose=False,
                n_jobs=1,
            )
            self.save_final_results(result)
            return result

        except KeyboardInterrupt:
            print("\n\nInterrupted -- saving progress...")
            self.save_progress()
            if self.best_params is not None:
                self.apply_params(dict(zip(self.param_names, self.best_params)))
                print("  Applied best-so-far parameters.")
            raise

        except Exception as e:
            print(f"\nOptimization error: {e}")
            import traceback; traceback.print_exc()
            self.save_progress()
            if self.best_params is not None:
                self.apply_params(dict(zip(self.param_names, self.best_params)))
            raise

    # =================================================================
    # Results
    # =================================================================

    def save_progress(self):
        phase_dir = os.path.join(OPT_DIR, f"phase{self.phase}")
        os.makedirs(phase_dir, exist_ok=True)
        path = os.path.join(phase_dir, 'progress.json')
        with open(path, 'w') as f:
            json.dump({
                'phase':         self.phase,
                'iteration':     self.iteration_count,
                'best_cost':     self.best_cost if self.best_cost < float('inf') else None,
                'best_metrics':  self.best_metrics,
                'best_params':   (dict(zip(self.param_names, self.best_params))
                                  if self.best_params else None),
                'history':       self.history,
            }, f, indent=2, default=str)
        print(f"  Progress saved -> {path}")

    def save_final_results(self, result):
        phase_dir = os.path.join(OPT_DIR, f"phase{self.phase}")
        os.makedirs(phase_dir, exist_ok=True)

        with open(os.path.join(phase_dir, 'result.pkl'), 'wb') as f:
            pickle.dump(result, f)

        self.save_progress()
        self.plot_progress()

        print(f"\n{'='*65}")
        print(f"PHASE {self.phase} OPTIMIZATION COMPLETE")
        print(f"{'='*65}")

        if result.fun < self.FAILURE_PENALTY:
            best_dict = dict(zip(self.param_names, result.x))
            m = self.best_metrics
            print(f"\n  Best cost: {result.fun:.6f}")
            if m:
                print(f"  Body RMS:  {np.degrees(m['body_rms']):.3f}°")
                print(f"  Leg  RMS:  {np.degrees(m['leg_rms']):.3f}°")
                print(f"  COM Z min: {m['com_z_min']*1000:.2f} mm")
                print(f"  COM Z std: {m['com_z_std']*1000:.3f} mm")
                print(f"  Pitch max: {m['pitch_deflection_max']*1000:.2f} mm")
            print(f"\n  Optimal parameters:")
            for name in self.param_names:
                init    = self.INITIAL_VALUES[name]
                optimal = best_dict[name]
                ratio   = optimal / init if init != 0 else float('inf')
                print(f"    {name:22s}: {optimal:.6e}  (×{ratio:.2f})")

            print(f"\n  Applying optimal parameters to XML...")
            self.apply_params(best_dict)

            # Save optimal params as a standalone JSON
            with open(os.path.join(phase_dir, 'optimal_params.json'), 'w') as f:
                json.dump(best_dict, f, indent=2)
        else:
            print("  WARNING: No stable solution found!")
            if self.best_params:
                self.apply_params(dict(zip(self.param_names, self.best_params)))
            else:
                shutil.copy2(XML_BACKUP, XML_PATH)

        ok   = sum(1 for h in self.history if h['status'] == 'ok')
        fail = len(self.history) - ok
        print(f"\n  Evaluations: {len(self.history)} total ({ok} stable, {fail} failed)")

    def plot_progress(self):
        phase_dir = os.path.join(OPT_DIR, f"phase{self.phase}")
        os.makedirs(phase_dir, exist_ok=True)

        ok = [h for h in self.history if h['status'] == 'ok']
        if len(ok) < 3:
            return

        iters     = [h['iter'] for h in ok]
        costs     = [h['cost'] for h in ok]
        body_errs = [np.degrees(h['body_rms']) for h in ok]
        leg_errs  = [np.degrees(h['leg_rms'])  for h in ok]
        com_z_min = [h['com_z_min']*1000       for h in ok]
        pitch_max = [h['pitch_max']*1000        for h in ok]

        run_best, bsf = [], float('inf')
        for c in costs:
            bsf = min(bsf, c)
            run_best.append(bsf)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Convergence
        ax = axes[0, 0]
        ax.plot(iters, costs, 'o', ms=3, alpha=0.3, color='gray')
        ax.plot(iters, run_best, '-', color='red', lw=2)
        ax.set(xlabel='Iter', ylabel='Cost', title='Convergence')
        ax.grid(True, alpha=0.3)

        # Body vs Leg error
        ax = axes[0, 1]
        sc = ax.scatter(body_errs, leg_errs, c=iters, cmap='viridis', s=15, alpha=0.7)
        ax.set(xlabel='Body RMS (°)', ylabel='Leg RMS (°)', title='Body vs Leg Error')
        plt.colorbar(sc, ax=ax, label='Iter')
        ax.grid(True, alpha=0.3)

        # COM Z minimum
        ax = axes[0, 2]
        ax.plot(iters, com_z_min, 'o', ms=3, alpha=0.5, color='blue')
        ax.axhline(5, color='red', ls='--', alpha=0.5, label='5mm threshold')
        ax.set(xlabel='Iter', ylabel='COM Z min (mm)', title='Penetration Check')
        ax.legend(); ax.grid(True, alpha=0.3)

        # Pitch compliance
        ax = axes[1, 0]
        ax.plot(iters, pitch_max, 'o', ms=3, alpha=0.5, color='green')
        ax.set(xlabel='Iter', ylabel='Pitch max range (mm)', title='Pitch Compliance')
        ax.grid(True, alpha=0.3)

        # Stability map
        ax = axes[1, 1]
        all_statuses = []
        for i in range(1, self.iteration_count + 1):
            m = [h for h in self.history if h['iter'] == i]
            all_statuses.append('green' if m and m[0]['status'] == 'ok' else 'red')
        ax.bar(range(1, len(all_statuses)+1), [1]*len(all_statuses),
               color=all_statuses, width=1.0)
        ax.set(xlabel='Iter', title='Stability Map (green=ok, red=fail)')
        ax.set_yticks([])

        # Key params
        ax = axes[1, 2]
        for pn, color in [('pitch_stiffness', 'blue'), ('pitch_damping', 'red'),
                          ('timestep', 'green')]:
            vals = [h.get(pn) for h in ok if h.get(pn) is not None]
            its  = [h['iter'] for h in ok if h.get(pn) is not None]
            if vals:
                ax.plot(its, vals, 'o', ms=3, alpha=0.5, color=color, label=pn)
        ax.set(xlabel='Iter', ylabel='Value', title='Key Parameters')
        ax.set_yscale('log')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        plt.suptitle(f'Phase {self.phase} — Comprehensive Optimization',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(phase_dir, 'progress.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved -> {phase_dir}/progress.png")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Parameter Optimization — FARMS Centipede')
    parser.add_argument('--phase', type=str, default='1',
                        choices=['1', '2', 'both'],
                        help='Phase 1=flat, 2=rough, both=sequential')
    parser.add_argument('--prior-weight', type=float, default=0.4)
    parser.add_argument('--body-weight',  type=float, default=0.5)
    parser.add_argument('--leg-weight',   type=float, default=0.5)
    parser.add_argument('--n-calls',      type=int,   default=150)
    parser.add_argument('--n-initial',    type=int,   default=25)
    parser.add_argument('--duration',     type=float, default=5.0)
    parser.add_argument('--terrain',      type=str,   default=None,
                        help='Path to rough terrain PNG for Phase 2')
    args = parser.parse_args()

    total_w = args.body_weight + args.leg_weight
    bw = args.body_weight / total_w
    lw = args.leg_weight / total_w

    ensure_flat_terrain()

    phases = [1, 2] if args.phase == 'both' else [int(args.phase)]

    for phase in phases:
        terrain = None
        if phase == 2:
            if args.terrain:
                terrain = args.terrain
            elif ROUGH_TERRAIN_DIRS:
                # Use medium roughness terrain
                mid_idx = len(ROUGH_TERRAIN_DIRS) // 2
                terrain = os.path.join(ROUGH_TERRAIN_DIRS[mid_idx], "1.png")
                print(f"  Auto-selected terrain: {terrain}")

        optimizer = ComprehensiveCentipedeOptimizer(
            phase=phase,
            prior_weight=args.prior_weight,
            body_weight=bw,
            leg_weight=lw,
            duration=args.duration,
            terrain_path=terrain,
        )

        # If Phase 2 and Phase 1 results exist, seed from them
        if phase == 2:
            p1_params_path = os.path.join(OPT_DIR, "phase1", "optimal_params.json")
            if os.path.exists(p1_params_path):
                with open(p1_params_path) as f:
                    p1_best = json.load(f)
                # Update initial values with Phase 1 best
                for k, v in p1_best.items():
                    if k in optimizer.INITIAL_VALUES:
                        optimizer.INITIAL_VALUES[k] = v
                optimizer.define_parameter_space()
                print(f"  Seeded Phase 2 from Phase 1 optimal: {p1_params_path}")

        try:
            result = optimizer.optimize(
                n_calls=args.n_calls,
                n_initial_points=args.n_initial)
        except KeyboardInterrupt:
            print(f"\nPhase {phase} interrupted. Progress saved.")
            break
        except Exception as e:
            print(f"\nPhase {phase} failed: {e}")
            import traceback; traceback.print_exc()
            break


if __name__ == "__main__":
    main()
