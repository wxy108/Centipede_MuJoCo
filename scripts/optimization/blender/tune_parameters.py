#!/usr/bin/env python3
# python3 tune_parameters.py --n-calls 200 --n-initial 20
"""
tune_parameters.py — Bayesian Optimization for MuJoCo Centipede Parameters
===========================================================================
15-parameter per-DOF PD servo framework.

The key addition over the previous 12-parameter version is per-DOF
velocity gains (kv = kd), which implement the derivative term of the
PD servo alongside the existing position gains (kp).

PD servo per joint:
    F = kp × (q_target − q) + kv × (q_dot_target − q_dot)

kv is supplied to MuJoCo velocity actuators as the gain; the controller
sets ctrl = q_dot_target (analytically computed from the sinusoidal
trajectory). When CPG arrives, the same velocity actuators are driven
by ẋ from the oscillator state — no changes needed to this optimizer.

Parameter set (15):
  Gains — position:
    1.  body_kp          body joint servo stiffness
    2.  leg_dof0_kp      yaw (active)
    3.  leg_dof1_kp      upper pitch (active)
    4.  leg_dof2_kp      lower pitch (passive hold)

  Gains — velocity (kd):
    5.  body_kv          body joint derivative gain
    6.  leg_dof0_kv      yaw derivative
    7.  leg_dof1_kv      upper pitch derivative
    8.  leg_dof2_kv      lower pitch derivative  ← PRIMARY contact absorber

  Force ranges:
    9.  body_fr          body actuator force limit
   10.  leg_fr           leg actuator force limit (all DOFs)

  Damping (passive mechanical, NOT the servo derivative):
   11.  body_damping     body joint viscous damping
   12.  leg_dof01_damping active DOF damping (small — servo handles tracking)

  Regularization:
   13.  body_armature    body solver regularization
   14.  leg_armature     leg solver regularization

  Contact:
   15.  solref_timeconst ground contact time constant

Note: leg_dof2_damping is NO LONGER in the search space.
      It is fixed at a small constant in generate_mjcf.py because
      leg_dof2_kv is now the primary absorber of contact impulses
      on the passive joint. Keeping both would create a redundant
      parameter that confuses the optimizer.

Usage:
    python tune_parameters.py --n-calls 200 --n-initial 20
    python tune_parameters.py --prior-weight 0.5 --duration 5
    python tune_parameters.py --body-weight 0.7 --leg-weight 0.3
"""

import argparse
import json
import os
import sys
import re
import shutil
import subprocess
import pickle
import numpy as np
import yaml

try:
    from skopt import gp_minimize
    from skopt.space import Real
except ImportError:
    print("Installing scikit-optimize...")
    subprocess.run([sys.executable, "-m", "pip", "install", "scikit-optimize"])
    from skopt import gp_minimize
    from skopt.space import Real

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════

BASE            = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GENERATE_SCRIPT = os.path.join(BASE, "archive", "blender_pipeline", "centipede_archive", "generate_mjcf.py")
XML_OUTPUT_GEN  = os.path.join(BASE, "archive", "blender_pipeline", "centipede_archive", "centipede.xml")
XML_OUTPUT      = os.path.join(BASE, "models", "blender", "centipede.xml")
CONTROL_DIR     = os.path.join(BASE, "controllers", "blender")
CONFIG_PATH     = os.path.join(BASE, "configs", "blender_controller.yaml")
TUNE_DIR        = os.path.join(BASE, "outputs", "data", "tuning")
OPT_DIR         = os.path.join(BASE, "outputs", "optimization", "blender")
GENERATE_BACKUP = GENERATE_SCRIPT + ".backup"


class CentipedeOptimizer:
    """Bayesian optimization for centipede PD servo parameters (15 params)."""

    # ── Physics-informed initial values ────────────────────────────────
    # kv initial = 2 × 0.7 × √(kp × I_estimated)
    # I_body ≈ 1e-7 kg·m²,  I_leg ≈ 1e-8 kg·m²
    INITIAL_VALUES = {
        # Position gains
        'body_kp':            0.009399,
        'leg_dof0_kp':        0.000813,
        'leg_dof1_kp':        0.000813,
        'leg_dof2_kp':        0.001880,
        # Velocity gains (kd) — set at ζ≈0.7 critical damping
        'body_kv':            4.30e-05,
        'leg_dof0_kv':        4.00e-06,
        'leg_dof1_kv':        4.00e-06,
        'leg_dof2_kv':        8.00e-06,   # 2× higher — passive contact absorber
        # Force ranges
        'body_fr':            0.6756,
        'leg_fr':             0.00705,
        # Passive mechanical damping (small — servo kv handles active damping)
        'body_damping':       2.82e-07,
        'leg_dof01_damping':  2.13e-06,
        # Regularization
        'body_armature':      3.48e-07,
        'leg_armature':       2.82e-08,
        # Contact
        'solref_timeconst':   0.02,
    }

    FAILURE_PENALTY = 10.0

    def __init__(self, prior_weight=0.3, body_weight=0.5, leg_weight=0.5,
                 duration=5.0):
        self.prior_weight = prior_weight
        self.body_weight  = body_weight
        self.leg_weight   = leg_weight
        self.duration     = duration

        os.makedirs(OPT_DIR,  exist_ok=True)
        os.makedirs(TUNE_DIR, exist_ok=True)

        if not os.path.exists(GENERATE_BACKUP):
            shutil.copy2(GENERATE_SCRIPT, GENERATE_BACKUP)
            print(f"Backed up generate_mjcf.py → {GENERATE_BACKUP}")

        self.define_parameter_space()

        self.iteration_count     = 0
        self.best_combined_error = float('inf')
        self.best_body_error     = float('inf')
        self.best_leg_error      = float('inf')
        self.best_params         = None
        self.history             = []

        print(f"\nInitialized optimizer (15 parameters):")
        print(f"  Prior weight: {prior_weight}")
        print(f"  Body weight:  {body_weight}  Leg weight: {leg_weight}")
        print(f"  Sim duration: {duration}s")
        print(f"\n  Initial values:")
        for name, val in self.INITIAL_VALUES.items():
            print(f"    {name:22s}: {val:.3e}")

    # ══════════════════════════════════════════════════════
    # Search space
    # ══════════════════════════════════════════════════════

    def define_parameter_space(self):
        """Log-uniform bounds, clamped per parameter type."""
        scale = 2.0 * (1.0 - self.prior_weight) + 0.5 * self.prior_weight

        CLAMPS = {
            'kp':       (1e-5,  1.0),
            'kv':       (1e-8,  1.0),
            'fr':       (1e-4,  10.0),
            'damping':  (1e-9,  0.1),
            'armature': (1e-10, 1e-2),
            'solref':   (0.002, 0.2),
        }

        def ptype(name):
            for k in ['kv', 'kp', 'fr', 'damping', 'armature', 'solref']:
                if k in name:
                    return k
            return 'kp'

        self.space       = []
        self.param_names = []
        for name, initial in self.INITIAL_VALUES.items():
            pt = ptype(name)
            lo = max(CLAMPS[pt][0], initial / (10 ** scale))
            hi = min(CLAMPS[pt][1], initial * (10 ** scale))
            self.space.append(Real(lo, hi, name=name, prior='log-uniform'))
            self.param_names.append(name)
            print(f"  {name:22s}: [{lo:.2e}, {hi:.2e}]  (init={initial:.2e})")

    # ══════════════════════════════════════════════════════
    # Apply parameters → regenerate XML
    # ══════════════════════════════════════════════════════

    def apply_params(self, params_dict):
        """Patch constants in generate_mjcf.py and regenerate centipede.xml."""
        with open(GENERATE_SCRIPT, 'r', encoding='utf-8') as f:
            content = f.read()

        p = params_dict

        def replace_const(text, var, value):
            return re.sub(
                rf'^({re.escape(var)}\s*=\s*).*$',
                rf'\g<1>{value}',
                text, count=1, flags=re.MULTILINE)

        # ── Scalar constants ─────────────────────────────────────────
        content = replace_const(content, 'POSITION_KP',        f"{p['body_kp']:.10e}")
        content = replace_const(content, 'FORCE_RANGE',        f"{p['body_fr']:.10e}")
        content = replace_const(content, 'BODY_JOINT_DAMPING', f"{p['body_damping']:.4e}")
        content = replace_const(content, 'LEG_DOF01_DAMPING',  f"{p['leg_dof01_damping']:.4e}")
        content = replace_const(content, 'VELOCITY_KV_BODY',   f"{p['body_kv']:.10e}")

        # ── Per-DOF kp ratios (relative to POSITION_KP) ──────────────
        bkp = p['body_kp'] if p['body_kp'] > 0 else 1e-9
        content = re.sub(
            r'(LEG_DOF0_KP\s*=\s*POSITION_KP\s*\*\s*)[\d.e+\-]+',
            rf'\g<1>{p["leg_dof0_kp"]/bkp:.8f}', content)
        content = re.sub(
            r'(LEG_DOF1_KP\s*=\s*POSITION_KP\s*\*\s*)[\d.e+\-]+',
            rf'\g<1>{p["leg_dof1_kp"]/bkp:.8f}', content)
        content = re.sub(
            r'(LEG_DOF2_KP\s*=\s*POSITION_KP\s*\*\s*)[\d.e+\-]+',
            rf'\g<1>{p["leg_dof2_kp"]/bkp:.8f}', content)

        # ── Per-DOF kv ratios (relative to VELOCITY_KV_BODY) ─────────
        bkv = p['body_kv'] if p['body_kv'] > 0 else 1e-9
        content = re.sub(
            r'(VELOCITY_KV_LEG_DOF0\s*=\s*VELOCITY_KV_BODY\s*\*\s*)[\d.e+\-]+',
            rf'\g<1>{p["leg_dof0_kv"]/bkv:.8f}', content)
        content = re.sub(
            r'(VELOCITY_KV_LEG_DOF1\s*=\s*VELOCITY_KV_BODY\s*\*\s*)[\d.e+\-]+',
            rf'\g<1>{p["leg_dof1_kv"]/bkv:.8f}', content)
        content = re.sub(
            r'(VELOCITY_KV_LEG_DOF2\s*=\s*VELOCITY_KV_BODY\s*\*\s*)[\d.e+\-]+',
            rf'\g<1>{p["leg_dof2_kv"]/bkv:.8f}', content)

        # ── Leg force range ratio ─────────────────────────────────────
        bfr          = p['body_fr'] if p['body_fr'] > 0 else 1e-9
        leg_fr_ratio = p['leg_fr'] / bfr
        content = re.sub(
            r'(FORCE_RANGE\*)([\d.e+\-]+)(,\s*FORCE_RANGE\*)([\d.e+\-]+)(,\s*FORCE_RANGE\*)([\d.e+\-]+)',
            rf'\g<1>{leg_fr_ratio:.8f}\g<3>{leg_fr_ratio:.8f}\g<5>{leg_fr_ratio:.8f}',
            content)

        # ── Armature ─────────────────────────────────────────────────
        armature_matches = list(re.finditer(r'armature="([^"]*)"', content))
        if len(armature_matches) >= 2:
            ba_str = f"{p['body_armature']:.4e}"
            la_str = f"{p['leg_armature']:.4e}"
            for m in reversed(armature_matches[1:]):
                content = content[:m.start(1)] + la_str + content[m.end(1):]
            m0 = armature_matches[0]
            content = content[:m0.start(1)] + ba_str + content[m0.end(1):]

        # ── SOLREF time constant ──────────────────────────────────────
        content = re.sub(
            r'(SOLREF\s*=\s*\[)\s*[\d.e+\-]+(\s*,\s*1\.0\s*\])',
            rf'\g<1>{p["solref_timeconst"]:.4f}\2',
            content)

        with open(GENERATE_SCRIPT, 'w', encoding='utf-8') as f:
            f.write(content)

        result = subprocess.run(
            [sys.executable, GENERATE_SCRIPT],
            capture_output=True, text=True,
            cwd=os.path.dirname(GENERATE_SCRIPT))
        if result.returncode != 0:
            print(f"    XML gen error: {result.stderr[-400:]}")
            return False

        if os.path.exists(XML_OUTPUT_GEN):
            shutil.copy2(XML_OUTPUT_GEN, XML_OUTPUT)
            return True
        return False

    # ══════════════════════════════════════════════════════
    # Run simulation
    # ══════════════════════════════════════════════════════

    def run_simulation(self, tag):
        output_dir  = os.path.join(TUNE_DIR, tag)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "results.npz")

        try:
            result = subprocess.run(
                [sys.executable, os.path.join(CONTROL_DIR, "run.py"),
                 "--headless", "--duration", str(self.duration),
                 "--output", output_path, "--config", CONFIG_PATH,
                 "--tag", tag],
                capture_output=True, text=True, cwd=CONTROL_DIR, timeout=120)
        except subprocess.TimeoutExpired:
            return None

        if "unstable" in result.stdout.lower() or "nan" in result.stdout.lower():
            return None
        if not os.path.exists(output_path):
            return None
        return output_path

    # ══════════════════════════════════════════════════════
    # Tracking error
    # ══════════════════════════════════════════════════════

    def compute_tracking_error(self, data_path):
        """
        Returns (body_rms, leg_rms, leg_rms_per_dof[3]) in radians.

        Skips the first WARMUP_TIME seconds to avoid penalising the
        startup transient — the controller ramps from rest so the first
        ~0.5s always has elevated error regardless of gains.
        """
        WARMUP_TIME = 0.5   # seconds to skip at start

        try:
            d           = np.load(data_path)
            times       = d['time']
            body_actual = d['body_joint_pos']   # (T, n_body_joints)
            leg_actual  = d['leg_joint_pos']    # (T, 19, 2, 3)
        except Exception as e:
            print(f"    NPZ load error: {e}")
            return None, None, None

        # Skip warmup
        mask = times >= WARMUP_TIME
        if mask.sum() < 10:
            return None, None, None
        times       = times[mask]
        body_actual = body_actual[mask]
        leg_actual  = leg_actual[mask]

        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        bw    = config['body_wave']
        lw    = config['leg_wave']
        omega = 2.0 * np.pi * bw['frequency']
        woff  = 2.0 * np.pi * bw['wave_number']
        speed = bw['speed']
        n_wj  = config.get('n_body_joints_wave', 19)
        N     = max(n_wj - 1, 1)

        # ── Body command ─────────────────────────────────────────────
        body_cmd = np.zeros_like(body_actual)
        for i in range(min(n_wj, body_actual.shape[1])):
            phi = woff * (i / N) * speed
            body_cmd[:, i] = bw['amplitude'] * np.sin(omega * times - phi)
        body_rms = np.sqrt(np.mean((body_actual - body_cmd) ** 2))

        # ── Leg command ──────────────────────────────────────────────
        def _pad(src, n):
            a = np.zeros(n)
            src = np.asarray(src, dtype=float)
            a[:min(len(src), n)] = src[:n]
            return a

        amps   = _pad(lw['amplitudes'],                3)
        poff   = _pad(lw['phase_offsets'],             3)
        dcoff  = _pad(lw.get('dc_offsets', [0.0]*3),  3)
        active = set(lw.get('active_dofs', [0, 1]))

        leg_cmd = np.zeros_like(leg_actual)
        for n in range(19):
            phi = woff * (n / N) * speed
            for si in range(2):
                sign = 1.0 if si == 0 else -1.0
                for dof in range(3):
                    if dof not in active:
                        leg_cmd[:, n, si, dof] = dcoff[dof]
                    else:
                        phase = omega * times - phi + poff[dof]
                        leg_cmd[:, n, si, dof] = sign * amps[dof] * np.sin(phase) + dcoff[dof]

        leg_rms = np.sqrt(np.mean((leg_actual - leg_cmd) ** 2))
        leg_rms_dof = np.array([
            np.sqrt(np.mean((leg_actual[:,:,:,d] - leg_cmd[:,:,:,d]) ** 2))
            for d in range(3)])

        return body_rms, leg_rms, leg_rms_dof

    # ══════════════════════════════════════════════════════
    # Objective
    # ══════════════════════════════════════════════════════

    def objective(self, params_list):
        self.iteration_count += 1
        tag         = f"opt_{self.iteration_count:04d}"
        params_dict = dict(zip(self.param_names, params_list))
        p           = params_dict

        # ── Hard constraints ─────────────────────────────────────────
        # Force range must give headroom above kp (avoids torque saturation)
        if p['body_fr'] < 5 * p['body_kp']:
            self._log(tag, p, 'constrained', None, None, None)
            print(f"  [{self.iteration_count:3d}] SKIP: body_fr < 5×body_kp")
            return self.FAILURE_PENALTY

        max_leg_kp = max(p['leg_dof0_kp'], p['leg_dof1_kp'], p['leg_dof2_kp'])
        if p['leg_fr'] < 5 * max_leg_kp:
            self._log(tag, p, 'constrained', None, None, None)
            print(f"  [{self.iteration_count:3d}] SKIP: leg_fr < 5×max_leg_kp")
            return self.FAILURE_PENALTY

        # kv must be positive (guaranteed by log-uniform, but sanity check)
        for kname in ('body_kv', 'leg_dof0_kv', 'leg_dof1_kv', 'leg_dof2_kv'):
            if p[kname] <= 0:
                self._log(tag, p, 'constrained', None, None, None)
                return self.FAILURE_PENALTY

        # ── Apply + regenerate ───────────────────────────────────────
        if not self.apply_params(params_dict):
            self._log(tag, p, 'gen_fail', None, None, None)
            print(f"  [{self.iteration_count:3d}] XML generation failed")
            return self.FAILURE_PENALTY

        # ── Run ──────────────────────────────────────────────────────
        data_path = self.run_simulation(tag)
        if data_path is None:
            self._log(tag, p, 'unstable', None, None, None)
            print(f"  [{self.iteration_count:3d}] "
                  f"kp_b={p['body_kp']:.3e}  kv_b={p['body_kv']:.2e}  "
                  f"kv_d2={p['leg_dof2_kv']:.2e}  → UNSTABLE")
            return self.FAILURE_PENALTY

        # ── Error ────────────────────────────────────────────────────
        body_rms, leg_rms, leg_rms_dof = self.compute_tracking_error(data_path)
        if body_rms is None:
            self._log(tag, p, 'error', None, None, None)
            return self.FAILURE_PENALTY

        cost    = self.body_weight * body_rms + self.leg_weight * leg_rms
        is_best = cost < self.best_combined_error

        if is_best:
            self.best_combined_error = cost
            self.best_body_error     = body_rms
            self.best_leg_error      = leg_rms
            self.best_params         = params_list.copy()

        self._log(tag, p, 'ok', body_rms, leg_rms, leg_rms_dof)

        print(f"  [{self.iteration_count:3d}] "
              f"kp_b={p['body_kp']:.3e}  kv_b={p['body_kv']:.2e}  "
              f"kv_d2={p['leg_dof2_kv']:.2e}  sol={p['solref_timeconst']:.3f}  "
              f"→ body={np.degrees(body_rms):.2f}°  "
              f"leg={np.degrees(leg_rms):.2f}°  "
              f"[d0={np.degrees(leg_rms_dof[0]):.2f}° "
              f"d1={np.degrees(leg_rms_dof[1]):.2f}° "
              f"d2={np.degrees(leg_rms_dof[2]):.2f}°]  "
              f"cost={cost:.5f}{'  ★' if is_best else ''}")

        if self.iteration_count % 10 == 0:
            self.save_progress()

        return cost

    def _log(self, tag, p, status, body_rms, leg_rms, leg_rms_dof):
        self.history.append({
            'iter':        self.iteration_count,
            'tag':         tag,
            'status':      status,
            'body_rms':    float(body_rms)         if body_rms    is not None else None,
            'leg_rms':     float(leg_rms)          if leg_rms     is not None else None,
            'leg_rms_dof0': float(leg_rms_dof[0]) if leg_rms_dof is not None else None,
            'leg_rms_dof1': float(leg_rms_dof[1]) if leg_rms_dof is not None else None,
            'leg_rms_dof2': float(leg_rms_dof[2]) if leg_rms_dof is not None else None,
            'cost': (float(self.body_weight * body_rms + self.leg_weight * leg_rms)
                     if body_rms is not None else self.FAILURE_PENALTY),
            **{k: float(v) for k, v in p.items()}
        })

    # ══════════════════════════════════════════════════════
    # Optimization runner
    # ══════════════════════════════════════════════════════

    def optimize(self, n_calls=200, n_initial_points=20):
        print("\n" + "=" * 60)
        print("STARTING BAYESIAN OPTIMIZATION  (15 parameters, PD per-DOF)")
        print("=" * 60)
        print(f"  Total evaluations:  {n_calls}")
        print(f"  Initial random:     {n_initial_points}")
        print(f"  Acquisition:        EI")
        print(f"  Failure penalty:    {self.FAILURE_PENALTY}")
        print(f"  Warmup skip:        0.5s")
        print()

        x0 = None
        if self.prior_weight > 0.2:
            x0 = [[self.INITIAL_VALUES[n] for n in self.param_names]]
            print("  Using initial configuration as first evaluation")

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
            print("\n\nInterrupted — saving progress...")
            self.save_progress()
            self.plot_progress()
            if self.best_params is not None:
                self.apply_params(dict(zip(self.param_names, self.best_params)))
            raise

        except Exception as e:
            print(f"\nOptimization error: {e}")
            self.save_progress()
            if self.best_params is not None:
                self.apply_params(dict(zip(self.param_names, self.best_params)))
            raise

    # ══════════════════════════════════════════════════════
    # Results & plotting
    # ══════════════════════════════════════════════════════

    def save_progress(self):
        path = os.path.join(OPT_DIR, 'optimization_progress.json')
        with open(path, 'w') as f:
            json.dump({
                'iteration':     self.iteration_count,
                'best_cost':     self.best_combined_error if self.best_combined_error < float('inf') else None,
                'best_body_rms': self.best_body_error if self.best_body_error < float('inf') else None,
                'best_leg_rms':  self.best_leg_error  if self.best_leg_error  < float('inf') else None,
                'best_params':   dict(zip(self.param_names, self.best_params)) if self.best_params else None,
                'history':       self.history,
            }, f, indent=2, default=str)

    def save_final_results(self, result):
        with open(os.path.join(OPT_DIR, 'optimization_result.pkl'), 'wb') as f:
            pickle.dump(result, f)

        self.save_progress()
        self.plot_progress()

        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)

        if result.fun < self.FAILURE_PENALTY:
            best_dict = dict(zip(self.param_names, result.x))
            print(f"\n  Best combined error: {result.fun:.6f}")
            print(f"  Body RMS: {np.degrees(self.best_body_error):.3f}°")
            print(f"  Leg  RMS: {np.degrees(self.best_leg_error):.3f}°")
            print(f"\n  Optimal parameters:")
            for name in self.param_names:
                init    = self.INITIAL_VALUES[name]
                optimal = best_dict[name]
                print(f"    {name:22s}: {optimal:.6e}  (×{optimal/init:.2f})")
            print(f"\n  Applying optimal parameters...")
            self.apply_params(best_dict)
        else:
            print("  WARNING: No stable solution found!")
            if self.best_params:
                self.apply_params(dict(zip(self.param_names, self.best_params)))
            else:
                shutil.copy2(GENERATE_BACKUP, GENERATE_SCRIPT)
                print("  Restored backup.")

        ok   = sum(1 for h in self.history if h['status'] == 'ok')
        fail = len(self.history) - ok
        print(f"\n  Evaluations: {len(self.history)} total  ({ok} stable, {fail} failed)")
        print(f"  Results: {OPT_DIR}/")

    def plot_progress(self):
        ok = [h for h in self.history if h['status'] == 'ok']
        if len(ok) < 3:
            return

        iters     = [h['iter']                      for h in ok]
        body_errs = [np.degrees(h['body_rms'])       for h in ok]
        leg_errs  = [np.degrees(h['leg_rms'])        for h in ok]
        costs     = [h['cost']                       for h in ok]
        d2_errs   = [np.degrees(h['leg_rms_dof2'])   for h in ok if h.get('leg_rms_dof2')]
        d2_iters  = [h['iter']                       for h in ok if h.get('leg_rms_dof2')]
        kv_body   = [h.get('body_kv')                for h in ok if h.get('body_kv')]
        kv_d2     = [h.get('leg_dof2_kv')            for h in ok if h.get('leg_dof2_kv')]
        kv_iters  = [h['iter']                       for h in ok if h.get('body_kv')]

        run_best, bsf = [], float('inf')
        for c in costs:
            bsf = min(bsf, c)
            run_best.append(bsf)

        fig = plt.figure(figsize=(22, 12))
        gs  = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)

        # Convergence
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(iters, costs, 'o', ms=3, alpha=0.4, color='gray', label='All')
        ax.plot(iters, run_best, '-', color='red', lw=2, label='Best')
        ax.set(xlabel='Iteration', ylabel='Cost', title='Convergence')
        ax.legend(); ax.grid(True, alpha=0.3)

        # Body vs Leg scatter
        ax = fig.add_subplot(gs[0, 1])
        sc = ax.scatter(body_errs, leg_errs, c=iters, cmap='viridis', s=20, alpha=0.7)
        ax.set(xlabel='Body RMS (°)', ylabel='Leg RMS (°)', title='Body vs Leg Error')
        plt.colorbar(sc, ax=ax, label='Iteration')
        ax.grid(True, alpha=0.3)

        # DOF2 error vs kv_dof2 — key PD diagnostic
        ax = fig.add_subplot(gs[0, 2])
        if d2_errs and kv_d2:
            # Align lengths
            n = min(len(d2_errs), len(kv_d2))
            ax.scatter(kv_d2[:n], d2_errs[:n], c=d2_iters[:n], cmap='plasma', s=20, alpha=0.7)
        ax.set(xlabel='leg_dof2_kv', ylabel='DOF2 RMS (°)',
               title='DOF2 Oscillation vs kv')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # All tracking errors
        ax = fig.add_subplot(gs[0, 3])
        ax.plot(iters, body_errs, 'o', ms=3, alpha=0.5, color='blue',   label='Body')
        ax.plot(iters, leg_errs,  'o', ms=3, alpha=0.5, color='red',    label='Leg')
        if d2_errs:
            ax.plot(d2_iters, d2_errs, 'o', ms=3, alpha=0.5, color='orange', label='DOF2')
        ax.set(xlabel='Iteration', ylabel='RMS (°)', title='Tracking Errors')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Key parameter evolutions
        for pi, (pname, title) in enumerate([
            ('body_kp',      'body_kp'),
            ('body_kv',      'body_kv'),
            ('leg_dof2_kv',  'leg_dof2_kv  (contact absorber)'),
            ('solref_timeconst', 'solref'),
        ]):
            ax    = fig.add_subplot(gs[1, pi])
            pvals = [h.get(pname) for h in ok if h.get(pname) is not None]
            pits  = [h['iter']    for h in ok if h.get(pname) is not None]
            clrs  = ['green' if h['cost'] == min(costs) else
                     ('blue' if h['cost'] < np.median(costs) else 'gray')
                     for h in ok if h.get(pname) is not None]
            ax.scatter(pits, pvals, c=clrs, s=15, alpha=0.6)
            if pname in self.INITIAL_VALUES:
                ax.axhline(self.INITIAL_VALUES[pname], color='orange',
                           ls='--', alpha=0.6, label='Initial')
            ax.set(xlabel='Iteration', ylabel=pname, title=title)
            ax.set_yscale('log')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.suptitle('Bayesian Optimization — 15-param PD per-DOF centipede',
                     fontsize=13, fontweight='bold')
        plt.savefig(os.path.join(OPT_DIR, 'optimization_progress.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Stability map
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        all_its = range(1, self.iteration_count + 1)
        smap = []
        for i in all_its:
            m = [h for h in self.history if h['iter'] == i]
            s = m[0]['status'] if m else ''
            smap.append('green' if s == 'ok' else 'red' if s == 'unstable' else 'orange')
        ax1.bar(list(all_its), [1]*len(smap), color=smap, width=1.0)
        ax1.set(xlabel='Iteration', title='Stability Map'); ax1.set_yticks([])

        ax2.hist(body_errs, bins=20, alpha=0.6, color='blue', label='Body')
        ax2.hist(leg_errs,  bins=20, alpha=0.6, color='red',  label='Leg')
        ax2.set(xlabel='RMS (°)', ylabel='Count', title='Error Distribution')
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OPT_DIR, 'error_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plots: {OPT_DIR}/")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='15-parameter Bayesian Optimization — PD per-DOF centipede')
    parser.add_argument('--prior-weight', type=float, default=0.3)
    parser.add_argument('--body-weight',  type=float, default=0.5)
    parser.add_argument('--leg-weight',   type=float, default=0.5)
    parser.add_argument('--n-calls',      type=int,   default=200)
    parser.add_argument('--n-initial',    type=int,   default=20)
    parser.add_argument('--duration',     type=float, default=5.0)
    args = parser.parse_args()

    total_w = args.body_weight + args.leg_weight
    optimizer = CentipedeOptimizer(
        prior_weight=args.prior_weight,
        body_weight=args.body_weight / total_w,
        leg_weight=args.leg_weight   / total_w,
        duration=args.duration,
    )

    try:
        result = optimizer.optimize(
            n_calls=args.n_calls,
            n_initial_points=args.n_initial)

        if result.fun < optimizer.FAILURE_PENALTY:
            print(f"\n  Run with optimal parameters:")
            print(f"    python {os.path.join(CONTROL_DIR, 'run.py')} --video test.mp4 --duration 10")
        print(f"\n  To restore backup:")
        print(f"    cp {GENERATE_BACKUP} {GENERATE_SCRIPT}")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved.")
    except Exception as e:
        print(f"\n\nFailed: {e}")
        raise


if __name__ == "__main__":
    main()
