#!/usr/bin/env python3
# python3 farms_tune_parameters.py --n-calls 200 --n-initial 20
"""
farms_tune_parameters.py — Bayesian Optimization for FARMS Centipede MuJoCo Model
==================================================================================
Patches centipede.xml actuator and joint parameters directly (no regeneration step).

The FARMS model uses MuJoCo <position> actuators with built-in PD:
    τ = kp × (q_target − q) − kv × q̇

16-parameter search space (from XML analysis):

  Actuator gains — position (kp):
    1.  body_kp              body yaw servo stiffness           (current: 1.8304)
    2.  leg_dof0_kp          hip yaw                            (current: 0.03578)
    3.  leg_dof1_kp          hip pitch                          (current: 0.03578)
    4.  leg_dof2_kp          tibia pitch                        (current: 0.03578)
    5.  leg_dof3_kp          tarsus + foot                      (current: 0.03578)

  Actuator gains — velocity (kv, PD damping):
    6.  body_kv              body yaw derivative                (current: 0.0915)
    7.  leg_dof0_kv          hip yaw derivative                 (current: 0.001789)
    8.  leg_dof1_kv          hip pitch derivative               (current: 0.001789)
    9.  leg_dof2_kv          tibia derivative                   (current: 0.001789)
   10.  leg_dof3_kv          tarsus + foot derivative           (current: 0.001789)

  Joint passive damping (6 distinct groups in XML):
   11.  body_yaw_damping     joint_body_* viscous               (current: 4.075e-6)
   12.  leg_dof01_damping    DOF0 + DOF1 passive damping        (current: ~1e-6)
   13.  leg_dof2_damping     tibia passive damping (3400x DOF0) (current: 3.027e-3)
   14.  leg_dof3_damping     tarsus + foot passive damping      (current: 1.724e-4)

  Passive pitch springs (joint_pitch_body_*):
   15.  pitch_stiffness      inter-segment pitch spring         (current: 1e-2)
   16.  pitch_damping        inter-segment pitch viscous        (current: 1e-6)

  Fixed (not optimized):
    - armature = 0.0001 (numerical regularization, not physical)
    - force limits disabled (forcelimited="false")

Usage:
    python farms_tune_parameters.py --n-calls 200 --n-initial 20
    python farms_tune_parameters.py --prior-weight 0.5 --duration 5
    python farms_tune_parameters.py --body-weight 0.7 --leg-weight 0.3
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
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "scikit-optimize", "--break-system-packages"])
    from skopt import gp_minimize
    from skopt.space import Real

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =====================================================================
# PATHS  — adjust if your layout differs
# =====================================================================

BASE        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
XML_PATH    = os.path.join(BASE, "models", "farms", "centipede.xml")
XML_BACKUP  = XML_PATH + ".backup"
CONTROL_DIR = os.path.join(BASE, "controllers", "farms")
CONFIG_PATH = os.path.join(BASE, "configs", "farms_controller.yaml")
TUNE_DIR    = os.path.join(BASE, "outputs", "data", "tuning_farms")
OPT_DIR     = os.path.join(BASE, "outputs", "optimization", "farms")


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


# =====================================================================
# Optimizer
# =====================================================================

class FARMSCentipedeOptimizer:
    """Bayesian optimization for FARMS centipede — 16 parameters."""

    INITIAL_VALUES = {
        # Actuator kp
        'body_kp':           1.8304,
        'leg_dof0_kp':       0.03578,
        'leg_dof1_kp':       0.03578,
        'leg_dof2_kp':       0.03578,
        'leg_dof3_kp':       0.03578,
        # Actuator kv (after 5% patch)
        'body_kv':           0.0915,
        'leg_dof0_kv':       0.001789,
        'leg_dof1_kv':       0.001789,
        'leg_dof2_kv':       0.001789,
        'leg_dof3_kv':       0.001789,
        # Joint passive damping
        'body_yaw_damping':  4.075e-06,
        'leg_dof01_damping': 1.0e-06,
        'leg_dof2_damping':  3.027e-03,
        'leg_dof3_damping':  1.724e-04,
        # Passive pitch springs (optimized for terrain compliance)
        'pitch_stiffness':   1.0e-04,
        'pitch_damping':     1.0e-03,
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

        if not os.path.exists(XML_BACKUP):
            shutil.copy2(XML_PATH, XML_BACKUP)
            print(f"Backed up centipede.xml -> {XML_BACKUP}")

        self.define_parameter_space()

        self.iteration_count     = 0
        self.best_combined_error = float('inf')
        self.best_body_error     = float('inf')
        self.best_leg_error      = float('inf')
        self.best_params         = None
        self.history             = []

        print(f"\nInitialized FARMS optimizer (16 parameters):")
        print(f"  Prior weight: {prior_weight}")
        print(f"  Body weight:  {body_weight}  Leg weight: {leg_weight}")
        print(f"  Sim duration: {duration}s")
        print(f"\n  Initial values:")
        for name, val in self.INITIAL_VALUES.items():
            print(f"    {name:22s}: {val:.4e}")

    # =================================================================
    # Search space
    # =================================================================

    def define_parameter_space(self):
        scale = 2.0 * (1.0 - self.prior_weight) + 0.5 * self.prior_weight

        CLAMPS = {
            'kp':        (1e-4,   100.0),
            'kv':        (1e-6,   10.0),
            'damping':   (1e-9,   0.1),
            'stiffness': (1e-5,   1.0),
        }

        def ptype(name):
            if 'stiffness' in name:
                return 'stiffness'
            for k in ['kv', 'kp', 'damping']:
                if k in name:
                    return k
            return 'kp'

        self.space       = []
        self.param_names = []
        print(f"\n  Search space (scale={scale:.2f}):")
        for name, initial in self.INITIAL_VALUES.items():
            pt = ptype(name)
            lo = max(CLAMPS[pt][0], initial / (10 ** scale))
            hi = min(CLAMPS[pt][1], initial * (10 ** scale))
            self.space.append(Real(lo, hi, name=name, prior='log-uniform'))
            self.param_names.append(name)
            print(f"    {name:22s}: [{lo:.2e}, {hi:.2e}]  (init={initial:.2e})")

    # =================================================================
    # Apply parameters -> patch XML
    # =================================================================

    def apply_params(self, params_dict):
        p = params_dict

        with open(XML_BACKUP, 'r', encoding='utf-8') as f:
            xml = f.read()

        # Disable force limits
        xml = re.sub(r'forcelimited="true"', 'forcelimited="false"', xml)

        # -- ACTUATOR KP/KV --

        # Body
        xml = _patch_actuator_attr(xml, r'act_joint_body_\d+', 'kp', p['body_kp'])
        xml = _patch_actuator_attr(xml, r'act_joint_body_\d+', 'kv', p['body_kv'])

        # Leg DOF 0
        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_0', 'kp', p['leg_dof0_kp'])
        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_0', 'kv', p['leg_dof0_kv'])

        # Leg DOF 1
        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_1', 'kp', p['leg_dof1_kp'])
        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_1', 'kv', p['leg_dof1_kv'])

        # Leg DOF 2
        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_2', 'kp', p['leg_dof2_kp'])
        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_2', 'kv', p['leg_dof2_kv'])

        # Leg DOF 3
        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_3', 'kp', p['leg_dof3_kp'])
        xml = _patch_actuator_attr(xml, r'act_joint_leg_\d+_[LR]_3', 'kv', p['leg_dof3_kv'])

        # Foot actuators = same as DOF3
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

        with open(XML_PATH, 'w', encoding='utf-8') as f:
            f.write(xml)
        return True

    # =================================================================
    # Run simulation
    # =================================================================

    def run_simulation(self, tag):
        output_dir  = os.path.join(TUNE_DIR, tag)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "results_FARMS.npz")

        try:
            result = subprocess.run(
                [sys.executable, os.path.join(CONTROL_DIR, "farms_run.py"),
                 "--headless", "--duration", str(self.duration),
                 "--output", output_path, "--config", CONFIG_PATH],
                capture_output=True, text=True, cwd=CONTROL_DIR, timeout=300)
        except subprocess.TimeoutExpired:
            return None

        stdout = result.stdout.lower()
        if "unstable" in stdout or "nan" in stdout:
            return None
        if result.returncode != 0:
            stderr_tail = result.stderr[-400:] if result.stderr else ""
            print(f"    Sim error: {stderr_tail}")
            return None
        if not os.path.exists(output_path):
            return None
        return output_path

    # =================================================================
    # Tracking error
    # =================================================================

    def compute_tracking_error(self, data_path):
        WARMUP_TIME = 0.5

        try:
            d           = np.load(data_path)
            times       = d['time']
            body_actual = d['body_jnt_pos']
            leg_actual  = d['leg_jnt_pos']
        except Exception as e:
            print(f"    NPZ load error: {e}")
            return None, None, None

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
        n_w   = bw['wave_number']
        speed = bw['speed']
        N     = max(18, 1)

        # Body command
        body_cmd = np.zeros_like(body_actual)
        for i in range(body_actual.shape[1]):
            phi_s = 2.0 * np.pi * n_w * speed * i / N
            body_cmd[:, i] = bw['amplitude'] * np.sin(omega * times - phi_s)
        body_rms = np.sqrt(np.mean((body_actual - body_cmd) ** 2))

        # Leg command
        n_dof = leg_actual.shape[-1]

        def _pad(src, n):
            a = np.zeros(n)
            src = np.asarray(src, dtype=float)
            a[:min(len(src), n)] = src[:n]
            return a

        amps   = _pad(lw['amplitudes'],    n_dof)
        poff   = _pad(lw['phase_offsets'],  n_dof)
        dcoff  = _pad(lw.get('dc_offsets', [0.0]*n_dof), n_dof)
        active = set(lw.get('active_dofs', [0, 1]))

        n_legs  = leg_actual.shape[1]
        leg_cmd = np.zeros_like(leg_actual)
        for n in range(n_legs):
            phi_s = 2.0 * np.pi * n_w * speed * n / N
            for si in range(2):
                sign = 1.0 if si == 0 else -1.0
                for dof in range(n_dof):
                    if dof not in active:
                        leg_cmd[:, n, si, dof] = dcoff[dof]
                    else:
                        phase = omega * times - phi_s + poff[dof]
                        leg_cmd[:, n, si, dof] = (sign * amps[dof] *
                                                  np.sin(phase) + dcoff[dof])

        leg_rms = np.sqrt(np.mean((leg_actual - leg_cmd) ** 2))
        leg_rms_dof = np.array([
            np.sqrt(np.mean((leg_actual[:,:,:,d] - leg_cmd[:,:,:,d]) ** 2))
            for d in range(n_dof)])

        return body_rms, leg_rms, leg_rms_dof

    # =================================================================
    # Objective
    # =================================================================

    def objective(self, params_list):
        self.iteration_count += 1
        tag         = f"opt_{self.iteration_count:04d}"
        params_dict = dict(zip(self.param_names, params_list))
        p           = params_dict

        if not self.apply_params(params_dict):
            self._log(tag, p, 'gen_fail', None, None, None)
            print(f"  [{self.iteration_count:3d}] XML patch failed")
            return self.FAILURE_PENALTY

        data_path = self.run_simulation(tag)
        if data_path is None:
            self._log(tag, p, 'unstable', None, None, None)
            print(f"  [{self.iteration_count:3d}] "
                  f"kp_b={p['body_kp']:.2e} kv_b={p['body_kv']:.2e} "
                  f"kp_l0={p['leg_dof0_kp']:.2e} kv_l0={p['leg_dof0_kv']:.2e} "
                  f"-> UNSTABLE")
            return self.FAILURE_PENALTY

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

        dof_str = "  ".join(f"d{d}={np.degrees(leg_rms_dof[d]):.1f}"
                            for d in range(len(leg_rms_dof)))
        print(f"  [{self.iteration_count:3d}] "
              f"kp_b={p['body_kp']:.2e} kv_b={p['body_kv']:.2e} "
              f"kp_l0={p['leg_dof0_kp']:.2e} kv_l0={p['leg_dof0_kv']:.2e} "
              f"d2_dmp={p['leg_dof2_damping']:.2e} "
              f"pitch_k={p['pitch_stiffness']:.2e}  "
              f"-> body={np.degrees(body_rms):.2f} "
              f"leg={np.degrees(leg_rms):.2f} [{dof_str}]  "
              f"cost={cost:.5f}{'  *' if is_best else ''}")

        if self.iteration_count % 10 == 0:
            self.save_progress()

        return cost

    def _log(self, tag, p, status, body_rms, leg_rms, leg_rms_dof):
        entry = {
            'iter':     self.iteration_count,
            'tag':      tag,
            'status':   status,
            'body_rms': float(body_rms) if body_rms is not None else None,
            'leg_rms':  float(leg_rms)  if leg_rms  is not None else None,
            'cost': (float(self.body_weight * body_rms + self.leg_weight * leg_rms)
                     if body_rms is not None else self.FAILURE_PENALTY),
            **{k: float(v) for k, v in p.items()},
        }
        if leg_rms_dof is not None:
            for d in range(len(leg_rms_dof)):
                entry[f'leg_rms_dof{d}'] = float(leg_rms_dof[d])
        self.history.append(entry)

    # =================================================================
    # Optimization runner
    # =================================================================

    def optimize(self, n_calls=200, n_initial_points=20):
        print("\n" + "=" * 65)
        print("STARTING BAYESIAN OPTIMIZATION  (FARMS centipede, 16 params)")
        print("=" * 65)
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
            print("\n\nInterrupted -- saving progress...")
            self.save_progress()
            self.plot_progress()
            if self.best_params is not None:
                self.apply_params(dict(zip(self.param_names, self.best_params)))
                print("  Applied best-so-far parameters to XML.")
            raise

        except Exception as e:
            print(f"\nOptimization error: {e}")
            self.save_progress()
            if self.best_params is not None:
                self.apply_params(dict(zip(self.param_names, self.best_params)))
            raise

    # =================================================================
    # Results & plotting
    # =================================================================

    def save_progress(self):
        path = os.path.join(OPT_DIR, 'optimization_progress.json')
        with open(path, 'w') as f:
            json.dump({
                'iteration':     self.iteration_count,
                'best_cost':     (self.best_combined_error
                                  if self.best_combined_error < float('inf')
                                  else None),
                'best_body_rms': (self.best_body_error
                                  if self.best_body_error < float('inf')
                                  else None),
                'best_leg_rms':  (self.best_leg_error
                                  if self.best_leg_error < float('inf')
                                  else None),
                'best_params':   (dict(zip(self.param_names, self.best_params))
                                  if self.best_params else None),
                'history':       self.history,
            }, f, indent=2, default=str)

    def save_final_results(self, result):
        with open(os.path.join(OPT_DIR, 'optimization_result.pkl'), 'wb') as f:
            pickle.dump(result, f)

        self.save_progress()
        self.plot_progress()

        print("\n" + "=" * 65)
        print("OPTIMIZATION COMPLETE")
        print("=" * 65)

        if result.fun < self.FAILURE_PENALTY:
            best_dict = dict(zip(self.param_names, result.x))
            print(f"\n  Best combined error: {result.fun:.6f}")
            print(f"  Body RMS: {np.degrees(self.best_body_error):.3f} deg")
            print(f"  Leg  RMS: {np.degrees(self.best_leg_error):.3f} deg")
            print(f"\n  Optimal parameters:")
            for name in self.param_names:
                init    = self.INITIAL_VALUES[name]
                optimal = best_dict[name]
                ratio   = optimal / init if init != 0 else float('inf')
                print(f"    {name:22s}: {optimal:.6e}  (x{ratio:.2f})")
            print(f"\n  Applying optimal parameters...")
            self.apply_params(best_dict)
        else:
            print("  WARNING: No stable solution found!")
            if self.best_params:
                self.apply_params(dict(zip(self.param_names, self.best_params)))
            else:
                shutil.copy2(XML_BACKUP, XML_PATH)
                print("  Restored backup.")

        ok   = sum(1 for h in self.history if h['status'] == 'ok')
        fail = len(self.history) - ok
        print(f"\n  Evaluations: {len(self.history)} total  "
              f"({ok} stable, {fail} failed)")
        print(f"  Results: {OPT_DIR}/")

    def plot_progress(self):
        ok = [h for h in self.history if h['status'] == 'ok']
        if len(ok) < 3:
            return

        iters     = [h['iter']                 for h in ok]
        body_errs = [np.degrees(h['body_rms']) for h in ok]
        leg_errs  = [np.degrees(h['leg_rms'])  for h in ok]
        costs     = [h['cost']                  for h in ok]

        run_best, bsf = [], float('inf')
        for c in costs:
            bsf = min(bsf, c)
            run_best.append(bsf)

        fig = plt.figure(figsize=(22, 12))
        gs  = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.35)

        # Row 1
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(iters, costs, 'o', ms=3, alpha=0.4, color='gray', label='All')
        ax.plot(iters, run_best, '-', color='red', lw=2, label='Best-so-far')
        ax.set(xlabel='Iteration', ylabel='Cost', title='Convergence')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[0, 1])
        sc = ax.scatter(body_errs, leg_errs, c=iters, cmap='viridis',
                        s=20, alpha=0.7)
        ax.set(xlabel='Body RMS (deg)', ylabel='Leg RMS (deg)',
               title='Body vs Leg Error')
        plt.colorbar(sc, ax=ax, label='Iteration')
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[0, 2])
        for d in range(4):
            key = f'leg_rms_dof{d}'
            vals = [np.degrees(h[key]) for h in ok if h.get(key) is not None]
            its  = [h['iter']          for h in ok if h.get(key) is not None]
            if vals:
                ax.plot(its, vals, 'o', ms=3, alpha=0.5, label=f'DOF{d}')
        ax.set(xlabel='Iteration', ylabel='RMS (deg)',
               title='Leg Per-DOF Tracking')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[0, 3])
        ax.plot(iters, body_errs, 'o', ms=3, alpha=0.5, color='blue',
                label='Body')
        ax.plot(iters, leg_errs,  'o', ms=3, alpha=0.5, color='red',
                label='Leg')
        ax.set(xlabel='Iteration', ylabel='RMS (deg)', title='All Errors')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Row 2: key param evolutions
        key_params = ['body_kp', 'leg_dof0_kp', 'leg_dof0_kv', 'leg_dof2_damping']
        for pi, pname in enumerate(key_params):
            ax    = fig.add_subplot(gs[1, pi])
            pvals = [h.get(pname) for h in ok if h.get(pname) is not None]
            pits  = [h['iter']    for h in ok if h.get(pname) is not None]
            best_c = min(costs)
            med_c  = np.median(costs)
            clrs  = ['green' if h['cost'] == best_c else
                     ('blue'  if h['cost'] < med_c else 'gray')
                     for h in ok if h.get(pname) is not None]
            ax.scatter(pits, pvals, c=clrs, s=15, alpha=0.6)
            if pname in self.INITIAL_VALUES:
                ax.axhline(self.INITIAL_VALUES[pname], color='orange',
                           ls='--', alpha=0.6, label='Initial')
            ax.set(xlabel='Iteration', ylabel=pname, title=pname)
            ax.set_yscale('log')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        plt.suptitle('Bayesian Optimization -- FARMS centipede 16-param PD',
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
            smap.append('green' if s == 'ok' else
                        'red' if s == 'unstable' else 'orange')
        ax1.bar(list(all_its), [1]*len(smap), color=smap, width=1.0)
        ax1.set(xlabel='Iteration', title='Stability Map')
        ax1.set_yticks([])

        ax2.hist(body_errs, bins=20, alpha=0.6, color='blue', label='Body')
        ax2.hist(leg_errs,  bins=20, alpha=0.6, color='red',  label='Leg')
        ax2.set(xlabel='RMS (deg)', ylabel='Count', title='Error Distribution')
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OPT_DIR, 'error_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plots saved -> {OPT_DIR}/")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bayesian Optimization -- FARMS centipede 16-param PD')
    parser.add_argument('--prior-weight', type=float, default=0.3)
    parser.add_argument('--body-weight',  type=float, default=0.5)
    parser.add_argument('--leg-weight',   type=float, default=0.5)
    parser.add_argument('--n-calls',      type=int,   default=200)
    parser.add_argument('--n-initial',    type=int,   default=20)
    parser.add_argument('--duration',     type=float, default=5.0)
    args = parser.parse_args()

    total_w = args.body_weight + args.leg_weight
    optimizer = FARMSCentipedeOptimizer(
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
            print(f"    python {os.path.join(CONTROL_DIR, 'farms_run.py')} "
                  f"--duration 10")
        print(f"\n  To restore original XML:")
        print(f"    copy {XML_BACKUP} {XML_PATH}")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved.")
    except Exception as e:
        print(f"\n\nFailed: {e}")
        raise


if __name__ == "__main__":
    main()
