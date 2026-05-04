"""
analyze_tracking.py — Compare commanded vs actual joint angles
==============================================================
Reads results.npz from 4-Data/ and config.yaml from the run folder,
reconstructs the commanded signal using the same equations as
SinusoidalTrajectoryGenerator in controller.py, and plots against
actual joint positions.

Key alignment with controller.py:
  - Reconstruction uses identical wave equations to SinusoidalTrajectoryGenerator.
  - Velocity commands are also reconstructed (A·ω·cos) and shown alongside
    position tracking — important now that we run PD servo with kv terms.
  - Warmup skip (default 0.5s) matches tune_parameters.py so reported RMS
    is comparable to what the optimizer minimizes.
  - The locals() padding bug from the previous version is fixed.

Location: 3-MUJOCO/2-Analysis/analyze_tracking.py

Usage:
    python analyze_tracking.py                          # latest run
    python analyze_tracking.py --run run_03_23_2026     # specific run
    python analyze_tracking.py --data ..\\4-Data\\run_03_23_2026\\results.npz
    python analyze_tracking.py --joints jb1 jb5 jb10   # specific body joints
    python analyze_tracking.py --legs 1 7 14            # specific leg numbers
    python analyze_tracking.py --save tracking.png      # save figures
    python analyze_tracking.py --no-warmup-skip         # include t=0 transient
    python analyze_tracking.py --no-velocity            # skip velocity plots
"""

import argparse
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml

# ── Paths relative to this script (in analysis/) ──
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.join(SCRIPT_DIR, "..")
DATA_DIR    = os.path.join(BASE_DIR, "outputs", "data")
CONTROL_DIR = os.path.join(BASE_DIR, "controllers", "blender")
DEFAULT_CONFIG = os.path.join(BASE_DIR, "configs", "blender_controller.yaml")

WARMUP_DEFAULT = 0.5   # seconds — must match tune_parameters.py WARMUP_TIME


# ═══════════════════════════════════════════════════════════════════════
# Wave reconstruction — mirrors SinusoidalTrajectoryGenerator exactly
# ═══════════════════════════════════════════════════════════════════════

def _pad3(arr):
    """Pad array to length 3 with zeros. Returns a new array — no aliasing."""
    out = np.zeros(3)
    src = np.asarray(arr, dtype=float)
    out[:min(len(src), 3)] = src[:3]
    return out


def _wave_params(config):
    """Extract and derive all wave parameters from config dict."""
    bw = config['body_wave']
    lw = config['leg_wave']
    freq        = float(bw['frequency'])
    n_wave      = float(bw['wave_number'])
    speed       = float(bw['speed'])
    n_bj_wave   = int(config.get('n_body_joints_wave', 19))
    omega       = 2.0 * np.pi * freq
    wave_offset = 2.0 * np.pi * n_wave
    N           = max(n_bj_wave - 1, 1)

    amps        = _pad3(lw['amplitudes'])
    poff        = _pad3(lw['phase_offsets'])
    dcoff       = _pad3(lw.get('dc_offsets', [0.0, 0.0, 0.0]))
    active_dofs = set(lw.get('active_dofs', [0, 1]))

    return dict(
        body_amp=float(bw['amplitude']),
        omega=omega, wave_offset=wave_offset, speed=speed,
        N=N, n_bj_wave=n_bj_wave,
        amps=amps, poff=poff, dcoff=dcoff, active_dofs=active_dofs,
    )


def reconstruct_body_commands(times, config):
    """
    Reconstruct body joint position and velocity commands.

    Matches SinusoidalTrajectoryGenerator.body_target() exactly:
        q     = A_body · sin(ωt − φᵢ)
        q_dot = A_body · ω · cos(ωt − φᵢ)

    Returns
    -------
    q     : (T, 20)   commanded positions [rad]
    q_dot : (T, 20)   commanded velocities [rad/s]
    """
    wp  = _wave_params(config)
    T   = len(times)
    q     = np.zeros((T, 20))
    q_dot = np.zeros((T, 20))

    for i in range(20):
        if i < wp['n_bj_wave']:
            phi   = wp['wave_offset'] * (i / wp['N']) * wp['speed']
            theta = wp['omega'] * times - phi
            q[:,   i] = wp['body_amp'] * np.sin(theta)
            q_dot[:, i] = wp['body_amp'] * wp['omega'] * np.cos(theta)

    return q, q_dot


def reconstruct_leg_commands(times, config):
    """
    Reconstruct leg joint position and velocity commands.

    Matches SinusoidalTrajectoryGenerator.leg_target() exactly
    (sinusoidal path — duty_factor=0 branch):
        LEFT:   q = +A·sin(θ)+dc,   q_dot = +A·ω·cos(θ)
        RIGHT:  q = −A·sin(θ)+dc,   q_dot = −A·ω·cos(θ)
        PASSIVE (dof not in active_dofs):
                q = dc,              q_dot = 0

    Returns
    -------
    q     : (T, 19, 2, 3)   commanded positions [rad]
    q_dot : (T, 19, 2, 3)   commanded velocities [rad/s]
    """
    wp  = _wave_params(config)
    T   = len(times)
    q     = np.zeros((T, 19, 2, 3))
    q_dot = np.zeros((T, 19, 2, 3))

    for n in range(19):
        phi_s = wp['wave_offset'] * (n / wp['N']) * wp['speed']

        for si in range(2):
            sign = 1.0 if si == 0 else -1.0

            for dof in range(3):
                if dof not in wp['active_dofs']:
                    # Passive: hold at dc offset, zero velocity
                    q[:, n, si, dof]     = wp['dcoff'][dof]
                    q_dot[:, n, si, dof] = 0.0
                else:
                    theta = wp['omega'] * times - phi_s + wp['poff'][dof]
                    q[:,    n, si, dof] = sign * wp['amps'][dof] * np.sin(theta) + wp['dcoff'][dof]
                    q_dot[:, n, si, dof] = sign * wp['amps'][dof] * wp['omega'] * np.cos(theta)

    return q, q_dot


# ═══════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════

def _rms(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def _make_rms_title(label, cmd, actual):
    rms = _rms(actual, cmd)
    return f"{label}  |  RMS: {rms:.4f} rad ({np.degrees(rms):.2f}°)"


# ═══════════════════════════════════════════════════════════════════════
# Body joint plots
# ═══════════════════════════════════════════════════════════════════════

def plot_body_tracking(times, q_cmd, q_dot_cmd, q_actual, q_dot_actual,
                       joint_indices, show_velocity, save_path=None):
    """
    Position (and optionally velocity) tracking for selected body joints.

    q_dot_actual is read from the NPZ `body_joint_vel` array if available,
    otherwise approximated by finite difference of q_actual.
    """
    n_plots = len(joint_indices)
    n_cols  = 2 if show_velocity else 1
    fig, axes = plt.subplots(n_plots, n_cols,
                              figsize=(7 * n_cols, 2.8 * n_plots),
                              sharex=True, squeeze=False)

    for row, ji in enumerate(joint_indices):
        label = f"jb{ji + 1}"

        # Position
        ax = axes[row, 0]
        ax.plot(times, q_cmd[:, ji],    'b-', lw=1.0, alpha=0.8, label='Command')
        ax.plot(times, q_actual[:, ji], 'r-', lw=1.0, alpha=0.8, label='Actual')
        ax.set_ylabel(f"{label} pos (rad)")
        ax.set_title(_make_rms_title(label + " position", q_cmd[:, ji], q_actual[:, ji]),
                     fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Velocity
        if show_velocity:
            ax = axes[row, 1]
            ax.plot(times, q_dot_cmd[:, ji],    'b-', lw=1.0, alpha=0.8, label='Cmd vel')
            ax.plot(times, q_dot_actual[:, ji], 'r-', lw=1.0, alpha=0.8, label='Act vel')
            ax.set_ylabel(f"{label} vel (rad/s)")
            ax.set_title(_make_rms_title(label + " velocity",
                                          q_dot_cmd[:, ji], q_dot_actual[:, ji]),
                         fontsize=9)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

    for c in range(n_cols):
        axes[-1, c].set_xlabel("Time (s)")

    fig.suptitle("Body Joint Tracking: Command vs Actual", fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        out = save_path.replace('.png', '_body.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Leg joint plots
# ═══════════════════════════════════════════════════════════════════════

def plot_leg_tracking(times, q_cmd, q_dot_cmd, q_actual, q_dot_actual,
                      leg_numbers, show_velocity, save_path=None):
    """
    Position (and optionally velocity) tracking for selected legs.
    Each leg gets a figure: 3 rows (DOFs) × 2 columns (left/right).
    If show_velocity: second figure per leg with velocity.
    """
    dof_names  = ['DOF 0 (yaw)', 'DOF 1 (upper pitch)', 'DOF 2 (lower pitch)']
    side_names = ['Left', 'Right']

    for leg_n in leg_numbers:
        # ── Position ──
        fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
        for dof in range(3):
            for si in range(2):
                ax = axes[dof, si]
                c  = q_cmd[:,    leg_n - 1, si, dof]
                a  = q_actual[:, leg_n - 1, si, dof]
                ax.plot(times, c, 'b-', lw=1.0, alpha=0.8, label='Command')
                ax.plot(times, a, 'r-', lw=1.0, alpha=0.8, label='Actual')
                ax.set_title(
                    f"{side_names[si]} {dof_names[dof]}  |  RMS: {_rms(c, a):.4f}",
                    fontsize=9)
                ax.grid(True, alpha=0.3)
                if dof == 0 and si == 0:
                    ax.legend(loc='upper right', fontsize=8)
        for dof in range(3):
            axes[dof, 0].set_ylabel("Angle (rad)")
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 1].set_xlabel("Time (s)")
        fig.suptitle(f"Leg {leg_n} Tracking: Command vs Actual", fontsize=13, fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path.replace('.png', f'_leg{leg_n}.png'),
                        dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path.replace('.png', f'_leg{leg_n}.png')}")

        # ── Velocity (separate figure) ──
        if show_velocity:
            fig2, axes2 = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
            for dof in range(3):
                for si in range(2):
                    ax  = axes2[dof, si]
                    cv  = q_dot_cmd[:,    leg_n - 1, si, dof]
                    av  = q_dot_actual[:, leg_n - 1, si, dof]
                    ax.plot(times, cv, 'b-', lw=1.0, alpha=0.8, label='Cmd vel')
                    ax.plot(times, av, 'r-', lw=1.0, alpha=0.8, label='Act vel')
                    ax.set_title(
                        f"{side_names[si]} {dof_names[dof]}  |  vel RMS: {_rms(cv, av):.4f}",
                        fontsize=9)
                    ax.grid(True, alpha=0.3)
                    if dof == 0 and si == 0:
                        ax.legend(loc='upper right', fontsize=8)
            for dof in range(3):
                axes2[dof, 0].set_ylabel("Angular vel (rad/s)")
            axes2[2, 0].set_xlabel("Time (s)")
            axes2[2, 1].set_xlabel("Time (s)")
            fig2.suptitle(f"Leg {leg_n} Velocity Tracking: Command vs Actual",
                          fontsize=13, fontweight='bold')
            plt.tight_layout()
            if save_path:
                fig2.savefig(save_path.replace('.png', f'_leg{leg_n}_vel.png'),
                             dpi=150, bbox_inches='tight')
                print(f"Saved: {save_path.replace('.png', f'_leg{leg_n}_vel.png')}")

    if not save_path:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Error summary
# ═══════════════════════════════════════════════════════════════════════

def plot_error_summary(times, body_q_cmd, body_q_actual,
                       leg_q_cmd,  leg_q_actual,
                       save_path=None):
    """
    Four-panel summary:
      1. Body position error rolling mean (selected joints)
      2. Leg position RMS per leg × DOF (bar chart)
      3. Leg DOF2 error over time (key oscillation diagnostic)
      4. Left vs Right asymmetry per leg (DOF1 pitch — gravity sag diagnostic)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    window = max(1, int(0.5 / dt))

    # ── Panel 1: body rolling error ───────────────────────────────────
    ax = axes[0, 0]
    body_error = body_q_actual - body_q_cmd
    for ji in range(0, 20, 4):
        err = np.abs(body_error[:, ji])
        if len(err) > window:
            rolling = np.convolve(err, np.ones(window) / window, mode='valid')
            t_r     = times[window - 1:]
        else:
            rolling, t_r = err, times
        ax.plot(t_r, rolling, lw=1.0, label=f"jb{ji + 1}")
    ax.set(ylabel="Abs error (rad)",
           title="Body joint tracking error (0.5s rolling mean)")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: leg position RMS per leg × DOF ───────────────────────
    ax    = axes[0, 1]
    x     = np.arange(1, 20)
    width = 0.25
    colors   = ['#2196F3', '#FF5722', '#4CAF50']
    dof_lbls = ['Yaw (DOF0)', 'Upper pitch (DOF1)', 'Lower pitch (DOF2)']
    for dof in range(3):
        rms_left  = np.array([_rms(leg_q_actual[:, n, 0, dof],
                                    leg_q_cmd[:,    n, 0, dof]) for n in range(19)])
        rms_right = np.array([_rms(leg_q_actual[:, n, 1, dof],
                                    leg_q_cmd[:,    n, 1, dof]) for n in range(19)])
        avg = (rms_left + rms_right) / 2
        ax.bar(x + (dof - 1) * width, avg, width, label=dof_lbls[dof],
               color=colors[dof], alpha=0.8)
    ax.set(xlabel="Leg number", ylabel="RMS error (rad)",
           title="Leg position RMS (L/R averaged) per DOF")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 3: DOF2 error over time (passive joint oscillation) ────
    ax = axes[1, 0]
    for n in [0, 6, 12, 18]:
        for si, lbl in [(0, 'L'), (1, 'R')]:
            err = np.abs(leg_q_actual[:, n, si, 2] - leg_q_cmd[:, n, si, 2])
            ax.plot(times, err, lw=0.8, alpha=0.7, label=f"leg{n+1}{lbl}")
    ax.set(xlabel="Time (s)", ylabel="Abs error (rad)",
           title="DOF2 (lower pitch) error — passive joint oscillation diagnostic")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Left vs Right asymmetry — DOF1 (gravity sag) ────────
    ax = axes[1, 1]
    rms_L = np.array([_rms(leg_q_actual[:, n, 0, 1],
                            leg_q_cmd[:,    n, 0, 1]) for n in range(19)])
    rms_R = np.array([_rms(leg_q_actual[:, n, 1, 1],
                            leg_q_cmd[:,    n, 1, 1]) for n in range(19)])
    ax.bar(x - width / 2, rms_L, width, label='Left',  color='steelblue', alpha=0.8)
    ax.bar(x + width / 2, rms_R, width, label='Right', color='tomato',    alpha=0.8)
    ax.set(xlabel="Leg number", ylabel="RMS error (rad)",
           title="DOF1 (upper pitch) L vs R asymmetry — gravity sag diagnostic")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        out = save_path.replace('.png', '_summary.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Phase portrait (position vs velocity)
# ═══════════════════════════════════════════════════════════════════════

def plot_phase_portrait(times, q_cmd, q_dot_cmd, q_actual, q_dot_actual,
                        joint_indices, save_path=None):
    """
    Phase portrait (q vs q_dot) for selected body joints.
    A well-tuned PD servo traces the command ellipse closely.
    Deviation shows both amplitude and phase errors simultaneously.
    """
    n = len(joint_indices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for col, ji in enumerate(joint_indices):
        ax = axes[0, col]
        ax.plot(q_cmd[:, ji],    q_dot_cmd[:, ji],    'b-',  lw=1.5, alpha=0.6, label='Command')
        ax.plot(q_actual[:, ji], q_dot_actual[:, ji], 'r-',  lw=0.8, alpha=0.7, label='Actual')
        # Mark start
        ax.plot(q_cmd[0, ji],    q_dot_cmd[0, ji],    'bs',  ms=6)
        ax.plot(q_actual[0, ji], q_dot_actual[0, ji], 'rs',  ms=6)
        ax.set(xlabel="Position (rad)", ylabel="Velocity (rad/s)",
               title=f"jb{ji+1} phase portrait")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('auto')

    fig.suptitle("Body Joint Phase Portrait (q vs q̇)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        out = save_path.replace('.png', '_phase.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Velocity loading helper
# ═══════════════════════════════════════════════════════════════════════

def _get_body_vel_actual(d, q_actual, times):
    """
    Return body joint velocity array (T, 20).
    Uses NPZ key 'body_joint_vel' if present, else finite-difference of position.
    """
    if 'body_joint_vel' in d:
        return d['body_joint_vel']
    dt = np.diff(times)
    dt = np.append(dt, dt[-1])          # repeat last dt to keep shape
    dq = np.gradient(q_actual, axis=0)  # 2nd-order central diff
    return dq / dt[:, None]


def _get_leg_vel_actual(d, q_actual, times):
    """
    Return leg joint velocity array (T, 19, 2, 3).
    Uses NPZ key 'leg_joint_vel' if present, else finite-difference.
    """
    if 'leg_joint_vel' in d:
        return d['leg_joint_vel']
    return np.gradient(q_actual, axis=0) / np.diff(times, append=times[-1])[:, None, None, None]


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def find_latest_run():
    runs = sorted(glob.glob(os.path.join(DATA_DIR, "run_*")))
    return runs[-1] if runs else None


def main():
    p = argparse.ArgumentParser(description="Analyze command vs actual joint tracking")
    p.add_argument("--data",           default=None)
    p.add_argument("--config",         default=None)
    p.add_argument("--run",            default=None,
                   help="Run folder name (e.g. run_03_23_2026)")
    p.add_argument("--joints",         nargs='+', default=None,
                   help="Body joints (e.g. jb1 jb5 jb10)")
    p.add_argument("--legs",           nargs='+', type=int, default=None,
                   help="Leg numbers (e.g. 1 7 14)")
    p.add_argument("--save",           default=None,
                   help="Save prefix (e.g. tracking.png)")
    p.add_argument("--no-summary",     action="store_true")
    p.add_argument("--no-velocity",    action="store_true",
                   help="Skip velocity tracking plots")
    p.add_argument("--phase-portrait", action="store_true",
                   help="Show phase portrait (q vs q_dot) for body joints")
    p.add_argument("--no-warmup-skip", action="store_true",
                   help="Include startup transient (t < 0.5s) in plots and stats")
    p.add_argument("--warmup",         type=float, default=WARMUP_DEFAULT,
                   help=f"Warmup duration to skip (default {WARMUP_DEFAULT}s)")
    args = p.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────
    if args.data:
        data_path   = args.data
        config_path = args.config or os.path.join(os.path.dirname(
                          os.path.abspath(data_path)), "config.yaml")
    elif args.run:
        run_dir     = os.path.join(DATA_DIR, args.run)
        data_path   = os.path.join(run_dir, "results.npz")
        config_path = args.config or os.path.join(run_dir, "config.yaml")
    else:
        latest = find_latest_run()
        if latest is None:
            print(f"ERROR: No run folders found in {DATA_DIR}")
            sys.exit(1)
        data_path   = os.path.join(latest, "results.npz")
        config_path = args.config or os.path.join(latest, "config.yaml")
        print(f"Using latest run: {os.path.basename(latest)}")

    if not os.path.exists(config_path):
        config_path = DEFAULT_CONFIG
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────
    print(f"Data:   {data_path}")
    print(f"Config: {config_path}")

    d           = np.load(data_path)
    times_all   = d['time']
    body_q_all  = d['body_joint_pos']   # (T, 20)
    leg_q_all   = d['leg_joint_pos']    # (T, 19, 2, 3)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    bw = config['body_wave']
    print(f"  {len(times_all)} frames,  t=[{times_all[0]:.2f}, {times_all[-1]:.2f}]s")
    print(f"  Wave: n={bw['wave_number']}, A_body={bw['amplitude']}, f={bw['frequency']}Hz")

    # ── Warmup skip ────────────────────────────────────────────────────
    warmup = 0.0 if args.no_warmup_skip else args.warmup
    mask   = times_all >= warmup
    if warmup > 0 and mask.sum() < 10:
        print(f"  WARNING: fewer than 10 frames after warmup skip ({warmup}s). "
              f"Using all frames.")
        mask = np.ones(len(times_all), dtype=bool)
    times   = times_all[mask]
    body_q  = body_q_all[mask]
    leg_q   = leg_q_all[mask]

    if warmup > 0:
        print(f"  Skipping first {warmup}s  ({(~mask).sum()} frames) — "
              f"matches tune_parameters.py WARMUP_TIME")

    # ── Commanded signals: prefer pre-recorded, fall back to reconstruction ──
    # run.py now saves body_joint_pos_cmd / vel_cmd directly from the controller
    # so there is no reconstruction error and no dependency on config.yaml values.
    if 'body_joint_pos_cmd' in d:
        body_q_cmd     = d['body_joint_pos_cmd'][mask]
        body_q_dot_cmd = d['body_joint_vel_cmd'][mask]
        leg_q_cmd      = d['leg_joint_pos_cmd'][mask]
        leg_q_dot_cmd  = d['leg_joint_vel_cmd'][mask]
        cmd_source     = "pre-recorded (from run.py controller)"
    else:
        print("  NOTE: commanded signals not in NPZ — reconstructing analytically "
              "(upgrade run.py to record them directly).")
        body_q_cmd, body_q_dot_cmd = reconstruct_body_commands(times, config)
        leg_q_cmd,  leg_q_dot_cmd  = reconstruct_leg_commands(times, config)
        cmd_source = "reconstructed from config.yaml"
    print(f"  Command source: {cmd_source}")

    # ── Actual velocities ──────────────────────────────────────────────
    show_vel = not args.no_velocity
    if show_vel:
        # Prefer sensor-recorded velocities (sv_jb*, sv_jl*) saved by run.py
        if 'body_joint_vel' in d:
            body_q_dot = d['body_joint_vel'][mask]
            leg_q_dot  = d['leg_joint_vel'][mask]
            vel_source = "sensor (sv_* from MuJoCo)"
        else:
            body_q_dot = _get_body_vel_actual(d, body_q_all, times_all)[mask]
            leg_q_dot  = _get_leg_vel_actual(d, leg_q_all, times_all)[mask]
            vel_source = "finite-difference (upgrade run.py to record sensors)"
        print(f"  Velocity source: {vel_source}")
    else:
        body_q_dot = body_q_dot_cmd  # won't be used
        leg_q_dot  = leg_q_dot_cmd

    # ── Overall RMS stats ──────────────────────────────────────────────
    body_rms = _rms(body_q, body_q_cmd)
    leg_rms  = _rms(leg_q,  leg_q_cmd)
    print(f"\n  Body position RMS: {body_rms:.4f} rad ({np.degrees(body_rms):.2f}°)")
    print(f"  Leg  position RMS: {leg_rms:.4f}  rad ({np.degrees(leg_rms):.2f}°)")

    # Per-DOF leg breakdown
    for dof, name in enumerate(['Yaw (DOF0)', 'Upper pitch (DOF1)', 'Lower pitch (DOF2)']):
        r = _rms(leg_q[:,:,:,dof], leg_q_cmd[:,:,:,dof])
        print(f"    Leg {name}: {r:.4f} rad ({np.degrees(r):.2f}°)")

    # Left vs Right asymmetry for DOF1
    r_L = _rms(leg_q[:,:,0,1], leg_q_cmd[:,:,0,1])
    r_R = _rms(leg_q[:,:,1,1], leg_q_cmd[:,:,1,1])
    print(f"\n  DOF1 Left  RMS: {np.degrees(r_L):.2f}°")
    print(f"  DOF1 Right RMS: {np.degrees(r_R):.2f}°  "
          f"({'★ asymmetry detected' if abs(r_R - r_L) > 0.03 else 'symmetric'})")

    # ── Parse joint/leg selections ─────────────────────────────────────
    if args.joints:
        body_idx = [int(j.replace('jb', '')) - 1 for j in args.joints]
    else:
        body_idx = [0, 4, 9, 14, 18]      # jb1, jb5, jb10, jb15, jb19

    leg_numbers = args.legs or [1, 7, 14]

    # ── Auto-save when running headless (Agg backend = no display) ────
    import matplotlib
    if args.save is None and matplotlib.get_backend().lower() == 'agg':
        out_dir   = os.path.dirname(os.path.abspath(data_path))
        args.save = os.path.join(out_dir, "tracking.png")
        print(f"\n  Headless server detected — auto-saving figures to {out_dir}/")
        print(f"  Files will be named: tracking_body.png, tracking_leg*.png, "
              f"tracking_summary.png, etc.")

    # ── Plots ──────────────────────────────────────────────────────────
    print(f"\nPlotting body joints: {['jb' + str(i+1) for i in body_idx]}")
    plot_body_tracking(times, body_q_cmd, body_q_dot_cmd,
                       body_q, body_q_dot,
                       body_idx, show_vel, args.save)
    plt.close('all')

    print(f"Plotting legs: {leg_numbers}")
    plot_leg_tracking(times, leg_q_cmd, leg_q_dot_cmd,
                      leg_q, leg_q_dot,
                      leg_numbers, show_vel, args.save)
    plt.close('all')

    if not args.no_summary:
        print("Plotting error summary...")
        plot_error_summary(times, body_q_cmd, body_q,
                           leg_q_cmd, leg_q, args.save)
        plt.close('all')

    if args.phase_portrait:
        print("Plotting phase portraits...")
        plot_phase_portrait(times, body_q_cmd, body_q_dot_cmd,
                            body_q, body_q_dot,
                            body_idx, args.save)
        plt.close('all')

    print("\nDone.")


if __name__ == "__main__":
    main()
