"""
analyze_tracking_FARMS.py — Command vs actual joint tracking for the FARMS centipede
======================================================================================
Reads results_FARMS.npz from 4-Data/ and farms_config.yaml, reconstructs
commanded signals using the same equations as FARMSTravelingWaveController,
and plots against actual joint positions.

Differences from analyze_tracking.py (Blender model):
  - 19 body joints  (joint_body_1..19, body_0 is welded)
  - 19 legs × 2 sides × 4 DOF  (DOF 0=yaw, 1=elevation, 2=tibia, 3=tarsus)
  - Active DOFs: 0 and 1 only; DOF2 held at 0, DOF3 held at π/6 (30°)
  - Leg index is 0-based (n=0..18)
  - NPZ keys: body_jnt_pos, leg_jnt_pos, com_pos, com_vel
  - All saved figures are suffixed _FARMS

Usage:
    python analyze_tracking_FARMS.py                         # latest run
    python analyze_tracking_FARMS.py --run run_03_29_2026_120000
    python analyze_tracking_FARMS.py --data ..\\4-Data\\run_*\\results_FARMS.npz
    python analyze_tracking_FARMS.py --joints 1 5 10        # body joint numbers
    python analyze_tracking_FARMS.py --legs 0 9 18          # leg indices (0-based)
    python analyze_tracking_FARMS.py --save tracking.png
    python analyze_tracking_FARMS.py --no-warmup-skip
"""

import argparse
import glob
import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import yaml

# ── Paths (script lives in 2-Analysis/) ───────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR       = os.path.join(SCRIPT_DIR, "..")
DATA_DIR       = os.path.join(BASE_DIR, "4-Data")
CONTROL_DIR    = os.path.join(BASE_DIR, "1-Control")
DEFAULT_CONFIG = os.path.join(CONTROL_DIR, "farms_config.yaml")

WARMUP_DEFAULT = 0.5   # seconds

# Model constants
N_BODY_JOINTS = 19
N_LEGS        = 19
N_LEG_DOF     = 4
DOF_NAMES     = ["Yaw (DOF0)", "Elevation (DOF1)", "Tibia pitch (DOF2)", "Tarsus (DOF3)"]
ACTIVE_DOFS   = (0, 1)


# ═══════════════════════════════════════════════════════════════════════
# Wave reconstruction — mirrors FARMSTravelingWaveController exactly
# ═══════════════════════════════════════════════════════════════════════

def _wave_params(config):
    bw = config["body_wave"]
    lw = config["leg_wave"]
    freq      = float(bw["frequency"])
    n_wave    = float(bw["wave_number"])
    speed     = float(bw["speed"])
    omega     = 2.0 * np.pi * freq
    # φ_s(i) = 2π · n_wave · speed · i / (N_BODY_JOINTS − 1)
    denom     = max(N_BODY_JOINTS - 1, 1)

    amps      = np.array(lw["amplitudes"],    dtype=float)   # (4,)
    poffs     = np.array(lw["phase_offsets"], dtype=float)   # (4,)
    dcoffs    = np.array(lw["dc_offsets"],    dtype=float)   # (4,)
    active    = set(lw["active_dofs"])

    return dict(
        body_amp=float(bw["amplitude"]),
        omega=omega, n_wave=n_wave, speed=speed, denom=denom,
        amps=amps, poffs=poffs, dcoffs=dcoffs, active=active,
    )


def _spatial_phase(wp, i):
    return 2.0 * np.pi * wp["n_wave"] * wp["speed"] * i / wp["denom"]


def reconstruct_body_commands(times, config):
    """
    q     : (T, 19)  commanded body yaw positions [rad]
    q_dot : (T, 19)  commanded body yaw velocities [rad/s]
    """
    wp    = _wave_params(config)
    T     = len(times)
    q     = np.zeros((T, N_BODY_JOINTS))
    q_dot = np.zeros((T, N_BODY_JOINTS))
    for i in range(N_BODY_JOINTS):
        phi   = _spatial_phase(wp, i)
        theta = wp["omega"] * times - phi
        q[:,   i] = wp["body_amp"] * np.sin(theta)
        q_dot[:, i] = wp["body_amp"] * wp["omega"] * np.cos(theta)
    return q, q_dot


def reconstruct_leg_commands(times, config):
    """
    q     : (T, 19, 2, 4)  commanded leg positions [rad]
    q_dot : (T, 19, 2, 4)  commanded leg velocities [rad/s]

    Convention:
      Left  (si=0): +A·sin(ωt − φ_s + offset) + dc
      Right (si=1): −A·sin(ωt − φ_s + offset) + dc
      Inactive DOF: dc only, zero velocity
    """
    wp    = _wave_params(config)
    T     = len(times)
    q     = np.zeros((T, N_LEGS, 2, N_LEG_DOF))
    q_dot = np.zeros((T, N_LEGS, 2, N_LEG_DOF))

    for n in range(N_LEGS):
        phi_s = _spatial_phase(wp, n)
        for si in range(2):
            sign = 1.0 if si == 0 else -1.0
            for dof in range(N_LEG_DOF):
                dc = wp["dcoffs"][dof]
                if dof not in wp["active"]:
                    q[:, n, si, dof]     = dc
                    q_dot[:, n, si, dof] = 0.0
                else:
                    theta = wp["omega"] * times - phi_s + wp["poffs"][dof]
                    q[:,    n, si, dof] = sign * wp["amps"][dof] * np.sin(theta) + dc
                    q_dot[:, n, si, dof] = sign * wp["amps"][dof] * wp["omega"] * np.cos(theta)
    return q, q_dot


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _rms(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def _finite_diff_vel(q, times):
    """Finite-difference velocity from position array along axis 0."""
    return np.gradient(q, times, axis=0)


def _save_name(save_path, suffix):
    """Insert suffix before .png and add _FARMS."""
    base = save_path.replace(".png", "")
    if not base.endswith("_FARMS"):
        base += "_FARMS"
    return base + suffix + ".png"


# ═══════════════════════════════════════════════════════════════════════
# Body joint plot
# ═══════════════════════════════════════════════════════════════════════

def plot_body_tracking(times, q_cmd, q_dot_cmd, q_actual, q_dot_actual,
                       joint_numbers, show_velocity, save_path=None):
    """
    joint_numbers : list of 1-based body joint numbers (1..19)
    """
    joint_indices = [j - 1 for j in joint_numbers]
    n_plots = len(joint_indices)
    n_cols  = 2 if show_velocity else 1
    fig, axes = plt.subplots(n_plots, n_cols,
                             figsize=(7 * n_cols, 2.8 * n_plots),
                             sharex=True, squeeze=False)

    for row, ji in enumerate(joint_indices):
        label = f"joint_body_{ji + 1}"

        ax = axes[row, 0]
        ax.plot(times, q_cmd[:, ji],    "b-", lw=1.0, alpha=0.8, label="Command")
        ax.plot(times, q_actual[:, ji], "r-", lw=1.0, alpha=0.8, label="Actual")
        rms = _rms(q_cmd[:, ji], q_actual[:, ji])
        ax.set_title(f"{label} pos  |  RMS {rms:.4f} rad ({np.degrees(rms):.2f}°)",
                     fontsize=9)
        ax.set_ylabel("Position (rad)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        if show_velocity:
            ax = axes[row, 1]
            ax.plot(times, q_dot_cmd[:, ji],    "b-", lw=1.0, alpha=0.8, label="Cmd vel")
            ax.plot(times, q_dot_actual[:, ji], "r-", lw=1.0, alpha=0.8, label="Act vel")
            rms_v = _rms(q_dot_cmd[:, ji], q_dot_actual[:, ji])
            ax.set_title(f"{label} vel  |  RMS {rms_v:.4f} rad/s", fontsize=9)
            ax.set_ylabel("Velocity (rad/s)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

    for c in range(n_cols):
        axes[-1, c].set_xlabel("Time (s)")
    fig.suptitle("Body Joint Tracking (FARMS model) — Command vs Actual",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        p = _save_name(save_path, "_body")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Leg joint plot
# ═══════════════════════════════════════════════════════════════════════

def plot_leg_tracking(times, q_cmd, q_dot_cmd, q_actual, q_dot_actual,
                      leg_indices, show_velocity, save_path=None):
    """
    leg_indices : list of 0-based leg indices (0..18)
    """
    side_names = ["Left", "Right"]

    for n in leg_indices:
        fig, axes = plt.subplots(N_LEG_DOF, 2, figsize=(14, 2.8 * N_LEG_DOF),
                                 sharex=True)
        for dof in range(N_LEG_DOF):
            for si in range(2):
                ax  = axes[dof, si]
                c   = q_cmd[:,    n, si, dof]
                a   = q_actual[:, n, si, dof]
                ax.plot(times, c, "b-", lw=1.0, alpha=0.8, label="Command")
                ax.plot(times, a, "r-", lw=1.0, alpha=0.8, label="Actual")
                rms = _rms(c, a)
                ax.set_title(
                    f"{side_names[si]} {DOF_NAMES[dof]}  |  RMS {rms:.4f} rad",
                    fontsize=9)
                ax.grid(True, alpha=0.3)
                if dof == 0 and si == 0:
                    ax.legend(loc="upper right", fontsize=8)
        for dof in range(N_LEG_DOF):
            axes[dof, 0].set_ylabel("Angle (rad)")
        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        fig.suptitle(f"Leg n={n} Tracking (FARMS) — Command vs Actual",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save_path:
            p = _save_name(save_path, f"_leg{n}")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"  Saved: {p}")

        if show_velocity:
            fig2, axes2 = plt.subplots(N_LEG_DOF, 2, figsize=(14, 2.8 * N_LEG_DOF),
                                       sharex=True)
            for dof in range(N_LEG_DOF):
                for si in range(2):
                    ax  = axes2[dof, si]
                    cv  = q_dot_cmd[:,    n, si, dof]
                    av  = q_dot_actual[:, n, si, dof]
                    ax.plot(times, cv, "b-", lw=1.0, alpha=0.8, label="Cmd vel")
                    ax.plot(times, av, "r-", lw=1.0, alpha=0.8, label="Act vel")
                    rms = _rms(cv, av)
                    ax.set_title(
                        f"{side_names[si]} {DOF_NAMES[dof]}  |  vel RMS {rms:.4f}",
                        fontsize=9)
                    ax.grid(True, alpha=0.3)
                    if dof == 0 and si == 0:
                        ax.legend(loc="upper right", fontsize=8)
            for dof in range(N_LEG_DOF):
                axes2[dof, 0].set_ylabel("Ang vel (rad/s)")
            axes2[-1, 0].set_xlabel("Time (s)")
            axes2[-1, 1].set_xlabel("Time (s)")
            fig2.suptitle(f"Leg n={n} Velocity Tracking (FARMS)",
                          fontsize=13, fontweight="bold")
            plt.tight_layout()
            if save_path:
                p = _save_name(save_path, f"_leg{n}_vel")
                fig2.savefig(p, dpi=150, bbox_inches="tight")
                print(f"  Saved: {p}")

    if not save_path:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Error summary
# ═══════════════════════════════════════════════════════════════════════

def plot_error_summary(times, body_q_cmd, body_q_actual,
                       leg_q_cmd, leg_q_actual,
                       save_path=None):
    """
    Four-panel summary:
      1. Body position rolling error
      2. Leg position RMS per leg × DOF (active DOFs only)
      3. DOF1 (elevation) error — key stance/swing diagnostic
      4. Left vs Right DOF1 asymmetry — gravity sag diagnostic
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    dt     = np.diff(times)
    dt     = np.append(dt, dt[-1])
    window = max(1, int(0.5 / (times[1] - times[0])))
    x      = np.arange(N_LEGS)   # 0-based leg indices

    # ── Panel 1: body rolling error ───────────────────────────────────
    ax = axes[0, 0]
    body_error = body_q_actual - body_q_cmd
    for ji in range(0, N_BODY_JOINTS, 4):
        err = np.abs(body_error[:, ji])
        if len(err) > window:
            rolling = np.convolve(err, np.ones(window) / window, mode="valid")
            t_r     = times[window - 1:]
        else:
            rolling, t_r = err, times
        ax.plot(t_r, rolling, lw=1.0, label=f"body_{ji + 1}")
    ax.set(ylabel="Abs error (rad)",
           title="Body joint tracking error (0.5s rolling mean)")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: leg RMS per leg × active DOF ─────────────────────────
    ax     = axes[0, 1]
    width  = 0.35
    colors = ["#2196F3", "#FF5722"]
    for di, dof in enumerate(ACTIVE_DOFS):
        rms_L = np.array([_rms(leg_q_actual[:, n, 0, dof],
                               leg_q_cmd[:,    n, 0, dof]) for n in range(N_LEGS)])
        rms_R = np.array([_rms(leg_q_actual[:, n, 1, dof],
                               leg_q_cmd[:,    n, 1, dof]) for n in range(N_LEGS)])
        avg   = (rms_L + rms_R) / 2
        ax.bar(x + (di - 0.5) * width, avg, width,
               label=DOF_NAMES[dof], color=colors[di], alpha=0.8)
    ax.set(xlabel="Leg index (0-based)", ylabel="RMS error (rad)",
           title="Leg position RMS (L/R averaged) — active DOFs")
    ax.set_xticks(x[::2])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: DOF1 (elevation) error over time ─────────────────────
    ax = axes[1, 0]
    for n in [0, 6, 12, 18]:
        for si, lbl in [(0, "L"), (1, "R")]:
            err = np.abs(leg_q_actual[:, n, si, 1] - leg_q_cmd[:, n, si, 1])
            ax.plot(times, err, lw=0.8, alpha=0.7, label=f"leg{n}{lbl}")
    ax.set(xlabel="Time (s)", ylabel="Abs error (rad)",
           title="DOF1 (elevation) tracking error — stance/swing diagnostic")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: DOF1 L vs R asymmetry ────────────────────────────────
    ax    = axes[1, 1]
    rms_L = np.array([_rms(leg_q_actual[:, n, 0, 1],
                           leg_q_cmd[:,    n, 0, 1]) for n in range(N_LEGS)])
    rms_R = np.array([_rms(leg_q_actual[:, n, 1, 1],
                           leg_q_cmd[:,    n, 1, 1]) for n in range(N_LEGS)])
    ax.bar(x - width / 2, rms_L, width, label="Left",  color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, rms_R, width, label="Right", color="tomato",    alpha=0.8)
    ax.set(xlabel="Leg index (0-based)", ylabel="RMS error (rad)",
           title="DOF1 (elevation) L vs R asymmetry — gravity sag diagnostic")
    ax.set_xticks(x[::2])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        p = _save_name(save_path, "_summary")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# COM trajectory
# ═══════════════════════════════════════════════════════════════════════

def plot_com_trajectory(times, com_pos, com_vel, save_path=None):
    """
    Two-panel: XY trajectory top view + X velocity over time.
    Gives a quick locomotion quality check.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # XY trajectory
    ax = axes[0]
    sc = ax.scatter(com_pos[:, 0], com_pos[:, 1],
                    c=times, cmap="viridis", s=2)
    plt.colorbar(sc, ax=ax, label="Time (s)")
    ax.set(xlabel="X (m)", ylabel="Y (m)",
           title="COM trajectory (top view)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # X velocity over time
    ax = axes[1]
    ax.plot(times, com_vel[:, 0], "b-", lw=1.0, label="X vel")
    ax.plot(times, com_vel[:, 1], "g-", lw=1.0, alpha=0.6, label="Y vel")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    mean_vx = np.mean(com_vel[:, 0])
    ax.axhline(mean_vx, color="r", lw=1.0, ls="--",
               label=f"Mean Vx = {mean_vx:.4f} m/s")
    ax.set(xlabel="Time (s)", ylabel="Velocity (m/s)",
           title="COM velocity")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("COM Trajectory (FARMS model)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        p = _save_name(save_path, "_com")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def find_latest_run():
    runs = sorted(glob.glob(os.path.join(DATA_DIR, "run_*")))
    return runs[-1] if runs else None


def main():
    p = argparse.ArgumentParser(
        description="Analyze FARMS centipede command vs actual tracking")
    p.add_argument("--data",           default=None,
                   help="Direct path to results_FARMS.npz")
    p.add_argument("--config",         default=None,
                   help="Direct path to farms_config.yaml")
    p.add_argument("--run",            default=None,
                   help="Run folder name e.g. run_03_29_2026_120000")
    p.add_argument("--joints",         nargs="+", type=int, default=None,
                   help="Body joint numbers to plot (1-based, e.g. 1 5 10 19)")
    p.add_argument("--legs",           nargs="+", type=int, default=None,
                   help="Leg indices to plot (0-based, e.g. 0 9 18)")
    p.add_argument("--save",           default=None,
                   help="Output prefix e.g. tracking.png → tracking_body_FARMS.png")
    p.add_argument("--no-summary",     action="store_true")
    p.add_argument("--no-velocity",    action="store_true")
    p.add_argument("--no-com",         action="store_true",
                   help="Skip COM trajectory plot")
    p.add_argument("--no-warmup-skip", action="store_true")
    p.add_argument("--warmup",         type=float, default=WARMUP_DEFAULT)
    args = p.parse_args()

    # ── Resolve paths ──────────────────────────────────────────────────
    if args.data:
        data_path   = args.data
        config_path = args.config or os.path.join(
            os.path.dirname(os.path.abspath(data_path)), "farms_config.yaml")
    elif args.run:
        run_dir     = os.path.join(DATA_DIR, args.run)
        data_path   = os.path.join(run_dir, "results_FARMS.npz")
        config_path = args.config or os.path.join(run_dir, "farms_config.yaml")
    else:
        latest = find_latest_run()
        if latest is None:
            print(f"ERROR: No run folders found in {DATA_DIR}"); sys.exit(1)
        data_path   = os.path.join(latest, "results_FARMS.npz")
        config_path = args.config or os.path.join(latest, "farms_config.yaml")
        print(f"Using latest run: {os.path.basename(latest)}")

    if not os.path.exists(config_path):
        config_path = DEFAULT_CONFIG
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    # ── Load ───────────────────────────────────────────────────────────
    print(f"Data:   {data_path}")
    print(f"Config: {config_path}")

    d          = np.load(data_path)
    times_all  = d["time"]
    # FARMSRecorder saves as body_jnt_pos / leg_jnt_pos
    body_q_all = d["body_jnt_pos"]   # (T, 19)
    leg_q_all  = d["leg_jnt_pos"]    # (T, 19, 2, 4)
    com_pos_all = d["com_pos"]        # (T, 3)
    com_vel_all = d["com_vel"]        # (T, 3)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    bw = config["body_wave"]
    print(f"  {len(times_all)} frames  "
          f"t=[{times_all[0]:.2f}, {times_all[-1]:.2f}]s")
    print(f"  Wave: n_wave={bw['wave_number']}, "
          f"A_body={bw['amplitude']}, f={bw['frequency']} Hz")

    # ── Warmup skip ────────────────────────────────────────────────────
    warmup = 0.0 if args.no_warmup_skip else args.warmup
    mask   = times_all >= warmup
    if warmup > 0 and mask.sum() < 10:
        print(f"  WARNING: <10 frames after warmup skip — using all frames.")
        mask = np.ones(len(times_all), dtype=bool)

    times    = times_all[mask]
    body_q   = body_q_all[mask]
    leg_q    = leg_q_all[mask]
    com_pos  = com_pos_all[mask]
    com_vel  = com_vel_all[mask]

    if warmup > 0:
        print(f"  Skipping first {warmup}s ({(~mask).sum()} frames)")

    # ── Commanded signals ──────────────────────────────────────────────
    body_q_cmd, body_q_dot_cmd = reconstruct_body_commands(times, config)
    leg_q_cmd,  leg_q_dot_cmd  = reconstruct_leg_commands(times, config)

    # ── Actual velocities (finite diff — no velocity sensors in recorder) ──
    show_vel    = not args.no_velocity
    body_q_dot  = _finite_diff_vel(body_q, times)
    leg_q_dot   = _finite_diff_vel(leg_q,  times)

    # ── RMS summary ────────────────────────────────────────────────────
    body_rms = _rms(body_q, body_q_cmd)
    leg_rms  = _rms(leg_q,  leg_q_cmd)
    print(f"\n  Body position RMS : {body_rms:.4f} rad ({np.degrees(body_rms):.2f}°)")
    print(f"  Leg  position RMS : {leg_rms:.4f} rad ({np.degrees(leg_rms):.2f}°)")

    for dof in range(N_LEG_DOF):
        r = _rms(leg_q[:,:,:,dof], leg_q_cmd[:,:,:,dof])
        print(f"    {DOF_NAMES[dof]}: {r:.4f} rad ({np.degrees(r):.2f}°)")

    r_L = _rms(leg_q[:,:,0,1], leg_q_cmd[:,:,0,1])
    r_R = _rms(leg_q[:,:,1,1], leg_q_cmd[:,:,1,1])
    asym = "★ asymmetry" if abs(r_R - r_L) > 0.03 else "symmetric"
    print(f"\n  DOF1 Left  RMS: {np.degrees(r_L):.2f}°")
    print(f"  DOF1 Right RMS: {np.degrees(r_R):.2f}°  ({asym})")

    mean_vx = np.mean(com_vel[:, 0])
    print(f"\n  Mean forward velocity: {mean_vx:.4f} m/s")

    # ── Selections ─────────────────────────────────────────────────────
    joint_numbers = args.joints or [1, 5, 10, 15, 19]
    leg_indices   = args.legs   or [0, 9, 18]

    # ── Auto-save: always save to timestamped folder under 2-Analysis/ ──
    import matplotlib
    if args.save is None:
        run_tag   = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        out_dir   = os.path.join(SCRIPT_DIR, f"{run_tag}_FARMS")
        os.makedirs(out_dir, exist_ok=True)
        args.save = os.path.join(out_dir, "tracking.png")
        print(f"\n  Saving all figures to: {out_dir}\\")

    # ── Plots ──────────────────────────────────────────────────────────
    print(f"\nPlotting body joints: {joint_numbers}")
    plot_body_tracking(times, body_q_cmd, body_q_dot_cmd,
                       body_q, body_q_dot,
                       joint_numbers, show_vel, args.save)
    plt.close("all")

    print(f"Plotting legs: {leg_indices}")
    plot_leg_tracking(times, leg_q_cmd, leg_q_dot_cmd,
                      leg_q, leg_q_dot,
                      leg_indices, show_vel, args.save)
    plt.close("all")

    if not args.no_com:
        print("Plotting COM trajectory...")
        plot_com_trajectory(times, com_pos, com_vel, args.save)
        plt.close("all")

    if not args.no_summary:
        print("Plotting error summary...")
        plot_error_summary(times, body_q_cmd, body_q,
                           leg_q_cmd, leg_q, args.save)
        plt.close("all")

    print("\nDone.")


if __name__ == "__main__":
    main()