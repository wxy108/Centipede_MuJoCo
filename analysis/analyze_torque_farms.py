"""
analyze_torque_farms.py — Torque analysis for the FARMS centipede
=================================================================
Reads results_FARMS.npz (which now contains torque data) and produces:

  1. Body torque time series — commanded vs actual per joint
  2. Body torque heatmap — all 19 joints over time (spatial-temporal view)
  3. Torque statistics — RMS, peak, mean for body and legs
  4. Torque distribution histogram
  5. Per-joint RMS bar chart
  6. Leg torque summary — active DOFs only

The "commanded torque" for body yaw is the impedance output:
    tau = kp * (target - q) - kv * qdot
written to data.ctrl[body_act_id].

The "actual torque" is data.actuator_force[body_act_id], which is what
MuJoCo actually applied after constraint solving.

For leg position actuators, "commanded" is the position target in
data.ctrl and "actual" is the actuator force MuJoCo computed.

Usage:
    python analyze_torque_farms.py                         # latest run
    python analyze_torque_farms.py --run run_04_03_2026_092759
    python analyze_torque_farms.py --data path/to/results_FARMS.npz
    python analyze_torque_farms.py --joints 1 5 10 15 19   # specific body joints
    python analyze_torque_farms.py --save output.png
    python analyze_torque_farms.py --no-warmup-skip
"""

import argparse
import glob
import os
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.join(SCRIPT_DIR, "..")
DATA_DIR   = os.path.join(BASE_DIR, "outputs", "data")

WARMUP_DEFAULT = 0.5   # seconds

# Model constants
N_BODY_JOINTS = 19
N_LEGS        = 19
N_LEG_DOF     = 4
DOF_NAMES     = ["Yaw (DOF0)", "Elevation (DOF1)", "Tibia (DOF2)", "Tarsus (DOF3)"]
ACTIVE_DOFS   = (0, 1)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _rms(a):
    return np.sqrt(np.mean(a ** 2))


def _save_name(save_path, suffix):
    base = save_path.replace(".png", "")
    if not base.endswith("_FARMS"):
        base += "_FARMS"
    return base + suffix + ".png"


# ═══════════════════════════════════════════════════════════════════════
# Plot 1: Body torque time series
# ═══════════════════════════════════════════════════════════════════════

def plot_body_torque_timeseries(times, cmd, act, joint_numbers, save_path=None):
    """
    cmd, act : (T, 19) — commanded and actual body torque
    joint_numbers : list of 1-based body joint numbers
    """
    indices = [j - 1 for j in joint_numbers]
    n = len(indices)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True, squeeze=False)

    for row, ji in enumerate(indices):
        ax = axes[row, 0]
        ax.plot(times, cmd[:, ji], "b-", lw=0.8, alpha=0.8, label="Commanded")
        ax.plot(times, act[:, ji], "r-", lw=0.8, alpha=0.7, label="Actual")
        err = cmd[:, ji] - act[:, ji]
        rms_err = _rms(err)
        rms_cmd = _rms(cmd[:, ji])
        ax.set_title(f"joint_body_{ji+1}  |  cmd RMS={rms_cmd:.5f}  "
                     f"err RMS={rms_err:.6f}  peak cmd={np.max(np.abs(cmd[:,ji])):.5f}",
                     fontsize=9)
        ax.set_ylabel("Torque (N·m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle("Body Yaw Torque — Commanded (impedance) vs Actual (MuJoCo)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        p = _save_name(save_path, "_body_torque")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Plot 2: Torque heatmap (spatial-temporal)
# ═══════════════════════════════════════════════════════════════════════

def plot_torque_heatmap(times, cmd, act, save_path=None):
    """
    cmd, act : (T, 19) — commanded and actual body torque
    Shows heatmap of commanded torque across all 19 joints over time.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    vmax = max(np.max(np.abs(cmd)), np.max(np.abs(act)))

    ax = axes[0]
    im = ax.imshow(cmd.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[times[0], times[-1], N_BODY_JOINTS + 0.5, 0.5],
                   interpolation="nearest")
    ax.set_ylabel("Body joint #")
    ax.set_title("Commanded torque (impedance output)")
    plt.colorbar(im, ax=ax, label="Torque (N·m)")

    ax = axes[1]
    im = ax.imshow(act.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[times[0], times[-1], N_BODY_JOINTS + 0.5, 0.5],
                   interpolation="nearest")
    ax.set_ylabel("Body joint #")
    ax.set_xlabel("Time (s)")
    ax.set_title("Actual torque (MuJoCo applied)")
    plt.colorbar(im, ax=ax, label="Torque (N·m)")

    fig.suptitle("Body Yaw Torque Heatmap — Spatial-Temporal View",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        p = _save_name(save_path, "_torque_heatmap")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Plot 3: Per-joint RMS bar chart
# ═══════════════════════════════════════════════════════════════════════

def plot_per_joint_rms(cmd, act, save_path=None):
    """Bar chart of RMS commanded and actual torque per body joint."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, N_BODY_JOINTS + 1)
    w = 0.35

    rms_cmd = np.array([_rms(cmd[:, j]) for j in range(N_BODY_JOINTS)])
    rms_act = np.array([_rms(act[:, j]) for j in range(N_BODY_JOINTS)])

    ax.bar(x - w/2, rms_cmd, w, label="Commanded", color="#2196F3", alpha=0.8)
    ax.bar(x + w/2, rms_act, w, label="Actual",    color="#FF5722", alpha=0.8)

    ax.set_xlabel("Body joint #")
    ax.set_ylabel("RMS torque (N·m)")
    ax.set_title("Per-Joint RMS Torque — Commanded vs Actual")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        p = _save_name(save_path, "_torque_rms_bar")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Plot 4: Torque distribution histogram
# ═══════════════════════════════════════════════════════════════════════

def plot_torque_distribution(cmd, act, save_path=None):
    """Histogram of all body commanded vs actual torque values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(cmd.ravel(), bins=80, color="#2196F3", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    rms = _rms(cmd.ravel())
    ax.set_title(f"Commanded torque distribution  |  RMS={rms:.5f}")
    ax.set_xlabel("Torque (N·m)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    err = (cmd - act).ravel()
    ax.hist(err, bins=80, color="#FF9800", alpha=0.7, edgecolor="none")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    rms_e = _rms(err)
    ax.set_title(f"Torque error (cmd − actual)  |  RMS={rms_e:.6f}")
    ax.set_xlabel("Error (N·m)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Body Torque Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        p = _save_name(save_path, "_torque_dist")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Plot 5: Leg torque summary
# ═══════════════════════════════════════════════════════════════════════

def plot_leg_torque_summary(leg_act, save_path=None):
    """
    leg_act : (T, 19, 2, 4) — actual leg actuator forces
    Bar chart: RMS actual torque per DOF, averaged across legs and sides.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-DOF average across legs and sides
    ax = axes[0]
    dof_rms = []
    for dof in range(N_LEG_DOF):
        vals = leg_act[:, :, :, dof].ravel()
        dof_rms.append(_rms(vals))
    colors = ["#2196F3" if d in ACTIVE_DOFS else "#BBBBBB" for d in range(N_LEG_DOF)]
    ax.bar(range(N_LEG_DOF), dof_rms, color=colors, alpha=0.8, edgecolor="gray")
    ax.set_xticks(range(N_LEG_DOF))
    ax.set_xticklabels(DOF_NAMES, fontsize=8)
    ax.set_ylabel("RMS actuator force")
    ax.set_title("Leg actuator RMS by DOF (blue = active)")
    ax.grid(True, alpha=0.3, axis="y")

    # Per-leg RMS (active DOFs only)
    ax = axes[1]
    x = np.arange(N_LEGS)
    for dof in ACTIVE_DOFS:
        rms_per_leg = []
        for n in range(N_LEGS):
            vals = leg_act[:, n, :, dof].ravel()
            rms_per_leg.append(_rms(vals))
        ax.plot(x, rms_per_leg, "o-", ms=4, label=DOF_NAMES[dof])
    ax.set_xlabel("Leg index (0-based)")
    ax.set_ylabel("RMS actuator force")
    ax.set_title("Per-leg RMS (active DOFs)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Leg Torque Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        p = _save_name(save_path, "_leg_torque")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Statistics printout
# ═══════════════════════════════════════════════════════════════════════

def print_torque_stats(body_cmd, body_act, leg_cmd, leg_act):
    """Print comprehensive torque statistics."""
    print("\n" + "=" * 65)
    print("  TORQUE STATISTICS")
    print("=" * 65)

    rms_cmd  = _rms(body_cmd.ravel())
    rms_act  = _rms(body_act.ravel())
    rms_err  = _rms((body_cmd - body_act).ravel())
    peak_cmd = np.max(np.abs(body_cmd))
    peak_act = np.max(np.abs(body_act))
    mean_abs = np.mean(np.abs(body_cmd))

    print(f"\n  Body yaw torque (impedance, 19 joints):")
    print(f"    Commanded RMS  : {rms_cmd:.6f} N·m")
    print(f"    Actual RMS     : {rms_act:.6f} N·m")
    print(f"    Error RMS      : {rms_err:.6f} N·m  ({rms_err/rms_cmd*100:.1f}% of cmd)")
    print(f"    Peak commanded : {peak_cmd:.6f} N·m")
    print(f"    Peak actual    : {peak_act:.6f} N·m")
    print(f"    Mean |cmd|     : {mean_abs:.6f} N·m")

    # Per-joint breakdown
    print(f"\n  Per-joint commanded torque RMS:")
    for j in range(N_BODY_JOINTS):
        r = _rms(body_cmd[:, j])
        print(f"    joint_body_{j+1:2d}: {r:.6f} N·m")

    # Leg torque
    print(f"\n  Leg actuator force (position control, 19 legs × 2 sides × 4 DOF):")
    for dof in range(N_LEG_DOF):
        r = _rms(leg_act[:, :, :, dof].ravel())
        tag = " ← active" if dof in ACTIVE_DOFS else "   passive"
        print(f"    {DOF_NAMES[dof]:>20s}: RMS={r:.6f}{tag}")

    total_body = np.sum(np.abs(body_cmd))
    total_leg  = np.sum(np.abs(leg_act[:, :, :, list(ACTIVE_DOFS)]))
    print(f"\n  Total effort (sum |torque|):")
    print(f"    Body: {total_body:.3f}")
    print(f"    Legs: {total_leg:.3f}")
    print(f"    Ratio body/legs: {total_body/max(total_leg, 1e-12):.2f}")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def find_latest_run():
    runs = sorted(glob.glob(os.path.join(DATA_DIR, "run_*")))
    return runs[-1] if runs else None


def main():
    p = argparse.ArgumentParser(
        description="Analyze torque data from FARMS centipede simulation")
    p.add_argument("--data",           default=None,
                   help="Direct path to results_FARMS.npz")
    p.add_argument("--run",            default=None,
                   help="Run folder name e.g. run_04_03_2026_092759")
    p.add_argument("--joints",         nargs="+", type=int, default=None,
                   help="Body joint numbers to plot (1-based, default: 1 5 10 15 19)")
    p.add_argument("--save",           default=None,
                   help="Output prefix e.g. torque.png → torque_body_torque_FARMS.png")
    p.add_argument("--no-warmup-skip", action="store_true")
    p.add_argument("--warmup",         type=float, default=WARMUP_DEFAULT)
    p.add_argument("--no-heatmap",     action="store_true")
    p.add_argument("--no-legs",        action="store_true")
    args = p.parse_args()

    # ── Resolve path ──────────────────────────────────────────────────
    if args.data:
        data_path = args.data
    elif args.run:
        data_path = os.path.join(DATA_DIR, args.run, "results_FARMS.npz")
    else:
        latest = find_latest_run()
        if latest is None:
            print(f"ERROR: No run folders found in {DATA_DIR}"); sys.exit(1)
        data_path = os.path.join(latest, "results_FARMS.npz")
        print(f"Using latest run: {os.path.basename(latest)}")

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found."); sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────
    print(f"Data: {data_path}")
    d = np.load(data_path)

    # Check if torque data exists
    if "body_cmd_torque" not in d:
        print("\nERROR: This NPZ file does not contain torque data.")
        print("You need to re-run the simulation with the updated recorder.")
        print("  python controllers/farms/run.py --video output.mp4 --duration 5 --headless")
        print("\nThe updated FARMSRecorder now saves:")
        print("  body_cmd_torque, body_act_torque, leg_cmd_torque, leg_act_torque")
        sys.exit(1)

    times    = d["time"]
    body_cmd = d["body_cmd_torque"]   # (T, 19)
    body_act = d["body_act_torque"]   # (T, 19)
    leg_cmd  = d["leg_cmd_torque"]    # (T, 19, 2, 4)
    leg_act  = d["leg_act_torque"]    # (T, 19, 2, 4)

    print(f"  {len(times)} frames  t=[{times[0]:.2f}, {times[-1]:.2f}]s")

    # ── Warmup skip ───────────────────────────────────────────────────
    warmup = 0.0 if args.no_warmup_skip else args.warmup
    mask   = times >= warmup
    if warmup > 0 and mask.sum() < 10:
        print(f"  WARNING: <10 frames after warmup skip — using all frames.")
        mask = np.ones(len(times), dtype=bool)

    times    = times[mask]
    body_cmd = body_cmd[mask]
    body_act = body_act[mask]
    leg_cmd  = leg_cmd[mask]
    leg_act  = leg_act[mask]

    if warmup > 0:
        print(f"  Skipping first {warmup}s ({(~mask).sum()} frames)")

    # ── Statistics ────────────────────────────────────────────────────
    print_torque_stats(body_cmd, body_act, leg_cmd, leg_act)

    # ── Auto-save directory ──────────────────────────────────────────
    if args.save is None:
        run_tag = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        out_dir = os.path.join(SCRIPT_DIR, f"{run_tag}_torque_FARMS")
        os.makedirs(out_dir, exist_ok=True)
        args.save = os.path.join(out_dir, "torque.png")
        print(f"\n  Saving all figures to: {out_dir}")

    # ── Plots ─────────────────────────────────────────────────────────
    joint_numbers = args.joints or [1, 5, 10, 15, 19]

    print(f"\nPlotting body torque time series for joints: {joint_numbers}")
    plot_body_torque_timeseries(times, body_cmd, body_act, joint_numbers, args.save)
    plt.close("all")

    if not args.no_heatmap:
        print("Plotting torque heatmap...")
        plot_torque_heatmap(times, body_cmd, body_act, args.save)
        plt.close("all")

    print("Plotting per-joint RMS bar chart...")
    plot_per_joint_rms(body_cmd, body_act, args.save)
    plt.close("all")

    print("Plotting torque distribution...")
    plot_torque_distribution(body_cmd, body_act, args.save)
    plt.close("all")

    if not args.no_legs:
        print("Plotting leg torque summary...")
        plot_leg_torque_summary(leg_act, args.save)
        plt.close("all")

    print("\nDone.")


if __name__ == "__main__":
    main()
