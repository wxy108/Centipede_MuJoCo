#!/usr/bin/env python3
"""
plot_bode.py — Bode-plot analysis of wavelength sweep results.

Reads a wavelength sweep results.json and produces publication-quality plots:

  1. Magnitude plot  — CoT vs wavelength (log-x)
  2. Phase plot      — terrain-slope → body-pitch phase lag vs wavelength
  3. Speed plot      — forward speed vs wavelength
  4. Pitch/Roll plot — max pitch & roll angles vs wavelength
  5. Combined Bode   — magnitude + phase stacked (the main figure)

Vertical dashed lines mark morphology reference wavelengths:
  L_w (world), L_b (body), L_s (segment), L_ell (leg)

Usage
-----
  python analysis/wavelength_sweep/plot_bode.py                    # auto-find latest sweep
  python analysis/wavelength_sweep/plot_bode.py --sweep-dir outputs/wavelength_sweep/sweep_20260408_163221
  python analysis/wavelength_sweep/plot_bode.py --all              # plot all sweeps found

Output
------
  outputs/figures/wavelength_sweep/  (PNG + PDF for each figure)
"""

import argparse
import json
import math
import os
import sys
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
SWEEP_DIR    = os.path.join(PROJECT_ROOT, "outputs", "wavelength_sweep")
FIG_DIR      = os.path.join(PROJECT_ROOT, "outputs", "figures", "wavelength_sweep")

# ── Style ─────────────────────────────────────────────────────────────────────
MORPH_COLORS = {
    "L_w":   "#888888",
    "L_b":   "#2196F3",
    "L_s":   "#FF9800",
    "L_ell": "#4CAF50",
}
MORPH_LABELS = {
    "L_w":   "$L_w$ (world)",
    "L_b":   "$L_b$ (body)",
    "L_s":   "$L_s$ (segment)",
    "L_ell": "$L_{\\ell}$ (leg)",
}


def load_sweep(sweep_dir):
    """Load results.json from a sweep directory."""
    path = os.path.join(sweep_dir, "results.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_latest_sweep():
    """Find the most recently created sweep directory."""
    dirs = sorted(glob(os.path.join(SWEEP_DIR, "sweep_*")))
    if not dirs:
        print(f"ERROR: No sweep directories found in {SWEEP_DIR}")
        sys.exit(1)
    return dirs[-1]


def _add_morphology_lines(ax, morphology, y_range=None):
    """Draw vertical dashed lines at morphology wavelengths."""
    for key in ["L_w", "L_b", "L_s", "L_ell"]:
        val_mm = morphology[key] * 1000
        ax.axvline(val_mm, color=MORPH_COLORS[key], linestyle="--",
                   linewidth=1.2, alpha=0.7, label=MORPH_LABELS[key])


def _configure_log_x(ax, label="Wavelength (mm)"):
    """Configure log-scale x-axis with nice formatting."""
    ax.set_xscale("log")
    ax.set_xlabel(label, fontsize=11)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="both", labelsize=10)
    ax.invert_xaxis()  # long wavelengths (low freq) on left, short on right


def _save_fig(fig, name, out_dir):
    """Save figure as both PNG and PDF."""
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {os.path.join(out_dir, name)}.{{png,pdf}}")


# ═══════════════════════════════════════════════════════════════════════════════
# Individual plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cot(results, morphology, out_dir, tag):
    """CoT magnitude plot."""
    survived = [r for r in results if r["survived"]]
    if not survived:
        print("  No surviving trials — skipping CoT plot")
        return

    wl = [r["wavelength_mm"] for r in survived]
    cot = [r["cot"] for r in survived]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(wl, cot, "o-", color="#E53935", markersize=6, linewidth=1.8,
            label="CoT", zorder=5)
    _add_morphology_lines(ax, morphology)
    _configure_log_x(ax)
    ax.set_ylabel("Cost of Transport", fontsize=11)
    ax.set_title("Magnitude: CoT vs Terrain Wavelength", fontsize=13)
    ax.legend(fontsize=9, loc="best", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    _save_fig(fig, f"cot_vs_wavelength_{tag}", out_dir)
    plt.close(fig)


def plot_phase(results, morphology, out_dir, tag):
    """Phase lag plot."""
    survived = [r for r in results if r["survived"]]
    valid = [r for r in survived
             if not math.isnan(r.get("phase_lag_deg", float("nan")))]
    if not valid:
        print("  No valid phase data — skipping phase plot")
        return

    wl = [r["wavelength_mm"] for r in valid]
    phase = [r["phase_lag_deg"] for r in valid]
    coherence = [r["phase_coherence"] for r in valid]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))

    # Phase lag
    ax1.plot(wl, phase, "s-", color="#1565C0", markersize=6, linewidth=1.8,
             label="Phase lag", zorder=5)
    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax1.axhline(90, color="red", linewidth=0.8, alpha=0.4, linestyle=":")
    ax1.axhline(-90, color="red", linewidth=0.8, alpha=0.4, linestyle=":")
    _add_morphology_lines(ax1, morphology)
    _configure_log_x(ax1)
    ax1.set_ylabel("Phase lag (degrees)", fontsize=11, color="#1565C0")
    ax1.set_title("Phase: Terrain Slope → Body Pitch Phase Lag", fontsize=13)

    # Coherence on secondary axis
    ax2 = ax1.twinx()
    ax2.fill_between(wl, coherence, alpha=0.15, color="#4CAF50")
    ax2.plot(wl, coherence, "^", color="#4CAF50", markersize=4, alpha=0.6,
             label="Coherence")
    ax2.set_ylabel("Coherence ($\\gamma^2$)", fontsize=10, color="#4CAF50")
    ax2.set_ylim(-0.05, 1.15)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best", ncol=2)
    ax1.grid(True, alpha=0.3, which="both")
    _save_fig(fig, f"phase_vs_wavelength_{tag}", out_dir)
    plt.close(fig)


def plot_speed(results, morphology, out_dir, tag):
    """Forward speed plot."""
    survived = [r for r in results if r["survived"]]
    if not survived:
        return

    wl = [r["wavelength_mm"] for r in survived]
    speed = [r["forward_speed"] * 1000 for r in survived]  # mm/s

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(wl, speed, "D-", color="#7B1FA2", markersize=6, linewidth=1.8,
            label="Forward speed", zorder=5)
    _add_morphology_lines(ax, morphology)
    _configure_log_x(ax)
    ax.set_ylabel("Forward speed (mm/s)", fontsize=11)
    ax.set_title("Speed vs Terrain Wavelength", fontsize=13)
    ax.legend(fontsize=9, loc="best", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    _save_fig(fig, f"speed_vs_wavelength_{tag}", out_dir)
    plt.close(fig)


def plot_pitch_roll(results, morphology, out_dir, tag):
    """Max pitch and roll angles."""
    survived = [r for r in results if r["survived"]]
    if not survived:
        return

    wl = [r["wavelength_mm"] for r in survived]
    pitch = [r["max_pitch_deg"] for r in survived]
    roll = [r["max_roll_deg"] for r in survived]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(wl, pitch, "o-", color="#E53935", markersize=5, linewidth=1.5,
            label="Max pitch")
    ax.plot(wl, roll, "s-", color="#1565C0", markersize=5, linewidth=1.5,
            label="Max roll")
    _add_morphology_lines(ax, morphology)
    _configure_log_x(ax)
    ax.set_ylabel("Angle (degrees)", fontsize=11)
    ax.set_title("Body Pitch & Roll vs Terrain Wavelength", fontsize=13)
    ax.legend(fontsize=9, loc="best", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    _save_fig(fig, f"pitch_roll_vs_wavelength_{tag}", out_dir)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Combined Bode plot (the main figure)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bode_combined(results, morphology, out_dir, tag, meta=None):
    """
    Stacked Bode plot: magnitude (CoT) on top, phase on bottom.
    Shared log-x axis.
    """
    survived = [r for r in results if r["survived"]]
    if not survived:
        print("  No surviving trials — skipping Bode plot")
        return

    wl = np.array([r["wavelength_mm"] for r in survived])
    cot = np.array([r["cot"] for r in survived])
    speed = np.array([r["forward_speed"] * 1000 for r in survived])

    # Phase (may have NaN)
    has_phase = any(
        not math.isnan(r.get("phase_lag_deg", float("nan")))
        for r in survived
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Centipede Frequency Response (Bode Plot)", fontsize=14, y=0.98)

    # ── Top: CoT magnitude ────────────────────────────────────────────────
    ax_cot = axes[0]
    ax_cot.plot(wl, cot, "o-", color="#E53935", markersize=6, linewidth=2,
                label="CoT", zorder=5)
    _add_morphology_lines(ax_cot, morphology)
    ax_cot.set_ylabel("Cost of Transport", fontsize=11)
    ax_cot.set_title("Magnitude", fontsize=12, loc="left", pad=4)
    ax_cot.legend(fontsize=8, loc="upper right", ncol=3)
    ax_cot.grid(True, alpha=0.3, which="both")

    # ── Middle: Speed ─────────────────────────────────────────────────────
    ax_spd = axes[1]
    ax_spd.plot(wl, speed, "D-", color="#7B1FA2", markersize=5, linewidth=1.8,
                label="Speed", zorder=5)
    _add_morphology_lines(ax_spd, morphology)
    ax_spd.set_ylabel("Speed (mm/s)", fontsize=11)
    ax_spd.set_title("Forward Speed", fontsize=12, loc="left", pad=4)
    ax_spd.legend(fontsize=8, loc="upper right", ncol=3)
    ax_spd.grid(True, alpha=0.3, which="both")

    # ── Bottom: Phase ─────────────────────────────────────────────────────
    ax_ph = axes[2]
    if has_phase:
        valid = [r for r in survived
                 if not math.isnan(r.get("phase_lag_deg", float("nan")))]
        wl_ph = [r["wavelength_mm"] for r in valid]
        phase = [r["phase_lag_deg"] for r in valid]
        ax_ph.plot(wl_ph, phase, "s-", color="#1565C0", markersize=6,
                   linewidth=2, label="Phase lag", zorder=5)
        ax_ph.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax_ph.axhline(90, color="red", linewidth=0.8, alpha=0.3, linestyle=":")
        ax_ph.axhline(-90, color="red", linewidth=0.8, alpha=0.3, linestyle=":")
    _add_morphology_lines(ax_ph, morphology)
    ax_ph.set_ylabel("Phase lag (deg)", fontsize=11)
    ax_ph.set_title("Phase: terrain slope → body pitch", fontsize=12,
                     loc="left", pad=4)
    ax_ph.legend(fontsize=8, loc="upper right", ncol=3)
    ax_ph.grid(True, alpha=0.3, which="both")

    # Shared x-axis config
    ax_ph.set_xscale("log")
    ax_ph.set_xlabel("Terrain Wavelength (mm)  [long → short]", fontsize=11)
    ax_ph.xaxis.set_major_formatter(ScalarFormatter())
    ax_ph.invert_xaxis()

    # Annotation: sweep parameters
    if meta:
        info = (f"n={meta.get('n_points','?')} pts, "
                f"dur={meta.get('duration','?')}s, "
                f"amp={meta.get('amplitude',0)*1000:.1f}mm")
        fig.text(0.99, 0.01, info, fontsize=8, ha="right", va="bottom",
                 color="gray", style="italic")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    _save_fig(fig, f"bode_combined_{tag}", out_dir)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_sweep(sweep_dir, out_dir):
    """Run all analyses on one sweep directory."""
    data = load_sweep(sweep_dir)
    tag = data["timestamp"]
    morphology = data["morphology"]
    results = data["results"]

    print(f"\nAnalyzing sweep: {sweep_dir}")
    print(f"  {len(results)} wavelengths, "
          f"{sum(1 for r in results if r['survived'])} survived")

    sweep_fig_dir = out_dir

    plot_cot(results, morphology, sweep_fig_dir, tag)
    plot_phase(results, morphology, sweep_fig_dir, tag)
    plot_speed(results, morphology, sweep_fig_dir, tag)
    plot_pitch_roll(results, morphology, sweep_fig_dir, tag)
    plot_bode_combined(results, morphology, sweep_fig_dir, tag, meta=data)

    print(f"  All figures saved to: {sweep_fig_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Bode-plot analysis from wavelength sweep results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sweep-dir", default=None,
                        help="Path to a specific sweep_* directory. "
                             "Default: auto-detect latest.")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all sweep directories found.")
    parser.add_argument("--out-dir", default=FIG_DIR,
                        help=f"Output directory for figures. Default: {FIG_DIR}")
    args = parser.parse_args()

    if args.all:
        dirs = sorted(glob(os.path.join(SWEEP_DIR, "sweep_*")))
        if not dirs:
            print(f"No sweep directories found in {SWEEP_DIR}")
            sys.exit(1)
        for d in dirs:
            analyze_sweep(d, args.out_dir)
    else:
        sweep_dir = args.sweep_dir or find_latest_sweep()
        if not os.path.isabs(sweep_dir):
            sweep_dir = os.path.join(PROJECT_ROOT, sweep_dir)
        analyze_sweep(sweep_dir, args.out_dir)

    print(f"\nDone. Figures in: {args.out_dir}")


if __name__ == "__main__":
    main()
