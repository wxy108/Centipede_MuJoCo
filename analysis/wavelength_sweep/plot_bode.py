#!/usr/bin/env python3
"""
plot_bode.py — Bode-plot analysis of wavelength sweep results.

Handles both single-trial and batch (multi-trial) sweep formats.
For batch data: plots mean +/- std error bands and individual trial scatter.

Figures produced:
  1. bode_combined   — CoT + Speed + Phase stacked (main figure)
  2. cot_vs_wavelength
  3. phase_vs_wavelength
  4. speed_vs_wavelength
  5. pitch_roll_vs_wavelength

Usage
-----
  python analysis/wavelength_sweep/plot_bode.py                    # latest sweep
  python analysis/wavelength_sweep/plot_bode.py --sweep-dir outputs/wavelength_sweep/sweep_XXX
  python analysis/wavelength_sweep/plot_bode.py --all

Output: outputs/figures/wavelength_sweep/  (PNG + PDF)
"""

import argparse
import json
import math
import os
import sys
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
SWEEP_DIR    = os.path.join(PROJECT_ROOT, "outputs", "wavelength_sweep")
FIG_DIR      = os.path.join(PROJECT_ROOT, "outputs", "figures", "wavelength_sweep")

# ── Style ─────────────────────────────────────────────────────────────────────
MORPH_COLORS = {
    "L_w": "#888888", "L_b": "#2196F3", "L_s": "#FF9800", "L_ell": "#4CAF50",
}
MORPH_LABELS = {
    "L_w": "$L_w$ (world)", "L_b": "$L_b$ (body)",
    "L_s": "$L_s$ (segment)", "L_ell": "$L_{\\ell}$ (leg)",
}


def load_sweep(sweep_dir):
    path = os.path.join(sweep_dir, "results.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_latest_sweep():
    dirs = sorted(glob(os.path.join(SWEEP_DIR, "sweep_*")))
    if not dirs:
        print(f"ERROR: No sweep directories found in {SWEEP_DIR}")
        sys.exit(1)
    return dirs[-1]


def is_batch(data):
    """Check if this is a batch (multi-trial) sweep."""
    return 'wavelength_results' in data


def _add_morphology_lines(ax, morphology):
    for key in ["L_w", "L_b", "L_s", "L_ell"]:
        val_mm = morphology[key] * 1000
        ax.axvline(val_mm, color=MORPH_COLORS[key], linestyle="--",
                   linewidth=1.2, alpha=0.7, label=MORPH_LABELS[key])


def _configure_log_x(ax, label="Wavelength (mm)"):
    ax.set_xscale("log")
    ax.set_xlabel(label, fontsize=11)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="both", labelsize=10)
    ax.invert_xaxis()


def _save_fig(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {os.path.join(out_dir, name)}.{{png,pdf}}")


# ═══════════════════════════════════════════════════════════════════════════════
# Extract data (handles both single and batch format)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_agg(data):
    """
    Extract aggregated per-wavelength data.
    Returns list of dicts with: wavelength_mm, cot_mean, cot_std, speed_mean, ...
    For single-trial data, std = 0.
    """
    if is_batch(data):
        return data['wavelength_results']
    else:
        # Old single-trial format: wrap each result as an "aggregated" entry
        out = []
        for r in data['results']:
            if not r.get('survived', True):
                continue
            out.append({
                'wavelength_mm': r['wavelength_mm'],
                'frequency': r['frequency'],
                'n_survived': 1, 'n_trials': 1, 'survival_rate': 1.0,
                'cot_mean': r['cot'], 'cot_std': 0,
                'cot_median': r['cot'], 'cot_min': r['cot'], 'cot_max': r['cot'],
                'speed_mean': r['forward_speed'], 'speed_std': 0,
                'max_pitch_mean': r['max_pitch_deg'], 'max_pitch_std': 0,
                'max_roll_mean': r['max_roll_deg'], 'max_roll_std': 0,
                'phase_lag_mean': r.get('phase_lag_deg', float('nan')),
                'phase_lag_std': 0,
            })
        return out


def extract_trials(data):
    """Extract raw per-trial data (batch only). Returns list of dicts or None."""
    if is_batch(data):
        return data.get('all_trials', [])
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Individual plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cot(agg, trials, morphology, out_dir, tag):
    valid = [r for r in agg if r.get('n_survived', 1) > 0]
    if not valid:
        return

    wl = [r['wavelength_mm'] for r in valid]
    cot_mean = [r['cot_mean'] for r in valid]
    cot_std = [r.get('cot_std', 0) for r in valid]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Scatter individual trials if available
    if trials:
        survived_trials = [t for t in trials if t.get('survived', True)]
        t_wl = [t['wavelength_mm'] for t in survived_trials]
        t_cot = [t['cot'] for t in survived_trials]
        ax.scatter(t_wl, t_cot, s=12, alpha=0.25, color="#E53935",
                   zorder=3, label="Individual trials")

    # Mean + error band
    ax.plot(wl, cot_mean, "o-", color="#E53935", markersize=7, linewidth=2,
            label="Mean CoT", zorder=5)
    if any(s > 0 for s in cot_std):
        cot_lo = [m - s for m, s in zip(cot_mean, cot_std)]
        cot_hi = [m + s for m, s in zip(cot_mean, cot_std)]
        ax.fill_between(wl, cot_lo, cot_hi, alpha=0.15, color="#E53935",
                        label="$\\pm 1\\sigma$")

    _add_morphology_lines(ax, morphology)
    _configure_log_x(ax)
    ax.set_ylabel("Cost of Transport", fontsize=11)
    ax.set_title("Magnitude: CoT vs Terrain Wavelength", fontsize=13)
    ax.legend(fontsize=9, loc="best", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    _save_fig(fig, f"cot_vs_wavelength_{tag}", out_dir)
    plt.close(fig)


def plot_phase(agg, trials, morphology, out_dir, tag):
    valid = [r for r in agg
             if r.get('n_survived', 1) > 0
             and not math.isnan(r.get('phase_lag_mean', float('nan')))]
    if not valid:
        print("  No valid phase data -- skipping")
        return

    wl = [r['wavelength_mm'] for r in valid]
    phase_mean = [r['phase_lag_mean'] for r in valid]
    phase_std = [r.get('phase_lag_std', 0) for r in valid]

    fig, ax = plt.subplots(figsize=(10, 5))

    if trials:
        valid_trials = [t for t in trials
                        if t.get('survived', True)
                        and not math.isnan(t.get('phase_lag_deg', float('nan')))]
        t_wl = [t['wavelength_mm'] for t in valid_trials]
        t_ph = [t['phase_lag_deg'] for t in valid_trials]
        ax.scatter(t_wl, t_ph, s=12, alpha=0.25, color="#1565C0", zorder=3)

    ax.plot(wl, phase_mean, "s-", color="#1565C0", markersize=7, linewidth=2,
            label="Mean phase lag", zorder=5)
    if any(s > 0 for s in phase_std):
        ph_lo = [m - s for m, s in zip(phase_mean, phase_std)]
        ph_hi = [m + s for m, s in zip(phase_mean, phase_std)]
        ax.fill_between(wl, ph_lo, ph_hi, alpha=0.15, color="#1565C0")

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.axhline(90, color="red", linewidth=0.8, alpha=0.3, linestyle=":")
    ax.axhline(-90, color="red", linewidth=0.8, alpha=0.3, linestyle=":")
    _add_morphology_lines(ax, morphology)
    _configure_log_x(ax)
    ax.set_ylabel("Phase lag (degrees)", fontsize=11)
    ax.set_title("Phase: Terrain Slope -> Body Pitch Phase Lag", fontsize=13)
    ax.legend(fontsize=9, loc="best", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    _save_fig(fig, f"phase_vs_wavelength_{tag}", out_dir)
    plt.close(fig)


def plot_speed(agg, trials, morphology, out_dir, tag):
    valid = [r for r in agg if r.get('n_survived', 1) > 0]
    if not valid:
        return

    wl = [r['wavelength_mm'] for r in valid]
    spd_mean = [r['speed_mean'] * 1000 for r in valid]
    spd_std = [r.get('speed_std', 0) * 1000 for r in valid]

    fig, ax = plt.subplots(figsize=(10, 5))

    if trials:
        survived_trials = [t for t in trials if t.get('survived', True)]
        t_wl = [t['wavelength_mm'] for t in survived_trials]
        t_spd = [t['forward_speed'] * 1000 for t in survived_trials]
        ax.scatter(t_wl, t_spd, s=12, alpha=0.25, color="#7B1FA2", zorder=3)

    ax.plot(wl, spd_mean, "D-", color="#7B1FA2", markersize=6, linewidth=1.8,
            label="Mean speed", zorder=5)
    if any(s > 0 for s in spd_std):
        lo = [m - s for m, s in zip(spd_mean, spd_std)]
        hi = [m + s for m, s in zip(spd_mean, spd_std)]
        ax.fill_between(wl, lo, hi, alpha=0.15, color="#7B1FA2")

    _add_morphology_lines(ax, morphology)
    _configure_log_x(ax)
    ax.set_ylabel("Forward speed (mm/s)", fontsize=11)
    ax.set_title("Speed vs Terrain Wavelength", fontsize=13)
    ax.legend(fontsize=9, loc="best", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    _save_fig(fig, f"speed_vs_wavelength_{tag}", out_dir)
    plt.close(fig)


def plot_pitch_roll(agg, trials, morphology, out_dir, tag):
    valid = [r for r in agg if r.get('n_survived', 1) > 0]
    if not valid:
        return

    wl = [r['wavelength_mm'] for r in valid]
    pitch_mean = [r['max_pitch_mean'] for r in valid]
    roll_mean = [r['max_roll_mean'] for r in valid]

    fig, ax = plt.subplots(figsize=(10, 5))

    if trials:
        survived_trials = [t for t in trials if t.get('survived', True)]
        t_wl = [t['wavelength_mm'] for t in survived_trials]
        ax.scatter(t_wl, [t['max_pitch_deg'] for t in survived_trials],
                   s=10, alpha=0.2, color="#E53935", zorder=3)
        ax.scatter(t_wl, [t['max_roll_deg'] for t in survived_trials],
                   s=10, alpha=0.2, color="#1565C0", zorder=3)

    ax.plot(wl, pitch_mean, "o-", color="#E53935", markersize=5, linewidth=1.5,
            label="Mean max pitch")
    ax.plot(wl, roll_mean, "s-", color="#1565C0", markersize=5, linewidth=1.5,
            label="Mean max roll")
    _add_morphology_lines(ax, morphology)
    _configure_log_x(ax)
    ax.set_ylabel("Angle (degrees)", fontsize=11)
    ax.set_title("Body Pitch & Roll vs Terrain Wavelength", fontsize=13)
    ax.legend(fontsize=9, loc="best", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    _save_fig(fig, f"pitch_roll_vs_wavelength_{tag}", out_dir)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Combined Bode plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bode_combined(agg, trials, morphology, out_dir, tag, meta=None):
    valid = [r for r in agg if r.get('n_survived', 1) > 0]
    if not valid:
        print("  No surviving trials -- skipping Bode plot")
        return

    wl = np.array([r['wavelength_mm'] for r in valid])
    cot_mean = np.array([r['cot_mean'] for r in valid])
    cot_std = np.array([r.get('cot_std', 0) for r in valid])
    spd_mean = np.array([r['speed_mean'] * 1000 for r in valid])
    spd_std = np.array([r.get('speed_std', 0) * 1000 for r in valid])

    has_phase = any(not math.isnan(r.get('phase_lag_mean', float('nan')))
                    for r in valid)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    fig.suptitle("Centipede Frequency Response (Bode Plot)", fontsize=14, y=0.98)

    # Scatter helper
    def scatter_trials(ax, key, color):
        if not trials:
            return
        survived_t = [t for t in trials if t.get('survived', True)]
        t_wl = [t['wavelength_mm'] for t in survived_t]
        t_val = [t[key] for t in survived_t]
        ax.scatter(t_wl, t_val, s=10, alpha=0.2, color=color, zorder=3)

    # ── Top: CoT ──────────────────────────────────────────────────────────
    ax = axes[0]
    scatter_trials(ax, 'cot', '#E53935')
    ax.plot(wl, cot_mean, "o-", color="#E53935", markersize=7, linewidth=2,
            label="Mean CoT", zorder=5)
    if np.any(cot_std > 0):
        ax.fill_between(wl, cot_mean - cot_std, cot_mean + cot_std,
                        alpha=0.15, color="#E53935", label="$\\pm 1\\sigma$")
    _add_morphology_lines(ax, morphology)
    ax.set_ylabel("Cost of Transport", fontsize=11)
    ax.set_title("Magnitude", fontsize=12, loc="left", pad=4)
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.3, which="both")

    # ── Middle: Speed ─────────────────────────────────────────────────────
    ax = axes[1]
    if trials:
        survived_t = [t for t in trials if t.get('survived', True)]
        ax.scatter([t['wavelength_mm'] for t in survived_t],
                   [t['forward_speed'] * 1000 for t in survived_t],
                   s=10, alpha=0.2, color="#7B1FA2", zorder=3)
    ax.plot(wl, spd_mean, "D-", color="#7B1FA2", markersize=5, linewidth=1.8,
            label="Mean speed", zorder=5)
    if np.any(spd_std > 0):
        ax.fill_between(wl, spd_mean - spd_std, spd_mean + spd_std,
                        alpha=0.15, color="#7B1FA2")
    _add_morphology_lines(ax, morphology)
    ax.set_ylabel("Speed (mm/s)", fontsize=11)
    ax.set_title("Forward Speed", fontsize=12, loc="left", pad=4)
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.3, which="both")

    # ── Bottom: Phase ─────────────────────────────────────────────────────
    ax = axes[2]
    if has_phase:
        ph_valid = [r for r in valid
                    if not math.isnan(r.get('phase_lag_mean', float('nan')))]
        wl_ph = [r['wavelength_mm'] for r in ph_valid]
        ph_mean = [r['phase_lag_mean'] for r in ph_valid]
        ph_std = [r.get('phase_lag_std', 0) for r in ph_valid]

        if trials:
            vt = [t for t in trials
                  if t.get('survived', True)
                  and not math.isnan(t.get('phase_lag_deg', float('nan')))]
            ax.scatter([t['wavelength_mm'] for t in vt],
                       [t['phase_lag_deg'] for t in vt],
                       s=10, alpha=0.2, color="#1565C0", zorder=3)

        ax.plot(wl_ph, ph_mean, "s-", color="#1565C0", markersize=7,
                linewidth=2, label="Mean phase lag", zorder=5)
        if any(s > 0 for s in ph_std):
            lo = [m - s for m, s in zip(ph_mean, ph_std)]
            hi = [m + s for m, s in zip(ph_mean, ph_std)]
            ax.fill_between(wl_ph, lo, hi, alpha=0.15, color="#1565C0")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.axhline(90, color="red", linewidth=0.8, alpha=0.3, linestyle=":")
        ax.axhline(-90, color="red", linewidth=0.8, alpha=0.3, linestyle=":")

    _add_morphology_lines(ax, morphology)
    ax.set_ylabel("Phase lag (deg)", fontsize=11)
    ax.set_title("Phase: terrain slope -> body pitch", fontsize=12,
                 loc="left", pad=4)
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.3, which="both")

    # Shared x
    ax.set_xscale("log")
    ax.set_xlabel("Terrain Wavelength (mm)  [long -> short]", fontsize=11)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.invert_xaxis()

    if meta:
        n_trials = meta.get('n_trials', 1)
        info = (f"n={meta.get('n_points','?')} wl x {n_trials} trials, "
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
    data = load_sweep(sweep_dir)
    tag = data["timestamp"]
    morphology = data["morphology"]

    agg = extract_agg(data)
    trials = extract_trials(data)

    n_wl = len(agg)
    n_trials = data.get('n_trials', 1)
    batch_str = f" x {n_trials} trials" if n_trials > 1 else ""

    print(f"\nAnalyzing sweep: {sweep_dir}")
    print(f"  {n_wl} wavelengths{batch_str}")

    plot_cot(agg, trials, morphology, out_dir, tag)
    plot_phase(agg, trials, morphology, out_dir, tag)
    plot_speed(agg, trials, morphology, out_dir, tag)
    plot_pitch_roll(agg, trials, morphology, out_dir, tag)
    plot_bode_combined(agg, trials, morphology, out_dir, tag, meta=data)

    print(f"  All figures saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Bode-plot analysis from wavelength sweep results.")
    parser.add_argument("--sweep-dir", default=None,
                        help="Path to a specific sweep_* directory.")
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
