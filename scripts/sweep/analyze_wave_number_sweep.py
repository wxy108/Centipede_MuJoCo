#!/usr/bin/env python3
"""
analyze_wave_number_sweep.py — Multi-metric analysis of a (k × λ) sweep.

Reads `results_all_trials.csv` from a `wave_number_sweep_*` run directory
(falls back to `results_aggregated.csv` if the per-trial file is missing),
applies an outlier trim per cell, and produces:

  - A 6-panel figure: speed, CoT, survival rate, max pitch, max roll,
    phase lag — all plotted vs terrain wavelength λ, one curve per k.
    Each curve marks the predicted morphology-anchored resonance band
    at λ_body(k) = body_length / k as a vertical line in matching color,
    AND in the figure legend. Panel letters (a)–(f).

  - A heatmap of forward speed across the (k × λ) grid.

  - A markdown summary: predicted body wavelengths, best (k, λ) per
    metric, k-marginal means, all-buckled cells.

Outputs land in `<sweep_dir>/analysis/`.

Trimming
--------
Each (k, λ) cell typically has N trials. By default, trials beyond
±2 σ of the cell median are excluded (robust to single bad runs), then
the mean and ±0.6 σ band are computed on the trimmed set. This produces
a tight band that represents the bulk of the trials, not every outlier.

Usage
-----
    # Most recent sweep, default settings (drop k=1.5, trim at 2σ, band 0.6σ)
    python scripts/sweep/analyze_wave_number_sweep.py

    # Specific sweep
    python scripts/sweep/analyze_wave_number_sweep.py outputs/wave_number_sweep/sweep_20260507_024045

    # Keep all wave numbers, no trimming
    python scripts/sweep/analyze_wave_number_sweep.py --exclude-k "" --trim-sigma 0

    # Tighter band
    python scripts/sweep/analyze_wave_number_sweep.py --band-sigma 0.4
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, ScalarFormatter

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
SWEEP_ROOT   = os.path.join(PROJECT_ROOT, "outputs", "wave_number_sweep")

# Canonical morphology constants — from configs/terrain.yaml / textbook
DEFAULT_BODY_LENGTH_M = 0.1025  # 102.5 mm full body length
DEFAULT_SEGMENT_LENGTH_M = 0.0054
DEFAULT_LEG_LENGTH_M = 0.0074


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_sweep(root: str = SWEEP_ROOT) -> str:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Sweep root does not exist: {root}")
    candidates = sorted(
        (d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No sweep directories under {root}")
    return os.path.join(root, candidates[0])


def _to_float(s):
    if s is None or s == "":
        return float("nan")
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _to_bool_float(s) -> float:
    """Parse a 'survived'-style flag that may be 'True'/'False'/'1'/'0'."""
    if s is None:
        return float("nan")
    sl = str(s).strip().lower()
    if sl in ("true", "t", "yes", "y", "1", "1.0"):
        return 1.0
    if sl in ("false", "f", "no", "n", "0", "0.0"):
        return 0.0
    # Fall back to numeric parsing; anything else becomes nan.
    return _to_float(s)


def load_all_trials(sweep_dir: str) -> list[dict]:
    """Read results_all_trials.csv (one row per trial)."""
    path = os.path.join(sweep_dir, "results_all_trials.csv")
    if not os.path.isfile(path):
        return []
    rows: list[dict] = []
    with open(path, "r") as f:
        for r in csv.DictReader(f):
            # NOTE: forward_speed in the CSV is in m/s (= distance_m / effective_time).
            # We convert to mm/s here so the rest of the analysis carries mm/s
            # natively and the y-axis labels match the data.
            speed_mps = _to_float(r.get("forward_speed"))
            rows.append({
                "wave_number":   _to_float(r.get("wave_number")),
                "wavelength_mm": _to_float(r.get("wavelength_mm")),
                "trial_idx":     _to_float(r.get("trial_idx")),
                "yaw_deg":       _to_float(r.get("yaw_deg")),
                # `survived` is written by wavelength_sweep.py as a Python bool
                # so the CSV stores "True"/"False" (not 1/0). Use the bool-aware
                # parser, otherwise every trial reads as NaN and gets filtered.
                "survived":      _to_bool_float(r.get("survived", "True")),
                "cot":           _to_float(r.get("cot")),
                "forward_speed": speed_mps * 1000.0 if math.isfinite(speed_mps) else float("nan"),
                "distance_m":    _to_float(r.get("distance_m")),
                "max_pitch_deg": _to_float(r.get("max_pitch_deg")),
                "mean_pitch_deg":_to_float(r.get("mean_pitch_deg")),
                "max_roll_deg":  _to_float(r.get("max_roll_deg")),
                "mean_roll_deg": _to_float(r.get("mean_roll_deg")),
                "energy_J":      _to_float(r.get("energy_J")),
                "phase_lag_deg": _to_float(r.get("phase_lag_deg")),
                "phase_coherence": _to_float(r.get("phase_coherence")),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Cell aggregation with outlier trimming
# ─────────────────────────────────────────────────────────────────────────────

# (column, output_label, lower_is_better, log_y, scale)
METRIC_DEFS = [
    ("forward_speed",  "Forward speed (mm/s)",       False, False, 1.0),
    ("cot",            "Cost of Transport",          True,  True,  1.0),
    ("survival_rate",  "Survival rate",              False, False, 1.0),
    ("max_pitch_deg",  "Max pitch (deg)",            True,  False, 1.0),
    ("max_roll_deg",   "Max roll (deg)",             True,  False, 1.0),
    ("phase_lag_deg",  "Phase lag (deg)",            False, False, 1.0),
]

PANEL_LETTERS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


def trimmed_stats(values: np.ndarray, trim_sigma: float
                  ) -> tuple[float, float, int, int]:
    """
    Robust mean and std after dropping outliers beyond ±trim_sigma of median.

    Returns (mean, std, n_kept, n_dropped). If trim_sigma <= 0 or n < 3,
    returns un-trimmed mean/std.
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float("nan"), float("nan"), 0, 0
    if trim_sigma <= 0 or v.size < 3:
        return float(v.mean()), float(v.std(ddof=0)), int(v.size), 0
    median = float(np.median(v))
    mad = float(np.median(np.abs(v - median)))
    # Convert MAD → robust sigma estimate (1.4826 factor for Gaussian).
    sigma_est = 1.4826 * mad if mad > 0 else float(v.std(ddof=0))
    if sigma_est == 0:
        return float(v.mean()), 0.0, int(v.size), 0
    keep_mask = np.abs(v - median) <= trim_sigma * sigma_est
    kept = v[keep_mask]
    n_dropped = int(v.size - kept.size)
    if kept.size == 0:
        return float("nan"), float("nan"), 0, int(v.size)
    return float(kept.mean()), float(kept.std(ddof=0)), int(kept.size), n_dropped


def aggregate_cells(trials: list[dict], trim_sigma: float
                    ) -> dict[tuple[float, float], dict]:
    """
    Group trials by (k, λ) cell, compute trimmed stats per metric.
    Returns {(k, λ): {metric: {mean, std, n_kept, n_dropped}, ...}}.
    survival_rate is mean of `survived` flags; not subject to trimming.
    """
    by_cell: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for t in trials:
        by_cell[(t["wave_number"], t["wavelength_mm"])].append(t)

    out: dict[tuple[float, float], dict] = {}
    for cell, ts in by_cell.items():
        cell_stats = {"n_total": len(ts)}
        # survival rate
        surv = np.array([t["survived"] for t in ts], dtype=float)
        cell_stats["survival_rate"] = {
            "mean": float(np.nanmean(surv)),
            "std":  float(np.nanstd(surv, ddof=0)),
            "n_kept":   int(np.sum(np.isfinite(surv))),
            "n_dropped": 0,
        }
        # other metrics computed only on surviving trials
        survived_only = [t for t in ts if t["survived"] >= 0.5]
        for col, _, _, _, _ in METRIC_DEFS:
            if col == "survival_rate":
                continue
            vals = np.array([t[col] for t in survived_only], dtype=float)
            mean, std, n_kept, n_dropped = trimmed_stats(vals, trim_sigma)
            cell_stats[col] = {
                "mean": mean, "std": std,
                "n_kept": n_kept, "n_dropped": n_dropped,
            }
        out[cell] = cell_stats
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def make_colors(wave_numbers: list[float]):
    if len(wave_numbers) <= 1:
        return {wave_numbers[0]: "#1f77b4"} if wave_numbers else {}
    cmap = cm.get_cmap("viridis", len(wave_numbers))
    sorted_k = sorted(wave_numbers)
    return {k: cmap(i) for i, k in enumerate(sorted_k)}


def panel_metric_vs_lambda(
    ax,
    by_cell: dict,
    wave_numbers: list[float],
    colors: dict,
    body_length_m: float,
    metric_col: str,
    ylabel: str,
    log_y: bool,
    band_sigma: float,
    panel_letter: str,
):
    body_length_mm = body_length_m * 1000.0

    for k in wave_numbers:
        # Collect this k's data, sorted by λ
        cells_k = [(lam, by_cell[(k, lam)]) for (kk, lam) in by_cell if kk == k]
        cells_k.sort(key=lambda x: x[0])
        if not cells_k:
            continue
        lambdas = np.array([c[0] for c in cells_k])
        means = np.array([c[1][metric_col]["mean"] for c in cells_k])
        stds  = np.array([c[1][metric_col]["std"]  for c in cells_k])

        # Drop NaN points so the line stays connected through valid cells only.
        mask = np.isfinite(means)
        lambdas, means, stds = lambdas[mask], means[mask], stds[mask]
        if lambdas.size == 0:
            continue

        c = colors[k]
        ax.plot(lambdas, means, "-o",
                color=c, lw=1.6, ms=4.5, mec="white", mew=0.6,
                label=f"k = {k:g}")
        # Band at ±band_sigma σ — narrower than ±1σ to represent the bulk
        if np.any(stds > 0):
            ax.fill_between(lambdas,
                            means - band_sigma * stds,
                            means + band_sigma * stds,
                            color=c, alpha=0.20, linewidth=0)

        # Body wavelength prediction line, color-matched to the curve
        lam_body = body_length_mm / k
        ax.axvline(lam_body, color=c, lw=1.0, ls="--", alpha=0.6)

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Terrain wavelength λ (mm)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, which="both", lw=0.4, alpha=0.45)
    ax.tick_params(which="both", labelsize=9)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
    # Panel letter in top-left
    ax.text(0.02, 0.95, panel_letter,
            transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")


def figure_metrics(
    by_cell: dict,
    wave_numbers: list[float],
    colors: dict,
    body_length_m: float,
    band_sigma: float,
    trim_sigma: float,
    sweep_name: str,
    out_path: str,
):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5))
    axes = axes.ravel()

    for ax, (col, ylabel, _, log_y, _), letter in zip(
        axes, METRIC_DEFS, PANEL_LETTERS
    ):
        panel_metric_vs_lambda(
            ax, by_cell, wave_numbers, colors, body_length_m,
            metric_col=col, ylabel=ylabel, log_y=log_y,
            band_sigma=band_sigma, panel_letter=letter,
        )

    # ── Combined legend: solid lines for k, plus a dashed-line entry ──
    handles = []
    for k in wave_numbers:
        handles.append(Line2D([0], [0], color=colors[k], lw=2,
                               marker="o", ms=5, mec="white", mew=0.6,
                               label=f"k = {k:g}"))
    handles.append(Line2D([0], [0], color="0.4", lw=1.0, ls="--",
                          label=r"Predicted body $\lambda_{\mathrm{body}}(k) = L_{\mathrm{body}}/k$"))
    fig.legend(handles=handles,
               loc="upper center",
               ncol=len(handles),
               bbox_to_anchor=(0.5, 0.985),
               frameon=False, fontsize=10)

    # Caption: data-prep notes
    body_mm = body_length_m * 1000.0
    caption = (
        f"Centipede locomotion: wave-number × terrain-wavelength sweep   ·   "
        f"{sweep_name}\n"
        f"L_body = {body_mm:.1f} mm   ·   "
        f"trim outliers > {trim_sigma:.1f}σ from cell median (robust MAD)   ·   "
        f"shaded band = mean ± {band_sigma:.2f}σ on trimmed trials"
    )
    fig.suptitle(caption, fontsize=11, y=0.998)
    fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    print(f"  wrote {out_path.replace('.png', '.pdf')}")


def figure_heatmap(
    by_cell: dict,
    wave_numbers: list[float],
    body_length_m: float,
    sweep_name: str,
    out_path: str,
):
    all_lams = sorted({lam for (_, lam) in by_cell.keys()})
    Z = np.full((len(wave_numbers), len(all_lams)), np.nan)
    for i, k in enumerate(wave_numbers):
        for j, lam in enumerate(all_lams):
            cell = by_cell.get((k, lam))
            if cell is not None:
                Z[i, j] = cell["forward_speed"]["mean"]

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    im = ax.imshow(Z, aspect="auto", cmap="viridis", origin="lower",
                   interpolation="nearest")
    ax.set_xticks(range(len(all_lams)))
    ax.set_xticklabels([f"{lam:g}" for lam in all_lams], rotation=45, fontsize=8)
    ax.set_yticks(range(len(wave_numbers)))
    ax.set_yticklabels([f"{k:g}" for k in wave_numbers], fontsize=10)
    ax.set_xlabel("Terrain wavelength λ (mm)", fontsize=11)
    ax.set_ylabel("Body wave-number k", fontsize=11)
    ax.set_title(f"Forward speed (mm/s) — heatmap across (k × λ)   ·   {sweep_name}",
                 fontsize=11)
    cbar = fig.colorbar(im, ax=ax, label="Forward speed (mm/s)")
    cbar.ax.tick_params(labelsize=9)

    # Cell value annotations
    if np.any(np.isfinite(Z)):
        zmax = float(np.nanmax(Z))
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                v = Z[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}",
                            ha="center", va="center",
                            color="white" if v < zmax * 0.55 else "black",
                            fontsize=7)

    # Red boxes at the closest-to-predicted-λ_body cell per k
    body_length_mm = body_length_m * 1000.0
    for i, k in enumerate(wave_numbers):
        lam_body = body_length_mm / k
        diffs = [abs(lam - lam_body) for lam in all_lams]
        j = int(np.argmin(diffs))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor="red",
                                    linewidth=1.8, alpha=0.85))
    # Label the red-box convention as a legend below the title
    ax.text(0.99, 1.02,
            r"Red boxes = closest λ to predicted $L_{\mathrm{body}}/k$",
            ha="right", va="bottom", transform=ax.transAxes,
            fontsize=8, style="italic", color="#700")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    print(f"  wrote {out_path.replace('.png', '.pdf')}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(
    by_cell: dict,
    wave_numbers: list[float],
    body_length_m: float,
    trim_sigma: float,
    band_sigma: float,
    excluded_k: list[float],
    out_path: str,
):
    body_length_mm = body_length_m * 1000.0
    lines = []
    lines.append("# Wave-number × wavelength sweep — analysis summary\n")
    lines.append(f"- Wave numbers retained: {wave_numbers}")
    if excluded_k:
        lines.append(f"- Wave numbers excluded: {excluded_k}")
    lines.append(f"- Cells analyzed: {len(by_cell)}")
    lines.append(f"- Trim threshold: ±{trim_sigma:.1f} σ from cell median (robust MAD)")
    lines.append(f"- Plot band width: ±{band_sigma:.2f} σ on trimmed trials")
    lines.append(f"- Body length L_body = {body_length_mm:.1f} mm")
    lines.append("")

    n_total_dropped = sum(c.get("forward_speed", {}).get("n_dropped", 0)
                          for c in by_cell.values())
    n_total_trials = sum(c.get("n_total", 0) for c in by_cell.values())
    n_all_buck = sum(1 for c in by_cell.values() if c["survival_rate"]["mean"] == 0)
    lines.append(f"- Total trial count across cells: {n_total_trials}")
    lines.append(f"- Trials dropped by trim filter (forward_speed): {n_total_dropped}")
    lines.append(f"- Cells where ALL trials buckled: {n_all_buck}\n")

    # Predicted body wavelengths
    lines.append("## Predicted body wavelengths\n")
    lines.append(f"λ_body(k) = L_body / k = {body_length_mm:.1f} mm / k\n")
    lines.append("| k | λ_body (mm) |")
    lines.append("|---|---|")
    for k in wave_numbers:
        lines.append(f"| {k:g} | {body_length_mm/k:.1f} |")
    lines.append("")

    # Best cells per metric
    def best(metric: str, lower_is_better: bool):
        best_cell, best_val = None, None
        for cell, stats in by_cell.items():
            if cell[0] not in wave_numbers:
                continue
            if stats["survival_rate"]["mean"] == 0:
                continue
            mu = stats[metric]["mean"]
            if not np.isfinite(mu):
                continue
            if best_val is None or (lower_is_better and mu < best_val) \
                                or (not lower_is_better and mu > best_val):
                best_val, best_cell = mu, cell
        return best_cell, best_val

    lines.append("## Best (k, λ) cell per metric\n")
    lines.append("| Metric          | Best k | Best λ (mm) | Value |")
    lines.append("|------------------|--------|-------------|-------|")
    for col, _, lower_is_better, _, _ in METRIC_DEFS:
        cell, val = best(col, lower_is_better)
        if cell is None:
            lines.append(f"| {col} | — | — | (no valid cells) |")
        else:
            lines.append(f"| {col} | {cell[0]:g} | {cell[1]:g} | {val:.3g} |")
    lines.append("")

    # k-marginal speed means (averaged over λ)
    lines.append("## k-marginal mean forward speed (averaged over λ)\n")
    lines.append("| k | mean speed (mm/s) | n cells |")
    lines.append("|---|---|---|")
    for k in wave_numbers:
        speeds = [stats["forward_speed"]["mean"]
                  for cell, stats in by_cell.items()
                  if cell[0] == k and np.isfinite(stats["forward_speed"]["mean"])]
        if speeds:
            lines.append(f"| {k:g} | {np.mean(speeds):.2f} | {len(speeds)} |")
    lines.append("")

    if n_all_buck:
        lines.append("## Cells where every trial buckled\n")
        lines.append("| k | λ (mm) |")
        lines.append("|---|---|")
        for cell, stats in sorted(by_cell.items()):
            if stats["survival_rate"]["mean"] == 0:
                lines.append(f"| {cell[0]:g} | {cell[1]:g} |")
        lines.append("")

    Path(out_path).write_text("\n".join(lines))
    print(f"  wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_exclude_k(s: str) -> list[float]:
    if s is None or s.strip() == "":
        return []
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir", nargs="?", default=None,
                   help="Path to sweep_<timestamp>/. Defaults to most recent.")
    p.add_argument("--body-length", type=float, default=DEFAULT_BODY_LENGTH_M,
                   help="Centipede body length in metres (default 0.1025 = 102.5 mm).")
    p.add_argument("--exclude-k", type=str, default="1.5",
                   help="Comma-separated wave-numbers to drop from plots "
                        "(default: '1.5'; pass '' to keep all).")
    p.add_argument("--trim-sigma", type=float, default=2.0,
                   help="Drop per-cell trials beyond ±σ × this from the median, "
                        "using robust MAD-based σ. 0 disables trimming.")
    p.add_argument("--band-sigma", type=float, default=0.6,
                   help="Width of the error band in trimmed σ units (default 0.6).")
    p.add_argument("--out-subdir", default="analysis",
                   help="Subdirectory under sweep_dir for outputs.")
    args = p.parse_args()

    sweep_dir = args.sweep_dir or find_latest_sweep()
    sweep_dir = os.path.abspath(sweep_dir)
    if not os.path.isdir(sweep_dir):
        print(f"ERROR: not a directory: {sweep_dir}", file=sys.stderr); sys.exit(1)

    sweep_name = os.path.basename(sweep_dir.rstrip(os.sep))
    print(f"[analyze] sweep dir : {sweep_dir}")
    print(f"[analyze] body L    : {args.body_length*1000:.1f} mm")
    print(f"[analyze] trim σ    : {args.trim_sigma}")
    print(f"[analyze] band σ    : {args.band_sigma}")

    # Load per-trial data, trim, and aggregate
    trials = load_all_trials(sweep_dir)
    if not trials:
        print(f"ERROR: results_all_trials.csv missing or empty.\n"
              f"       Path: {os.path.join(sweep_dir, 'results_all_trials.csv')}",
              file=sys.stderr)
        sys.exit(1)
    print(f"[analyze] loaded {len(trials)} trials")

    excluded_k = parse_exclude_k(args.exclude_k)
    if excluded_k:
        before = len(trials)
        trials = [t for t in trials if t["wave_number"] not in excluded_k]
        print(f"[analyze] dropped k ∈ {excluded_k}: {before - len(trials)} trials")

    by_cell = aggregate_cells(trials, args.trim_sigma)
    wave_numbers = sorted({c[0] for c in by_cell.keys()})
    n_lams = len({c[1] for c in by_cell.keys()})
    print(f"[analyze] kept k    : {wave_numbers}")
    print(f"[analyze] cells     : {len(by_cell)}  ({len(wave_numbers)} k × {n_lams} λ)")

    # ── Diagnostic: per-cell survival + sample speed values ──
    n_total_trials = sum(s["n_total"] for s in by_cell.values())
    n_alive_trials = sum(int(s["survival_rate"]["mean"] * s["n_total"])
                         for s in by_cell.values())
    print(f"[analyze] survival  : {n_alive_trials}/{n_total_trials} trials survived")
    if n_alive_trials == 0:
        print("[analyze] WARNING: zero surviving trials — check the 'survived' "
              "column parsing. CSV may use unexpected boolean format.",
              file=sys.stderr)
    # Show one cell as a sanity check
    if by_cell:
        sample_key = next(iter(sorted(by_cell.keys())))
        sample = by_cell[sample_key]
        sp = sample.get("forward_speed", {})
        print(f"[analyze] sample (k={sample_key[0]:g}, λ={sample_key[1]:g}): "
              f"speed mean={sp.get('mean', float('nan')):.2f} mm/s, "
              f"std={sp.get('std', float('nan')):.2f}, "
              f"n_kept={sp.get('n_kept', 0)}, n_dropped={sp.get('n_dropped', 0)}")

    out_dir = os.path.join(sweep_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    colors = make_colors(wave_numbers)

    print("\n[analyze] writing figures + summary ...")
    figure_metrics(by_cell, wave_numbers, colors, args.body_length,
                   args.band_sigma, args.trim_sigma, sweep_name,
                   os.path.join(out_dir, "metrics_vs_wavelength.png"))
    figure_heatmap(by_cell, wave_numbers, args.body_length, sweep_name,
                   os.path.join(out_dir, "speed_heatmap.png"))
    write_summary(by_cell, wave_numbers, args.body_length,
                  args.trim_sigma, args.band_sigma, excluded_k,
                  os.path.join(out_dir, "summary.md"))

    print(f"\n[analyze] DONE. Outputs under: {out_dir}")


if __name__ == "__main__":
    main()
