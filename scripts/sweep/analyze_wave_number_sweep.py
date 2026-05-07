#!/usr/bin/env python3
"""
analyze_wave_number_sweep.py — Multi-metric analysis of a (k × λ) sweep.

Reads `results_aggregated.csv` from a `wave_number_sweep_*` run directory
and produces:

  - A 6-panel figure: speed, CoT, survival rate, max pitch, max roll,
    phase lag — all plotted vs terrain wavelength λ, one curve per k.
    Each curve marks the predicted morphology-anchored resonance band
    at λ_body(k) = body_length / k as a vertical span in matching color.

  - A heatmap figure of forward speed across the full (k × λ) grid.

  - A markdown summary table with: best (k, λ) per metric, k-marginal
    means, λ-marginal means, and any all-failed cells flagged.

Outputs land in `<sweep_dir>/analysis/`.

Usage
-----
    # Most recent sweep
    python scripts/sweep/analyze_wave_number_sweep.py

    # Specific sweep
    python scripts/sweep/analyze_wave_number_sweep.py outputs/wave_number_sweep/sweep_20260507_024045

    # Override body length (default 0.1025 m = 102.5 mm from morphology config)
    python scripts/sweep/analyze_wave_number_sweep.py SWEEP_DIR --body-length 0.1025
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import LogLocator, ScalarFormatter

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
SWEEP_ROOT   = os.path.join(PROJECT_ROOT, "outputs", "wave_number_sweep")

# Canonical morphology constants — from configs/terrain.yaml / textbook
DEFAULT_BODY_LENGTH_M = 0.1025  # 102.5 mm full body length
DEFAULT_SEGMENT_LENGTH_M = 0.0054  # 5.4 mm inter-segment spacing
DEFAULT_LEG_LENGTH_M = 0.0074  # 7.4 mm leg


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


def load_aggregated_csv(sweep_dir: str) -> list[dict]:
    """Return list of row-dicts from results_aggregated.csv."""
    csv_path = os.path.join(sweep_dir, "results_aggregated.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Aggregated CSV not found: {csv_path}")

    rows: list[dict] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cleaned = {}
            for k, v in r.items():
                if v == "" or v is None:
                    cleaned[k] = float("nan")
                else:
                    try:
                        cleaned[k] = float(v)
                    except ValueError:
                        cleaned[k] = v
            rows.append(cleaned)
    return rows


def organize_by_wave_number(rows: list[dict]) -> dict[float, list[dict]]:
    """Group aggregated rows by wave_number, sort each group by wavelength."""
    by_k: dict[float, list[dict]] = {}
    for r in rows:
        k = float(r["wave_number"])
        by_k.setdefault(k, []).append(r)
    for k in by_k:
        by_k[k].sort(key=lambda x: float(x["wavelength_mm"]))
    return by_k


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

# Color palette: viridis is colorblind-friendly and orderly along k.
def make_colors(wave_numbers: list[float]):
    if len(wave_numbers) <= 1:
        return {wave_numbers[0]: "#1f77b4"} if wave_numbers else {}
    cmap = cm.get_cmap("viridis", len(wave_numbers))
    sorted_k = sorted(wave_numbers)
    return {k: cmap(i) for i, k in enumerate(sorted_k)}


METRIC_DEFS = [
    # (column_mean, column_std, axis_label, title, log_y, lower_is_better)
    ("speed_mean",      "speed_std",      "Forward speed (mm/s)",       "Forward speed",          False, False),
    ("cot_mean",        "cot_std",        "Cost of Transport",          "Cost of Transport",       True,  True),
    ("survival_rate",   None,             "Survival rate",              "Trial survival",          False, False),
    ("max_pitch_mean",  "max_pitch_std",  "Max pitch (deg)",            "Body pitch deviation",    False, True),
    ("max_roll_mean",   "max_roll_std",   "Max roll (deg)",             "Body roll deviation",     False, True),
    ("phase_lag_mean",  "phase_lag_std",  "Phase lag (deg)",            "Terrain → body phase lag", False, False),
]


def panel_metric_vs_lambda(
    ax,
    by_k: dict[float, list[dict]],
    colors: dict[float, tuple],
    body_length_m: float,
    metric_mean: str,
    metric_std: str | None,
    ylabel: str,
    title: str,
    log_y: bool,
):
    """One subplot: metric vs λ, with curves per k and body-wavelength bands."""
    # Convert body length to mm for plotting on the same axis as terrain λ
    body_length_mm = body_length_m * 1000.0

    # Collect speed across all cells for axis sanity (avoid pathological auto-range)
    all_lambdas, all_means = [], []
    for k, rows in sorted(by_k.items()):
        lambdas = np.array([float(r["wavelength_mm"]) for r in rows])
        means   = np.array([float(r.get(metric_mean, np.nan)) for r in rows])
        if metric_std is not None:
            stds = np.array([float(r.get(metric_std, np.nan)) for r in rows])
        else:
            stds = np.zeros_like(means)
        # Filter out NaN-only points so they don't break the line
        mask = np.isfinite(means)
        lambdas, means, stds = lambdas[mask], means[mask], stds[mask]
        if len(lambdas) == 0:
            continue

        # Curve + error band
        c = colors[k]
        ax.plot(lambdas, means, "-o",
                color=c, lw=1.6, ms=4, mec="white", mew=0.5,
                label=f"k = {k:g}")
        if np.any(stds > 0):
            ax.fill_between(lambdas, means - stds, means + stds,
                            color=c, alpha=0.18, linewidth=0)

        all_lambdas.append(lambdas)
        all_means.append(means)

        # Body-wavelength prediction for this k
        lam_body = body_length_mm / k
        ax.axvline(lam_body, color=c, lw=1.0, ls="--", alpha=0.55)

    # Cosmetics
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Terrain wavelength λ (mm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", lw=0.4, alpha=0.45)
    ax.tick_params(which="both", labelsize=9)
    # Use scalar formatter so log-x reads as "10, 100" not "10^1, 10^2"
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))


def figure_metrics(
    by_k: dict[float, list[dict]],
    colors: dict[float, tuple],
    body_length_m: float,
    out_path: str,
):
    """6-panel figure: each metric vs λ with body-wavelength annotations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ax, (mu, std, ylabel, title, log_y, _) in zip(axes, METRIC_DEFS):
        panel_metric_vs_lambda(
            ax, by_k, colors, body_length_m,
            metric_mean=mu, metric_std=std,
            ylabel=ylabel, title=title, log_y=log_y,
        )

    # One legend, top right outside plot area
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 0.99), frameon=False, fontsize=10)

    # Caption explaining the body-wavelength dashed lines
    fig.text(0.5, 0.005,
             "Dashed vertical lines mark predicted body wavelength "
             "λ_body(k) = L_body / k = {:.0f} mm / k.  "
             "Resonance hypothesis: speed peaks (and CoT troughs) align with these markers."
             .format(body_length_m * 1000.0),
             ha="center", fontsize=9, style="italic", color="#444")

    fig.suptitle("Centipede locomotion: wave-number × terrain-wavelength sweep",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    print(f"  wrote {out_path.replace('.png', '.pdf')}")


def figure_heatmap(
    by_k: dict[float, list[dict]],
    body_length_m: float,
    out_path: str,
):
    """Speed heatmap across (k, λ)."""
    wave_numbers = sorted(by_k.keys())
    # Use the first k's wavelength axis as canonical, but expand if needed
    all_lams = sorted({float(r["wavelength_mm"]) for rows in by_k.values() for r in rows})

    Z = np.full((len(wave_numbers), len(all_lams)), np.nan)
    for i, k in enumerate(wave_numbers):
        rows = by_k[k]
        for r in rows:
            lam = float(r["wavelength_mm"])
            mu = float(r.get("speed_mean", np.nan))
            if lam in all_lams:
                Z[i, all_lams.index(lam)] = mu

    fig, ax = plt.subplots(figsize=(11, 4.8))
    im = ax.imshow(Z, aspect="auto", cmap="viridis", origin="lower",
                   interpolation="nearest")
    ax.set_xticks(range(len(all_lams)))
    ax.set_xticklabels([f"{lam:g}" for lam in all_lams], rotation=45, fontsize=8)
    ax.set_yticks(range(len(wave_numbers)))
    ax.set_yticklabels([f"{k:g}" for k in wave_numbers], fontsize=9)
    ax.set_xlabel("Terrain wavelength λ (mm)")
    ax.set_ylabel("Body wave-number k")
    ax.set_title("Forward speed (mm/s) — heatmap across (k × λ)")
    cbar = fig.colorbar(im, ax=ax, label="Speed (mm/s)")
    cbar.ax.tick_params(labelsize=9)

    # Annotate cells with their speed
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0f}",
                        ha="center", va="center",
                        color="white" if v < np.nanmax(Z) * 0.55 else "black",
                        fontsize=7)

    # Mark predicted body-wavelength lambda for each k
    body_length_mm = body_length_m * 1000.0
    for i, k in enumerate(wave_numbers):
        lam_body = body_length_mm / k
        # Find the index closest to lam_body in all_lams
        diffs = [abs(lam - lam_body) for lam in all_lams]
        j = int(np.argmin(diffs))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor="red",
                                    linewidth=1.8, alpha=0.85))
    ax.text(0.99, 1.02,
            "Red boxes = closest λ to predicted L_body/k.",
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

def write_summary(rows: list[dict], by_k: dict, body_length_m: float, out_path: str):
    """Markdown summary: best cells, marginals, body-λ predictions."""
    body_length_mm = body_length_m * 1000.0
    lines = []
    lines.append("# Wave-number × wavelength sweep — analysis summary\n")
    lines.append(f"- Cells (k × λ): {len(rows)}")
    n_buck = sum(1 for r in rows if float(r.get("survival_rate", 1)) < 1)
    lines.append(f"- Cells with at least one buckle: {n_buck}")
    n_all_buck = sum(1 for r in rows if float(r.get("survival_rate", 1)) == 0)
    lines.append(f"- Cells where ALL trials buckled: {n_all_buck}\n")

    # Predicted body wavelengths per k
    lines.append("## Predicted body wavelengths\n")
    lines.append(f"Body length L_body = {body_length_mm:.1f} mm, λ_body(k) = L_body / k:\n")
    lines.append("| k | λ_body (mm) |")
    lines.append("|---|---|")
    for k in sorted(by_k.keys()):
        lines.append(f"| {k:g} | {body_length_mm/k:.1f} |")
    lines.append("")

    # Best (k, λ) per metric (max speed, min CoT, max survival, etc.)
    def best(metric: str, lower_is_better: bool):
        valid = [r for r in rows if np.isfinite(float(r.get(metric, np.nan)))
                                and float(r.get("survival_rate", 0)) > 0]
        if not valid:
            return None
        key = lambda r: float(r[metric])
        return min(valid, key=key) if lower_is_better else max(valid, key=key)

    lines.append("## Best cells per metric\n")
    lines.append("| Metric                | Best k | Best λ (mm) | Value |")
    lines.append("|------------------------|--------|-------------|-------|")
    for col, lower in [("speed_mean",     False),
                       ("cot_mean",       True),
                       ("survival_rate",  False),
                       ("max_pitch_mean", True),
                       ("max_roll_mean",  True)]:
        b = best(col, lower)
        if b is None:
            lines.append(f"| {col} | — | — | (no valid cells) |")
        else:
            lines.append(f"| {col} | {float(b['wave_number']):g} | "
                         f"{float(b['wavelength_mm']):g} | "
                         f"{float(b[col]):.3g} |")
    lines.append("")

    # Marginal means (averaged over the other axis)
    lines.append("## k-marginal speed means (averaged over λ)\n")
    lines.append("| k | mean speed (mm/s) | n cells |")
    lines.append("|---|---|---|")
    for k in sorted(by_k.keys()):
        speeds = [float(r["speed_mean"]) for r in by_k[k]
                  if np.isfinite(float(r.get("speed_mean", np.nan)))]
        if speeds:
            lines.append(f"| {k:g} | {np.mean(speeds):.2f} | {len(speeds)} |")
    lines.append("")

    # All-buckled cells
    if n_all_buck:
        lines.append("## Cells where every trial buckled (excluded from headline plots)\n")
        lines.append("| k | λ (mm) |")
        lines.append("|---|---|")
        for r in rows:
            if float(r.get("survival_rate", 0)) == 0:
                lines.append(f"| {float(r['wave_number']):g} | "
                             f"{float(r['wavelength_mm']):g} |")
        lines.append("")

    Path(out_path).write_text("\n".join(lines))
    print(f"  wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep_dir", nargs="?", default=None,
                   help="Path to sweep_<timestamp> dir. Defaults to most recent.")
    p.add_argument("--body-length", type=float, default=DEFAULT_BODY_LENGTH_M,
                   help="Centipede body length in metres "
                        "(default: 0.1025 = 102.5 mm).")
    p.add_argument("--out-subdir", default="analysis",
                   help="Subdirectory under sweep_dir for figures/summary.")
    args = p.parse_args()

    sweep_dir = args.sweep_dir or find_latest_sweep()
    sweep_dir = os.path.abspath(sweep_dir)
    if not os.path.isdir(sweep_dir):
        print(f"ERROR: not a directory: {sweep_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[analyze] sweep dir: {sweep_dir}")
    print(f"[analyze] body length: {args.body_length*1000:.1f} mm")

    rows = load_aggregated_csv(sweep_dir)
    print(f"[analyze] loaded {len(rows)} (k, λ) cells")
    if not rows:
        print("ERROR: no rows in aggregated CSV", file=sys.stderr)
        sys.exit(1)

    by_k = organize_by_wave_number(rows)
    wave_numbers = sorted(by_k.keys())
    n_lams = len({float(r["wavelength_mm"]) for r in rows})
    print(f"[analyze] wave numbers ({len(wave_numbers)}): {wave_numbers}")
    print(f"[analyze] wavelengths  ({n_lams}): "
          f"{sorted({float(r['wavelength_mm']) for r in rows})}")

    out_dir = os.path.join(sweep_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    colors = make_colors(wave_numbers)

    print("\n[analyze] writing figures and summary ...")
    figure_metrics(by_k, colors, args.body_length,
                   os.path.join(out_dir, "metrics_vs_wavelength.png"))
    figure_heatmap(by_k, args.body_length,
                   os.path.join(out_dir, "speed_heatmap.png"))
    write_summary(rows, by_k, args.body_length,
                  os.path.join(out_dir, "summary.md"))

    print(f"\n[analyze] DONE. Outputs under: {out_dir}")


if __name__ == "__main__":
    main()
