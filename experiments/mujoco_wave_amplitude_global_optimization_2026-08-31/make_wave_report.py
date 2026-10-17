"""Build final compact tables, figure, and evidence report."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "data"
FIGURES = ROOT / "outputs" / "figures"


STATUS_LABELS = {
    "long_safe_grid_global": "4 s global + 6 s pass",
    "long_safe_ranked_runner_up": "6 s-safe ranked runner-up",
    "no_6s_safe_among_4s_feasible": "4 s global only; no 6 s-safe point",
    "no_4s_feasible_grid_point": "no 4 s feasible point",
}


def main():
    payload = json.loads(
        (DATA / "final_selection_summary.json").read_text(encoding="utf-8")
    )
    rows = []
    for wave in range(1, 10):
        item = payload["waves"][str(wave)]
        selected = item["selected"]
        if selected is None:
            rows.append(
                {
                    "wave": wave,
                    "status": item["status"],
                    "rank": math.nan,
                    "body": math.nan,
                    "yaw": math.nan,
                    "pitch": math.nan,
                    "grid_mm": math.nan,
                    "long_mm": math.nan,
                    "feasible_count": 0,
                    "retention": math.nan,
                    "survival": math.nan,
                }
            )
            continue
        random = selected["random_contact"]
        rows.append(
            {
                "wave": wave,
                "status": item["status"],
                "rank": int(selected["grid_rank"]),
                "body": float(selected["body_yaw_amplitude_deg"]),
                "yaw": float(selected["leg_yaw_amplitude_deg"]),
                "pitch": float(selected["leg_pitch_amplitude_deg"]),
                "grid_mm": 1000.0
                * float(selected["grid_row"]["forward_displacement_m"]),
                "long_mm": 1000.0
                * float(selected["long_row"]["forward_displacement_m"]),
                "feasible_count": int(item["grid_feasible_count"]),
                "retention": 100.0
                * float(random["mean_retention_vs_nominal"]),
                "survival": 100.0
                * float(random["hard_gate_survival_fraction"]),
            }
        )

    FIGURES.mkdir(parents=True, exist_ok=True)
    waves = np.asarray([row["wave"] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.7), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(waves, [row["body"] for row in rows], "o-", label="body yaw")
    axis.plot(waves, [row["yaw"] for row in rows], "o-", label="leg yaw")
    axis.plot(waves, [row["pitch"] for row in rows], "o-", label="leg pitch")
    axis.set_ylabel("Selected amplitude (deg)")
    axis.legend(frameon=False, ncol=3, fontsize=8)

    axis = axes[0, 1]
    axis.plot(waves, [row["grid_mm"] for row in rows], "o-", label="4 s grid")
    axis.plot(waves, [row["long_mm"] for row in rows], "s--", label="6 s check")
    for row in rows:
        if row["status"] == "long_safe_ranked_runner_up":
            axis.annotate(
                f"rank {int(row['rank'])}",
                (row["wave"], row["grid_mm"]),
                xytext=(3, 7),
                textcoords="offset points",
                fontsize=7,
            )
    axis.set_ylabel("Forward displacement (mm)")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    axis.bar(waves, [row["feasible_count"] for row in rows], color="#2878b5")
    axis.set_ylabel("4 s feasible grid points / 343")

    axis = axes[1, 1]
    axis.plot(
        waves,
        [row["retention"] for row in rows],
        "o-",
        color="#d45f00",
        label="q=0.3 displacement retention",
    )
    axis.plot(
        waves,
        [row["survival"] for row in rows],
        "s--",
        color="#4a7c2b",
        label="q=0.3 hard-gate survival",
    )
    axis.axhline(100.0, color="0.75", linewidth=1)
    axis.set_ylabel("Random-contact result (%)")
    axis.legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.set_xlabel("Coupled body/leg wave number")
        axis.set_xticks(waves)
        axis.grid(alpha=0.2)
    figure = FIGURES / "final_wave_amplitude_summary.png"
    fig.savefig(figure, dpi=220)
    plt.close(fig)

    lines = [
        "# Flat-ground wave/amplitude optimization",
        "",
        "The body and legs share one wave number. Only body-yaw, leg-yaw, and "
        "leg-pitch amplitudes vary. The 343-point grid for every wave is "
        "complete and unique. A selected point must pass all 4 s gates and is "
        "then checked with the same gates at 6 s.",
        "",
        "| wave | status | grid rank | body yaw | leg yaw | leg pitch | 4 s D | 6 s D | feasible | q=.3 retention | q=.3 gate survival |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if math.isnan(row["rank"]):
            lines.append(
                f"| {row['wave']} | {STATUS_LABELS[row['status']]} | — | — | — | — | — | — | 0/343 | — | — |"
            )
        else:
            lines.append(
                f"| {row['wave']} | {STATUS_LABELS[row['status']]} | "
                f"{int(row['rank'])} | {row['body']:.0f}° | {row['yaw']:.0f}° | "
                f"{row['pitch']:.0f}° | {row['grid_mm']:.1f} mm | "
                f"{row['long_mm']:.1f} mm | {row['feasible_count']}/343 | "
                f"{row['retention']:.1f}% | {row['survival']:.0f}% |"
            )
    lines.extend(
        [
            "",
            "Wave 1 and 2 have valid 4 s finite-grid optima but no 4 s-feasible "
            "candidate survives every 6 s gate. Wave 8 and 9 have no 4 s "
            "feasible point because every point violates the chassis-drag gate. "
            "Wave 3 and 5 therefore use rank-2 choices: all higher-ranked points "
            "were checked and failed the 6 s gate.",
            "",
            "Random contact erasure is post-selection validation, never part of "
            "parameter selection. At q=0.3 no tested selected gait survives the "
            "entire strict gate set, even though forward displacement is partly "
            "retained.",
            "",
            f"Frozen model: `{payload['model']}`",
            "",
            f"SHA-256: `{payload['model_sha256']}`",
            "",
            f"Compiled mass: `{payload['model_total_mass_kg']:.16g} kg`; "
            f"nonzero armatures: `{payload['armature_nonzero_count']}`; "
            f"step: `{payload['physics_timestep_s']} s`; "
            f"integrator: `{payload['integrator']}`.",
        ]
    )
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows, "figure": str(figure)}, indent=2))


if __name__ == "__main__":
    main()
