"""Plot absolute displacement and loss for the completed N_wave=1--9 scan."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = (
    ROOT
    / "outputs"
    / "strong_independent_erasure"
    / "all_waves_1_to_9"
    / "trials.csv"
)
OUTPUT = (
    ROOT
    / "outputs"
    / "strong_independent_erasure"
    / "all_waves_1_to_9"
    / "requested_figures"
)
WAVES = tuple(range(1, 10))
MULTIPLICITY = {
    1: "1–2",
    2: "2–3",
    3: "3–4",
    4: "2–3",
    5: "1–2",
    6: "6–7",
    7: "1–2",
    8: "2–3",
    9: "9–10",
}
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">")


def load_rows() -> list[dict[str, str]]:
    with DATA.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows: list[dict[str, str]]) -> dict[int, list[dict[str, float]]]:
    baseline = {
        wave: float(
            next(
                row["absolute_forward_displacement_m"]
                for row in rows
                if int(row["wave"]) == wave
                and float(row["noise_level_nu"]) == 0.0
            )
        )
        for wave in WAVES
    }
    result: dict[int, list[dict[str, float]]] = {}
    for wave in WAVES:
        result[wave] = []
        levels = sorted(
            {
                float(row["noise_level_nu"])
                for row in rows
                if int(row["wave"]) == wave
            }
        )
        for noise in levels:
            cell = [
                row
                for row in rows
                if int(row["wave"]) == wave
                and float(row["noise_level_nu"]) == noise
            ]
            stable_displacement = np.asarray(
                [
                    float(row["absolute_forward_displacement_m"])
                    for row in cell
                    if row["stable"].lower() == "true"
                ],
                dtype=float,
            )
            loss = (baseline[wave] - stable_displacement) / baseline[wave]
            result[wave].append(
                {
                    "noise": noise,
                    "mean_displacement": float(np.mean(stable_displacement)),
                    "sd_displacement": float(np.std(stable_displacement, ddof=1))
                    if len(stable_displacement) > 1
                    else 0.0,
                    "mean_loss": float(np.mean(loss)),
                    "sd_loss": float(np.std(loss, ddof=1))
                    if len(loss) > 1
                    else 0.0,
                    "stable_n": float(len(stable_displacement)),
                    "total_n": float(len(cell)),
                }
            )
    return result


def style(axis: plt.Axes, ylabel: str) -> None:
    axis.set_xlabel(r"Noise level $\nu$ (erased-time fraction)")
    axis.set_ylabel(ylabel)
    axis.set_xlim(-0.025, 1.025)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2, linewidth=0.7)


def plot_metric(
    summary: dict[int, list[dict[str, float]]],
    mean_key: str,
    sd_key: str,
    ylabel: str,
    filename: str,
) -> None:
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.94, len(WAVES)))
    fig, axis = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
    for index, wave in enumerate(WAVES):
        cell = summary[wave]
        axis.errorbar(
            [row["noise"] for row in cell],
            [row[mean_key] for row in cell],
            yerr=[row[sd_key] for row in cell],
            marker=MARKERS[index],
            linestyle="-",
            linewidth=1.7,
            markersize=5.3,
            capsize=2.3,
            color=colors[index],
            label=rf"$N_{{\mathrm{{wave}}}}={wave},\ m_g={MULTIPLICITY[wave]}$",
        )
    style(axis, ylabel)
    if mean_key == "mean_loss":
        axis.axhline(0.0, color="0.45", linewidth=0.8)
        axis.set_ylim(-0.035, 1.035)
    axis.legend(frameon=False, fontsize=7.4, ncol=2)
    fig.savefig(OUTPUT / filename, dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = aggregate(load_rows())
    plot_metric(
        summary,
        "mean_displacement",
        "sd_displacement",
        "Forward displacement (m)",
        "forward_displacement_vs_noise_all_waves.png",
    )
    plot_metric(
        summary,
        "mean_loss",
        "sd_loss",
        "Mean fractional performance loss",
        "mean_fractional_performance_loss_vs_noise_all_waves.png",
    )


if __name__ == "__main__":
    main()
