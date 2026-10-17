from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / ".python_packages"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "data"
FIGURES = ROOT / "outputs" / "figures"


def main() -> None:
    rows = json.loads((DATA / "gait_naturality_audit.json").read_text(encoding="utf-8"))
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [
        "clean 1.5", "aggressive 1.5", "Isaac anchor", "old Model w3",
        *[f"rejected w{i}" for i in range(1, 10)],
    ]
    x = np.arange(len(rows))
    anchor = np.asarray([not str(r["case"]).startswith("rejected") for r in rows])
    colors = np.where(anchor, "#2878b5", "#d1495b")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.bar(x - 0.25, [r["body_yaw_deg"] for r in rows], width=0.25, label="body yaw")
    ax.bar(x, [r["leg_yaw_deg"] for r in rows], width=0.25, label="leg yaw")
    ax.bar(x + 0.25, [r["leg_pitch_deg"] for r in rows], width=0.25, label="leg pitch")
    ax.set_ylabel("commanded amplitude (deg)")
    ax.set_title("The rejected search concentrated on large leg excursions")
    ax.legend(frameon=False, ncol=3)

    ax = axes[0, 1]
    ax.bar(x, [r["forward_displacement_mm"] for r in rows], color=colors)
    ax.set_ylabel("6 s forward displacement (mm)")
    ax.set_title("Displacement alone rewards the rejected gaits")

    ax = axes[1, 0]
    ax.bar(x - 0.18, [r["leg_yaw_speed_p99_rad_s"] for r in rows], width=0.36, label="leg yaw")
    ax.bar(x + 0.18, [r["leg_pitch_speed_p99_rad_s"] for r in rows], width=0.36, label="leg pitch")
    ax.axhline(8.0, color="#2878b5", linestyle="--", linewidth=1.5, label="yaw pilot gate")
    ax.axhline(7.0, color="#d1495b", linestyle=":", linewidth=2.0, label="pitch pilot gate")
    ax.set_ylabel("joint-speed P99 (rad/s)")
    ax.set_title("High-wave winners contain fast contact-driven pitch motion")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 1]
    ax.bar(x - 0.18, [r["swing_foot_height_p95_mm"] for r in rows], width=0.36, label="swing height P95 (mm)")
    ax.bar(x + 0.18, [r["leg_pitch_tracking_error_rms_deg"] for r in rows], width=0.36, label="pitch tracking RMS (deg)")
    ax.axhline(5.5, color="#2878b5", linestyle="--", linewidth=1.5)
    ax.axhline(10.0, color="#d1495b", linestyle=":", linewidth=2.0)
    ax.set_title("The visual oversweep has measurable kinematic signatures")
    ax.legend(frameon=False)

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=50, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Centipede Model gait audit: historical anchors vs rejected displacement winners", fontsize=16)
    path = FIGURES / "historical_anchors_vs_rejected_gaits.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()

