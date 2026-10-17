"""Measure coherent roll in the archived 2026-07-18 RL flat low-wave baseline.

This is a read-only replay audit.  It uses the same yaw-independent segment
up-vector convention as flat_roll_diagnosis_2026-06-30.py.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RL_ROOT = Path(r"C:\Users\wxy22\Documents\Centipede_IsaacLab_RL")
TRACE = (
    RL_ROOT
    / "outputs/analysis/p8_local_refine_lr0_2026-07-18/data"
    / "metrics_LR0low_b0_alllink.npz"
)
AUDIT = (
    RL_ROOT
    / "outputs/analysis/p8_demo_video_pack_2026-07-18/videos"
    / "flat_lowwave_wave1p5/BASELINE__flat_lowwave_wave1p5_side_audit.json"
)
OUT_DATA = ROOT / "outputs/data/historical_centipede_rl_flat_roll_audit.json"
OUT_FIGURE = ROOT / "outputs/figures/historical_centipede_rl_flat_roll_audit.png"


def up_vector_roll_deg(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=float)
    q /= np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    up_y = 2.0 * (y * z - x * w)
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arctan2(up_y, up_z))


def p95(values):
    return float(np.quantile(np.asarray(values, dtype=float), 0.95))


def main():
    trace = np.load(TRACE, allow_pickle=True)
    names = trace["body_names"].astype(str)
    spine = [int(np.where(names == f"link_body_{i}")[0][0]) for i in range(21)]
    env = 0
    alive = np.asarray(trace["alive"][:, env], dtype=bool)
    quats = np.asarray(trace["body_quat_w"][:, env, spine, :], dtype=float)[alive]
    roll = up_vector_roll_deg(quats)
    common = np.mean(roll, axis=1)
    front7 = np.mean(roll[:, :7], axis=1)
    span = np.ptp(roll, axis=1)
    robust_p2p = np.quantile(roll, 0.975, axis=0) - np.quantile(roll, 0.025, axis=0)
    dt = float(trace["dt_pol"])
    time_s = np.arange(len(roll), dtype=float) * dt
    video_audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    metrics = {
        "all_spine_common_mode_abs_p95_deg": p95(np.abs(common)),
        "front7_common_mode_abs_p95_deg": p95(np.abs(front7)),
        "all_spine_world_roll_span_p95_deg": p95(span),
        "all_chassis_abs_roll_p95_deg": p95(np.abs(roll).reshape(-1)),
        "head_roll_robust_p2p_deg": float(robust_p2p[0]),
        "mid_link10_roll_robust_p2p_deg": float(robust_p2p[10]),
        "tail_link20_roll_robust_p2p_deg": float(robust_p2p[20]),
        "per_link_roll_robust_p2p_deg": robust_p2p.tolist(),
    }
    output = {
        "purpose": "archived Centipede RL flat wave=1.5 coherent-roll audit",
        "source_trace": str(TRACE),
        "source_env": env,
        "source_dt_s": dt,
        "source_frames_alive": int(alive.sum()),
        "source_video_audit_pass": bool(video_audit["audit"]["PASS"]),
        "source_video_hud_final": video_audit.get("hud_final"),
        "roll_convention": "atan2(2*(y*z-x*w), 1-2*(x*x+y*y)) on wxyz quaternions",
        "rl_metrics": metrics,
        "model_or_controller_modified": False,
    }
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    OUT_DATA.write_text(json.dumps(output, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].plot(np.arange(21), robust_p2p, marker="o")
    axes[0].set_xlabel("spine link index (0=head, 20=tail)")
    axes[0].set_ylabel("yaw-independent roll robust p2p (deg)")
    axes[0].set_title("Archived RL baseline: distributed segment twist")
    axes[0].grid(alpha=0.3)
    axes[1].plot(time_s, roll[:, 0], label="head link 0", alpha=0.8)
    axes[1].plot(time_s, roll[:, 10], label="middle link 10", alpha=0.8)
    axes[1].plot(time_s, roll[:, 20], label="tail link 20", alpha=0.8)
    axes[1].plot(time_s, common, label="signed whole-spine mean", color="black", linewidth=2)
    axes[1].set_xlabel("recorded time (s)")
    axes[1].set_ylabel("roll (deg)")
    axes[1].set_title("Flat wave=1.5: coherent mode remains near level")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.savefig(OUT_FIGURE, dpi=180)
    plt.close(fig)
    print(json.dumps({"data": str(OUT_DATA), "figure": str(OUT_FIGURE), "rl": metrics}, indent=2))


if __name__ == "__main__":
    main()
