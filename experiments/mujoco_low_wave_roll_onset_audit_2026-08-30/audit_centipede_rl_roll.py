"""Measure flat low-wave roll directly from the frozen Centipede RL pose trace."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RL_ROOT = Path(r"C:\Users\wxy22\Documents\Centipede_IsaacLab_RL_Clean")
TRACE = RL_ROOT / "outputs/demo/flat/terrain_seed_244949/open_loop_w1.5/all_link_world_poses.npz"
RL_METRICS = RL_ROOT / "outputs/demo/flat/terrain_seed_244949/open_loop_w1.5/metrics.json"
MUJOCO_COMMON = ROOT / "outputs/data/flat_low_wave_roll_common_mode.json"
OUT_DATA = ROOT / "outputs/data/centipede_rl_flat_roll_audit.json"
OUT_FIGURE = ROOT / "outputs/figures/centipede_rl_flat_roll_audit.png"


def percentile(values, q=0.95):
    return float(np.quantile(np.asarray(values, dtype=float), q))


def signed_up_vector_roll_deg(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=float)
    q /= np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    # World-y component of the segment's local up vector (R[1, 2]).
    # This is the same yaw-independent convention used by the frozen
    # Centipede RL flat-roll audit from 2026-06-30.
    r21 = 2.0 * (y * z - w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arctan2(r21, r22))


def main():
    trace = np.load(TRACE, allow_pickle=True)
    names = trace["body_names"].astype(str)
    spine_indices = [int(np.where(names == f"link_body_{i}")[0][0]) for i in range(21)]
    times = np.asarray(trace["time_s"], dtype=float)
    quats = np.asarray(trace["body_quat_w"][:, 0, spine_indices, :], dtype=float)
    roll = signed_up_vector_roll_deg(quats)
    mask = times >= 1.5
    eval_roll = roll[mask]
    eval_times = times[mask]
    common = np.mean(eval_roll, axis=1)
    front7 = np.mean(eval_roll[:, :7], axis=1)
    per_link_p2p = np.quantile(eval_roll, 0.975, axis=0) - np.quantile(
        eval_roll, 0.025, axis=0
    )
    per_link_abs_p95 = np.quantile(np.abs(eval_roll), 0.95, axis=0)
    span = np.ptp(eval_roll, axis=1)

    rl_run = json.loads(RL_METRICS.read_text(encoding="utf-8"))
    mujoco = json.loads(MUJOCO_COMMON.read_text(encoding="utf-8"))
    output = {
        "purpose": "direct yaw-independent up-vector roll audit of Centipede RL flat wave=1.5",
        "source_trace": str(TRACE),
        "source_mode": rl_run.get("mode"),
        "source_fixed_wave": rl_run.get("fixed_wave"),
        "source_duration_s": rl_run.get("duration_s"),
        "eval_start_s": 1.5,
        "spine_link_count": 21,
        "rl_metrics": {
            "all_spine_common_mode_abs_p95_deg": percentile(np.abs(common)),
            "front7_common_mode_abs_p95_deg": percentile(np.abs(front7)),
            "all_spine_world_roll_span_p95_deg": percentile(span),
            "all_chassis_abs_roll_p95_deg": percentile(np.abs(eval_roll).reshape(-1)),
            "head_roll_robust_p2p_deg": float(per_link_p2p[0]),
            "mid_link10_roll_robust_p2p_deg": float(per_link_p2p[10]),
            "tail_link20_roll_robust_p2p_deg": float(per_link_p2p[20]),
            "per_link_roll_robust_p2p_deg": per_link_p2p.tolist(),
            "per_link_abs_roll_p95_deg": per_link_abs_p95.tolist(),
        },
        "mujoco_reference_rows": mujoco["rows"],
        "model_or_controller_modified": False,
    }
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    OUT_DATA.write_text(json.dumps(output, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].plot(np.arange(21), per_link_p2p, marker="o")
    axes[0].set_xlabel("spine link index (0=head, 20=tail)")
    axes[0].set_ylabel("yaw-independent roll robust p2p (deg)")
    axes[0].set_title("Centipede RL flat wave=1.5: distributed twist")
    axes[0].grid(alpha=0.3)
    axes[1].plot(eval_times, eval_roll[:, 0], label="head link 0", alpha=0.9)
    axes[1].plot(eval_times, eval_roll[:, 10], label="middle link 10", alpha=0.9)
    axes[1].plot(eval_times, eval_roll[:, 20], label="tail link 20", alpha=0.9)
    axes[1].plot(eval_times, common, label="signed whole-spine mean", color="black", linewidth=2)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("roll (deg)")
    axes[1].set_title("Open-loop trace: coherent roll remains small")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.savefig(OUT_FIGURE, dpi=180)
    plt.close(fig)
    print(json.dumps({"data": str(OUT_DATA), "figure": str(OUT_FIGURE), "rl": output["rl_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
