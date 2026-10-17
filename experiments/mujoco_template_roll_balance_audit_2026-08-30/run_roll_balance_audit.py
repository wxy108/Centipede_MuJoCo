"""Per-segment flat-ground roll-balance audit for the abstract Centipede Model.

This experiment is diagnostic only.  It compares the selected locked-head
flat controller (coupled body/leg wave=1.5) with a coupled wave=1.0 causal
ablation while holding the selected flat leg amplitudes fixed.  It never
selects or overwrites a production model/controller and never writes to the
Centipede Robot project.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
import ctypes
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# MuJoCo 3.10's optional bundled plugin DLL loader can block on this Windows
# host.  This model does not reference any plugin instances, so skip only
# those optional DLL handles during import; the MuJoCo core library and all
# simulation APIs are loaded normally.
_original_cdll = ctypes.CDLL


def _cdll_without_optional_mujoco_plugins(path, *args, **kwargs):
    normalized = os.path.normcase(str(path))
    if "site-packages\\mujoco\\plugin" in normalized:
        return None
    return _original_cdll(path, *args, **kwargs)


ctypes.CDLL = _cdll_without_optional_mujoco_plugins
try:
    import mujoco
finally:
    ctypes.CDLL = _original_cdll


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parents[1]
TIP_SCRIPT = MODEL_ROOT / "experiments" / "mujoco_template_tip_compliance_2026-08-27" / "run_tip_compliance.py"
LOCKED_ROOT = MODEL_ROOT / "experiments" / "mujoco_template_locked_head_2026-08-27"
MODEL = LOCKED_ROOT / "models" / "centipede_flat_locked_head_m29p0.xml"
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"
FIGURES = OUTPUTS / "figures"
REPORT = ROOT / "ROLL_BALANCE_AUDIT.md"
SEGMENTS = 19
SIDES = ("L", "R")
GROUPS = {
    "front_0_5": np.arange(0, 6, dtype=int),
    "middle_6_12": np.arange(6, 13, dtype=int),
    "rear_13_18": np.arange(13, 19, dtype=int),
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tip = load_module(TIP_SCRIPT, "centipede_roll_balance_tip")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def pct(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.quantile(values, q)) if values.size else float("nan")


def fundamental_amplitude(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Fit DC + sin/cos at the commanded temporal frequency."""
    omega = 2.0 * math.pi * float(tip.base.FREQUENCY_HZ)
    design = np.column_stack((np.ones_like(times), np.sin(omega * times), np.cos(omega * times)))
    flat = values.reshape(len(times), -1)
    coefficients, *_ = np.linalg.lstsq(design, flat, rcond=None)
    amplitude = np.hypot(coefficients[1], coefficients[2])
    return amplitude.reshape(values.shape[1:])


def foot_identity(model: mujoco.MjModel, geom_id: int) -> tuple[int, int] | None:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    match = re.fullmatch(r"foot_(\d+)_(0|1)_collision", name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def flat_leg_amplitudes() -> np.ndarray:
    return np.asarray(
        [
            tip.base.LEG_YAW_AMPLITUDE_RAD * float(tip.ctrl.LEG_YAW_AMPLITUDE_SCALE),
            tip.base.LEG_PITCH_AMPLITUDE_RAD * float(tip.ctrl.LEG_PITCH_AMPLITUDE_SCALE),
        ],
        dtype=float,
    )


def configure_controller(model: mujoco.MjModel, index, wave: float):
    controller = tip.LeanController(model, index, wave)
    # LeanController otherwise interprets any non-1.5 wave as the rough profile.
    # Overwrite only amplitudes so this remains a wave-number-only ablation.
    controller.leg_amplitudes = flat_leg_amplitudes()
    return controller


def run_case(label: str, wave: float, duration_s: float = 6.0) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    index = tip.LeanIndex(model)
    tip.configure_model(model, index)
    controller = configure_controller(model, index, wave)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    ground = tip.stage1.ground_geom_id(model)
    root_body = tip.base.name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_0")
    pitch_active = index.leg_positions[:, :, 1]
    pitch_qpos = index.qpos_addresses[pitch_active]
    eval_start_s = 1.5
    dt = float(model.opt.timestep)
    stride = max(1, int(round(0.001 / dt)))

    times: list[float] = []
    targets: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    torques: list[np.ndarray] = []
    foot_fz: list[np.ndarray] = []
    foot_roll_moment: list[np.ndarray] = []
    contact_flags: list[np.ndarray] = []
    left_counts: list[int] = []
    right_counts: list[int] = []
    total_roll_moments: list[float] = []
    mean_rolls: list[float] = []
    start_x = end_x = None
    step_index = 0

    while data.time < duration_s - 0.5 * dt:
        torque = controller.step(data)
        target_pitch = controller.target[pitch_active].copy()
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f"Non-finite state in {label} at t={data.time}")
        if data.time < eval_start_s or step_index % stride:
            step_index += 1
            continue

        center = tip.base.chassis_centroid(data, index)
        if start_x is None:
            start_x = float(center[0])
        end_x = float(center[0])
        com = np.asarray(data.subtree_com[root_body], dtype=float).copy()
        head_rotation = np.asarray(data.xmat[index.head_body_id], dtype=float).reshape(3, 3)
        forward_axis = head_rotation[:, 0]
        sample_fz = np.zeros((SEGMENTS, 2), dtype=float)
        sample_moment = np.zeros((SEGMENTS, 2), dtype=float)
        sample_contact = np.zeros((SEGMENTS, 2), dtype=bool)
        total_moment = np.zeros(3, dtype=float)

        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 == ground:
                robot_geom, robot_is_geom2 = geom2, True
            elif geom2 == ground:
                robot_geom, robot_is_geom2 = geom1, False
            else:
                continue
            force6 = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, contact_id, force6)
            frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            force_on_geom2_world = frame.T @ force6[:3]
            force_world = force_on_geom2_world if robot_is_geom2 else -force_on_geom2_world
            moment_world = np.cross(np.asarray(contact.pos, dtype=float) - com, force_world)
            total_moment += moment_world
            identity = foot_identity(model, robot_geom)
            if identity is not None:
                segment, side = identity
                sample_fz[segment, side] += float(force_world[2])
                sample_moment[segment, side] += float(np.dot(moment_world, forward_axis))
                sample_contact[segment, side] = True

        times.append(float(data.time))
        targets.append(target_pitch)
        actuals.append(np.asarray(data.qpos[pitch_qpos], dtype=float).copy())
        torques.append(np.asarray(torque[pitch_active], dtype=float).copy())
        foot_fz.append(sample_fz)
        foot_roll_moment.append(sample_moment)
        contact_flags.append(sample_contact)
        left_counts.append(int(np.count_nonzero(sample_contact[:, 0])))
        right_counts.append(int(np.count_nonzero(sample_contact[:, 1])))
        total_roll_moments.append(float(np.dot(total_moment, forward_axis)))
        mean_rolls.append(float(np.mean(tip.signed_world_roll_deg(data, index.chassis_body_ids))))
        step_index += 1

    arrays = {
        "time_s": np.asarray(times, dtype=float),
        "target_pitch_rad": np.asarray(targets, dtype=float),
        "actual_pitch_rad": np.asarray(actuals, dtype=float),
        "pitch_torque_nm": np.asarray(torques, dtype=float),
        "foot_fz_n": np.asarray(foot_fz, dtype=float),
        "foot_roll_moment_nm": np.asarray(foot_roll_moment, dtype=float),
        "contact": np.asarray(contact_flags, dtype=bool),
        "left_count": np.asarray(left_counts, dtype=int),
        "right_count": np.asarray(right_counts, dtype=int),
        "total_roll_moment_nm": np.asarray(total_roll_moments, dtype=float),
        "mean_body_roll_deg": np.asarray(mean_rolls, dtype=float),
    }
    if len(arrays["time_s"]) < 100:
        raise RuntimeError(f"Insufficient samples for {label}")

    time = arrays["time_s"]
    target_amp = fundamental_amplitude(arrays["target_pitch_rad"], time)
    actual_amp = fundamental_amplitude(arrays["actual_pitch_rad"], time)
    tracking_ratio = actual_amp / np.maximum(target_amp, 1e-15)
    fz = arrays["foot_fz_n"]
    contact = arrays["contact"]
    roll_moment = arrays["foot_roll_moment_nm"]
    equal_count = arrays["left_count"] == arrays["right_count"]
    left_fz = np.sum(fz[:, :, 0], axis=1)
    right_fz = np.sum(fz[:, :, 1], axis=1)
    assigned_fz = left_fz + right_fz
    force_imbalance = (right_fz - left_fz) / np.maximum(np.abs(assigned_fz), 1e-12)
    selected_moment = arrays["total_roll_moment_nm"][equal_count]
    selected_imbalance = force_imbalance[equal_count]

    per_segment: list[dict[str, object]] = []
    for segment in range(SEGMENTS):
        row: dict[str, object] = {"segment": segment}
        for side_index, side in enumerate(SIDES):
            active_force = fz[:, segment, side_index]
            row[side] = {
                "target_pitch_fundamental_amplitude_deg": math.degrees(target_amp[segment, side_index]),
                "actual_pitch_fundamental_amplitude_deg": math.degrees(actual_amp[segment, side_index]),
                "tracking_ratio": float(tracking_ratio[segment, side_index]),
                "contact_fraction": float(np.mean(contact[:, segment, side_index])),
                "mean_vertical_support_mn": 1000.0 * float(np.mean(active_force)),
                "vertical_support_p95_mn": 1000.0 * pct(active_force, 0.95),
                "mean_abs_roll_moment_unm": 1.0e6 * float(np.mean(np.abs(roll_moment[:, segment, side_index]))),
                "pitch_torque_abs_p95_unm": 1.0e6 * pct(np.abs(arrays["pitch_torque_nm"][:, segment, side_index]), 0.95),
            }
        per_segment.append(row)

    group_metrics: dict[str, object] = {}
    for group_name, ids in GROUPS.items():
        group_metrics[group_name] = {
            "target_pitch_amplitude_deg_mean": float(np.degrees(np.mean(target_amp[ids]))),
            "actual_pitch_amplitude_deg_mean": float(np.degrees(np.mean(actual_amp[ids]))),
            "tracking_ratio_mean": float(np.mean(tracking_ratio[ids])),
            "contact_fraction_mean": float(np.mean(contact[:, ids, :])),
            "vertical_support_mn_mean_per_leg": 1000.0 * float(np.mean(fz[:, ids, :])),
            "vertical_support_mn_total_mean": 1000.0 * float(np.mean(np.sum(fz[:, ids, :], axis=(1, 2)))),
            "pitch_torque_abs_p95_unm": 1.0e6 * pct(np.abs(arrays["pitch_torque_nm"][:, ids, :]), 0.95),
        }

    result = {
        "label": label,
        "diagnostic_only": True,
        "model": str(MODEL),
        "model_sha256": sha256(MODEL),
        "wave_body": wave,
        "wave_leg": wave,
        "wave_numbers_coupled": True,
        "flat_leg_amplitudes_held_fixed": True,
        "duration_s": duration_s,
        "eval_start_s": eval_start_s,
        "forward_displacement_signed_world_x_mm": 1000.0 * float((end_x or 0.0) - (start_x or 0.0)),
        "forward_displacement_magnitude_mm": 1000.0 * abs(float((end_x or 0.0) - (start_x or 0.0))),
        "mean_body_roll_abs_p95_deg": pct(np.abs(arrays["mean_body_roll_deg"]), 0.95),
        "contact_roll_moment_abs_p95_unm": 1.0e6 * pct(np.abs(arrays["total_roll_moment_nm"]), 0.95),
        "mean_left_contacting_feet": float(np.mean(arrays["left_count"])),
        "mean_right_contacting_feet": float(np.mean(arrays["right_count"])),
        "equal_left_right_contact_count_fraction": float(np.mean(equal_count)),
        "when_counts_equal": {
            "sample_count": int(np.count_nonzero(equal_count)),
            "lr_vertical_force_imbalance_abs_p95": pct(np.abs(selected_imbalance), 0.95),
            "contact_roll_moment_abs_p95_unm": 1.0e6 * pct(np.abs(selected_moment), 0.95),
            "contact_roll_moment_abs_max_unm": 1.0e6 * float(np.max(np.abs(selected_moment))) if selected_moment.size else float("nan"),
        },
        "per_segment": per_segment,
        "groups": group_metrics,
    }
    return result, arrays


def make_figure(cases: list[tuple[dict[str, object], dict[str, np.ndarray]]]) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    segments = np.arange(SEGMENTS)
    colors = {"selected_wave1p5": "#225ea8", "diagnostic_wave1p0": "#d95f0e"}
    for result, arrays in cases:
        label = str(result["label"])
        color = colors[label]
        time = arrays["time_s"]
        target_amp = np.degrees(fundamental_amplitude(arrays["target_pitch_rad"], time))
        actual_amp = np.degrees(fundamental_amplitude(arrays["actual_pitch_rad"], time))
        axes[0, 0].plot(segments, np.mean(actual_amp, axis=1), "o-", color=color, label=label)
        if label == "selected_wave1p5":
            axes[0, 0].plot(segments, np.mean(target_amp, axis=1), "--", color="#222222", label="target")
        support = 1000.0 * np.mean(arrays["foot_fz_n"], axis=0)
        axes[0, 1].plot(segments, support[:, 0], "o-", color=color, alpha=0.95, label=f"{label} L")
        axes[0, 1].plot(segments, support[:, 1], "s--", color=color, alpha=0.75, label=f"{label} R")
        axes[1, 0].plot(time, arrays["mean_body_roll_deg"], color=color, alpha=0.85, label=label)
        axes[1, 1].plot(time, 1.0e6 * arrays["total_roll_moment_nm"], color=color, alpha=0.7, label=label)

    axes[0, 0].set(title="Actual leg-pitch amplitude by segment", xlabel="segment (head to tail)", ylabel="fundamental amplitude (deg)")
    axes[0, 1].set(title="Mean vertical support by foot", xlabel="segment (head to tail)", ylabel="mean Fz (mN)")
    axes[1, 0].set(title="Whole-body mean roll", xlabel="time (s)", ylabel="roll (deg)")
    axes[1, 1].set(title="Ground-contact moment about head-forward axis", xlabel="time (s)", ylabel="moment (uN m)")
    for axis in axes.flat:
        axis.legend(fontsize=8)
    path = FIGURES / "roll_balance_wave1p5_vs_wave1p0.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(output: dict[str, object]) -> None:
    selected, diagnostic = output["cases"]
    selected_groups = selected["groups"]
    diagnostic_groups = diagnostic["groups"]
    selected_front_delta = (
        selected_groups["front_0_5"]["actual_pitch_amplitude_deg_mean"]
        - selected_groups["middle_6_12"]["actual_pitch_amplitude_deg_mean"]
    )
    lines = [
        "# Roll-balance audit",
        "",
        "This is an evidence-only experiment on the abstract Centipede Model. It does not change the selected model or controller.",
        "",
        "## Baseline and rollback status",
        "",
        "- The selected flat baseline remains coupled body/leg `wave=1.5`; `wave=1.0` below is diagnostic only.",
        "- No first-pitch (`q1`) stiffness/damping override is applied.",
        "- The locked-head model XML is read-only in this experiment.",
        "",
        "## Main findings",
        "",
        f"- Target leg-pitch amplitude is spatially uniform after warm-up: front/middle/rear means are "
        f"{selected_groups['front_0_5']['target_pitch_amplitude_deg_mean']:.3f}/"
        f"{selected_groups['middle_6_12']['target_pitch_amplitude_deg_mean']:.3f}/"
        f"{selected_groups['rear_13_18']['target_pitch_amplitude_deg_mean']:.3f} deg.",
        f"- Actual amplitude front/middle/rear is "
        f"{selected_groups['front_0_5']['actual_pitch_amplitude_deg_mean']:.3f}/"
        f"{selected_groups['middle_6_12']['actual_pitch_amplitude_deg_mean']:.3f}/"
        f"{selected_groups['rear_13_18']['actual_pitch_amplitude_deg_mean']:.3f} deg.",
        f"  The front-minus-middle difference is only {selected_front_delta:.3f} deg; it is not a front-half amplitude collapse. "
        f"In the wave=1.0 ablation the ordering reverses (front/middle "
        f"{diagnostic_groups['front_0_5']['actual_pitch_amplitude_deg_mean']:.3f}/"
        f"{diagnostic_groups['middle_6_12']['actual_pitch_amplitude_deg_mean']:.3f} deg).",
        f"- In the selected gait, left/right contact counts are exactly equal in "
        f"{100.0 * selected['equal_left_right_contact_count_fraction']:.1f}% of steady-state samples. "
        f"Even within those samples, the P95 left/right vertical-force imbalance is "
        f"{selected['when_counts_equal']['lr_vertical_force_imbalance_abs_p95']:.3f} and the P95 contact roll moment is "
        f"{selected['when_counts_equal']['contact_roll_moment_abs_p95_unm']:.3f} uN m.",
        "- Equal contact count is therefore not a zero-roll condition. Zero roll moment would require force magnitude, lever arm, friction force, timing, and local body orientation to cancel pairwise.",
        f"- The signed mean-across-links roll P95 is {selected['mean_body_roll_abs_p95_deg']:.3f} deg at wave=1.5 and "
        f"{diagnostic['mean_body_roll_abs_p95_deg']:.3f} deg at wave=1.0. This metric can fall through cancellation "
        f"between differently tilted links, so it is diagnostic only and cannot select the wave=1.0 gait.",
        "",
        "## Mechanics",
        "",
        "For the instantaneous body-forward axis, the contact contribution is `tau_roll = sum((r x F) dot e_forward)`. "
        "Its dominant vertical-force term is approximately `sum(y_i Fz_i)`, but lateral friction and vertical offsets also contribute. "
        "The user's link-plus-leg 'spindle' analogy is mechanically sound: leg-ground reaction can lift and rotate each compliant body link, and passive body-roll joints allow an imbalance to accumulate into a tilted support surface.",
        "",
        f"Figure: `{output['figure']}`",
        f"Data: `{output['data']}`",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in (DATA, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    sentinels_before = tip.robot_sentinels()
    cases = []
    for label, wave in (("selected_wave1p5", 1.5), ("diagnostic_wave1p0", 1.0)):
        print(f"[audit] START {label}", flush=True)
        result, arrays = run_case(label, wave)
        np.savez_compressed(DATA / f"{label}_traces.npz", **arrays)
        cases.append((result, arrays))
        print(
            f"[audit] DONE {label}: rollP95={result['mean_body_roll_abs_p95_deg']:.3f} deg, "
            f"equal-count={100.0 * result['equal_left_right_contact_count_fraction']:.1f}%",
            flush=True,
        )
    figure = make_figure(cases)
    output_path = DATA / "roll_balance_audit.json"
    output = {
        "purpose": "test whether front leg-pitch amplitude and equal left/right contact counts explain whole-body roll",
        "selection_changed": False,
        "q1_pitch_impedance_override_applied": False,
        "production_flat_wave": 1.5,
        "diagnostic_wave": 1.0,
        "cases": [result for result, _ in cases],
        "figure": str(figure),
        "data": str(output_path),
        "robot_sentinels_before": sentinels_before,
        "robot_sentinels_after": tip.robot_sentinels(),
    }
    output["robot_sentinels_unchanged"] = output["robot_sentinels_before"] == output["robot_sentinels_after"]
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_report(output)
    print(json.dumps({"data": str(output_path), "figure": str(figure), "report": str(REPORT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
