"""Read-only causal audit of when flat low-wave roll first appeared.

No MJCF is generated or modified.  Historical models are replayed in place.
The only interventions are runtime controller-table swaps used to separate
controller inheritance from plant/contact/topology changes.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# The current MuJoCo 3.10 package can block while loading optional bundled
# plugins on this host.  These historical models contain no plugin instances.
_original_cdll = ctypes.CDLL


def _cdll_without_optional_plugins(path, *args, **kwargs):
    if "site-packages\\mujoco\\plugin" in os.path.normcase(str(path)):
        return None
    return _original_cdll(path, *args, **kwargs)


ctypes.CDLL = _cdll_without_optional_plugins
try:
    import mujoco
finally:
    ctypes.CDLL = _original_cdll


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parents[1]
EXPERIMENTS = MODEL_ROOT / "experiments"
CTRL_SCRIPT = EXPERIMENTS / "mujoco_zero_armature_controller_optimization_2026-08-26" / "controller_optimization.py"
BODY_ROLL_SCRIPT = EXPERIMENTS / "mujoco_template_body_roll_2026-08-27" / "run_body_roll.py"
STAGE1_MODEL = EXPERIMENTS / "mujoco_template_fidelity_stage1_2026-08-26" / "models" / "centipede_flat_stiff_dt0p25ms_q2pm30p0deg_restoring_k30_c18.xml"
Q2_MODEL = EXPERIMENTS / "mujoco_template_q2_compliance_2026-08-26" / "models" / "centipede_flat_stiff_dt0p25ms_q2pm30p0deg_bounded_k20_c18.xml"
BIO_SELECTION = EXPERIMENTS / "mujoco_biophysical_terrain_contact_audit_2026-08-26" / "outputs" / "data" / "controller_selection.json"
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"
FIGURES = OUTPUTS / "figures"
REPORT = ROOT / "LOW_WAVE_ROLL_DIAGNOSIS.md"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ctrl = load_module(CTRL_SCRIPT, "centipede_roll_onset_old_controller")
bodyroll = load_module(BODY_ROLL_SCRIPT, "centipede_roll_onset_bodyroll")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def pct(values, q: float) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.quantile(array, q)) if array.size else float("nan")


def fundamental_amplitude_deg(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    omega = 2.0 * math.pi * float(ctrl.base.FREQUENCY_HZ)
    design = np.column_stack((np.ones_like(times), np.sin(omega * times), np.cos(omega * times)))
    flat = values.reshape(len(times), -1)
    coefficients, *_ = np.linalg.lstsq(design, flat, rcond=None)
    return np.degrees(np.hypot(coefficients[1], coefficients[2])).reshape(values.shape[1:])


def normalized_impedance(raw: dict[str, object]) -> dict[str, tuple[float, float]]:
    return {name: (float(pair[0]), float(pair[1])) for name, pair in raw.items()}


def replay_original_plant(
    label: str,
    impedance: dict[str, tuple[float, float]],
    video: Path | None = None,
    camera_azimuth: float = 90.0,
    camera_elevation: float = -4.0,
    camera_distance: float = 0.19,
) -> dict[str, object]:
    """Replay the pre-Stage1 plant while capturing modern roll metrics."""
    original_posture = ctrl.base.posture_metrics
    times: list[float] = []
    local_abs: list[float] = []
    cumulative_span: list[float] = []
    world_abs: list[float] = []
    head_abs: list[float] = []
    world_common_abs: list[float] = []
    world_front_common_abs: list[float] = []
    world_span: list[float] = []
    leg_pitch: list[np.ndarray] = []

    def capture(data, index):
        result = original_posture(data, index)
        if float(data.time) < 1.5:
            return result
        q_local = np.asarray(
            data.qpos[index.qpos_addresses[index.roll_positions]], dtype=float
        )
        local_deg = np.degrees(q_local)
        local_abs.extend(np.abs(local_deg).tolist())
        cumulative_span.append(float(np.ptp(np.concatenate(([0.0], np.cumsum(local_deg))))))
        rotations = data.xmat[index.chassis_body_ids].reshape(-1, 3, 3)
        gravity_local = np.einsum(
            "nij,j->ni", rotations.transpose(0, 2, 1), np.asarray([0.0, 0.0, -1.0])
        )
        world = np.degrees(np.arctan2(-gravity_local[:, 1], -gravity_local[:, 2]))
        world_abs.extend(np.abs(world).tolist())
        head_abs.append(float(abs(world[0])))
        world_common_abs.append(float(abs(np.mean(world))))
        world_front_common_abs.append(float(abs(np.mean(world[:7]))))
        world_span.append(float(np.ptp(world)))
        pitch_positions = index.leg_positions[:, :, 1]
        leg_pitch.append(np.asarray(data.qpos[index.qpos_addresses[pitch_positions]], dtype=float).copy())
        times.append(float(data.time))
        return result

    ctrl.base.posture_metrics = capture
    try:
        row = ctrl.run_flat_lift_diagnostic(
            impedance,
            duration_s=6.0,
            eval_start_s=1.5,
            side_video=video,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            camera_distance=camera_distance,
        )
    finally:
        ctrl.base.posture_metrics = original_posture
    amplitudes = fundamental_amplitude_deg(np.asarray(leg_pitch), np.asarray(times))
    row.update(
        {
            "label": label,
            "intervention": "controller table only; historical plant unchanged",
            "plant": str(ctrl.BASE_MODEL),
            "plant_sha256": sha256(Path(ctrl.BASE_MODEL)),
            "timestep_s": 0.001,
            "q2_mode": "historical active implicit PD",
            "body_roll_local_abs_p95_deg": pct(local_abs, 0.95),
            "body_roll_cumulative_span_p95_deg": pct(cumulative_span, 0.95),
            "chassis_world_roll_abs_p95_deg": pct(world_abs, 0.95),
            "head_world_roll_abs_p95_deg": pct(head_abs, 0.95),
            "chassis_world_roll_common_mode_abs_p95_deg": pct(world_common_abs, 0.95),
            "front7_world_roll_common_mode_abs_p95_deg": pct(world_front_common_abs, 0.95),
            "chassis_world_roll_span_p95_deg": pct(world_span, 0.95),
            "leg_pitch_amplitude_front_deg": float(np.mean(amplitudes[0:6])),
            "leg_pitch_amplitude_middle_deg": float(np.mean(amplitudes[6:13])),
            "leg_pitch_amplitude_rear_deg": float(np.mean(amplitudes[13:19])),
        }
    )
    return row


def replay_stage1_plant(
    label: str,
    model_path: Path,
    impedance: dict[str, tuple[float, float]],
    q2_label: str,
    video: Path | None = None,
    camera_azimuth: float = 90.0,
    camera_elevation: float = -4.0,
    camera_distance: float = 0.19,
    candidate: str = "g1_r30",
) -> dict[str, object]:
    """Replay the post-contact-calibration plant with a supplied table."""
    original_selected = bodyroll.stage1.selected_impedance
    original_posture = bodyroll.stage1.base.posture_metrics
    world_common_abs: list[float] = []
    world_front_common_abs: list[float] = []
    world_span: list[float] = []

    def selected_override():
        return dict(impedance)

    def common_mode_capture(data, index):
        result = original_posture(data, index)
        if float(data.time) < 1.5:
            return result
        rotations = data.xmat[index.chassis_body_ids].reshape(-1, 3, 3)
        gravity_local = np.einsum(
            "nij,j->ni", rotations.transpose(0, 2, 1), np.asarray([0.0, 0.0, -1.0])
        )
        world = np.degrees(np.arctan2(-gravity_local[:, 1], -gravity_local[:, 2]))
        world_common_abs.append(float(abs(np.mean(world))))
        world_front_common_abs.append(float(abs(np.mean(world[:7]))))
        world_span.append(float(np.ptp(world)))
        return result

    bodyroll.stage1.selected_impedance = selected_override
    bodyroll.stage1.base.posture_metrics = common_mode_capture
    try:
        row = bodyroll.run_with_roll_capture(
            candidate,
            "flat",
            model_path,
            6.0,
            video=video,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            camera_distance=camera_distance,
        )
    finally:
        bodyroll.stage1.selected_impedance = original_selected
        bodyroll.stage1.base.posture_metrics = original_posture
    row.update(
        {
            "label": label,
            "intervention": "runtime controller table only; historical MJCF unchanged",
            "plant": str(model_path),
            "plant_sha256": sha256(model_path),
            "q2_historical_label": q2_label,
            "chassis_world_roll_common_mode_abs_p95_deg": pct(world_common_abs, 0.95),
            "front7_world_roll_common_mode_abs_p95_deg": pct(world_front_common_abs, 0.95),
            "chassis_world_roll_span_p95_deg_recomputed": pct(world_span, 0.95),
        }
    )
    return row


def sentinel_hashes() -> dict[str, str]:
    archive = MODEL_ROOT.parent / "Centipede_robot_MUJOCO" / "exp_data" / "reliability" / "_archive" / "legacy_steps_01_20"
    paths = {
        "step18_run_configuration": archive / "step18_body_wave_only_random_pitch" / "run_configuration.json",
        "step18_trials": archive / "step18_body_wave_only_random_pitch" / "trials.csv",
        "step19_run_configuration": archive / "step19_phi_pi_over_2_body_wave_only_random_pitch" / "run_configuration.json",
        "step19_trials": archive / "step19_phi_pi_over_2_body_wave_only_random_pitch" / "trials.csv",
    }
    return {name: sha256(path) if path.exists() else "MISSING" for name, path in paths.items()}


def make_figure(rows: list[dict[str, object]]) -> Path:
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(labels))
    world = [float(row["chassis_world_roll_abs_p95_deg"]) for row in rows]
    local = [float(row["body_roll_local_abs_p95_deg"]) for row in rows]
    displacement = [1000.0 * float(row["forward_displacement_m"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, world, width, label="chassis world-roll P95")
    axes[0].bar(x + width / 2, local, width, label="local body-roll P95")
    axes[0].set_ylabel("degrees")
    axes[0].set_title("Flat wave=1.5 roll across historical boundaries")
    axes[0].legend()
    axes[1].bar(x, displacement, color="#3182bd")
    axes[1].set_ylabel("forward displacement (mm / 4.5 s eval)")
    axes[1].set_title("Locomotion retained during the same replays")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.3)
    path = FIGURES / "flat_low_wave_roll_onset.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    for path in (DATA, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    sentinels_before = sentinel_hashes()
    old_impedance = normalized_impedance(ctrl.final_impedance())
    selection = json.loads(BIO_SELECTION.read_text(encoding="utf-8"))
    inherited_impedance = normalized_impedance(selection["selected_impedance"])

    rows = []
    cases = (
        ("A old plant + old flat controller", lambda: replay_original_plant("A old plant + old flat controller", old_impedance)),
        ("B old plant + inherited contact-audit controller", lambda: replay_original_plant("B old plant + inherited contact-audit controller", inherited_impedance)),
        ("C Stage1 k30 plant + old flat controller", lambda: replay_stage1_plant("C Stage1 k30 plant + old flat controller", STAGE1_MODEL, old_impedance, "restoring_k30_c18")),
        ("D Stage1 k30 plant + inherited controller", lambda: replay_stage1_plant("D Stage1 k30 plant + inherited controller", STAGE1_MODEL, inherited_impedance, "restoring_k30_c18")),
        ("E q2 k20 plant + inherited controller", lambda: replay_stage1_plant("E q2 k20 plant + inherited controller", Q2_MODEL, inherited_impedance, "bounded_k20_c18")),
    )
    for name, run in cases:
        print(f"[diagnosis] START {name}", flush=True)
        row = run()
        rows.append(row)
        print(
            f"[diagnosis] DONE {name}: world={row['chassis_world_roll_abs_p95_deg']:.3f} deg, "
            f"local={row['body_roll_local_abs_p95_deg']:.3f} deg, "
            f"disp={1000.0 * row['forward_displacement_m']:.2f} mm",
            flush=True,
        )

    gains = {}
    for family in ("body_roll", "legbase_pitch", "legbase_roll", "knee"):
        old = old_impedance[family]
        new = inherited_impedance[family]
        gains[family] = {
            "old_kp": old[0],
            "old_kv": old[1],
            "inherited_kp": new[0],
            "inherited_kv": new[1],
            "kp_ratio_inherited_to_old": new[0] / old[0],
            "kv_ratio_inherited_to_old": new[1] / old[1],
        }
    figure = make_figure(rows)
    output_path = DATA / "flat_low_wave_roll_onset.json"
    output = {
        "purpose": "diagnose when flat low-wave whole-body roll first appeared without changing a model",
        "model_files_modified": False,
        "production_controller_modified": False,
        "wave": 1.5,
        "rows": rows,
        "controller_gain_boundary": gains,
        "contact_audit_selection_status": selection.get("selected_status"),
        "figure": str(figure),
        "robot_sentinels_before": sentinels_before,
        "robot_sentinels_after": sentinel_hashes(),
    }
    output["robot_sentinels_unchanged"] = output["robot_sentinels_before"] == output["robot_sentinels_after"]
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"data": str(output_path), "figure": str(figure)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
