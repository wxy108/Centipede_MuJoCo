"""Separate coherent whole-body roll from segment-wise twist in historical runs."""

import json

import diagnose_roll_onset as audit
import numpy as np


PASSIVE_MODEL = (
    audit.EXPERIMENTS
    / "mujoco_template_passive_body_roll_2026-08-27"
    / "models"
    / "centipede_flat_passive_k4_coupled_wave.xml"
)
PASSIVE_SCRIPT = (
    audit.EXPERIMENTS
    / "mujoco_template_passive_body_roll_2026-08-27"
    / "run_passive_body_roll.py"
)
passive = audit.load_module(PASSIVE_SCRIPT, "centipede_roll_onset_passive_runner")
OUTPUT = audit.DATA / "flat_low_wave_roll_common_mode.json"


def compact(row):
    keys = (
        "label",
        "chassis_world_roll_abs_p95_deg",
        "head_world_roll_abs_p95_deg",
        "chassis_world_roll_common_mode_abs_p95_deg",
        "front7_world_roll_common_mode_abs_p95_deg",
        "chassis_world_roll_span_p95_deg",
        "chassis_world_roll_span_p95_deg_recomputed",
        "body_roll_local_abs_p95_deg",
        "forward_displacement_m",
    )
    return {key: row[key] for key in keys if key in row}


def save(rows):
    output = {
        "purpose": "separate coherent roll from segment-wise roll; no model changes",
        "models_modified": False,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")


def run_passive_common():
    original_posture = passive.stage1.base.posture_metrics
    common_abs = []
    front_abs = []
    spans = []

    def capture(data, index):
        result = original_posture(data, index)
        if float(data.time) < 1.5:
            return result
        rotations = data.xmat[index.chassis_body_ids].reshape(-1, 3, 3)
        gravity_local = np.einsum(
            "nij,j->ni", rotations.transpose(0, 2, 1), np.asarray([0.0, 0.0, -1.0])
        )
        world = np.degrees(np.arctan2(-gravity_local[:, 1], -gravity_local[:, 2]))
        common_abs.append(float(abs(np.mean(world))))
        front_abs.append(float(abs(np.mean(world[:7]))))
        spans.append(float(np.ptp(world)))
        return result

    passive.stage1.base.posture_metrics = capture
    try:
        row = passive.run_case("passive_k4", "flat", PASSIVE_MODEL, 6.0)
    finally:
        passive.stage1.base.posture_metrics = original_posture
    row["label"] = "P passive k4 body-roll"
    row["chassis_world_roll_common_mode_abs_p95_deg"] = audit.pct(common_abs, 0.95)
    row["front7_world_roll_common_mode_abs_p95_deg"] = audit.pct(front_abs, 0.95)
    row["chassis_world_roll_span_p95_deg_recomputed"] = audit.pct(spans, 0.95)
    return row


def main():
    old_impedance = audit.normalized_impedance(audit.ctrl.final_impedance())
    selection = json.loads(audit.BIO_SELECTION.read_text(encoding="utf-8"))
    inherited = audit.normalized_impedance(selection["selected_impedance"])
    rows = []
    print("[common-mode] old active plant", flush=True)
    rows.append(compact(audit.replay_original_plant("A old active", old_impedance)))
    save(rows)
    print("[common-mode] q2 active baseline", flush=True)
    rows.append(
        compact(
            audit.replay_stage1_plant(
                "E q2 k20 active body-roll",
                audit.Q2_MODEL,
                inherited,
                "bounded_k20_c18",
            )
        )
    )
    save(rows)
    print("[common-mode] passive k4 body-roll", flush=True)
    rows.append(compact(run_passive_common()))
    save(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
