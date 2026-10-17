"""Validate multi-fidelity control updates against stride-1 certification."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "run_broad_sobol_pilot.py"
OUTPUT = ROOT / "outputs" / "data" / "control_stride_equivalence.json"


def load_pilot():
    spec = importlib.util.spec_from_file_location("centipede_stride_pilot", PILOT)
    if spec is None or spec.loader is None:
        raise RuntimeError(PILOT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    pilot = load_pilot()
    clean = dict(pilot.anchors())["accepted_clean_amplitudes"]
    isaac = dict(pilot.anchors())["isaac_yaml_amplitudes"]
    rejected = pilot.rejected_lookup()[3.0]
    cases = [
        ("accepted_clean_w1p5", 1.5, *clean, 6.0, 1.5),
        ("isaac_anchor_w1p5", 1.5, *isaac, 6.0, 1.5),
        ("rejected_negative_w3", 3.0, *rejected, 6.0, 1.5),
    ]
    results = {}
    for stride in (1, 4):
        evaluator = pilot.PilotEvaluator(stride)
        for name, wave, body, yaw, pitch, duration, eval_start in cases:
            started = time.perf_counter()
            row = evaluator.evaluate(
                (f"{name}_stride{stride}", "stride_benchmark", wave, body, yaw, pitch, duration, eval_start)
            )
            row["wall_time_s"] = time.perf_counter() - started
            results[f"{name}_stride{stride}"] = row
            print(
                name,
                "stride",
                stride,
                "Dmm",
                round(1000.0 * row["forward_displacement_m"], 3),
                "base",
                row["provisional_base_feasible"],
                "natural",
                row["feasible_v2_pilot"],
                "wall_s",
                round(row["wall_time_s"], 2),
                flush=True,
            )

    comparisons = {}
    metric_keys = (
        "forward_displacement_m",
        "path_efficiency",
        "coherent_roll_p95_deg",
        "front7_roll_p95_deg",
        "chassis_ground_contact_fraction",
        "contact_penetration_p95_mm",
        "body_nonsinusoidal_residual_p95",
        "leg_yaw_speed_p99_rad_s",
        "leg_pitch_speed_p99_rad_s",
        "leg_pitch_tracking_error_rms_deg",
        "swing_foot_height_p95_mm",
        "contact_duty_mean",
    )
    for name, *_ in cases:
        full = results[f"{name}_stride1"]
        fast = results[f"{name}_stride4"]
        differences = {}
        for key in metric_keys:
            if key not in full or key not in fast:
                continue
            a, b = float(full[key]), float(fast[key])
            differences[key] = {
                "stride1": a,
                "stride4": b,
                "absolute_difference": b - a,
                "relative_difference": (b - a) / max(abs(a), 1e-12),
            }
        comparisons[name] = {
            "base_gate_parity": full["base_gates_v2"] == fast["base_gates_v2"],
            "naturality_gate_parity": full["naturality_gates"] == fast["naturality_gates"],
            "final_feasible_parity": full["feasible_v2_pilot"] == fast["feasible_v2_pilot"],
            "speedup": full["wall_time_s"] / max(fast["wall_time_s"], 1e-12),
            "metrics": differences,
        }
    payload = {
        "scope": "Centipede Model only; flat q=0",
        "physics_timestep_s": 0.0005,
        "stride1_control_update_s": 0.0005,
        "stride4_control_update_s": 0.002,
        "requirement": "stride 4 is screening-only; all finalists are re-evaluated at stride 1",
        "results": results,
        "comparisons": comparisons,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
