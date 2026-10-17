"""Verify that proof-based early exit preserves feasible candidates."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "data"
PILOT = ROOT / "run_broad_sobol_pilot.py"
OUTPUT = DATA / "proof_early_exit_equivalence.json"


def load_pilot():
    spec = importlib.util.spec_from_file_location("centipede_proof_pilot", PILOT)
    if spec is None or spec.loader is None:
        raise RuntimeError(PILOT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def certified_rows():
    rows = []
    for path in sorted(DATA.glob("broad_sobol_pilot_trials_cert_*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def main():
    pilot = load_pilot()
    clean = dict(pilot.anchors())["accepted_clean_amplitudes"]
    prior = json.loads((DATA / "control_stride_equivalence.json").read_text(encoding="utf-8"))
    clean_reference = prior["results"]["accepted_clean_w1p5_stride1"]
    chassis_reference = next(
        row
        for row in certified_rows()
        if not row["base_gates_v2"]["accepted_clean_chassis_contact_envelope"]
    )
    evaluator = pilot.PilotEvaluator(1, True)
    tasks = [
        ("accepted_clean", 1.5, *clean),
        (
            "known_chassis_failure",
            float(chassis_reference["wave"]),
            float(chassis_reference["body_yaw_amplitude_deg"]),
            float(chassis_reference["leg_yaw_amplitude_deg"]),
            float(chassis_reference["leg_pitch_amplitude_deg"]),
        ),
    ]
    results = {}
    for name, wave, body, yaw, pitch in tasks:
        started = time.perf_counter()
        row = evaluator.evaluate(
            (f"proof_{name}", "proof_early_exit_benchmark", wave, body, yaw, pitch, 6.0, 1.5)
        )
        row["wall_time_s"] = time.perf_counter() - started
        results[name] = row
        print(
            name,
            "Dmm",
            round(1000.0 * row["forward_displacement_m"], 3),
            "base",
            row["provisional_base_feasible"],
            "natural",
            row["feasible_v2_pilot"],
            "early",
            row.get("early_terminated_by"),
            "wall_s",
            round(row["wall_time_s"], 2),
            flush=True,
        )
    payload = {
        "accepted_clean": {
            "reference_displacement_m": clean_reference["forward_displacement_m"],
            "proof_displacement_m": results["accepted_clean"]["forward_displacement_m"],
            "displacement_difference_m": results["accepted_clean"]["forward_displacement_m"]
            - clean_reference["forward_displacement_m"],
            "reference_feasible": clean_reference["feasible_v2_pilot"],
            "proof_feasible": results["accepted_clean"]["feasible_v2_pilot"],
            "early_terminated_by": results["accepted_clean"].get("early_terminated_by"),
        },
        "known_chassis_failure": {
            "reference_candidate_id": chassis_reference["candidate_id"],
            "reference_feasible": chassis_reference["feasible_v2_pilot"],
            "proof_feasible": results["known_chassis_failure"]["feasible_v2_pilot"],
            "reference_chassis_fraction": chassis_reference["chassis_ground_contact_fraction"],
            "proof_early_terminated_by": results["known_chassis_failure"].get("early_terminated_by"),
            "proof_wall_time_s": results["known_chassis_failure"]["wall_time_s"],
        },
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
