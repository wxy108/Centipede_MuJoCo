"""Cold-replay nominal winners and quantify integer-wave random erasure.

Nominal selection is never changed by this script.  Random contact is a
separate q=0.35, ten-seed robustness report using the established independent
short-burst collision-disable protocol.
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "data"
PILOT_PATH = ROOT / "run_broad_sobol_pilot.py"
SELECTION = DATA / "final_certified_selection.json"
ERASURE_Q = 0.35
ERASURE_SEEDS = tuple(range(10))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pilot = load_module(PILOT_PATH, "centipede_final_validator")


def v2_base_gates(row: dict) -> dict[str, bool]:
    required = {
        name: bool(passed)
        for name, passed in row["gates"].items()
        if name not in {"tip_speed", "no_chassis_drag"}
    }
    required["accepted_clean_chassis_contact_envelope"] = (
        float(row["chassis_ground_contact_fraction"]) <= 0.08
    )
    return required


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    report = {
        "scope": "Centipede Model only; Robot Step 18/19 untouched",
        "nominal_contract": "cold stride-1 six-second replay with the same complete V2 gates",
        "random_contact_contract": {
            "q": ERASURE_Q,
            "seeds": list(ERASURE_SEEDS),
            "interpretation": "separate robustness validation; never part of nominal amplitude optimization",
        },
        "waves": {},
    }
    for wave_text, item in selection["waves"].items():
        selected = item.get("selected")
        if selected is None:
            report["waves"][wave_text] = {"status": "no nominal V2-feasible candidate"}
            continue
        wave = float(selected["wave"])
        body = float(selected["body_yaw_amplitude_deg"])
        leg_yaw = float(selected["leg_yaw_amplitude_deg"])
        leg_pitch = float(selected["leg_pitch_amplitude_deg"])
        nominal = pilot.PilotEvaluator(1).evaluate(
            (f"validate_{selected['candidate_id']}", "cold_final_replay", wave, body, leg_yaw, leg_pitch, 6.0, 1.5)
        )
        expected = float(selected["forward_displacement_m"])
        exact_repeat = abs(float(nominal["forward_displacement_m"]) - expected) <= 1e-12
        if not nominal["feasible_v2_pilot"] or not exact_repeat:
            raise RuntimeError(
                f"nominal cold replay failed for wave {wave:g}: feasible={nominal['feasible_v2_pilot']} exact={exact_repeat}"
            )
        wave_report = {
            "status": "nominal certified",
            "candidate_id": selected["candidate_id"],
            "amplitudes_deg": [body, leg_yaw, leg_pitch],
            "nominal_forward_displacement_m": float(nominal["forward_displacement_m"]),
            "exact_repeat": exact_repeat,
            "nominal_base_gates": nominal["base_gates_v2"],
            "nominal_naturality_gates": nominal["naturality_gates"],
        }
        if math.isclose(wave, round(wave)):
            evaluator = pilot.opt.FastEvaluator()
            erasure_rows = []
            for seed in ERASURE_SEEDS:
                row = evaluator.evaluate((wave, body, leg_yaw, leg_pitch, 6.0, 1.5, ERASURE_Q, seed))
                gates = v2_base_gates(row)
                erasure_rows.append(
                    {
                        "seed": seed,
                        "forward_displacement_m": float(row["forward_displacement_m"]),
                        "retention": float(row["forward_displacement_m"]) / max(expected, 1e-12),
                        "v2_base_gate_pass": all(gates.values()),
                        "failed_v2_base_gates": [name for name, passed in gates.items() if not passed],
                        "triggered_events": int(row["erasure_triggered_events"]),
                        "opportunities": int(row["erasure_opportunities"]),
                        "realized_erased_load": float(row["realized_erased_load"]),
                    }
                )
            retention = [row["retention"] for row in erasure_rows]
            wave_report["random_erasure_q035"] = {
                "seed_rows": erasure_rows,
                "strict_v2_base_survival_count": sum(row["v2_base_gate_pass"] for row in erasure_rows),
                "seed_count": len(erasure_rows),
                "retention_mean": statistics.fmean(retention),
                "retention_p05": percentile(retention, 0.05),
                "retention_min": min(retention),
            }
        report["waves"][wave_text] = wave_report
        print(f"validated wave {wave:g}", flush=True)
    output = DATA / "final_selection_cold_replay_and_erasure.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
