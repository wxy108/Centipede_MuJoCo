"""Merge and certify the finite broad-plus-refinement candidate bank.

This script does not simulate, edit the model, or relax any gate.  It only
deduplicates completed stride-1, six-second rows and ranks displacement among
rows that already passed every V2 base and naturality gate.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "data"
WAVES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def false_names(mapping: dict) -> list[str]:
    return [name for name, passed in mapping.items() if not bool(passed)]


def diagnostic_key(row: dict):
    """Prefer candidates closest to the hard feasible set, then distance."""
    if row.get("feasible_v2_pilot"):
        tier, failures = 0, 0
    elif row.get("provisional_base_feasible"):
        tier, failures = 1, len(false_names(row.get("naturality_gates", {})))
    else:
        tier, failures = 2, len(false_names(row.get("base_gates_v2", {})))
    return (tier, failures, -float(row.get("forward_displacement_m", -math.inf)))


def compact(row: dict | None) -> dict | None:
    if row is None:
        return None
    keep = (
        "candidate_id",
        "source",
        "refinement_stage",
        "wave",
        "body_yaw_amplitude_deg",
        "leg_yaw_amplitude_deg",
        "leg_pitch_amplitude_deg",
        "duration_s",
        "eval_start_s",
        "control_stride",
        "forward_displacement_m",
        "provisional_base_feasible",
        "feasible_v2_pilot",
        "base_gates_v2",
        "naturality_gates",
    )
    return {key: row[key] for key in keep if key in row}


def main() -> None:
    paths = [DATA / "broad_sobol_pilot_certified_trials.jsonl"]
    paths.extend(sorted(DATA.glob("adaptive_refinement_trials_s[123]_*.jsonl")))
    rows_by_id: dict[str, dict] = {}
    duplicates = 0
    for path in paths:
        for row in read_jsonl(path):
            candidate_id = str(row["candidate_id"])
            existing = rows_by_id.get(candidate_id)
            if existing is not None:
                if existing != row:
                    raise RuntimeError(f"conflicting duplicate candidate {candidate_id}")
                duplicates += 1
                continue
            rows_by_id[candidate_id] = row
    rows = list(rows_by_id.values())
    if not rows:
        raise RuntimeError("no candidate rows found")

    invalid_certification_rows = [
        row["candidate_id"]
        for row in rows
        if row.get("feasible_v2_pilot")
        and (
            int(row.get("control_stride", -1)) != 1
            or float(row.get("duration_s", 0.0)) < 6.0
            or not all(row.get("base_gates_v2", {}).values())
            or not all(row.get("naturality_gates", {}).values())
        )
    ]
    if invalid_certification_rows:
        raise RuntimeError(f"invalid certified rows: {invalid_certification_rows[:5]}")

    wave_payload = {}
    table_rows = []
    for wave in WAVES:
        group = [row for row in rows if math.isclose(float(row["wave"]), wave)]
        feasible = [row for row in group if bool(row.get("feasible_v2_pilot"))]
        winner = max(feasible, key=lambda row: float(row["forward_displacement_m"])) if feasible else None
        diagnostic = min(group, key=diagnostic_key) if group else None
        stages = {}
        for row in group:
            label = str(row.get("refinement_stage", "broad"))
            stages[label] = stages.get(label, 0) + 1
        item = {
            "status": "certified_feasible" if winner else "no_feasible_candidate_in_declared_bank",
            "candidate_count": len(group),
            "base_feasible_count": sum(bool(row.get("provisional_base_feasible")) for row in group),
            "natural_feasible_count": len(feasible),
            "counts_by_stage": stages,
            "selected": compact(winner),
            "diagnostic_nearest": compact(diagnostic),
        }
        wave_payload[f"{wave:g}"] = item
        selected = item["selected"]
        table_rows.append(
            {
                "wave": wave,
                "status": item["status"],
                "candidate_count": item["candidate_count"],
                "base_feasible_count": item["base_feasible_count"],
                "natural_feasible_count": item["natural_feasible_count"],
                "candidate_id": selected.get("candidate_id") if selected else "",
                "body_yaw_deg": selected.get("body_yaw_amplitude_deg") if selected else "",
                "leg_yaw_deg": selected.get("leg_yaw_amplitude_deg") if selected else "",
                "leg_pitch_deg": selected.get("leg_pitch_amplitude_deg") if selected else "",
                "forward_displacement_mm": 1000.0 * float(selected["forward_displacement_m"]) if selected else "",
            }
        )

    payload = {
        "scope": "Centipede Model only; Centipede Robot Step 18/19 untouched",
        "claim": "maximum displacement within the completed declared finite broad-plus-adaptive candidate bank, after all V2 hard gates",
        "continuous_global_optimum_claim": False,
        "ranking": "hard-gate feasibility first; forward displacement only among feasible rows",
        "candidate_files": [str(path) for path in paths if path.exists()],
        "unique_candidate_count": len(rows),
        "identical_duplicate_rows_ignored": duplicates,
        "waves": wave_payload,
    }
    json_path = DATA / "final_certified_selection.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = DATA / "final_certified_selection.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps({"rows": len(rows), "duplicates": duplicates, "json": str(json_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
