"""Merge isolated pilot shards and rebuild canonical CSV/summary outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "data"
JSONL = DATA / "broad_sobol_pilot_certified_trials.jsonl"
CSV = DATA / "broad_sobol_pilot_certified_trials.csv"
SUMMARY = DATA / "broad_sobol_pilot_certified_summary.json"
WAVES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0)


def canonical(row):
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def main():
    rows_by_id = {}
    # Only the clean, stride-1, direct-process certification shards belong in
    # the canonical pilot.  Earlier partial/process-pool diagnostics remain on
    # disk but are intentionally excluded.
    sources = sorted(DATA.glob("broad_sobol_pilot_trials_cert_*.jsonl"))
    for path in sources:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = row["candidate_id"]
            previous = rows_by_id.get(candidate_id)
            if previous is not None and canonical(previous) != canonical(row):
                raise RuntimeError(f"conflicting duplicate {candidate_id} in {path}:{line_number}")
            rows_by_id[candidate_id] = row

    rows = sorted(rows_by_id.values(), key=lambda r: (float(r["wave"]), str(r["candidate_id"])))
    JSONL.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")

    flat = []
    fieldnames = []
    for row in rows:
        record = {
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        flat.append(record)
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)

    summary = {
        "metadata": {
            "scope": "Centipede Model only; Robot Step 18/19 untouched",
            "waves": WAVES,
            "domain_deg": {"body_yaw": [0, 36], "leg_yaw": [0, 90], "leg_pitch": [0, 60]},
            "duration_s": 6.0,
            "eval_start_s": 1.5,
            "candidate_count": len(rows),
            "expected_candidate_count": 2039,
            "continuous_global_claim": False,
            "status": "broad pilot; adaptive refinement required",
            "source_shards": [str(p) for p in sources],
        },
        "waves": {},
    }
    for wave in WAVES:
        group = [r for r in rows if math.isclose(float(r["wave"]), wave)]
        feasible = [r for r in group if r["feasible_v2_pilot"]]
        feasible.sort(
            key=lambda r: (
                -float(r["forward_displacement_m"]),
                -float(r["path_efficiency"]),
                float(r["body_yaw_amplitude_deg"])
                + float(r["leg_yaw_amplitude_deg"])
                + float(r["leg_pitch_amplitude_deg"]),
            )
        )
        summary["waves"][f"{wave:g}"] = {
            "trials": len(group),
            "base_feasible": sum(bool(r["provisional_base_feasible"]) for r in group),
            "natural_feasible": len(feasible),
            "pilot_best": feasible[0] if feasible else None,
            "top10": feasible[:10],
        }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"sources": len(sources), "rows": len(rows), "expected": 2039}, indent=2))


if __name__ == "__main__":
    main()
