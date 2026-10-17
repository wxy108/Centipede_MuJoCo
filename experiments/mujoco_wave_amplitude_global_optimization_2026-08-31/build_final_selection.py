"""Combine grid, long-horizon, and random-contact evidence into final selections."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "data"


def read(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def main():
    validation = read("winner_validation_summary.json")
    runnerups = read("runnerup_long_summary.json")
    reselected = read("reselected_random_summary.json")
    waves = {}
    compact = []
    for wave in range(1, 10):
        key = str(wave)
        primary = validation["waves"][key]
        if primary["winner"] is None:
            diagnostic = primary["diagnostic_best"]
            failed = [
                gate for gate, value in diagnostic["gates"].items() if not value
            ]
            item = {
                "status": "no_4s_feasible_grid_point",
                "grid_feasible_count": 0,
                "selected": None,
                "diagnostic_best": diagnostic,
                "diagnostic_failed_gates": failed,
            }
            waves[key] = item
            compact.append(
                {
                    "wave": wave,
                    "status": item["status"],
                    "grid_rank": "",
                    "body_yaw_deg": "",
                    "leg_yaw_deg": "",
                    "leg_pitch_deg": "",
                    "grid_displacement_mm": "",
                    "long_displacement_mm": "",
                    "grid_feasible_count": 0,
                    "random_mean_displacement_mm": "",
                    "random_retention": "",
                    "random_gate_survival": "",
                    "diagnostic_failed_gates": ";".join(failed),
                }
            )
            continue

        selected_rank = 1
        grid_row = primary["winner"]
        long_row = primary["nominal_long_validation"]
        status = "long_safe_grid_global"
        random_result = primary["random_contact"]
        if not bool(long_row["feasible"]):
            runner = runnerups["waves"].get(key)
            choice = runner.get("selected_long_safe") if runner else None
            if choice is None:
                status = "no_6s_safe_among_4s_feasible"
            else:
                selected_rank = int(choice["grid_rank"])
                grid_row = choice["grid_row"]
                long_row = choice["long_row"]
                status = "long_safe_ranked_runner_up"
                random_result = reselected["waves"][key]

        selected = {
            "grid_rank": selected_rank,
            "body_yaw_amplitude_deg": float(grid_row["body_yaw_amplitude_deg"]),
            "leg_yaw_amplitude_deg": float(grid_row["leg_yaw_amplitude_deg"]),
            "leg_pitch_amplitude_deg": float(grid_row["leg_pitch_amplitude_deg"]),
            "grid_row": grid_row,
            "long_row": long_row,
            "long_safe": bool(long_row["feasible"]),
            "maximum_4s_displacement_subject_to_4s_and_6s_gates": bool(
                long_row["feasible"]
            ),
            "random_contact": {
                "q": 0.3,
                "seed_count": 5,
                "forward_displacement_mean_m": float(
                    random_result["forward_displacement_mean_m"]
                ),
                "forward_displacement_min_m": float(
                    random_result["forward_displacement_min_m"]
                ),
                "mean_retention_vs_nominal": float(
                    random_result["mean_retention_vs_nominal"]
                ),
                "hard_gate_survival_fraction": float(
                    random_result["hard_gate_survival_fraction"]
                ),
                "mean_realized_erased_load": float(
                    random_result["mean_realized_erased_load"]
                ),
            },
        }
        item = {
            "status": status,
            "grid_feasible_count": int(primary["feasible_count"]),
            "primary_4s_grid_winner": primary["winner"],
            "selected": selected,
        }
        waves[key] = item
        compact.append(
            {
                "wave": wave,
                "status": status,
                "grid_rank": selected_rank,
                "body_yaw_deg": selected["body_yaw_amplitude_deg"],
                "leg_yaw_deg": selected["leg_yaw_amplitude_deg"],
                "leg_pitch_deg": selected["leg_pitch_amplitude_deg"],
                "grid_displacement_mm": 1000.0
                * float(grid_row["forward_displacement_m"]),
                "long_displacement_mm": 1000.0
                * float(long_row["forward_displacement_m"]),
                "grid_feasible_count": int(primary["feasible_count"]),
                "random_mean_displacement_mm": 1000.0
                * selected["random_contact"]["forward_displacement_mean_m"],
                "random_retention": selected["random_contact"][
                    "mean_retention_vs_nominal"
                ],
                "random_gate_survival": selected["random_contact"][
                    "hard_gate_survival_fraction"
                ],
                "diagnostic_failed_gates": "",
            }
        )

    payload = {
        "selection_rule": (
            "maximize 4 s forward displacement over the complete finite grid "
            "after all 4 s gates, then require the selected point to pass the "
            "same gates at 6 s; test higher-ranked candidates before accepting "
            "a runner-up"
        ),
        "continuous_domain_global_claimed": False,
        "random_contact_role": "post-selection robustness only",
        "model": validation["model"],
        "model_sha256": validation["model_sha256"],
        "armature_nonzero_count": validation["armature_nonzero_count"],
        "model_total_mass_kg": validation["model_total_mass_kg"],
        "physics_timestep_s": validation["physics_timestep_s"],
        "integrator": validation["integrator"],
        "robot_sentinels_unchanged": bool(
            validation["robot_sentinels_unchanged"]
            and runnerups["robot_sentinels_unchanged"]
            and reselected["robot_sentinels_unchanged"]
        ),
        "waves": waves,
    }
    (DATA / "final_selection_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (DATA / "final_selection_compact.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact[0]))
        writer.writeheader()
        writer.writerows(compact)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
