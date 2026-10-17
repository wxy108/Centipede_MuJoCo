"""Longer-horizon and random-contact validation of the frozen grid winners.

Selection is based only on the complete nominal flat-ground grid.  This file
does not retune parameters under random erasure: it validates the frozen
winner for each wave with the earlier seed-locked segment-pair protocol.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OPTIMIZER = ROOT / "run_wave_amplitude_global.py"
DATA = ROOT / "outputs" / "data"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


opt = load_module(OPTIMIZER, "centipede_wave_amplitude_optimizer")
_EVALUATOR = None


def worker_init() -> None:
    global _EVALUATOR
    _EVALUATOR = opt.FastEvaluator()


def worker_evaluate(task):
    if _EVALUATOR is None:
        worker_init()
    return _EVALUATOR.evaluate(task)


def validation_key(row_or_task):
    if isinstance(row_or_task, dict):
        return (
            int(row_or_task["wave"]),
            float(row_or_task["erasure_q"]),
            int(row_or_task["erasure_seed"]),
            float(row_or_task["duration_s"]),
            float(row_or_task["eval_start_s"]),
        )
    return (
        int(row_or_task[0]),
        float(row_or_task[6]),
        int(row_or_task[7]),
        float(row_or_task[4]),
        float(row_or_task[5]),
    )


def grid_winners(rows):
    result = {}
    expected_parameters = {
        (body, leg_yaw, leg_pitch)
        for body in opt.BODY_YAW_DEG
        for leg_yaw in opt.LEG_YAW_DEG
        for leg_pitch in opt.LEG_PITCH_DEG
    }
    for wave in opt.WAVES:
        wave_rows = [row for row in rows if int(row["wave"]) == wave]
        if len(wave_rows) != opt.GRID_SIZE_PER_WAVE:
            raise RuntimeError(
                f"wave {wave}: expected {opt.GRID_SIZE_PER_WAVE} grid rows, "
                f"found {len(wave_rows)}"
            )
        observed_parameters = {
            (
                float(row["body_yaw_amplitude_deg"]),
                float(row["leg_yaw_amplitude_deg"]),
                float(row["leg_pitch_amplitude_deg"]),
            )
            for row in wave_rows
        }
        if observed_parameters != expected_parameters:
            missing = expected_parameters - observed_parameters
            extra = observed_parameters - expected_parameters
            raise RuntimeError(
                f"wave {wave}: non-bijective grid; missing={len(missing)}, "
                f"extra={len(extra)}, duplicates="
                f"{len(wave_rows) - len(observed_parameters)}"
            )
        candidates = [row for row in wave_rows if bool(row["feasible"])]
        candidates.sort(
            key=lambda row: (
                -float(row["forward_displacement_m"]),
                -float(row["path_efficiency"]),
                float(row["body_yaw_amplitude_deg"])
                + float(row["leg_yaw_amplitude_deg"])
                + float(row["leg_pitch_amplitude_deg"]),
            )
        )
        if not candidates:
            diagnostics = sorted(
                wave_rows,
                key=lambda row: (
                    sum(not bool(value) for value in row["gates"].values()),
                    -float(row["forward_displacement_m"]),
                ),
            )
            result[wave] = {
                "winner": None,
                "feasible_count": 0,
                "finite_grid_global": False,
                "status": "no_feasible_grid_point",
                "diagnostic_best": diagnostics[0],
            }
            continue
        winner = candidates[0]
        boundary = {
            "body_yaw": float(winner["body_yaw_amplitude_deg"])
            in (min(opt.BODY_YAW_DEG), max(opt.BODY_YAW_DEG)),
            "leg_yaw": float(winner["leg_yaw_amplitude_deg"])
            in (min(opt.LEG_YAW_DEG), max(opt.LEG_YAW_DEG)),
            "leg_pitch": float(winner["leg_pitch_amplitude_deg"])
            in (min(opt.LEG_PITCH_DEG), max(opt.LEG_PITCH_DEG)),
        }
        result[wave] = {
            "winner": winner,
            "feasible_count": len(candidates),
            "finite_grid_global": True,
            "winner_on_grid_boundary": boundary,
            "interior_optimum": not any(boundary.values()),
        }
    return result


def summarize(winners, validation_rows, duration_s, eval_start_s, q, seeds):
    audit_model = opt.mujoco.MjModel.from_xml_path(str(opt.MODEL))
    by_wave = {}
    for wave in opt.WAVES:
        selected = winners[wave]
        if selected["winner"] is None:
            by_wave[str(wave)] = selected
            continue
        nominal = next(
            row
            for row in validation_rows
            if int(row["wave"]) == wave and float(row["erasure_q"]) == 0.0
        )
        erased = [
            row
            for row in validation_rows
            if int(row["wave"]) == wave and float(row["erasure_q"]) == q
        ]
        displacements = np.asarray(
            [float(row["forward_displacement_m"]) for row in erased], dtype=float
        )
        by_wave[str(wave)] = {
            **selected,
            "nominal_long_validation": nominal,
            "random_contact": {
                "q": q,
                "seeds": seeds,
                "trial_count": len(erased),
                "forward_displacement_mean_m": float(np.mean(displacements)),
                "forward_displacement_std_m": float(np.std(displacements)),
                "forward_displacement_min_m": float(np.min(displacements)),
                "mean_retention_vs_nominal": float(
                    np.mean(displacements)
                    / max(float(nominal["forward_displacement_m"]), 1e-12)
                ),
                "hard_gate_survival_fraction": float(
                    np.mean([bool(row["feasible"]) for row in erased])
                ),
                "mean_realized_erased_load": float(
                    np.mean([float(row["realized_erased_load"]) for row in erased])
                ),
                "rows": erased,
            },
        }
    return {
        "selection_protocol": {
            "terrain": "flat only",
            "selection_erasure_q": 0.0,
            "objective": "maximum forward displacement after every hard gate",
            "claim": "exact global optimum on the pre-registered finite grid",
            "continuous_domain_global_claimed": False,
            "grid_points_per_wave": opt.GRID_SIZE_PER_WAVE,
        },
        "validation_protocol": {
            "duration_s": duration_s,
            "eval_start_s": eval_start_s,
            "random_erasure_q": q,
            "random_erasure_seeds": seeds,
            "random_erasure_role": "post-selection robustness only",
            "support_window_s": opt.SUPPORT_WINDOW_S,
            "minimum_burst_fraction": opt.MINIMUM_BURST_FRACTION,
            "spatial_object": "both complete leg collision subtrees of one segment pair",
        },
        "model": str(opt.MODEL),
        "model_sha256": opt.tip.sha256(opt.MODEL),
        "armature_nonzero_count": int(np.count_nonzero(audit_model.dof_armature)),
        "model_total_mass_kg": float(np.sum(audit_model.body_mass)),
        "physics_timestep_s": float(audit_model.opt.timestep),
        "integrator": opt.mujoco.mjtIntegrator(audit_model.opt.integrator).name,
        "waves": by_wave,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--eval-start", type=float, default=1.5)
    parser.add_argument("--q", type=float, default=0.3)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    grid_rows = opt.load_rows(DATA / "grid_trials.jsonl")
    winners = grid_winners(grid_rows)

    tasks = []
    for wave, selected in winners.items():
        if selected["winner"] is None:
            continue
        winner = selected["winner"]
        prefix = (
            wave,
            float(winner["body_yaw_amplitude_deg"]),
            float(winner["leg_yaw_amplitude_deg"]),
            float(winner["leg_pitch_amplitude_deg"]),
            args.duration,
            args.eval_start,
        )
        tasks.append((*prefix, 0.0, 0))
        tasks.extend((*prefix, args.q, seed) for seed in range(args.seeds))

    DATA.mkdir(parents=True, exist_ok=True)
    checkpoint = DATA / "winner_validation_trials.jsonl"
    existing = opt.load_rows(checkpoint)
    completed = {validation_key(row) for row in existing}
    pending = [task for task in tasks if validation_key(task) not in completed]
    before = opt.tip.robot_sentinels()
    started = time.perf_counter()
    if pending:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, args.workers),
            mp_context=context,
            initializer=worker_init,
        ) as executor, checkpoint.open("a", encoding="utf-8") as handle:
            futures = [executor.submit(worker_evaluate, task) for task in pending]
            for done, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                row = future.result()
                existing.append(row)
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                handle.flush()
                print(
                    f"validated {done}/{len(pending)} new trials; "
                    f"{(time.perf_counter() - started) / done:.3f} s/trial",
                    flush=True,
                )

    existing.sort(key=validation_key)
    opt.write_csv(DATA / "winner_validation_trials.csv", existing)
    summary = summarize(
        winners, existing, args.duration, args.eval_start, args.q, args.seeds
    )
    summary["robot_sentinels_unchanged"] = before == opt.tip.robot_sentinels()
    summary["elapsed_wall_s"] = time.perf_counter() - started
    (DATA / "winner_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
