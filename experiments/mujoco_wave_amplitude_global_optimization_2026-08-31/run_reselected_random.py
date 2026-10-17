"""Random-contact validation for long-safe runner-up selections."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "data"
OPTIMIZER = ROOT / "run_wave_amplitude_global.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


opt = load_module(OPTIMIZER, "centipede_reselected_random_optimizer")
_EVALUATOR = None


def worker_init():
    global _EVALUATOR
    _EVALUATOR = opt.FastEvaluator()


def worker(task):
    if _EVALUATOR is None:
        worker_init()
    return _EVALUATOR.evaluate(task)


def main():
    runnerups = json.loads(
        (DATA / "runnerup_long_summary.json").read_text(encoding="utf-8")
    )
    tasks = []
    selected = {}
    for wave in (3, 5):
        choice = runnerups["waves"][str(wave)]["selected_long_safe"]
        if choice is None:
            raise RuntimeError(f"wave {wave} has no long-safe selection")
        grid = choice["grid_row"]
        selected[wave] = choice
        for seed in range(5):
            tasks.append(
                (
                    wave,
                    float(grid["body_yaw_amplitude_deg"]),
                    float(grid["leg_yaw_amplitude_deg"]),
                    float(grid["leg_pitch_amplitude_deg"]),
                    6.0,
                    1.5,
                    0.3,
                    seed,
                )
            )
    before = opt.tip.robot_sentinels()
    context = multiprocessing.get_context("spawn")
    rows = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=8, mp_context=context, initializer=worker_init
    ) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for done, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            rows.append(future.result())
            print(f"reselected random {done}/{len(tasks)}", flush=True)
    rows.sort(key=lambda row: (int(row["wave"]), int(row["erasure_seed"])))
    opt.write_csv(DATA / "reselected_random_trials.csv", rows)
    summary = {}
    for wave in (3, 5):
        wave_rows = [row for row in rows if int(row["wave"]) == wave]
        nominal = selected[wave]["long_row"]
        displacement = np.asarray(
            [float(row["forward_displacement_m"]) for row in wave_rows]
        )
        summary[str(wave)] = {
            "selection": selected[wave],
            "random_rows": wave_rows,
            "forward_displacement_mean_m": float(np.mean(displacement)),
            "forward_displacement_min_m": float(np.min(displacement)),
            "mean_retention_vs_nominal": float(
                np.mean(displacement)
                / max(float(nominal["forward_displacement_m"]), 1e-12)
            ),
            "hard_gate_survival_fraction": float(
                np.mean([bool(row["feasible"]) for row in wave_rows])
            ),
            "mean_realized_erased_load": float(
                np.mean([float(row["realized_erased_load"]) for row in wave_rows])
            ),
        }
    payload = {
        "purpose": "post-selection random-contact validation of 6 s long-safe runner-ups",
        "q": 0.3,
        "seeds": list(range(5)),
        "waves": summary,
        "robot_sentinels_unchanged": before == opt.tip.robot_sentinels(),
    }
    (DATA / "reselected_random_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
