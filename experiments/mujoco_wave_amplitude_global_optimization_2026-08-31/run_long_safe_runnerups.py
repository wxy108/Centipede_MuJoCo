"""Validate ranked runner-ups when a 4 s grid winner fails at 6 s."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import multiprocessing
import sys
import time
from pathlib import Path


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


opt = load_module(OPTIMIZER, "centipede_long_safe_optimizer")
_EVALUATOR = None


def worker_init():
    global _EVALUATOR
    _EVALUATOR = opt.FastEvaluator()


def worker_evaluate(task):
    if _EVALUATOR is None:
        worker_init()
    return _EVALUATOR.evaluate(task)


def key(row_or_task):
    if isinstance(row_or_task, dict):
        return (
            int(row_or_task["wave"]),
            float(row_or_task["body_yaw_amplitude_deg"]),
            float(row_or_task["leg_yaw_amplitude_deg"]),
            float(row_or_task["leg_pitch_amplitude_deg"]),
            float(row_or_task["duration_s"]),
        )
    return (
        int(row_or_task[0]),
        float(row_or_task[1]),
        float(row_or_task[2]),
        float(row_or_task[3]),
        float(row_or_task[4]),
    )


def ranked_candidates(grid_rows, wave):
    candidates = [
        row
        for row in grid_rows
        if int(row["wave"]) == wave and bool(row["feasible"])
    ]
    candidates.sort(
        key=lambda row: (
            -float(row["forward_displacement_m"]),
            -float(row["path_efficiency"]),
            float(row["body_yaw_amplitude_deg"])
            + float(row["leg_yaw_amplitude_deg"])
            + float(row["leg_pitch_amplitude_deg"]),
        )
    )
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waves", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--eval-start", type=float, default=1.5)
    args = parser.parse_args()

    grid_rows = opt.load_rows(DATA / "grid_trials.jsonl")
    sentinels_before = opt.tip.robot_sentinels()
    primary_rows = opt.load_rows(DATA / "winner_validation_trials.jsonl")
    primary_nominal = {
        int(row["wave"]): row
        for row in primary_rows
        if float(row["erasure_q"]) == 0.0
        and float(row["duration_s"]) == args.duration
    }
    ranked = {wave: ranked_candidates(grid_rows, wave) for wave in args.waves}
    tasks = []
    for wave, candidates in ranked.items():
        for candidate in candidates[1 : args.top]:
            tasks.append(
                (
                    wave,
                    float(candidate["body_yaw_amplitude_deg"]),
                    float(candidate["leg_yaw_amplitude_deg"]),
                    float(candidate["leg_pitch_amplitude_deg"]),
                    args.duration,
                    args.eval_start,
                    0.0,
                    0,
                )
            )

    checkpoint = DATA / "runnerup_long_trials.jsonl"
    rows = opt.load_rows(checkpoint)
    completed = {key(row) for row in rows}
    pending = [task for task in tasks if key(task) not in completed]
    started = time.perf_counter()
    if pending:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=worker_init,
        ) as executor, checkpoint.open("a", encoding="utf-8") as handle:
            futures = [executor.submit(worker_evaluate, task) for task in pending]
            for done, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                row = future.result()
                rows.append(row)
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                handle.flush()
                print(f"runner-up {done}/{len(pending)}", flush=True)

    rows.sort(key=key)
    opt.write_csv(DATA / "runnerup_long_trials.csv", rows)
    lookup = {key(row): row for row in rows}
    summary = {}
    for wave, candidates in ranked.items():
        evaluated = []
        for rank, candidate in enumerate(candidates[: args.top], start=1):
            if rank == 1:
                row = primary_nominal[wave]
            else:
                trial_key = (
                    wave,
                    float(candidate["body_yaw_amplitude_deg"]),
                    float(candidate["leg_yaw_amplitude_deg"]),
                    float(candidate["leg_pitch_amplitude_deg"]),
                    args.duration,
                )
                row = lookup[trial_key]
            evaluated.append(
                {
                    "grid_rank": rank,
                    "grid_row": candidate,
                    "long_row": row,
                }
            )
        selected = next(
            (item for item in evaluated if bool(item["long_row"]["feasible"])),
            None,
        )
        summary[str(wave)] = {
            "grid_feasible_count": len(candidates),
            "tested_count": len(evaluated),
            "selected_long_safe": selected,
            "evaluated": evaluated,
        }
    payload = {
        "criterion": "first 4 s displacement-ranked grid-feasible candidate that also passes all 6 s gates",
        "duration_s": args.duration,
        "eval_start_s": args.eval_start,
        "top_requested": args.top,
        "waves": summary,
        "robot_sentinels_unchanged": sentinels_before
        == opt.tip.robot_sentinels(),
        "elapsed_wall_s": time.perf_counter() - started,
    }
    (DATA / "runnerup_long_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
