"""Run a deterministic adaptive-refinement task bank with stride-1 physics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "run_broad_sobol_pilot.py"
DATA = ROOT / "outputs" / "data"


def load_pilot():
    spec = importlib.util.spec_from_file_location("centipede_refinement_pilot", PILOT)
    if spec is None or spec.loader is None:
        raise RuntimeError(PILOT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--task-modulo", type=int, default=1)
    parser.add_argument("--task-remainder", type=int, default=0)
    parser.add_argument(
        "--proof-early-exit",
        action="store_true",
        help=(
            "stop only after an irreversible base-gate failure; every base-pass "
            "and every feasible candidate still receives the complete 6 s replay"
        ),
    )
    args = parser.parse_args()
    if args.task_modulo < 1 or not 0 <= args.task_remainder < args.task_modulo:
        raise ValueError("task remainder must satisfy 0 <= remainder < modulo")
    safe_tag = "".join(c for c in args.output_tag if c.isalnum() or c in "-_")
    if not safe_tag:
        raise ValueError("invalid output tag")

    payload = json.loads(args.tasks.read_text(encoding="utf-8"))
    selected = [
        item
        for task_index, item in enumerate(payload["tasks"])
        if task_index % args.task_modulo == args.task_remainder
    ]
    output = DATA / f"adaptive_refinement_trials_{safe_tag}.jsonl"
    completed = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(json.loads(line)["candidate_id"])
    todo = [item for item in selected if item["candidate_id"] not in completed]
    print(f"tasks total={len(selected)} completed={len(completed)} todo={len(todo)}", flush=True)

    pilot = load_pilot()
    evaluator = pilot.PilotEvaluator(1, args.proof_early_exit)
    started = time.time()
    with output.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(todo, 1):
            row = evaluator.evaluate(
                (
                    item["candidate_id"],
                    item["source"],
                    float(item["wave"]),
                    float(item["body_yaw_amplitude_deg"]),
                    float(item["leg_yaw_amplitude_deg"]),
                    float(item["leg_pitch_amplitude_deg"]),
                    float(item.get("duration_s", 6.0)),
                    float(item.get("eval_start_s", 1.5)),
                )
            )
            row["refinement_stage"] = payload["stage"]
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(todo):
                print(
                    f"progress {index}/{len(todo)} elapsed={(time.time()-started)/60:.1f}min",
                    flush=True,
                )


if __name__ == "__main__":
    main()
