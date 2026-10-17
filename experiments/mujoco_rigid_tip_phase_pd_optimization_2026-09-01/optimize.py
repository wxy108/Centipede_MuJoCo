"""Gate-first phase-collocation and PD search for the one-piece rigid leg."""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import sys
import time
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RIGID_ROOT = PROJECT / "experiments" / "mujoco_rigid_tip_ablation_2026-09-01"
OUT = ROOT / "outputs"
DATA = OUT / "data"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rigid = load(RIGID_ROOT / "run_ablation.py", "rigid_phase_pd_runner")
np = rigid.np

FIELDS = (
    "yaw_k",
    "yaw_d",
    "pitch_k",
    "pitch_d",
    "roll_k",
    "roll_d",
    "leg_yaw_k",
    "leg_yaw_d",
    "leg_pitch_k",
    "leg_pitch_d",
)
STIFFNESS_FIELDS = {"yaw_k", "pitch_k", "roll_k", "leg_yaw_k", "leg_pitch_k"}


def source_reference(wave: float) -> dict:
    name = (
        "legyaw_refine_k0p29_w2p5_t6_dt0p00025.json"
        if wave == 2.5
        else "basin15_yk2_pd4_rk1_w3_t6_dt0p00025.json"
    )
    return json.loads(
        (
            PROJECT
            / "experiments"
            / "mujoco_nearground_pd_search_2026-09-01"
            / "outputs"
            / "common"
            / name
        ).read_text(encoding="utf-8")
    )


def relative_gates(row: dict) -> dict[str, bool]:
    metrics = row["metrics"]
    ref = source_reference(float(row["case"]["wave"]))["metrics"]
    eps = 1.0e-12
    return {
        "relative_displacement": metrics["displacement_mm"] >= 0.90 * ref["displacement_mm"],
        "relative_work_per_distance": metrics["work_per_distance_n"] <= 1.25 * ref["work_per_distance_n"],
        "relative_loaded_foot_slip": metrics["loaded_foot_slip_p95_mm_s"] <= 1.10 * ref["loaded_foot_slip_p95_mm_s"],
        "relative_lr_force_imbalance": metrics["left_right_force_imbalance_rms_bw"] <= 1.10 * ref["left_right_force_imbalance_rms_bw"],
        "relative_normal_impulse": metrics["normal_impulse_10ms_bw_s"] <= 1.10 * ref["normal_impulse_10ms_bw_s"],
        "relative_chassis_impulse": metrics["chassis_impulse_10ms_bw_s"] <= max(1.10 * ref["chassis_impulse_10ms_bw_s"], eps),
    }


def annotate(row: dict) -> dict:
    rel = relative_gates(row)
    row["relative_gates"] = rel
    row["all_registered_gates_pass"] = bool(
        row["feasible_applicable_gates"] and all(rel.values())
    )
    row["failed_registered_gates"] = row["failed_applicable_gates"] + [
        name for name, passed in rel.items() if not passed
    ]
    path = DATA / f"{row['id']}.json"
    path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def irreversible_precheck(task: tuple) -> dict | None:
    """Stop only after an already-failed peak/change gate cannot recover."""
    wave, cfg, fraction, duration, dt_scale, _trace, prefix, _precheck = task
    started = time.monotonic()
    ev = rigid.RigidTipEvaluator()
    model, data, index = ev.model, ev.data, ev.index
    rigid.pilot.apply_impedance(model, index, cfg)
    model.opt.timestep *= dt_scale
    dt = float(model.opt.timestep)
    stride = max(1, int(round(0.001 / dt)))
    lag = int(round(0.010 / (dt * stride)))
    case = rigid.fixed_case(wave)
    controller = rigid.CollocatedLegPhaseController(
        model,
        index,
        wave,
        case["body"],
        case["leg_yaw"],
        case["leg_pitch"],
        leg_phase_fraction=fraction,
    )
    body_ids = np.stack(
        (
            index.active_joint_ids[index.body_positions],
            index.active_joint_ids[index.pitch_positions],
            index.roll_joint_ids,
        ),
        axis=-1,
    )
    body_qpos = model.jnt_qposadr[body_ids]
    body_dof = model.jnt_dofadr[body_ids]
    history_q = deque(maxlen=lag + 1)
    history_omega = deque(maxlen=lag + 1)
    rigid.mujoco.mj_resetData(model, data)
    rigid.mujoco.mj_forward(model, data)
    initialization = rigid.near.near_ground(ev)
    failures: list[dict] = []
    steps = int(round(duration / dt))
    for step in range(steps):
        controller.step(data)
        rigid.mujoco.mj_step(model, data)
        torque = float(np.max(np.abs(data.actuator_force[index.active_actuator_ids])))
        if torque > 2.0e-4 or not np.isfinite(torque):
            failures.append({"gate": "torque", "value": torque, "limit": 2.0e-4})
        if data.time >= 1.5 - 0.5 * dt:
            speed = np.max(np.abs(data.qvel[body_dof]), axis=0)
            checks = (
                ("body_yaw_speed", float(speed[0]), 12.0),
                ("body_pitch_speed", float(speed[1]), 8.0),
                ("body_roll_speed", float(speed[2]), 5.0),
                (
                    "active_speed",
                    float(np.max(np.abs(data.qvel[index.dof_addresses]))),
                    100.0,
                ),
            )
            failures.extend(
                {"gate": name, "value": value, "limit": limit}
                for name, value, limit in checks
                if value > limit or not np.isfinite(value)
            )
            if step % stride == 0:
                history_q.append(data.qpos[body_qpos].copy())
                history_omega.append(data.cvel[index.chassis_body_ids, :3].copy())
                if len(history_q) == lag + 1:
                    delta_q = np.degrees(np.max(np.abs(history_q[-1] - history_q[0]), axis=0))
                    delta_omega = float(
                        np.max(
                            np.linalg.norm(history_omega[-1] - history_omega[0], axis=-1)
                        )
                    )
                    change_checks = (
                        ("body_velocity_jump", delta_omega, 10.0),
                        ("pitch_10ms", float(delta_q[1]), 2.0),
                        ("roll_10ms", float(delta_q[2]), 2.0),
                    )
                    failures.extend(
                        {"gate": name, "value": value, "limit": limit}
                        for name, value, limit in change_checks
                        if value > limit or not np.isfinite(value)
                    )
        if failures:
            break
    if not failures:
        return None
    phase_tag = f"_lf{fraction:g}" if fraction else ""
    tag = f"{prefix}_{cfg['name']}{phase_tag}_w{wave:g}_t{duration:g}_dt{dt:g}".replace(".", "p")
    row = {
        "id": tag,
        "config": cfg,
        "case": case,
        "duration": duration,
        "dt": dt,
        "phase_collocation": {
            "leg_phase_fraction": float(fraction),
            "leg_phase_offset_deg": float(fraction * 360.0 * wave / 18.0),
            "applied_to": ["leg_yaw", "leg_pitch"],
            "body_and_leg_wave_number_coupled": True,
        },
        "metrics": None,
        "proven_infeasible": True,
        "precheck": True,
        "checked_until_s": float(data.time),
        "failure_evidence": failures,
        "failed_applicable_gates": [item["gate"] for item in failures],
        "failed_registered_gates": [item["gate"] for item in failures],
        "all_registered_gates_pass": False,
        "objective": "not evaluated: irreversible hard gate already failed",
        "remaining_gates": "not evaluated",
        "initialization": initialization,
        "elapsed_s": time.monotonic() - started,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{tag}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def run_task(task: tuple) -> dict:
    wave, cfg, fraction, duration, dt_scale, trace, prefix, use_precheck = task
    if use_precheck:
        rejected = irreversible_precheck(task)
        if rejected is not None:
            return rejected
    row = rigid.run(
        wave,
        duration,
        dt_scale,
        trace,
        config_override=cfg,
        leg_phase_fraction=fraction,
        output_dir=DATA,
        tag_prefix=prefix,
    )
    return annotate(row)


def phase_tasks() -> list[tuple]:
    fractions = (0.0, 0.25, 0.35, 0.425, 0.5, 0.575, 0.65, 0.75)
    tasks = []
    for wave in (2.5, 3.0):
        base = rigid.selected_config(wave)
        for fraction in fractions:
            cfg = dict(base, name=f"phase_only_f{fraction:g}".replace(".", "p"))
            tasks.append((wave, cfg, fraction, 6.0, 1.0, False, "phase_screen", False))
    return tasks


def sobol_tasks(points_per_wave: int = 64) -> list[tuple]:
    from scipy.stats import qmc

    # Multipliers relative to each wave's previously selected compliant-tip PD.
    # Rigid removal mainly raises impact/speed, so damping gets the wider range.
    stiffness_bounds = (0.45, 1.80)
    damping_bounds = (0.50, 4.00)
    tasks = []
    for wave, seed in ((2.5, 250901), (3.0, 300901)):
        base = rigid.selected_config(wave)
        sampler = qmc.Sobol(d=11, scramble=True, seed=seed)
        exponent = int(round(math.log2(points_per_wave)))
        if 2**exponent != points_per_wave:
            raise ValueError("points_per_wave must be a power of two")
        unit = sampler.random_base2(exponent)
        for index, sample in enumerate(unit):
            cfg = dict(base)
            for column, field in enumerate(FIELDS):
                low, high = (
                    stiffness_bounds if field in STIFFNESS_FIELDS else damping_bounds
                )
                factor = math.exp(math.log(low) + float(sample[column]) * math.log(high / low))
                cfg[field] = float(base[field]) * factor
            fraction = 0.35 + float(sample[-1]) * 0.30
            cfg["name"] = f"sobol_{index:03d}"
            tasks.append((wave, cfg, fraction, 6.0, 1.0, False, "phase_pd", True))
    return tasks


def refine_tasks(rows: list[dict], points_per_wave: int = 32) -> list[tuple]:
    from scipy.stats import qmc

    tasks = []
    for wave, seed in ((2.5, 251001), (3.0, 301001)):
        feasible = [
            row for row in rows
            if float(row["case"]["wave"]) == wave and row.get("all_registered_gates_pass")
        ]
        if not feasible:
            continue
        center = max(feasible, key=lambda row: row["metrics"]["displacement_mm"])
        sampler = qmc.Sobol(d=11, scramble=True, seed=seed)
        exponent = int(round(math.log2(points_per_wave)))
        unit = sampler.random_base2(exponent)
        for index, sample in enumerate(unit):
            cfg = dict(center["config"])
            for column, field in enumerate(FIELDS):
                # Narrow symmetric log band: x0 / 1.35 ... x0 * 1.35.
                factor = math.exp((2.0 * float(sample[column]) - 1.0) * math.log(1.35))
                cfg[field] = float(center["config"][field]) * factor
            f0 = float(center["phase_collocation"]["leg_phase_fraction"])
            fraction = min(0.65, max(0.35, f0 + (2.0 * float(sample[-1]) - 1.0) * 0.06))
            cfg["name"] = f"refine_{index:03d}"
            tasks.append((wave, cfg, fraction, 6.0, 1.0, False, "phase_pd_refine", True))
    return tasks


def run_many(tasks: list[tuple], workers: int) -> list[dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"pending={len(tasks)} workers={workers}", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_task, task) for task in tasks]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "done": done,
                        "total": len(tasks),
                        "id": row["id"],
                        "wave": row["case"]["wave"],
                        "D_mm": row["metrics"]["displacement_mm"] if row.get("metrics") else None,
                        "registered_pass": row["all_registered_gates_pass"],
                        "failed": row["failed_registered_gates"],
                        "checked_until_s": row.get("checked_until_s", row.get("duration")),
                        "elapsed_s": row["elapsed_s"],
                    }
                ),
                flush=True,
            )
    return rows


def load_rows(prefixes: tuple[str, ...] = ("phase_screen_", "phase_pd_")) -> list[dict]:
    rows = []
    for path in DATA.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("id", "").startswith(prefixes) and row.get("metrics"):
            rows.append(row)
    return rows


def write_summary(rows: list[dict]) -> dict:
    summary: dict[str, object] = {
        "scope": "Centipede Model rigid one-piece leg only",
        "robot_reference_read_only": True,
        "amplitudes_fixed": True,
        "body_leg_wave_number_coupled": True,
        "objective": "maximize displacement among all-gates feasible candidates only",
        "continuous_global_optimum_proven": False,
        "registered_candidate_count": len(rows),
        "waves": {},
    }
    for wave in (2.5, 3.0):
        group = [row for row in rows if float(row["case"]["wave"]) == wave]
        feasible = [row for row in group if row.get("all_registered_gates_pass")]
        summary["waves"][f"{wave:g}"] = {
            "tested": len(group),
            "feasible": len(feasible),
            "best_tested_feasible": (
                max(feasible, key=lambda row: row["metrics"]["displacement_mm"])
                if feasible else None
            ),
            "failure_counts": {
                gate: sum(gate in row.get("failed_registered_gates", []) for row in group)
                for gate in sorted(
                    {gate for row in group for gate in row.get("failed_registered_gates", [])}
                )
            },
        }
    (OUT / "search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def validate(summary: dict) -> list[dict]:
    tasks = []
    for wave in (2.5, 3.0):
        best = summary["waves"][f"{wave:g}"]["best_tested_feasible"]
        if best is None:
            continue
        fraction = float(best["phase_collocation"]["leg_phase_fraction"])
        cfg = dict(best["config"], name=f"selected_w{wave:g}".replace(".", "p"))
        tasks.extend(
            [
                (wave, cfg, fraction, 12.0, 1.0, True, "validation_long", False),
                (wave, cfg, fraction, 6.0, 0.5, True, "validation_halfdt", False),
            ]
        )
    return run_many(tasks, min(4, len(tasks))) if tasks else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("phase", "sobol", "refine", "summary", "validate", "all"), default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--points", type=int, default=64)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("phase", "all"):
        run_many(phase_tasks(), args.workers)
    if args.stage in ("sobol", "all"):
        run_many(sobol_tasks(args.points), args.workers)
    rows = load_rows()
    if args.stage in ("refine", "all"):
        run_many(refine_tasks(rows), args.workers)
        rows = load_rows()
    summary = write_summary(rows)
    if args.stage in ("validate", "all"):
        validation = validate(summary)
        (OUT / "validation_results.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
