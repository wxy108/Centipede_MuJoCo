"""Local gate-first PD + segment-center phase search with the compliant tip."""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PREVIOUS = PROJECT / "experiments" / "mujoco_nearground_pd_search_2026-09-01"
OUT = ROOT / "outputs"
DATA = OUT / "data"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


near = load(PREVIOUS / "run_search.py", "compliant_phase_near_source")
np, mujoco, opt, pilot = near.np, near.mujoco, near.opt, near.pilot
ORIGINAL_CONTROLLER = opt.ThreeAmplitudeController
ACTIVE_LEG_PHASE_FRACTION = 0.0


class CollocatedLegPhaseController(ORIGINAL_CONTROLLER):
    """Sample leg commands at segment centers, keeping one coupled wave."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.leg_phase_fraction = float(ACTIVE_LEG_PHASE_FRACTION)
        self.leg_phase_offset_rad = (
            self.leg_phase_fraction * 2.0 * math.pi * self.wave / 18.0
        )

    def step(self, data):
        t = float(data.time)
        dt = float(self.model.opt.timestep)
        omega = 2.0 * math.pi * opt.base.FREQUENCY_HZ
        target = self.target
        target.fill(0.0)
        body_phase = omega * t - self.spatial_phase
        blend = self.blend(t)
        body_target = blend * self.body_amplitude * np.sin(body_phase)
        body_target[0] = 0.0
        target[self.index.body_positions] = body_target
        target[self.index.pitch_positions] = 0.0
        leg_phase = body_phase + self.leg_phase_offset_rad
        phases = leg_phase[:, None, None] + self.leg_offsets[None, None, :]
        leg_target = (
            blend[:, None, None]
            * self.side_sign[None, :, None]
            * self.leg_amplitudes[None, None, :]
            * np.sin(phases)
        )
        target[self.index.leg_positions.reshape(-1)] = leg_target.reshape(-1)
        if self.previous_target is None:
            target_velocity = np.zeros_like(target)
        else:
            target_velocity = (target - self.previous_target) / dt
        q = data.qpos[self.index.qpos_addresses]
        qdot = data.qvel[self.index.dof_addresses]
        composite = target + (self.index.kv / self.index.kp) * target_velocity
        torque = self.index.kp * (target - q) + self.index.kv * (
            target_velocity - qdot
        )
        data.ctrl[:] = 0.0
        data.ctrl[self.index.active_actuator_ids] = composite
        self.previous_target = target.copy()
        return float(np.max(np.abs(torque)))


# Process-local substitution only; no source controller or XML is edited.
opt.ThreeAmplitudeController = CollocatedLegPhaseController

FIELDS = (
    "yaw_k", "yaw_d", "pitch_k", "pitch_d", "roll_k", "roll_d",
    "leg_yaw_k", "leg_yaw_d", "leg_pitch_k", "leg_pitch_d",
)
STIFFNESS_FIELDS = {"yaw_k", "pitch_k", "roll_k", "leg_yaw_k", "leg_pitch_k"}


def selected_config(wave: float) -> dict:
    filename = (
        "legyaw_refine_k0p29_w2p5_t6_dt0p00025.json"
        if wave == 2.5 else "basin15_yk2_pd4_rk1_w3_t6_dt0p00025.json"
    )
    return json.loads((PREVIOUS / "outputs" / "common" / filename).read_text())["config"]


def fixed_case(wave: float) -> dict:
    return dict(near.amplitude_case(wave, "common"))


def reference_path(wave: float, duration: float, dt: float) -> Path:
    stem = "legyaw_refine_k0p29" if wave == 2.5 else "basin15_yk2_pd4_rk1"
    tag = f"{stem}_w{wave:g}_t{duration:g}_dt{dt:g}".replace(".", "p")
    return PREVIOUS / "outputs" / "common" / f"{tag}.json"


def relative_gates(row: dict) -> dict[str, bool]:
    ref_file = reference_path(
        float(row["case"]["wave"]), float(row["duration"]), float(row["dt"])
    )
    ref = json.loads(ref_file.read_text())["metrics"]
    metrics = row["metrics"]
    eps = 1.0e-12
    return {
        "relative_displacement": metrics["displacement_mm"] >= 0.90 * ref["displacement_mm"],
        "relative_work_per_distance": metrics["work_per_distance_n"] <= 1.25 * ref["work_per_distance_n"],
        "relative_loaded_foot_slip": metrics["loaded_foot_slip_p95_mm_s"] <= 1.10 * ref["loaded_foot_slip_p95_mm_s"],
        "relative_lr_force_imbalance": metrics["left_right_force_imbalance_rms_bw"] <= 1.10 * ref["left_right_force_imbalance_rms_bw"],
        "relative_normal_impulse": metrics["normal_impulse_10ms_bw_s"] <= 1.10 * ref["normal_impulse_10ms_bw_s"],
        "relative_chassis_impulse": metrics["chassis_impulse_10ms_bw_s"] <= max(1.10 * ref["chassis_impulse_10ms_bw_s"], eps),
    }


def phase_metadata(wave: float, fraction: float) -> dict:
    return {
        "body_joint_phase": "omega*t - j*(2*pi*wave/18)",
        "leg_segment_phase": "body_joint_sample + f*(2*pi*wave/18)",
        "leg_phase_fraction": float(fraction),
        "leg_phase_offset_deg": float(fraction * 360.0 * wave / 18.0),
        "geometry_midpoint_fraction": 0.5,
        "applied_to": ["leg_yaw", "leg_pitch"],
        "body_and_leg_wave_number_coupled": True,
    }


def annotate(row: dict, wave: float, fraction: float, stage: str) -> dict:
    row["search_stage"] = stage
    row["phase_collocation"] = phase_metadata(wave, fraction)
    if row.get("metrics") is None:
        row["all_registered_gates_pass"] = False
        row["failed_registered_gates"] = [
            item["gate"] for item in row.get("failure_evidence", [])
        ]
    else:
        absolute = dict(row["gates"])
        absolute["yaw_tracking"] = row["metrics"]["leg_yaw_tracking_rms_deg"] <= 10.0
        relative = relative_gates(row)
        row["gates"] = absolute
        row["relative_gates"] = relative
        row["all_registered_gates_pass"] = bool(
            all(absolute.values()) and all(relative.values())
        )
        row["failed_registered_gates"] = [
            name for name, passed in {**absolute, **relative}.items() if not passed
        ]
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{row['id']}.json").write_text(json.dumps(row, indent=2))
    return row


def run_task(task: tuple) -> dict:
    global ACTIVE_LEG_PHASE_FRACTION
    wave, cfg, fraction, duration, dt_scale, trace, stage, use_precheck = task
    ACTIVE_LEG_PHASE_FRACTION = float(fraction)
    if use_precheck:
        near.OUT = OUT
        rejected = near.precheck(cfg, wave, duration, dt_scale, "data")
        if rejected is not None:
            return annotate(rejected, wave, fraction, stage)
    row = pilot.run_task(
        (cfg, wave, duration, dt_scale, trace),
        initialize=near.near_ground,
        case_override=fixed_case(wave),
        output_dir=DATA,
    )
    return annotate(row, wave, fraction, stage)


def phase_tasks() -> list[tuple]:
    tasks = []
    for wave in (2.5, 3.0):
        base = selected_config(wave)
        for fraction in (0.0, 0.35, 0.425, 0.5, 0.575, 0.65):
            name = f"phase_only_f{fraction:g}".replace(".", "p")
            tasks.append((wave, dict(base, name=name), fraction, 6.0, 1.0, False, "phase", False))
    return tasks


def sobol_tasks(points: int) -> list[tuple]:
    from scipy.stats import qmc

    if points < 2 or points & (points - 1):
        raise ValueError("points must be a power of two")
    tasks = []
    for wave, seed in ((2.5, 250902), (3.0, 300902)):
        base = selected_config(wave)
        unit = qmc.Sobol(d=11, scramble=True, seed=seed).random_base2(int(math.log2(points)))
        # Include exact geometric midpoint with the already validated PD.
        tasks.append((wave, dict(base, name="anchor_midpoint"), 0.5, 6.0, 1.0, False, "sobol", True))
        for index, sample in enumerate(unit):
            cfg = dict(base)
            for column, field in enumerate(FIELDS):
                low, high = (0.70, 1.35) if field in STIFFNESS_FIELDS else (0.70, 1.80)
                factor = math.exp(math.log(low) + float(sample[column]) * math.log(high / low))
                cfg[field] = float(base[field]) * factor
            fraction = 0.35 + 0.30 * float(sample[-1])
            cfg["name"] = f"sobol_{index:03d}"
            tasks.append((wave, cfg, fraction, 6.0, 1.0, False, "sobol", True))
    return tasks


def targeted_wave3_tasks() -> list[tuple]:
    """Single-pass candidates aimed at the three marginal f=0.35 failures."""
    base = selected_config(3.0)
    specifications = (
        # name, f, pitch_k factor, pitch_d factor, leg-yaw k/d factors,
        # leg-pitch k/d factors
        ("target_00", 0.35, 1.00, 1.25, 1.00, 1.25, 1.00, 1.00),
        ("target_01", 0.35, 1.00, 1.50, 0.90, 1.50, 1.00, 1.00),
        ("target_02", 0.35, 0.85, 1.50, 0.87, 1.50, 1.00, 1.20),
        ("target_03", 0.375, 0.85, 1.60, 0.87, 1.60, 0.95, 1.30),
        ("target_04", 0.40, 0.80, 1.70, 0.82, 1.70, 0.90, 1.50),
        ("target_05", 0.425, 0.75, 1.80, 0.80, 1.80, 0.88, 1.70),
        ("target_06", 0.45, 0.75, 2.00, 0.78, 1.90, 0.85, 1.90),
        ("target_07", 0.50, 0.70, 2.00, 0.75, 2.00, 0.80, 2.00),
    )
    tasks = []
    for name, fraction, pk, pd, lyk, lyd, lpk, lpd in specifications:
        cfg = dict(
            base,
            name=name,
            pitch_k=float(base["pitch_k"]) * pk,
            pitch_d=float(base["pitch_d"]) * pd,
            leg_yaw_k=float(base["leg_yaw_k"]) * lyk,
            leg_yaw_d=float(base["leg_yaw_d"]) * lyd,
            leg_pitch_k=float(base["leg_pitch_k"]) * lpk,
            leg_pitch_d=float(base["leg_pitch_d"]) * lpd,
        )
        tasks.append((3.0, cfg, fraction, 6.0, 1.0, False, "targeted", False))
    return tasks


def recovery_tasks() -> list[tuple]:
    """Robustness follow-up after the first long/half-step validation."""
    tasks = []
    # The two lower-displacement wave-2.5 nominal winners have zero chassis
    # contact and larger active/tip-speed margins than the first winner.
    for stem in ("sobol_027_w2p5_t6_dt0p00025", "sobol_012_w2p5_t6_dt0p00025"):
        row = json.loads((DATA / f"{stem}.json").read_text())
        cfg = dict(row["config"], name=f"recovery_{row['config']['name']}")
        fraction = float(row["phase_collocation"]["leg_phase_fraction"])
        tasks.extend([
            (2.5, cfg, fraction, 12.0, 1.0, True, "recovery", False),
            (2.5, cfg, fraction, 6.0, 0.5, True, "recovery", False),
        ])

    source = json.loads((DATA / "target_02_w3_t6_dt0p00025.json").read_text())
    center = source["config"]
    variants = (
        ("robust_00", 1.00, 1.00, 1.00, 1.00, 0.90, 1.25),
        ("robust_01", 0.90, 1.10, 0.95, 1.10, 0.90, 1.35),
        ("robust_02", 0.85, 1.20, 0.90, 1.20, 0.80, 1.50),
        ("robust_03", 0.90, 1.15, 0.92, 1.15, 0.85, 1.40),
    )
    for name, pk, pd, lyk, lyd, lpk, lpd in variants:
        cfg = dict(
            center,
            name=name,
            pitch_k=float(center["pitch_k"]) * pk,
            pitch_d=float(center["pitch_d"]) * pd,
            leg_yaw_k=float(center["leg_yaw_k"]) * lyk,
            leg_yaw_d=float(center["leg_yaw_d"]) * lyd,
            leg_pitch_k=float(center["leg_pitch_k"]) * lpk,
            leg_pitch_d=float(center["leg_pitch_d"]) * lpd,
        )
        tasks.append((3.0, cfg, 0.35, 12.0, 1.0, True, "recovery", False))
    return tasks


def margin_tasks() -> list[tuple]:
    """Final narrow margin checks; tip compliance itself remains fixed."""
    tasks = []
    w25 = json.loads((DATA / "sobol_027_w2p5_t6_dt0p00025.json").read_text())
    center25 = w25["config"]
    for index, (roll_k_factor, roll_d_factor) in enumerate(
        ((0.85, 1.00), (0.75, 1.10), (0.65, 1.25), (0.80, 1.30))
    ):
        cfg = dict(
            center25,
            name=f"margin25_{index:02d}",
            roll_k=float(center25["roll_k"]) * roll_k_factor,
            roll_d=float(center25["roll_d"]) * roll_d_factor,
        )
        tasks.append((2.5, cfg, float(w25["phase_collocation"]["leg_phase_fraction"]), 6.0, 0.5, True, "margin", False))

    w3 = json.loads((DATA / "target_02_w3_t6_dt0p00025.json").read_text())
    center3 = w3["config"]
    variants = (
        ("margin3_00", 0.80, 1.00, 0.75, 0.80),
        ("margin3_01", 0.70, 1.20, 0.65, 0.70),
        ("margin3_02", 0.80, 1.30, 0.60, 0.60),
        ("margin3_03", 0.60, 1.50, 0.50, 0.50),
    )
    for name, rk, rd, lpk, lpd in variants:
        cfg = dict(
            center3,
            name=name,
            roll_k=float(center3["roll_k"]) * rk,
            roll_d=float(center3["roll_d"]) * rd,
            leg_pitch_k=float(center3["leg_pitch_k"]) * lpk,
            leg_pitch_d=float(center3["leg_pitch_d"]) * lpd,
        )
        tasks.append((3.0, cfg, 0.35, 12.0, 1.0, True, "margin", False))
    return tasks


def fallback_tasks() -> list[tuple]:
    """One amended nonzero-phase confirmation after robustness failures."""
    tasks = []
    w25 = json.loads((DATA / "sobol_027_w2p5_t6_dt0p00025.json").read_text())
    cfg25 = dict(w25["config"], name="fallback_w25_f0p30")
    tasks.extend([
        (2.5, cfg25, 0.30, 12.0, 1.0, True, "fallback", False),
        (2.5, cfg25, 0.30, 6.0, 0.5, True, "fallback", False),
    ])
    w3 = json.loads((DATA / "target_02_w3_t6_dt0p00025.json").read_text())
    cfg3 = dict(w3["config"], name="fallback_w3_f0p30")
    tasks.extend([
        (3.0, cfg3, 0.30, 12.0, 1.0, True, "fallback", False),
        (3.0, cfg3, 0.30, 6.0, 0.5, True, "fallback", False),
    ])
    return tasks


def refine_tasks(rows: list[dict], points: int = 16) -> list[tuple]:
    from scipy.stats import qmc

    tasks = []
    for wave, seed in ((2.5, 251102), (3.0, 301102)):
        feasible = [
            row for row in rows
            if row.get("all_registered_gates_pass") and float(row["case"]["wave"]) == wave
        ]
        if not feasible:
            continue
        center = max(feasible, key=lambda row: row["metrics"]["displacement_mm"])
        unit = qmc.Sobol(d=11, scramble=True, seed=seed).random_base2(int(math.log2(points)))
        for index, sample in enumerate(unit):
            cfg = dict(center["config"])
            for column, field in enumerate(FIELDS):
                factor = math.exp((2.0 * float(sample[column]) - 1.0) * math.log(1.20))
                cfg[field] = float(center["config"][field]) * factor
            f0 = float(center["phase_collocation"]["leg_phase_fraction"])
            fraction = min(0.65, max(0.35, f0 + (2.0 * float(sample[-1]) - 1.0) * 0.04))
            cfg["name"] = f"refine_{index:03d}"
            tasks.append((wave, cfg, fraction, 6.0, 1.0, False, "refine", True))
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
            print(json.dumps({
                "done": done, "total": len(tasks), "id": row["id"],
                "wave": row["case"]["wave"],
                "D_mm": row["metrics"]["displacement_mm"] if row.get("metrics") else None,
                "pass": row["all_registered_gates_pass"],
                "failed": row["failed_registered_gates"],
                "elapsed_s": row["elapsed_s"],
            }), flush=True)
    return rows


def load_rows() -> list[dict]:
    rows = []
    for path in DATA.glob("*.json"):
        row = json.loads(path.read_text())
        if row.get("search_stage") in {"phase", "sobol", "targeted", "refine"} and row.get("metrics"):
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    result = {
        "scope": "latest compliant-tip Centipede Model only",
        "rigid_tip_used": False,
        "robot_reference_read_only": True,
        "amplitudes_fixed": True,
        "tip_parameters_fixed": True,
        "body_leg_wave_number_coupled": True,
        "continuous_global_optimum_proven": False,
        "objective": "maximum displacement among all registered gates passing",
        "waves": {},
    }
    for wave in (2.5, 3.0):
        group = [row for row in rows if float(row["case"]["wave"]) == wave]
        eligible = [
            row for row in group
            if 0.35 <= float(row["phase_collocation"]["leg_phase_fraction"]) <= 0.65
        ]
        feasible = [row for row in eligible if row.get("all_registered_gates_pass")]
        shared_phase = [
            row for row in group
            if float(row["phase_collocation"]["leg_phase_fraction"]) == 0.0
        ]
        result["waves"][f"{wave:g}"] = {
            "full_evaluations": len(group),
            "phase_corrected_eligible_evaluations": len(eligible),
            "feasible": len(feasible),
            "best_tested_feasible": max(
                feasible, key=lambda row: row["metrics"]["displacement_mm"]
            ) if feasible else None,
            "shared_phase_reproduction_diagnostic": shared_phase[0] if shared_phase else None,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "search_summary.json").write_text(json.dumps(result, indent=2))
    return result


def validation_tasks(summary: dict) -> list[tuple]:
    tasks = []
    for wave in (2.5, 3.0):
        best = summary["waves"][f"{wave:g}"]["best_tested_feasible"]
        if best is None:
            continue
        cfg = dict(best["config"], name=f"selected_w{wave:g}".replace(".", "p"))
        fraction = float(best["phase_collocation"]["leg_phase_fraction"])
        tasks.extend([
            (wave, cfg, fraction, 12.0, 1.0, True, "validation", False),
            (wave, cfg, fraction, 6.0, 0.5, True, "validation", False),
        ])
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("phase", "sobol", "targeted", "recovery", "margin", "fallback", "refine", "summary", "validate", "all"), default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--points", type=int, default=32)
    args = parser.parse_args()
    if args.stage in ("phase", "all"):
        run_many(phase_tasks(), args.workers)
    if args.stage in ("sobol", "all"):
        run_many(sobol_tasks(args.points), args.workers)
    if args.stage == "targeted":
        run_many(targeted_wave3_tasks(), args.workers)
    if args.stage == "recovery":
        run_many(recovery_tasks(), args.workers)
    if args.stage == "margin":
        run_many(margin_tasks(), args.workers)
    if args.stage == "fallback":
        run_many(fallback_tasks(), args.workers)
    rows = load_rows()
    if args.stage in ("refine", "all"):
        tasks = refine_tasks(rows)
        if tasks:
            run_many(tasks, args.workers)
            rows = load_rows()
    summary = summarize(rows)
    if args.stage in ("validate", "all"):
        tasks = validation_tasks(summary)
        if tasks:
            validation = run_many(tasks, min(4, len(tasks)))
            (OUT / "validation_results.json").write_text(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
