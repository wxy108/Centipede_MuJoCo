"""Gate-first distal-tip flexion search for the Centipede Model.

This experiment deliberately keeps the historical ``*_roll`` joint names so
the established evaluator can index the same 38 passive DOFs.  In this branch
their axis is tip-local Z and their physical meaning is distal flexion.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
TRUSTED_SITE = Path(sys.executable).resolve().parent / "Lib" / "site-packages"
if TRUSTED_SITE.exists():
    sys.path.insert(0, str(TRUSTED_SITE))
import numpy as np

PROJECT_PACKAGES = PROJECT / ".python_packages"
if PROJECT_PACKAGES.exists():
    sys.path.append(str(PROJECT_PACKAGES))
import mujoco

SOURCE_MODELS = PROJECT / "experiments" / "mujoco_template_locked_head_2026-08-27" / "models"
PD_ROOT = PROJECT / "experiments" / "mujoco_nearground_pd_search_2026-09-01"
PD_LEDGER = PD_ROOT / "outputs" / "selected_pd_profiles.json"
PD_RUNNER = PROJECT / "experiments" / "mujoco_pd_multimetric_stability_2026-09-01" / "run_pd_pilot.py"
NEAR_RUNNER = PD_ROOT / "run_search.py"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
DATA = OUT / "data"
TRIALS = OUT / "trials"

OLD_K = 4.5e-6
OLD_C = 1.15375965086e-9
STAGE_A_K = (1e-6, 4.5e-6, 2e-5)
STAGE_A_DAMPING_SCALE = (0.5, 1.0, 2.0)
STAGE_B_PAIRS = (
    (1.2e-5, 1.0), (1.2e-5, 2.0),
    (1.6e-5, 1.0), (1.6e-5, 2.0),
    (2.0e-5, 1.5),
    (2.5e-5, 1.0), (2.5e-5, 2.0),
    (3.2e-5, 1.0), (3.2e-5, 2.0),
)
STAGE_C_PAIRS = (
    (1.6e-5, 3.2), (1.6e-5, 4.2),
    (2.0e-5, 3.2), (2.0e-5, 4.2),
    (2.5e-5, 3.2), (2.5e-5, 4.2), (2.5e-5, 6.4),
)
NEW_AXIS_EFFECTIVE_INERTIA = 3.326499300917627e-13
WAVES = (2.5, 3.0)
LIMIT_DEG = 38.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pilot = load_module(PD_RUNNER, "tip_flexion_pd_runner")
near = load_module(NEAR_RUNNER, "tip_flexion_near_runner")
opt = pilot.opt


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def number(value: float) -> str:
    return f"{value:.12g}"


def slug(value: float) -> str:
    return number(value).replace("-", "m").replace(".", "p")


def candidate_name(k: float, damping_scale: float, limit_deg: float = LIMIT_DEG) -> str:
    return f"localz_k{slug(k)}_d{slug(damping_scale)}_r{slug(limit_deg)}"


def damping_for(k: float, damping_scale: float) -> float:
    # Preserve the historical sqrt(k) damping law, then explore a declared scale.
    return OLD_C * math.sqrt(k / OLD_K) * damping_scale


def source_model(terrain: str) -> Path:
    return SOURCE_MODELS / f"centipede_{terrain}_locked_head_m29p0.xml"


def build_model(terrain: str, k: float, damping_scale: float, limit_deg: float = LIMIT_DEG) -> dict:
    source = source_model(terrain)
    tree = ET.parse(source)
    root = tree.getroot()
    name = candidate_name(k, damping_scale, limit_deg)
    root.set("model", f"centipede_tip_flexion_{terrain}_{name}")
    joints = [j for j in root.iter("joint") if j.get("name", "").startswith("joint_tip_")]
    if len(joints) != 38:
        raise RuntimeError(f"expected 38 tip joints in {source}, got {len(joints)}")
    c = damping_for(k, damping_scale)
    limit = math.radians(limit_deg)
    for joint in joints:
        joint.set("axis", "0 0 1")
        joint.set("stiffness", number(k))
        joint.set("damping", number(c))
        joint.set("range", f"{-limit:.12g} {limit:.12g}")
        joint.set("springref", "0")
        joint.set("armature", "0")
    MODELS.mkdir(parents=True, exist_ok=True)
    target = MODELS / f"centipede_{terrain}_{name}.xml"
    ET.indent(tree, space="  ")
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return dict(
        candidate=name,
        terrain=terrain,
        source=str(source),
        model=str(target),
        model_sha256=sha256(target),
        tip_joint_count=len(joints),
        axis_tip_local=[0.0, 0.0, 1.0],
        stiffness_nm_rad=k,
        damping_nms_rad=c,
        damping_scale=damping_scale,
        limit_deg=limit_deg,
    )


def stage_a_bank() -> list[dict]:
    return [
        dict(name=candidate_name(k, ds), stiffness_nm_rad=k, damping_scale=ds,
             damping_nms_rad=damping_for(k, ds), limit_deg=LIMIT_DEG)
        for k in STAGE_A_K for ds in STAGE_A_DAMPING_SCALE
    ]


def stage_b_bank() -> list[dict]:
    return [
        dict(name=candidate_name(k, ds), stiffness_nm_rad=k, damping_scale=ds,
             damping_nms_rad=damping_for(k, ds), limit_deg=LIMIT_DEG)
        for k, ds in STAGE_B_PAIRS
    ]


def stage_c_bank() -> list[dict]:
    return [
        dict(name=candidate_name(k, ds), stiffness_nm_rad=k, damping_scale=ds,
             damping_nms_rad=damping_for(k, ds), limit_deg=LIMIT_DEG,
             estimated_damping_ratio=damping_for(k, ds) / (2.0 * math.sqrt(k * NEW_AXIS_EFFECTIVE_INERTIA)))
        for k, ds in STAGE_C_PAIRS
    ]


def registered_bank(stage: str) -> list[dict]:
    if stage == "a":
        return stage_a_bank()
    if stage == "b":
        return stage_b_bank()
    return stage_c_bank()


def generate(stage: str = "a") -> list[dict]:
    manifest = []
    bank = registered_bank(stage)
    for item in bank:
        for terrain in ("flat", "rough"):
            manifest.append(build_model(terrain, item["stiffness_nm_rad"], item["damping_scale"], item["limit_deg"]))
    DATA.mkdir(parents=True, exist_ok=True)
    payload = dict(
        scope="Centipede Model only; Robot tree and Robot Step 18/19 untouched",
        semantic_joint="distal_tip_flexion_passive_gliding",
        legacy_identifier_suffix="_roll retained only for evaluator compatibility",
        stage=stage,
        bank=bank,
        models=manifest,
    )
    (DATA / f"stage_{stage}_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest


def pd_config(wave: float) -> dict:
    ledger = json.loads(PD_LEDGER.read_text(encoding="utf-8"))
    raw = ledger["selected_profiles"][f"{wave:g}"]["multipliers_relative_to_baseline"]
    cfg = dict(raw)
    cfg["name"] = ledger["selected_profiles"][f"{wave:g}"]["config_name"]
    return cfg


def amplitude_case(wave: float) -> dict:
    ledger = json.loads(PD_LEDGER.read_text(encoding="utf-8"))
    a = ledger["amplitudes_frozen_deg"]
    return dict(wave=wave, body=a["body"], leg_yaw=a["leg_yaw"], leg_pitch=a["leg_pitch"])


def rough_near_ground(ev, clearance: float = 25e-6) -> dict:
    """Root-only hfield placement using ground-contact onset bisection.

    ``mj_geomDistance`` is not a reliable signed-distance oracle for all
    hfield/mesh pairs.  The root is therefore bracketed between a collision-
    free and a ground-contacting height, bisected to sub-nanometre vertical
    tolerance, and lifted by the declared clearance.  Relative joints and all
    velocities remain unchanged.
    """
    model, data, index = ev.model, ev.data, ev.index
    original = data.qpos.copy()
    first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_body_1")
    q = np.empty(4)
    mujoco.mju_negQuat(q, model.body_quat[first])
    data.qpos[3:7] = q

    def ground_contact() -> bool:
        mujoco.mj_forward(model, data)
        return any(
            int(data.contact[j].geom1) == ev.ground or int(data.contact[j].geom2) == ev.ground
            for j in range(data.ncon)
        )

    # Find a collision-free upper bracket.
    upper = float(data.qpos[2])
    data.qpos[2] = upper
    for _ in range(20):
        if not ground_contact():
            break
        upper += 0.005
        data.qpos[2] = upper
    else:
        raise RuntimeError("could not find collision-free rough-terrain root height")

    # Lower until the first ground collision provides the lower bracket.
    lower = upper
    for _ in range(200):
        lower -= 0.0005
        data.qpos[2] = lower
        if ground_contact():
            break
    else:
        raise RuntimeError("could not bracket rough-terrain ground contact")

    # lower contacts, upper does not.
    for _ in range(48):
        middle = 0.5 * (lower + upper)
        data.qpos[2] = middle
        if ground_contact():
            lower = middle
        else:
            upper = middle
    onset_free_z = upper
    data.qpos[2] = onset_free_z + clearance
    mujoco.mj_forward(model, data)
    if ground_contact():
        raise RuntimeError("rough near-ground placement retained a ground contact")
    if not np.array_equal(original[7:], data.qpos[7:]):
        raise RuntimeError("rough initialization changed a relative joint")
    if np.count_nonzero(data.qvel):
        raise RuntimeError("rough initialization introduced velocity")
    body_z = data.xpos[index.chassis_body_ids[1:], 2]
    return dict(
        method="root_only_hfield_contact_onset_bisection",
        target_clearance_m=clearance,
        onset_free_root_z_m=onset_free_z,
        root_after_z_m=float(data.qpos[2]),
        onset_bracket_width_m=float(upper - lower),
        root_before=original[:7].tolist(), root_after=data.qpos[:7].tolist(),
        chassis_origin_height_spread_m=float(np.ptp(body_z)),
        all_relative_joint_positions_unchanged=True,
        initial_velocity_zero=True,
        settling="unchanged zero-target interval t=0..1s",
    )


def run_one(task: tuple[dict, float, str, float, float, bool]) -> dict:
    candidate, wave, terrain, duration, dt_scale, trace = task
    model = source_model(terrain) if candidate.get("source_old_axis") else MODELS / f"centipede_{terrain}_{candidate['name']}.xml"
    if not model.exists() and not candidate.get("source_old_axis"):
        build_model(terrain, candidate["stiffness_nm_rad"], candidate["damping_scale"], candidate["limit_deg"])
    output = TRIALS / terrain / candidate["name"]
    expected_tag = f"{candidate['name']}_w{wave:g}_t{duration:g}_dt{0.00025 * dt_scale:g}".replace(".", "p")
    cached = output / f"{expected_tag}.json"
    if cached.exists() and (not trace or cached.with_suffix(".npz").exists()):
        record = json.loads(cached.read_text(encoding="utf-8"))
        if record.get("model_sha256") == sha256(model) and record.get("metrics"):
            return record
    opt.MODEL = model
    cfg = pd_config(wave)
    initializer = near.near_ground if terrain == "flat" else rough_near_ground
    result = pilot.run_task(
        (cfg, wave, duration, dt_scale, trace),
        initialize=initializer,
        case_override=amplitude_case(wave),
        output_dir=output,
    )
    result["model_candidate"] = candidate
    result["terrain"] = terrain
    result["semantic_joint"] = "distal_tip_flexion_passive_gliding"
    result["model_path"] = str(model)
    result["model_sha256"] = sha256(model)
    tag = f"{candidate['name']}_w{wave:g}_t{duration:g}_dt{result['dt']:g}".replace(".", "p")
    result["experiment_id"] = tag
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"{tag}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    # Keep a predictable NPZ alias when run_task emitted one under its legacy id.
    if trace:
        legacy = output / f"{result['id']}.npz"
        alias = output / f"{tag}.npz"
        if legacy.exists():
            legacy.replace(alias)
    return result


def run_bank(stage: str, terrain: str, duration: float, dt_scale: float, workers: int, trace: bool) -> list[dict]:
    generate(stage)
    bank = registered_bank(stage)
    tasks = [
        (candidate, wave, terrain, duration, dt_scale, trace)
        for candidate in bank for wave in WAVES
    ]
    results = []
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(run_one, tasks):
            results.append(result)
            failed = [key for key, passed in result["gates"].items() if not passed]
            print(json.dumps(dict(
                done=len(results), total=len(tasks), candidate=result["model_candidate"]["name"],
                wave=result["case"]["wave"], terrain=terrain,
                displacement_mm=result["metrics"]["displacement_mm"], failed=failed,
            )), flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    tag = f"stage_{stage}_{terrain}_t{duration:g}_dt{0.00025 * dt_scale:g}".replace(".", "p")
    ledger = dict(tag=tag, elapsed_s=time.monotonic() - started, results=results)
    (DATA / f"{tag}.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    summarize(results, stage, terrain, duration, dt_scale)
    return results


def run_validation(candidate_id: str, terrain: str, duration: float, dt_scale: float,
                   workers: int, trace: bool) -> list[dict]:
    bank = {item["name"]: item for item in stage_a_bank() + stage_b_bank() + stage_c_bank()}
    bank["source_old_axis"] = dict(
        name="source_old_axis", source_old_axis=True, stiffness_nm_rad=OLD_K,
        damping_scale=1.0, damping_nms_rad=OLD_C, limit_deg=LIMIT_DEG,
    )
    if candidate_id not in bank:
        raise KeyError(f"unknown registered candidate {candidate_id}")
    candidate = bank[candidate_id]
    if not candidate.get("source_old_axis"):
        for ground in ("flat", "rough"):
            model = MODELS / f"centipede_{ground}_{candidate_id}.xml"
            if not model.exists():
                build_model(ground, candidate["stiffness_nm_rad"], candidate["damping_scale"], candidate["limit_deg"])
    tasks = [(candidate, wave, terrain, duration, dt_scale, trace) for wave in WAVES]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_one, tasks))
    DATA.mkdir(parents=True, exist_ok=True)
    tag = f"validation_{candidate_id}_{terrain}_t{duration:g}_dt{0.00025 * dt_scale:g}".replace(".", "p")
    payload = dict(tag=tag, candidate=candidate, results=results)
    (DATA / f"{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for result in results:
        print(json.dumps(dict(
            candidate=candidate_id, terrain=terrain, wave=result["case"]["wave"],
            duration=duration, dt=result["dt"], displacement_mm=result["metrics"]["displacement_mm"],
            failed=[key for key, passed in result["gates"].items() if not passed],
        )), flush=True)
    return results


def run_validation_set(candidate_ids: list[str], terrain: str, duration: float,
                       dt_scale: float, workers: int, trace: bool) -> list[dict]:
    bank = {item["name"]: item for item in stage_a_bank() + stage_b_bank() + stage_c_bank()}
    missing = [name for name in candidate_ids if name not in bank]
    if missing:
        raise KeyError(f"unknown registered candidates: {missing}")
    tasks = []
    for name in candidate_ids:
        candidate = bank[name]
        for ground in ("flat", "rough"):
            model = MODELS / f"centipede_{ground}_{name}.xml"
            if not model.exists():
                build_model(ground, candidate["stiffness_nm_rad"], candidate["damping_scale"], candidate["limit_deg"])
        tasks.extend((candidate, wave, terrain, duration, dt_scale, trace) for wave in WAVES)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_one, tasks))
    tag = f"validation_set_{terrain}_t{duration:g}_dt{0.00025 * dt_scale:g}".replace(".", "p")
    payload = dict(tag=tag, candidate_ids=candidate_ids, results=results)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for result in results:
        print(json.dumps(dict(
            candidate=result["model_candidate"]["name"], terrain=terrain,
            wave=result["case"]["wave"], displacement_mm=result["metrics"]["displacement_mm"],
            failed=[key for key, passed in result["gates"].items() if not passed],
        )), flush=True)
    return results


def summarize(results: list[dict], stage: str, terrain: str, duration: float, dt_scale: float) -> dict:
    old = {2.5: 149.9622501702571, 3.0: 129.274380217374}
    grouped: dict[str, list[dict]] = {}
    for result in results:
        grouped.setdefault(result["model_candidate"]["name"], []).append(result)
    rows = []
    for name, values in grouped.items():
        feasible = all(all(v["gates"].values()) for v in values) and len(values) == len(WAVES)
        retention = [v["metrics"]["displacement_mm"] / old[float(v["case"]["wave"])] for v in values]
        rows.append(dict(
            candidate=name,
            all_wave_gates_pass=feasible,
            minimum_displacement_retention=min(retention),
            maximum_tip_speed_peak=max(v["metrics"]["tip_speed_peak"] for v in values),
            maximum_ground_force_peak_bw=max(v["metrics"]["ground_force_peak_bw"] for v in values),
            failed={f"{v['case']['wave']:g}": [k for k, p in v["gates"].items() if not p] for v in values},
            waves={f"{v['case']['wave']:g}": v["metrics"] for v in values},
        ))
    rows.sort(key=lambda row: (
        not row["all_wave_gates_pass"],
        -row["minimum_displacement_retention"],
        row["maximum_tip_speed_peak"],
        row["maximum_ground_force_peak_bw"],
    ))
    summary = dict(
        stage=stage, terrain=terrain,
        duration_s=duration,
        timestep_s=0.00025 * dt_scale,
        selection_rule="hard gates first; then max min displacement retention, min tip speed, min force",
        continuous_global_proof=False,
        ranking=rows,
    )
    tag = f"stage_{stage}_{terrain}_summary_t{duration:g}_dt{0.00025 * dt_scale:g}".replace(".", "p")
    (DATA / f"{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def audit_model(path: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(path))
    index = opt.tip.LeanIndex(model)
    axes = model.jnt_axis[index.tip_joint_ids]
    q = np.abs(model.jnt_range[index.tip_joint_ids])
    return dict(
        model=str(path), sha256=sha256(path), nq=model.nq, nv=model.nv, nu=model.nu,
        nbody=model.nbody, ngeom=model.ngeom,
        total_mass_g=float(model.body_mass.sum() * 1000),
        maximum_armature=float(np.max(np.abs(model.dof_armature))),
        tip_joint_count=int(index.tip_joint_ids.size),
        unique_tip_axes=np.unique(np.round(axes, 12), axis=0).tolist(),
        tip_stiffness_unique=np.unique(model.jnt_stiffness[index.tip_joint_ids]).tolist(),
        tip_damping_unique=np.unique(model.dof_damping[index.tip_dof_addresses]).tolist(),
        tip_limit_abs_deg_unique=np.unique(np.round(np.degrees(q), 9)).tolist(),
    )


def audit(stage: str = "a") -> dict:
    manifest = generate(stage)
    source = source_model("flat")
    source_audit = audit_model(source)
    candidates = [audit_model(Path(item["model"])) for item in manifest]
    payload = dict(source=source_audit, candidates=candidates)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"plant_audit_stage_{stage}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "audit", "search", "validate"))
    parser.add_argument("--stage", choices=("a", "b", "c"), default="a")
    parser.add_argument("--terrain", choices=("flat", "rough"), default="flat")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--dt-scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--candidate", nargs="+", default=["localz_k2p5em05_d1_r38"])
    args = parser.parse_args()
    if args.command == "generate":
        print(json.dumps(generate(args.stage), indent=2))
    elif args.command == "audit":
        audit(args.stage)
    elif args.command == "search":
        run_bank(args.stage, args.terrain, args.duration, args.dt_scale, args.workers, args.trace)
    else:
        if len(args.candidate) == 1:
            run_validation(args.candidate[0], args.terrain, args.duration, args.dt_scale,
                           min(args.workers, len(WAVES)), args.trace)
        else:
            run_validation_set(args.candidate, args.terrain, args.duration, args.dt_scale,
                               min(args.workers, len(args.candidate) * len(WAVES)), args.trace)


if __name__ == "__main__":
    main()
