"""Paired rigid-tip ablation with the frozen controller, amplitudes and PD profiles."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PREVIOUS = PROJECT / "experiments" / "mujoco_nearground_pd_search_2026-09-01"
MODEL = ROOT / "models" / "centipede_flat_locked_head_m29p0_rigid_tip.xml"
OUT = ROOT / "outputs" / "data"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


near = load(PREVIOUS / "run_search.py", "rigid_tip_near_source")
np, mujoco, opt, pilot = near.np, near.mujoco, near.opt, near.pilot
base, tip = opt.base, opt.tip


@dataclass
class RigidTipIndex:
    model: mujoco.MjModel

    def __post_init__(self) -> None:
        impedance = tip.stage1.selected_impedance()
        actuator_for_joint = {
            int(self.model.actuator_trnid[actuator_id, 0]): actuator_id
            for actuator_id in range(self.model.nu)
        }
        joint_ids, joint_names, actuator_ids, qpos, dofs, kp, kv = [], [], [], [], [], [], []

        def add(name: str, family: str) -> int:
            joint_id = base.name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id not in actuator_for_joint:
                raise RuntimeError(f"missing active actuator for {name}")
            position = len(joint_ids)
            joint_ids.append(joint_id)
            joint_names.append(name)
            actuator_ids.append(actuator_for_joint[joint_id])
            qpos.append(int(self.model.jnt_qposadr[joint_id]))
            dofs.append(int(self.model.jnt_dofadr[joint_id]))
            kp.append(float(impedance[family][0]))
            kv.append(float(impedance[family][1]))
            return position

        self.body_positions = np.asarray(
            [add(f"joint_body_{index}", "body_yaw") for index in range(1, 20)], dtype=int
        )
        self.locked_head = True
        self.pitch_segment_indices = tuple(range(1, 20))
        self.pitch_positions = np.asarray(
            [add(f"joint_pitch_body_{index}", "body_pitch") for index in range(1, 20)],
            dtype=int,
        )
        self.leg_positions = np.empty((19, 2, 2), dtype=int)
        for segment in range(19):
            for side_index, side in enumerate(("L", "R")):
                self.leg_positions[segment, side_index, 0] = add(
                    f"joint_leg_{segment}_{side}_0", "legbase_yaw"
                )
                self.leg_positions[segment, side_index, 1] = add(
                    f"joint_leg_{segment}_{side}_1", "legbase_pitch"
                )
        if len(joint_ids) != 114:
            raise RuntimeError(f"expected 114 active joints, got {len(joint_ids)}")
        self.active_joint_ids = np.asarray(joint_ids, dtype=int)
        self.active_joint_names = joint_names
        self.active_actuator_ids = np.asarray(actuator_ids, dtype=int)
        self.qpos_addresses = np.asarray(qpos, dtype=int)
        self.dof_addresses = np.asarray(dofs, dtype=int)
        self.kp = np.asarray(kp, dtype=float)
        self.kv = np.asarray(kv, dtype=float)
        self.roll_joint_ids = np.asarray(
            [
                base.name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_roll_body_{i}")
                for i in range(1, 20)
            ],
            dtype=int,
        )
        self.roll_qpos_addresses = self.model.jnt_qposadr[self.roll_joint_ids].astype(int)
        self.roll_dof_addresses = self.model.jnt_dofadr[self.roll_joint_ids].astype(int)
        self.tip_joint_ids = np.asarray([], dtype=int)
        self.tip_qpos_addresses = np.asarray([], dtype=int)
        self.tip_dof_addresses = np.asarray([], dtype=int)
        self.chassis_body_ids = np.asarray(
            [
                base.name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"link_body_{i}")
                for i in range(21)
            ],
            dtype=int,
        )
        self.head_body_id = int(self.chassis_body_ids[0])


class RigidTipEvaluator:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.index = RigidTipIndex(self.model)
        tip.configure_model(self.model, self.index)
        self.data = mujoco.MjData(self.model)
        self.ground = int(tip.stage1.ground_geom_id(self.model))
        self.foot_geoms = np.asarray(
            [
                base.name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"foot_{segment}_{side}_collision",
                )
                for segment in range(19)
                for side in range(2)
            ],
            dtype=int,
        )


def selected_config(wave: float) -> dict[str, object]:
    name = (
        "legyaw_refine_k0p29_w2p5_t6_dt0p00025.json"
        if wave == 2.5
        else "basin15_yk2_pd4_rk1_w3_t6_dt0p00025.json"
    )
    return json.loads((PREVIOUS / "outputs" / "common" / name).read_text())["config"]


def fixed_case(wave: float) -> dict[str, float]:
    return dict(near.amplitude_case(wave, "common"))


def run(wave: float, duration: float = 6.0, dt_scale: float = 1.0, trace: bool = True):
    started = time.monotonic()
    cfg = selected_config(wave)
    ev = RigidTipEvaluator()
    m, d, ix = ev.model, ev.data, ev.index
    pilot.apply_impedance(m, ix, cfg)
    m.opt.timestep *= dt_scale
    dt = float(m.opt.timestep)
    stride = max(1, int(round(0.001 / dt)))
    case = fixed_case(wave)
    ctl = opt.ThreeAmplitudeController(
        m, ix, wave, case["body"], case["leg_yaw"], case["leg_pitch"]
    )
    body_ids = np.stack(
        (
            ix.active_joint_ids[ix.body_positions],
            ix.active_joint_ids[ix.pitch_positions],
            ix.roll_joint_ids,
        ),
        axis=-1,
    )
    qa, va = m.jnt_qposadr[body_ids], m.jnt_dofadr[body_ids]
    legqa = ix.qpos_addresses[ix.leg_positions.reshape(-1)]
    legva = ix.dof_addresses[ix.leg_positions.reshape(-1)]
    foot_sites = np.asarray(
        [
            base.name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"site_foot_{segment}_{side}")
            for segment in range(19)
            for side in range(2)
        ],
        dtype=int,
    )
    foot_map = {int(geom): index for index, geom in enumerate(ev.foot_geoms)}
    body_map = {int(body): index for index, body in enumerate(ix.chassis_body_ids)}
    n = int(round(duration / dt))
    weight = float(m.body_mass.sum() * 9.81)
    force = np.zeros((n, 60), dtype=np.float64)
    pen = np.zeros(n)
    torque = np.zeros(n)
    all_speed = np.zeros(n)
    abswork = np.zeros(n)
    poswork = np.zeros(n)
    body_speed = np.zeros((n, 3))
    accmax = np.zeros((n, 3))
    foot_mask = np.zeros((n, 38), dtype=bool)
    times, bq, bv, om, xpos, rootmat, worldroll = [], [], [], [], [], [], []
    lq, lv, lt, fq, fs, qpos_record = [], [], [], [], [], []
    prevel = np.empty(m.nv)
    contact_force = np.zeros(6)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    initialization = near.near_ground(ev)
    finite = True
    for k in range(n):
        prevel[:] = d.qvel
        if k % stride == 0:
            times.append(k * dt)
            bq.append(d.qpos[qa].copy())
            bv.append(d.qvel[va].copy())
            lq.append(d.qpos[legqa].copy())
            lv.append(d.qvel[legva].copy())
            qpos_record.append(d.qpos.copy())
        ctl.step(d)
        mujoco.mj_step(m, d)
        torque[k] = np.max(np.abs(d.actuator_force[ix.active_actuator_ids]))
        power = d.actuator_force[ix.active_actuator_ids] * prevel[ix.dof_addresses]
        abswork[k] = np.sum(np.abs(power)) * dt
        poswork[k] = np.sum(np.maximum(power, 0)) * dt
        body_speed[k] = np.max(np.abs(d.qvel[va]), axis=0)
        accmax[k] = np.max(np.abs(d.qacc[va]), axis=0)
        all_speed[k] = np.max(np.abs(d.qvel[ix.dof_addresses]))
        for contact_id in range(d.ncon):
            contact = d.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            other = geom2 if geom1 == ev.ground else (geom1 if geom2 == ev.ground else -1)
            if other < 0:
                continue
            mujoco.mj_contactForce(m, d, contact_id, contact_force)
            normal = max(0.0, float(contact_force[0]))
            if other in foot_map:
                foot = foot_map[other]
                force[k, foot] += normal
                foot_mask[k, foot] = True
            elif int(m.geom_bodyid[other]) in body_map:
                force[k, 38 + body_map[int(m.geom_bodyid[other])]] += normal
            else:
                force[k, 59] += normal
            pen[k] = max(pen[k], -float(contact.dist))
        if k % stride == 0:
            lt.append(ctl.target[ix.leg_positions.reshape(-1)].copy())
            om.append(d.cvel[ix.chassis_body_ids, :3].copy())
            xpos.append(d.xpos[ix.chassis_body_ids].copy())
            rootmat.append(d.xmat[int(ix.chassis_body_ids[0])].reshape(3, 3).copy())
            worldroll.append(tip.signed_world_roll_deg(d, ix.chassis_body_ids))
            fq.append(d.site_xpos[foot_sites].copy())
            fs.append(k)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            finite = False
            break

    a = dict(
        t=np.asarray(times),
        bq=np.asarray(bq),
        bv=np.asarray(bv),
        om=np.asarray(om),
        xpos=np.asarray(xpos),
        rootmat=np.asarray(rootmat),
        worldroll=np.asarray(worldroll),
        lq=np.asarray(lq),
        lv=np.asarray(lv),
        lt=np.asarray(lt),
        fq=np.asarray(fq),
        fs=np.asarray(fs, dtype=int),
        qpos=np.asarray(qpos_record),
    )
    good = a["t"] >= 1.5
    startstep = int(round(1.5 / dt))
    fine = slice(startstep, k + 1)
    t = a["t"][good]
    x = a["xpos"][good]
    center = x.mean(axis=1)
    path = float(np.linalg.norm(np.diff(center[:, :2], axis=0), axis=1).sum())
    displacement = float(base.FORWARD_SIGN * (center[-1, 0] - center[0, 0]))
    bq1, bv1, om1 = a["bq"][good], a["bv"][good], a["om"][good]
    lag = int(round(0.01 / (dt * stride)))
    dq = np.degrees(np.abs(bq1[lag:] - bq1[:-lag]))
    dom = np.linalg.norm(om1[lag:] - om1[:-lag], axis=-1)
    rm = a["rootmat"][good]
    rootroll = np.degrees(np.arctan2(rm[:, 2, 1], rm[:, 2, 2]))
    wr = np.radians(a["worldroll"][good])
    coherent = np.degrees(np.arctan2(np.sin(wr).mean(axis=1), np.cos(wr).mean(axis=1)))
    front = np.degrees(
        np.arctan2(np.sin(wr[:, :7]).mean(axis=1), np.cos(wr[:, :7]).mean(axis=1))
    )
    f = force[fine]
    total = f.sum(axis=1)
    chassis = f[:, 38:59].sum(axis=1)
    head_link1 = f[:, 38:40].sum(axis=1)
    loaded = f[:, :38] > 1e-7
    sample_idx = a["fs"][good]
    contacts = foot_mask[sample_idx]
    legq = a["lq"][good].reshape(-1, 19, 2, 2)
    legv = a["lv"][good].reshape(-1, 19, 2, 2)
    target = a["lt"][good].reshape(-1, 19, 2, 2)
    oldstride = max(1, int(round(0.01 / (dt * stride))))
    tt = t[::oldstride]
    yaw = legq[..., 0].reshape(len(t), -1)
    pitch = legq[..., 1].reshape(len(t), -1)
    yawv = legv[..., 0].reshape(len(t), -1)
    pitchv = legv[..., 1].reshape(len(t), -1)
    cycle_d = []
    for begin in np.arange(2.0, duration - 0.999, 1.0):
        i = int(np.argmin(np.abs(t - begin)))
        j = int(np.argmin(np.abs(t - (begin + 1.0))))
        cycle_d.append(float(base.FORWARD_SIGN * (center[j, 0] - center[i, 0])))
    fv = np.diff(a["fq"][good], axis=0) / (dt * stride)
    slip = np.linalg.norm(fv[:, :, :2], axis=-1)[loaded[sample_idx[1:] - startstep]]
    mtr = dict(
        displacement_mm=displacement * 1000,
        path_efficiency=displacement / max(path, 1e-15),
        min_upright=float(rm[:, 2, 2].min()),
        spine_tortuosity_max=float(
            np.max(
                np.linalg.norm(np.diff(x, axis=1), axis=2).sum(axis=1)
                / np.maximum(np.linalg.norm(x[:, -1] - x[:, 0], axis=1), 1e-15)
            )
        ),
        root_roll_abs_peak_deg=float(np.abs(rootroll).max()),
        root_roll_abs_p95_deg=pilot.pct(np.abs(rootroll), 0.95),
        coherent_roll_p95_deg=pilot.pct(np.abs(coherent), 0.95),
        front_roll_p95_deg=pilot.pct(np.abs(front), 0.95),
        center_height_ptp_mm=float(np.ptp(center[:, 2]) * 1000),
        center_vertical_speed_p99_m_s=pilot.pct(
            np.abs(np.diff(center[:, 2]) / (dt * stride)), 0.99
        ),
        body_yaw_speed_peak=float(body_speed[fine, 0].max()),
        body_pitch_speed_peak=float(body_speed[fine, 1].max()),
        body_roll_speed_peak=float(body_speed[fine, 2].max()),
        body_deltaomega_10ms_peak=float(dom.max()),
        body_pitch_delta10ms_deg=float(dq[:, :, 1].max()),
        body_roll_delta10ms_deg=float(dq[:, :, 2].max()),
        body_accel_peak_rad_s2=accmax[fine].max(axis=0).tolist(),
        ground_force_peak_bw=float(total.max() / weight),
        ground_force_p99_bw=pilot.pct(total / weight, 0.99),
        chassis_force_peak_bw=float(chassis.max() / weight),
        foot_force_peak_bw=float(f[:, :38].max() / weight),
        head_link1_force_peak_bw=float(head_link1.max() / weight),
        head_link1_contact_fraction=float(np.mean(head_link1 > 1e-7)),
        normal_impulse_10ms_bw_s=pilot.window_integral(total / weight, dt),
        chassis_impulse_10ms_bw_s=pilot.window_integral(chassis / weight, dt),
        head_link1_impulse_10ms_bw_s=pilot.window_integral(head_link1 / weight, dt),
        chassis_contact_fraction=float(np.mean(chassis > 1e-7)),
        mean_contacting_feet=float(contacts.sum(axis=1).mean()),
        ballistic_gap_s=pilot.max_run(~loaded.any(axis=1)) * dt,
        penetration_p95_mm=pilot.pct(pen[fine][pen[fine] > 0] * 1000, 0.95),
        penetration_peak_mm=float(pen[fine].max() * 1000),
        active_speed_peak=float(all_speed[fine].max()),
        torque_peak_nm=float(torque.max()),
        torque_steady_peak_nm=float(torque[fine].max()),
        absolute_work_mJ=float(abswork[fine].sum() * 1000),
        positive_work_mJ=float(poswork[fine].sum() * 1000),
        work_per_distance_n=float(abswork[fine].sum() / max(displacement, 1e-15)),
        cycle_displacement_mm=[value * 1000 for value in cycle_d],
        cycle_displacement_cv=float(np.std(cycle_d) / max(np.mean(cycle_d), 1e-15)),
        loaded_foot_slip_p95_mm_s=pilot.pct(slip * 1000, 0.95),
        left_right_force_imbalance_rms_bw=float(
            np.sqrt(np.mean((f[:, :38:2].sum(axis=1) - f[:, 1:38:2].sum(axis=1)) ** 2))
            / weight
        ),
        body_wave_residual_p95=pilot.pct(pilot.pilot.sinusoid_residual(tt, bq1[::oldstride, 1:, 0]), 0.95),
        leg_yaw_wave_residual_p95=pilot.pct(pilot.pilot.sinusoid_residual(tt, yaw[::oldstride]), 0.95),
        leg_pitch_wave_residual_p95=pilot.pct(pilot.pilot.sinusoid_residual(tt, pitch[::oldstride]), 0.95),
        leg_yaw_speed_p99=pilot.pct(np.abs(yawv), 0.99),
        leg_pitch_speed_p99=pilot.pct(np.abs(pitchv), 0.99),
        leg_yaw_reversals_p95=pilot.pct(pilot.pilot.reversal_rate(tt, yawv[::oldstride]), 0.95),
        leg_pitch_reversals_p95=pilot.pct(pilot.pilot.reversal_rate(tt, pitchv[::oldstride]), 0.95),
        leg_pitch_tracking_rms_deg=float(
            np.degrees(np.sqrt(np.mean((legq[..., 1] - target[..., 1]) ** 2)))
        ),
        leg_yaw_tracking_rms_deg=float(
            np.degrees(np.sqrt(np.mean((legq[..., 0] - target[..., 0]) ** 2)))
        ),
        swing_foot_height_p95_mm=pilot.pct(a["fq"][good, :, 2][~contacts] * 1000, 0.95),
        contact_duty_mean=float(contacts.mean()),
        left_right_duty_difference=float(
            np.abs(contacts.mean(axis=0).reshape(19, 2)[:, 0] - contacts.mean(axis=0).reshape(19, 2)[:, 1]).mean()
        ),
    )
    warnings = tip.stage1.warning_counts(d)
    gates = dict(
        finite_warning_free=finite and not warnings,
        positive_forward=displacement > 0,
        path_efficiency=mtr["path_efficiency"] >= 0.8,
        upright=mtr["min_upright"] >= math.cos(math.radians(50)),
        spine=mtr["spine_tortuosity_max"] <= 1.2,
        root_roll_peak=mtr["root_roll_abs_peak_deg"] <= 20,
        height=mtr["center_height_ptp_mm"] <= 2,
        vertical_speed=mtr["center_vertical_speed_p99_m_s"] <= 0.1,
        coherent_roll=mtr["coherent_roll_p95_deg"] <= 10,
        front_roll=mtr["front_roll_p95_deg"] <= 12,
        body_yaw_speed=mtr["body_yaw_speed_peak"] <= 12,
        body_pitch_speed=mtr["body_pitch_speed_peak"] <= 8,
        body_roll_speed=mtr["body_roll_speed_peak"] <= 5,
        body_velocity_jump=mtr["body_deltaomega_10ms_peak"] <= 10,
        pitch_10ms=mtr["body_pitch_delta10ms_deg"] <= 2,
        roll_10ms=mtr["body_roll_delta10ms_deg"] <= 2,
        total_force=mtr["ground_force_peak_bw"] <= 2.5,
        chassis_force=mtr["chassis_force_peak_bw"] <= 1,
        foot_force=mtr["foot_force_peak_bw"] <= 1,
        chassis_contact=mtr["chassis_contact_fraction"] <= 0.03,
        feet_count=6 <= mtr["mean_contacting_feet"] <= 28,
        no_ballistic=mtr["ballistic_gap_s"] <= 0.08,
        penetration=mtr["penetration_p95_mm"] <= 0.1,
        active_speed=mtr["active_speed_peak"] <= 100,
        torque=mtr["torque_peak_nm"] <= 2e-4,
        cycle_consistency=mtr["cycle_displacement_cv"] <= 0.15,
        body_wave=mtr["body_wave_residual_p95"] <= 0.4,
        leg_yaw_wave=mtr["leg_yaw_wave_residual_p95"] <= 0.18,
        leg_pitch_wave=mtr["leg_pitch_wave_residual_p95"] <= 0.4,
        leg_yaw_speed=mtr["leg_yaw_speed_p99"] <= 8,
        leg_pitch_speed=mtr["leg_pitch_speed_p99"] <= 7,
        leg_yaw_reversals=mtr["leg_yaw_reversals_p95"] <= 3,
        leg_pitch_reversals=mtr["leg_pitch_reversals_p95"] <= 6.25,
        pitch_tracking=mtr["leg_pitch_tracking_rms_deg"] <= 10,
        yaw_tracking=mtr["leg_yaw_tracking_rms_deg"] <= 10,
        swing_height=mtr["swing_foot_height_p95_mm"] <= 5.5,
        duty=0.15 <= mtr["contact_duty_mean"] <= 0.65,
        lr_duty=mtr["left_right_duty_difference"] <= 0.1,
    )
    rest_mask = (a["t"] >= 0.9) & (a["t"] < 1.0)
    rest = dict(
        body_speed_peak_rad_s=float(np.abs(a["bv"][rest_mask]).max()),
        center_height_ptp_mm=float(np.ptp(a["xpos"][rest_mask].mean(axis=1)[:, 2]) * 1000),
        normal_force_peak_bw=float(
            force[int(round(0.9 / dt)) : int(round(1.0 / dt))].sum(axis=1).max() / weight
        ),
    )
    tag = f"rigid_tip_{cfg['name']}_w{wave:g}_t{duration:g}_dt{dt:g}".replace(".", "p")
    result = dict(
        id=tag,
        topology="rigid compound leg; no tip or knee DOF",
        model=str(MODEL),
        model_sha256=tip.sha256(MODEL),
        config=cfg,
        case=case,
        duration=duration,
        dt=dt,
        metrics=mtr,
        gates=gates,
        feasible_applicable_gates=all(gates.values()),
        failed_applicable_gates=[key for key, value in gates.items() if not value],
        not_applicable_original_gates=["tip_speed", "tip_peak", "tip_p95", "tip_stops"],
        initialization=initialization,
        rest_before_gait=rest,
        warnings=warnings,
        elapsed_s=time.monotonic() - started,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    if trace:
        np.savez_compressed(
            OUT / f"{tag}.npz",
            **{
                key: value.astype(np.float32) if value.dtype.kind == "f" else value
                for key, value in a.items()
            },
            force=force.astype(np.float32),
            torque=torque.astype(np.float32),
        )
    (OUT / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--waves", nargs="+", type=float, default=[2.5, 3.0])
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--dt-scale", type=float, default=1.0)
    args = parser.parse_args()
    for requested_wave in args.waves:
        row = run(requested_wave, args.duration, args.dt_scale, True)
        print(
            json.dumps(
                dict(
                    id=row["id"],
                    displacement_mm=row["metrics"]["displacement_mm"],
                    applicable_gates_pass=row["feasible_applicable_gates"],
                    failed=row["failed_applicable_gates"],
                )
            ),
            flush=True,
        )
