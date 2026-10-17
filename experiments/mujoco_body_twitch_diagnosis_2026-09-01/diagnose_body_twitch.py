"""Read-only replay diagnostics: no model, controller or selection changes.

Each sample is the start-of-step state. Contact forces and accelerations are
from the same mj_step evaluation; copied q/qvel precede integration. All
physics steps are logged, with the original timestep and CPG update rate.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(Path(sys.executable).resolve().parent / "Lib" / "site-packages"))
import numpy as np
sys.path.append(str(PROJECT / ".python_packages"))
import mujoco


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SOURCE = PROJECT / "experiments/mujoco_wave_amplitude_natural_gait_optimization_2026-08-31"
demo = load(SOURCE / "render_audit_demos.py", "twitch_original_renderer")
opt = demo.opt
OUT = ROOT / "outputs"


def cases():
    clean = demo.clean_case(1.5)
    clean["id"] = "accepted_w1p5"
    result = [clean]
    selection = json.loads((SOURCE / "outputs/data/final_certified_selection.json").read_text())
    for wave in (1.5, 2.5, 3.0):
        r = selection["waves"][f"{wave:g}"]["selected"]
        result.append(dict(id=f"selected_w{wave:g}".replace(".", "p"),
                           label=r["candidate_id"], wave=wave,
                           body=r["body_yaw_amplitude_deg"],
                           leg_yaw=r["leg_yaw_amplitude_deg"],
                           leg_pitch=r["leg_pitch_amplitude_deg"]))
    return result


def describe(x):
    a = np.abs(np.asarray(x))
    return dict(p95=float(np.quantile(a, .95)), p99=float(np.quantile(a, .99)), max=float(a.max()))


def run(case):
    ev = opt.FastEvaluator()
    m, d, ix = ev.model, ev.data, ev.index
    dt = float(m.opt.timestep)
    n = int(round(6.0 / dt))
    ids = np.stack((ix.active_joint_ids[ix.body_positions],
                    ix.active_joint_ids[ix.pitch_positions], ix.roll_joint_ids), axis=-1)
    qa, va = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
    ctrl = opt.ThreeAmplitudeController(m, ix, case["wave"], case["body"], case["leg_yaw"], case["leg_pitch"])
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    arrays = dict(time=np.arange(n)*dt, body_q=np.empty((n,19,3)),
                  body_qvel=np.empty((n,19,3)), body_qacc=np.empty((n,19,3)),
                  omega_world=np.empty((n,21,3)), xpos=np.empty((n,21,3)),
                  quat=np.empty((n,21,4)), body_contact_n=np.zeros((n,21)),
                  foot_contact_n=np.zeros((n,38)), other_contact_n=np.zeros(n),
                  total_contact_n=np.zeros(n), penetration=np.zeros(n),
                  limits=np.zeros(n,dtype=int), target_speed_max=np.zeros(n),
                  active_torque_max=np.zeros(n), tip_q=np.empty((n,38)),
                  tip_qvel=np.empty((n,38)), leg_qvel=np.empty((n,76)),
                  warnings=np.zeros(n,dtype=int))
    cb = {int(b):i for i,b in enumerate(ix.chassis_body_ids)}
    fg = {int(g):i for i,g in enumerate(ev.foot_geoms)}
    force = np.zeros(6)
    previous_target = None
    full_mass = np.empty((m.nv,m.nv))
    mass_checks = []
    for k in range(n):
        arrays["body_q"][k] = d.qpos[qa]
        arrays["body_qvel"][k] = d.qvel[va]
        arrays["tip_q"][k] = d.qpos[ix.tip_qpos_addresses]
        arrays["tip_qvel"][k] = d.qvel[ix.tip_dof_addresses]
        arrays["leg_qvel"][k] = d.qvel[ix.dof_addresses[ix.leg_positions.reshape(-1)]]
        ctrl.step(d)
        if previous_target is not None:
            arrays["target_speed_max"][k] = np.max(np.abs((ctrl.target-previous_target)/dt))
        previous_target = ctrl.target.copy()
        mujoco.mj_step(m,d)
        arrays["body_qacc"][k] = d.qacc[va]
        arrays["omega_world"][k] = d.cvel[ix.chassis_body_ids,:3]
        arrays["xpos"][k] = d.xpos[ix.chassis_body_ids]
        arrays["quat"][k] = d.xquat[ix.chassis_body_ids]
        arrays["active_torque_max"][k] = np.max(np.abs(d.actuator_force[ix.active_actuator_ids]))
        arrays["warnings"][k] = sum(w.number for w in d.warning)
        arrays["limits"][k] = np.count_nonzero(d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
        for j in range(d.ncon):
            c = d.contact[j]
            g1,g2 = int(c.geom1),int(c.geom2)
            other = g2 if g1 == ev.ground else (g1 if g2 == ev.ground else -1)
            if other < 0:
                continue
            mujoco.mj_contactForce(m,d,j,force)
            normal = max(0.,float(force[0]))
            arrays["total_contact_n"][k] += normal
            arrays["penetration"][k] = max(arrays["penetration"][k],-float(c.dist))
            if other in fg:
                arrays["foot_contact_n"][k,fg[other]] += normal
            elif int(m.geom_bodyid[other]) in cb:
                arrays["body_contact_n"][k,cb[int(m.geom_bodyid[other])]] += normal
            else:
                arrays["other_contact_n"][k] += normal
        if k in {int(round(s/dt)) for s in (1.5,2,2.5,3,3.5,4,4.5,5,5.5)}:
            mujoco.mj_fullM(m,full_mass,d.qM)
            eig = np.linalg.eigvalsh(full_mass)
            mass_checks.append(dict(t=k*dt,min_eigenvalue=float(eig[0]),max_eigenvalue=float(eig[-1])))
    np.savez_compressed(OUT / f"{case['id']}_trace.npz", **arrays)
    report = dict(case=case, model=str(opt.MODEL), model_sha256=opt.tip.sha256(opt.MODEL),
                  dt=dt, total_mass_g=float(m.body_mass.sum()*1000),
                  armature_nonzero_count=int(np.count_nonzero(m.dof_armature)),
                  warnings=int(arrays["warnings"].max()), mass_matrix_spot_checks=mass_checks,
                  body_dof_order=["yaw_active","pitch_zero_target_PD","roll_passive"],
                  first_joint_gains=dict(yaw_kp=float(ix.kp[ix.body_positions[1]]),
                                         pitch_kp=float(ix.kp[ix.pitch_positions[0]]),
                                         pitch_kv=float(ix.kv[ix.pitch_positions[0]]),
                                         roll_k=float(m.jnt_stiffness[ix.roll_joint_ids[0]]),
                                         roll_c=float(m.dof_damping[ix.roll_dof_addresses[0]])), windows={})
    for label, start, end in (("startup",1.,1.5),("steady",1.5,6.)):
        mask = (arrays["time"]>=start)&(arrays["time"]<end)
        t = arrays["time"][mask]
        qv = arrays["body_qvel"][mask]
        om = arrays["omega_world"][mask]
        qs = arrays["quat"][mask]
        lag = int(round(.01/dt))
        lag_frame = int(round(1/30/dt))
        delta = np.linalg.norm(om[lag:]-om[:-lag],axis=-1)
        frame_angle = np.degrees(2*np.arccos(np.clip(np.abs(np.sum(qs[lag_frame:]*qs[:-lag_frame],axis=-1)),0,1)))
        contact = arrays["body_contact_n"][mask]
        total = arrays["total_contact_n"][mask]
        foot = arrays["foot_contact_n"][mask]
        centerz = arrays["xpos"][mask,:,2].mean(axis=1)
        weight = m.body_mass.sum()*9.81
        events = []
        score = delta.max(axis=1).copy()
        for _ in range(8):
            h = int(np.argmax(score))
            seg = int(np.argmax(delta[h]))
            j = h+lag
            lo,hi = max(0,j-lag),min(len(t),j+lag+1)
            events.append(dict(t=float(t[j]), segment=seg, delta_omega_10ms=float(delta[h,seg]),
                               frame_angle_deg=float(frame_angle[max(0,min(j,len(frame_angle)-1)),seg]),
                               chassis_contact_segments=np.flatnonzero(contact[lo:hi].max(axis=0)>1e-7).tolist(),
                               ground_force_peak_bodyweights=float(total[lo:hi].max()/weight),
                               contacting_feet_before=int(np.count_nonzero(foot[lo]>1e-7)),
                               contacting_feet_after=int(np.count_nonzero(foot[hi-1]>1e-7))))
            score[max(0,h-int(.10/dt)):min(len(score),h+int(.10/dt))] = -1
        windows = dict(body_relative_speed_rad_s={axis:describe(qv[:,:,i]) for i,axis in enumerate(("yaw","pitch","roll"))},
                       body_relative_accel_rad_s2={axis:describe(arrays["body_qacc"][mask,:,i]) for i,axis in enumerate(("yaw","pitch","roll"))},
                       world_body_omega_rad_s=describe(np.linalg.norm(om,axis=-1)),
                       world_body_deltaomega_10ms_rad_s=describe(delta),
                       world_body_rotation_per_30fps_frame_deg=describe(frame_angle),
                       total_ground_force_bodyweights=describe(total/weight),
                       body_contact_fraction=float(np.mean(contact.max(axis=1)>1e-7)),
                       body_contact_by_segment_fraction={str(i):float(np.mean(contact[:,i]>1e-7)) for i in range(21) if np.any(contact[:,i]>1e-7)},
                       body_contact_peak_mN=float(contact.max()*1000),
                       chassis_height_ptp_mm=float(np.ptp(centerz)*1000),
                       target_speed_rad_s=describe(arrays["target_speed_max"][mask]),
                       joint_limit_fraction=float(np.mean(arrays["limits"][mask]>0)),
                       tip_speed_rad_s=describe(arrays["tip_qvel"][mask]),
                       tip_angle_deg=describe(np.degrees(arrays["tip_q"][mask])),
                       top_distinct_10ms_events=events)
        report["windows"][label]=windows
    path=OUT / f"{case['id']}_summary.json"
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(id=case["id"],dt=dt,warnings=report["warnings"],steady=report["windows"]["steady"])),flush=True)
    return report


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    chosen = [c for c in cases() if not sys.argv[1:] or c["id"] in sys.argv[1:]]
    for c in chosen:
        run(c)


if __name__ == "__main__":
    main()
