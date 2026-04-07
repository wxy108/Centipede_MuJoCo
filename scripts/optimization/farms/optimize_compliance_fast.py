#!/usr/bin/env python3
"""
optimize_compliance_fast.py — Global optimization for max compliance (fast mode)
=================================================================================
Uses dt=0.004 + Euler integrator for ~5s/eval (vs ~90s at full resolution).
Winners are verified at full resolution (dt=0.0005, RK4) in Stage 3.

Criterion (MINIMIZE):
  cost = W_TORQUE * torque_rms       # want LOW torque = soft
       + W_TRACK  * tracking_rms_deg # want GOOD wave tracking
       - W_FWD    * forward_mm       # want FORWARD motion
       + penalties for stuck/diverged/buckled

Method:
  Stage 1 — Latin Hypercube (25 pts, 1.5s fast sims) → map landscape
  Stage 2 — Differential Evolution (popsize=10, ~15 gens, 1.5s fast sims)
  Stage 3 — Verify top 3 at full resolution (3s, dt=0.0005 RK4)
"""

import os, sys, re, math, time, json
import numpy as np

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", ".."))
sys.path.insert(0, os.path.join(BASE, "controllers", "farms"))

import mujoco
mujoco.set_mju_user_warning(lambda msg: None)

from impedance_controller import ImpedanceTravelingWaveController, load_config
from kinematics import N_BODY_JOINTS, N_LEGS, N_LEG_DOF
import io, contextlib

XML_PATH     = os.path.join(BASE, "models", "farms", "centipede.xml")
XML_FAST     = os.path.join(BASE, "models", "farms", "centipede_fast.xml")
CFG_PATH     = os.path.join(BASE, "configs", "farms_controller.yaml")
OUT_DIR      = os.path.join(BASE, "outputs", "optimization", "compliance")
os.makedirs(OUT_DIR, exist_ok=True)

TERRAIN_ROUGH = os.path.join(BASE, "terrain", "output",
                             "low0.0060_mid0.0030_high0.0020_s0", "1.png")
TERRAIN_FLAT  = os.path.join(BASE, "terrain", "output", "flat_terrain.png")

DAMP_RATIO = 0.4
W_TORQUE = 200.0
W_TRACK  = 2.0
W_FWD    = 1.0
STUCK_PENALTY = 500.0

eval_log = []
eval_count = 0

# ── XML helpers ───────────────────────────────────────────────────────

def make_fast_xml(xml_str):
    """For fast mode: keep original physics (dt=0.0005, RK4) — reliability > speed."""
    return xml_str  # no changes; we use shorter durations instead

def patch_roll(xml_str, body_rk, leg_rk):
    body_rd = body_rk * DAMP_RATIO
    leg_rd  = leg_rk * DAMP_RATIO
    def fix_body(m):
        s = m.group(0)
        s = re.sub(r'stiffness="[^"]*"', f'stiffness="{body_rk:.6e}"', s)
        s = re.sub(r'damping="[^"]*"',   f'damping="{body_rd:.6e}"', s)
        return s
    xml_str = re.sub(r'<joint\s+name="joint_roll_body_\d+"[^/]*/>', fix_body, xml_str)
    xml_str = re.sub(r'<joint\s+name="joint_roll_passive_\d+"[^/]*/>', fix_body, xml_str)
    def fix_leg(m):
        s = m.group(0)
        s = re.sub(r'stiffness="[^"]*"', f'stiffness="{leg_rk:.6e}"', s)
        s = re.sub(r'damping="[^"]*"',   f'damping="{leg_rd:.6e}"', s)
        return s
    xml_str = re.sub(r'<joint\s+name="joint_roll_leg_\d+_[LR]"[^/]*/>', fix_leg, xml_str)
    return xml_str

def patch_terrain(xml_str, terrain_png, z_max=0.04):
    m = re.search(r'<hfield\s+name="terrain"\s+file="([^"]*)"', xml_str)
    if m:
        xml_str = xml_str.replace(f'file="{m.group(1)}"', f'file="{terrain_png}"')
    def fix_sz(m):
        parts = m.group(2).split()
        if len(parts) >= 3: parts[2] = f"{z_max:.6g}"
        return f'{m.group(1)}{" ".join(parts)}"'
    xml_str = re.sub(r'(<hfield[^>]*\bsize=")([^"]*)"', fix_sz, xml_str)
    return xml_str

# ── Simulation core ───────────────────────────────────────────────────

def simulate(body_kp, body_kv, body_roll_k, leg_roll_k,
             xml_base, terrain_png, z_max, duration, fast=True):
    global eval_count
    eval_count += 1

    xml = patch_roll(xml_base, body_roll_k, leg_roll_k)
    xml = patch_terrain(xml, terrain_png, z_max)
    if fast:
        xml = make_fast_xml(xml)

    out_xml = XML_FAST if fast else XML_PATH
    with open(out_xml, 'w') as f:
        f.write(xml)
    try:
        model = mujoco.MjModel.from_xml_path(out_xml)
    except:
        return None

    data = mujoco.MjData(model)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ctrl = ImpedanceTravelingWaveController(
                model, CFG_PATH, body_kp=body_kp, body_kv=body_kv)
    except:
        return None

    cfg = load_config(CFG_PATH)
    bw = cfg['body_wave']
    amp = bw['amplitude']; om = 2*math.pi*bw['frequency']
    nw = bw['wave_number']; sp = bw['speed']

    pitch_ids = []
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm and ('joint_pitch_body' in nm):
            pitch_ids.append(i)

    n_steps = int(duration / model.opt.timestep)
    rec_dt = 0.02; last_rec = -np.inf
    T, FWD, TORQUES, TRK = [], [], [], []
    buckled = False

    for s in range(n_steps):
        try:
            ctrl.step(model, data)
            mujoco.mj_step(model, data)
        except:
            return {"status": "diverged"}

        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time; t = data.time
            com = ctrl.idx.com_pos(data)
            if np.any(np.isnan(com)):
                return {"status": "diverged"}
            T.append(t); FWD.append(com[0])
            trqs = np.array([data.ctrl[ctrl.idx.body_act_ids[i]]
                             for i in range(N_BODY_JOINTS)])
            TORQUES.append(trqs)
            mse = 0.0
            for i in range(N_BODY_JOINTS):
                phase = om*t - 2*math.pi*nw*sp*i/max(N_BODY_JOINTS-1,1)
                target = amp*math.sin(phase)
                actual = ctrl.idx.body_joint_pos(data, i+1)
                mse += (target - actual)**2
            TRK.append(mse / N_BODY_JOINTS)

        if s % 50 == 0:
            if np.any(np.isnan(data.qpos[:7])) or np.any(np.abs(data.qpos[:3])>5):
                return {"status": "diverged"}
            for jid in pitch_ids:
                if abs(data.qpos[model.jnt_qposadr[jid]]) > np.radians(55):
                    buckled = True

    T = np.array(T); FWD = np.array(FWD); TORQUES = np.array(TORQUES)
    TRK = np.array(TRK)
    m = T > 0.1
    if m.sum() < 3:
        return {"status": "too_short"}

    fwd_mm  = (FWD[m][-1] - FWD[m][0]) * 1000
    trq_rms = float(np.sqrt(np.mean(TORQUES[m]**2)))
    trk_rms = float(np.degrees(np.sqrt(np.mean(TRK[m]))))
    trq_max = float(np.max(np.abs(TORQUES[m])))

    # Also capture roll joint activity
    roll_ids = []
    for i in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm and 'joint_roll_' in nm:
            roll_ids.append(i)
    roll_max = max(abs(data.qpos[model.jnt_qposadr[j]]) for j in roll_ids) if roll_ids else 0
    roll_std = np.std([data.qpos[model.jnt_qposadr[j]] for j in roll_ids]) if roll_ids else 0

    return {
        "status": "buckled" if buckled else "ok",
        "fwd_mm": round(fwd_mm, 2),
        "torque_rms": round(trq_rms, 6),
        "torque_max": round(trq_max, 6),
        "trk_rms_deg": round(trk_rms, 3),
        "roll_max_deg": round(float(np.degrees(roll_max)), 2),
        "roll_std_deg": round(float(np.degrees(roll_std)), 2),
    }


def cost_fn(x, xml_base, terrain_png, z_max, duration, fast=True):
    body_kp     = x[0]
    body_kv     = x[1]
    body_roll_k = 10**x[2]
    leg_roll_k  = 10**x[3]
    if leg_roll_k < body_roll_k:
        leg_roll_k = body_roll_k

    res = simulate(body_kp, body_kv, body_roll_k, leg_roll_k,
                   xml_base, terrain_png, z_max, duration, fast=fast)

    tag = f"[{eval_count:3d}]"
    base_entry = {
        'body_kp': round(x[0], 4), 'body_kv': round(x[1], 4),
        'body_roll_k': round(body_roll_k, 6), 'leg_roll_k': round(leg_roll_k, 6),
    }

    if res is None or res.get('status') in ('diverged', 'too_short'):
        print(f"  {tag} kp={x[0]:.2f} kv={x[1]:.3f} brk={body_roll_k:.1e} lrk={leg_roll_k:.1e} | DIVERGED")
        base_entry.update({'cost': STUCK_PENALTY*2, 'status': 'diverged'})
        eval_log.append(base_entry)
        return STUCK_PENALTY * 2
    if res.get('status') == 'buckled':
        print(f"  {tag} kp={x[0]:.2f} kv={x[1]:.3f} brk={body_roll_k:.1e} lrk={leg_roll_k:.1e} | BUCKLED")
        base_entry.update({'cost': STUCK_PENALTY*1.5, 'status': 'buckled', **res})
        eval_log.append(base_entry)
        return STUCK_PENALTY * 1.5
    if 'fwd_mm' not in res:
        base_entry.update({'cost': STUCK_PENALTY*2, 'status': 'error'})
        eval_log.append(base_entry)
        return STUCK_PENALTY * 2

    fwd = res['fwd_mm']
    trq = res['torque_rms']
    trk = res['trk_rms_deg']

    if fwd < 1.0:
        cost = STUCK_PENALTY
        print(f"  {tag} kp={x[0]:.2f} kv={x[1]:.3f} brk={body_roll_k:.1e} lrk={leg_roll_k:.1e} | STUCK fwd={fwd:.1f}mm")
        base_entry.update({'cost': cost, **res})
        eval_log.append(base_entry)
        return cost

    cost = W_TORQUE * trq + W_TRACK * trk - W_FWD * fwd
    base_entry.update({'cost': round(cost, 2), **res})
    eval_log.append(base_entry)

    print(f"  {tag} kp={x[0]:.2f} kv={x[1]:.3f} "
          f"brk={body_roll_k:.1e} lrk={leg_roll_k:.1e} | "
          f"fwd={fwd:6.1f}mm trq={trq:.4f} trk={trk:4.1f}° "
          f"roll={res.get('roll_max_deg',0):.1f}° cost={cost:7.1f}")

    return cost


# ═══════════════════════════════════════════════════════════════════════
#  STAGES
# ═══════════════════════════════════════════════════════════════════════

BOUNDS = [
    (0.1, 3.0),       # body_kp
    (0.005, 0.2),     # body_kv
    (-2.5, -0.7),     # log10(body_roll_k): 3e-3 .. 0.2
    (-2.0, -0.3),     # log10(leg_roll_k):  1e-2 .. 0.5
]


def stage1_lhs(xml_base, n_samples=25, duration=0.5):
    from scipy.stats.qmc import LatinHypercube
    global eval_log
    eval_log = []

    print("\n" + "="*72)
    print(f"  STAGE 1: Latin Hypercube ({n_samples} pts, {duration}s fast sims)")
    print("="*72)

    bnds = np.array(BOUNDS)
    sampler = LatinHypercube(d=4, seed=42)
    samples = sampler.random(n=n_samples)
    X = bnds[:, 0] + samples * (bnds[:, 1] - bnds[:, 0])

    costs = []
    t0 = time.time()
    for x in X:
        c = cost_fn(x, xml_base, TERRAIN_ROUGH, 0.04, duration, fast=True)
        costs.append(c)

    wall = time.time() - t0
    eval_log.sort(key=lambda r: r['cost'])

    with open(os.path.join(OUT_DIR, "stage1_lhs.json"), 'w') as f:
        json.dump(eval_log, f, indent=2)

    print(f"\n  Stage 1 done: {len(eval_log)} valid evals in {wall:.0f}s")
    ok_evals = [r for r in eval_log if r.get('status') == 'ok']
    print(f"  Top 5 (of {len(ok_evals)} stable):")
    for r in ok_evals[:5]:
        print(f"    kp={r['body_kp']:.2f} kv={r['body_kv']:.3f} "
              f"brk={r['body_roll_k']:.1e} lrk={r['leg_roll_k']:.1e} "
              f"fwd={r.get('fwd_mm',0):.1f}mm trq={r.get('torque_rms',0):.4f} "
              f"trk={r.get('trk_rms_deg',0):.1f}° cost={r['cost']:.1f}")

    return eval_log


def stage2_de(xml_base, lhs_results, duration=0.5, maxiter=12, popsize=10):
    from scipy.optimize import differential_evolution
    global eval_log
    eval_log = []

    print("\n" + "="*72)
    print(f"  STAGE 2: Differential Evolution (maxiter={maxiter}, pop={popsize})")
    print("="*72)

    # Seed with top LHS results
    init_pop = []
    for r in lhs_results[:min(popsize, len(lhs_results))]:
        x = [r['body_kp'], r['body_kv'],
             np.log10(r['body_roll_k']), np.log10(r['leg_roll_k'])]
        init_pop.append(x)
    rng = np.random.default_rng(123)
    bnds = np.array(BOUNDS)
    while len(init_pop) < popsize:
        x = [rng.uniform(b[0], b[1]) for b in bnds]
        init_pop.append(x)
    init_pop = np.array(init_pop[:popsize])

    t0 = time.time()
    result = differential_evolution(
        lambda x: cost_fn(x, xml_base, TERRAIN_ROUGH, 0.04, duration, fast=True),
        bounds=BOUNDS,
        init=init_pop,
        maxiter=maxiter,
        popsize=popsize,
        seed=42,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
    )
    wall = time.time() - t0

    best = result.x
    winner = {
        'body_kp': round(float(best[0]), 4),
        'body_kv': round(float(best[1]), 4),
        'body_roll_k': round(10**best[2], 6),
        'leg_roll_k': round(max(10**best[3], 10**best[2]), 6),
        'cost': round(result.fun, 2),
        'nfev': result.nfev, 'nit': result.nit,
    }

    eval_log.sort(key=lambda r: r['cost'])
    out = {'winner': winner, 'all_evals': eval_log}
    with open(os.path.join(OUT_DIR, "stage2_de.json"), 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n  Stage 2 done: {result.nfev} evals, {result.nit} iters in {wall:.0f}s")
    print(f"  WINNER: kp={winner['body_kp']:.2f} kv={winner['body_kv']:.3f} "
          f"brk={winner['body_roll_k']:.1e} lrk={winner['leg_roll_k']:.1e} "
          f"cost={winner['cost']:.1f}")

    # Get top 3 unique
    seen = set()
    top3 = []
    for r in eval_log:
        key = f"{r['body_kp']:.1f}_{r['body_kv']:.2f}_{r['body_roll_k']:.1e}_{r['leg_roll_k']:.1e}"
        if key not in seen:
            seen.add(key); top3.append(r)
        if len(top3) >= 3: break

    return top3


def stage3_verify(xml_base, top3, duration=3.0):
    """Verify at FULL resolution (dt=0.0005, RK4)."""
    global eval_log
    eval_log = []

    print("\n" + "="*72)
    print(f"  STAGE 3: Full-resolution verification ({duration}s, dt=0.0005 RK4)")
    print("="*72)

    results = []
    for cand in top3:
        kp = cand['body_kp']; kv = cand['body_kv']
        brk = cand['body_roll_k']; lrk = cand['leg_roll_k']

        for tname, tpng, zmax in [
            ("flat",  TERRAIN_FLAT,  0.001),
            ("rough", TERRAIN_ROUGH, 0.04),
        ]:
            t0 = time.time()
            res = simulate(kp, kv, brk, lrk, xml_base, tpng, zmax,
                           duration, fast=False)
            wall = time.time() - t0
            entry = {
                'body_kp': kp, 'body_kv': kv,
                'body_roll_k': brk, 'leg_roll_k': lrk,
                'terrain': tname,
            }
            if res:
                entry.update(res)
            else:
                entry['status'] = 'crash'
            entry['wall'] = round(wall, 1)
            results.append(entry)
            print(f"  kp={kp:.2f} kv={kv:.3f} brk={brk:.1e} lrk={lrk:.1e} "
                  f"{tname:>6s} | fwd={entry.get('fwd_mm','N/A'):>7}mm "
                  f"trq={entry.get('torque_rms','N/A')} "
                  f"trk={entry.get('trk_rms_deg','N/A')}° "
                  f"roll={entry.get('roll_max_deg','N/A')}° "
                  f"{entry.get('status','?')} | {wall:.0f}s")

    with open(os.path.join(OUT_DIR, "stage3_verify.json"), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0, help="1/2/3 or 0=all")
    parser.add_argument("--lhs", type=int, default=25)
    parser.add_argument("--de-iter", type=int, default=12)
    parser.add_argument("--de-pop", type=int, default=10)
    args = parser.parse_args()

    with open(XML_PATH, 'r') as f:
        base_xml = f.read()

    if args.stage in (0, 1):
        lhs = stage1_lhs(base_xml, n_samples=args.lhs, duration=1.5)
        if args.stage == 1:
            with open(XML_PATH, 'w') as f: f.write(base_xml)
            print(f"\n  XML restored. Stage 1 results in: {OUT_DIR}")
            sys.exit(0)
    else:
        with open(os.path.join(OUT_DIR, "stage1_lhs.json")) as f:
            lhs = json.load(f)

    if args.stage in (0, 2):
        top3 = stage2_de(base_xml, lhs, maxiter=args.de_iter, popsize=args.de_pop)
        if args.stage == 2:
            with open(XML_PATH, 'w') as f: f.write(base_xml)
            print(f"\n  XML restored. Stage 2 results in: {OUT_DIR}")
            sys.exit(0)
    else:
        with open(os.path.join(OUT_DIR, "stage2_de.json")) as f:
            top3 = json.load(f)['all_evals'][:3]

    if args.stage in (0, 3):
        stage3_verify(base_xml, top3, duration=3.0)

    # Restore XML
    with open(XML_PATH, 'w') as f:
        f.write(base_xml)
    print(f"\n  XML restored. All results in: {OUT_DIR}")
