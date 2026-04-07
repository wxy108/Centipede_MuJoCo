#!/usr/bin/env python3
"""
optimize_compliance.py — Global optimization for maximum compliance
=====================================================================
Goal: Make the centipede as SOFT as possible while still:
  (a) following the commanded traveling wave
  (b) locomoting forward (not stuck)
  (c) remaining stable (no divergence / buckling)

Criterion (MINIMIZE):
  cost = w_torque * torque_rms           # penalize high effort (want soft)
       + w_track  * tracking_rms_deg     # penalize poor wave following
       - w_fwd    * forward_mm           # reward forward progress
       + penalty  (if stuck/buckled/diverged)

Parameters (4D continuous):
  x[0] = body_kp       ∈ [0.1, 3.0]     impedance spring gain
  x[1] = body_kv       ∈ [0.005, 0.2]   impedance damping gain
  x[2] = log10(body_roll_k) ∈ [-3, -1]  body roll stiffness (log scale)
  x[3] = log10(leg_roll_k)  ∈ [-2.5, -0.5]  leg roll stiffness (log scale)

Method:
  Stage 1 — Latin Hypercube sample (20 pts, 1s sims) → map landscape
  Stage 2 — Differential Evolution from LHS best (refined, 1.5s sims)
  Stage 3 — Verify top 3 candidates (3s flat+rough)
"""

import os, sys, re, math, time, json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, os.path.join(BASE, "controllers", "farms"))

import mujoco
# Suppress MuJoCo warnings during optimization (unstable configs expected)
mujoco.set_mju_user_warning(lambda msg: None)

from impedance_controller import ImpedanceTravelingWaveController, load_config
from kinematics import FARMSModelIndex, N_BODY_JOINTS, N_LEGS, N_LEG_DOF

XML_PATH = os.path.join(BASE, "models", "farms", "centipede.xml")
CFG_PATH = os.path.join(BASE, "configs", "farms_controller.yaml")
OUT_DIR  = os.path.join(BASE, "outputs", "optimization", "compliance")
os.makedirs(OUT_DIR, exist_ok=True)

TERRAIN_ROUGH = os.path.join(BASE, "terrain", "output",
                             "low0.0060_mid0.0030_high0.0020_s0", "1.png")
TERRAIN_FLAT  = os.path.join(BASE, "terrain", "output", "flat_terrain.png")

DAMP_RATIO = 0.4

# Cost weights
W_TORQUE = 200.0   # penalize torque heavily (we want soft)
W_TRACK  = 1.0     # penalize tracking error
W_FWD    = 1.0     # reward forward motion
STUCK_PENALTY = 500.0

# ── XML patching ──────────────────────────────────────────────────────

def patch_xml_roll(xml_str, body_rk, leg_rk):
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

def patch_xml_terrain(xml_str, terrain_png, z_max=0.04):
    m = re.search(r'<hfield\s+name="terrain"\s+file="([^"]*)"', xml_str)
    if m:
        xml_str = xml_str.replace(f'file="{m.group(1)}"', f'file="{terrain_png}"')
    def fix_size(m):
        parts = m.group(2).split()
        if len(parts) >= 3: parts[2] = f"{z_max:.6g}"
        return f'{m.group(1)}{" ".join(parts)}"'
    xml_str = re.sub(r'(<hfield[^>]*\bsize=")([^"]*)"', fix_size, xml_str)
    return xml_str

# ── Simulation ────────────────────────────────────────────────────────

eval_count = 0

def simulate(body_kp, body_kv, body_roll_k, leg_roll_k,
             xml_base, terrain_png, z_max, duration):
    """
    Returns dict with metrics, or None on crash.
    """
    global eval_count
    eval_count += 1

    xml = patch_xml_roll(xml_base, body_roll_k, leg_roll_k)
    xml = patch_xml_terrain(xml, terrain_png, z_max)

    with open(XML_PATH, 'w') as f:
        f.write(xml)
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        return None

    data = mujoco.MjData(model)
    try:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            ctrl = ImpedanceTravelingWaveController(
                model, CFG_PATH, body_kp=body_kp, body_kv=body_kv)
    except Exception:
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
    rec_dt = 0.01; last_rec = -np.inf
    T, FWD, TORQUES, TRK = [], [], [], []
    buckled = False

    for s in range(n_steps):
        try:
            ctrl.step(model, data)
        except Exception:
            return {"status": "diverged"}
        if data.time - last_rec >= rec_dt - 1e-10:
            last_rec = data.time; t = data.time; T.append(t)
            com = ctrl.idx.com_pos(data)
            if np.any(np.isnan(com)):
                return {"status": "diverged"}
            FWD.append(com[0])
            trqs = np.array([data.ctrl[ctrl.idx.body_act_ids[i]]
                             for i in range(N_BODY_JOINTS)])
            TORQUES.append(trqs)
            errs = []
            for i in range(N_BODY_JOINTS):
                phase = om * t - 2*math.pi*nw*sp*i / max(N_BODY_JOINTS-1, 1)
                target = amp * math.sin(phase)
                actual = ctrl.idx.body_joint_pos(data, i+1)
                errs.append((target - actual)**2)
            TRK.append(np.mean(errs))
        try:
            mujoco.mj_step(model, data)
        except Exception:
            return {"status": "diverged"}
        if s % 100 == 0:
            if np.any(np.isnan(data.qpos[:7])) or np.any(np.abs(data.qpos[:3]) > 5):
                return {"status": "diverged"}
            for jid in pitch_ids:
                if abs(data.qpos[model.jnt_qposadr[jid]]) > np.radians(55):
                    buckled = True

    T = np.array(T); FWD = np.array(FWD); TORQUES = np.array(TORQUES)
    TRK = np.array(TRK)
    m = T > 0.5
    if m.sum() < 5:
        return {"status": "too_short"}

    fwd_mm   = (FWD[m][-1] - FWD[m][0]) * 1000
    trq_rms  = float(np.sqrt(np.mean(TORQUES[m]**2)))
    trk_rms  = float(np.degrees(np.sqrt(np.mean(TRK[m]))))
    trq_max  = float(np.max(np.abs(TORQUES[m])))

    return {
        "status": "buckled" if buckled else "ok",
        "fwd_mm": round(fwd_mm, 2),
        "torque_rms": round(trq_rms, 6),
        "torque_max": round(trq_max, 6),
        "trk_rms_deg": round(trk_rms, 3),
    }


def cost_function(x, xml_base, terrain_png, z_max, duration):
    """
    x = [body_kp, body_kv, log10(body_roll_k), log10(leg_roll_k)]
    Returns scalar cost to MINIMIZE.
    """
    body_kp = x[0]
    body_kv = x[1]
    body_roll_k = 10**x[2]
    leg_roll_k  = 10**x[3]

    # Constraint: leg roll >= body roll (legs need more stiffness)
    if leg_roll_k < body_roll_k:
        leg_roll_k = body_roll_k

    res = simulate(body_kp, body_kv, body_roll_k, leg_roll_k,
                   xml_base, terrain_png, z_max, duration)

    if res is None or res.get('status') == 'diverged':
        return STUCK_PENALTY * 2

    if res.get('status') == 'buckled':
        return STUCK_PENALTY * 1.5

    if 'fwd_mm' not in res:
        return STUCK_PENALTY * 2

    fwd = res['fwd_mm']
    trq = res['torque_rms']
    trk = res['trk_rms_deg']

    # Stuck check: if forward < 2mm in evaluation period, penalize
    if fwd < 2.0:
        return STUCK_PENALTY

    # Main cost: want LOW torque, LOW tracking error, HIGH forward distance
    cost = (W_TORQUE * trq
            + W_TRACK * trk
            - W_FWD * fwd)

    tag = f"[{eval_count:3d}]"
    print(f"  {tag} kp={body_kp:.2f} kv={body_kv:.3f} "
          f"brk={body_roll_k:.1e} lrk={leg_roll_k:.1e} → "
          f"fwd={fwd:6.1f}mm trq={trq:.4f} trk={trk:4.1f}° cost={cost:7.1f}")

    return cost


# ═══════════════════════════════════════════════════════════════════════
#  STAGE 1: Latin Hypercube exploration
# ═══════════════════════════════════════════════════════════════════════

def stage1_lhs(xml_base, n_samples=20, duration=1.0):
    """Explore 4D parameter space with LHS."""
    from scipy.stats.qmc import LatinHypercube

    print("\n" + "="*72)
    print(f"  STAGE 1: Latin Hypercube exploration ({n_samples} samples, {duration}s sims)")
    print("="*72)

    bounds = np.array([
        [0.1, 3.0],      # body_kp
        [0.005, 0.2],    # body_kv
        [-3.0, -1.0],    # log10(body_roll_k)
        [-2.5, -0.5],    # log10(leg_roll_k)
    ])

    sampler = LatinHypercube(d=4, seed=42)
    samples = sampler.random(n=n_samples)

    # Scale to bounds
    X = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])

    results = []
    t0 = time.time()
    for i, x in enumerate(X):
        cost = cost_function(x, xml_base, TERRAIN_ROUGH, 0.04, duration)
        results.append({
            'body_kp': round(float(x[0]), 4),
            'body_kv': round(float(x[1]), 4),
            'body_roll_k': round(10**x[2], 6),
            'leg_roll_k': round(10**x[3], 6),
            'cost': round(cost, 2),
            'x': x.tolist(),
        })

    wall = time.time() - t0
    results.sort(key=lambda r: r['cost'])

    with open(os.path.join(OUT_DIR, "stage1_lhs.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Stage 1 done in {wall:.0f}s")
    print(f"  Top 5 candidates:")
    for r in results[:5]:
        print(f"    kp={r['body_kp']:.2f} kv={r['body_kv']:.3f} "
              f"brk={r['body_roll_k']:.1e} lrk={r['leg_roll_k']:.1e} "
              f"cost={r['cost']:.1f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  STAGE 2: Differential Evolution refinement
# ═══════════════════════════════════════════════════════════════════════

def stage2_de(xml_base, lhs_results, duration=1.5, maxiter=15, popsize=8):
    """
    Refine using scipy differential_evolution,
    seeded with best LHS points.
    """
    from scipy.optimize import differential_evolution

    print("\n" + "="*72)
    print(f"  STAGE 2: Differential Evolution (maxiter={maxiter}, pop={popsize}, "
          f"{duration}s sims)")
    print("="*72)

    bounds = [
        (0.1, 3.0),      # body_kp
        (0.005, 0.2),    # body_kv
        (-3.0, -1.0),    # log10(body_roll_k)
        (-2.5, -0.5),    # log10(leg_roll_k)
    ]

    # Build initial population from top LHS results
    init_pop = []
    for r in lhs_results[:min(popsize, len(lhs_results))]:
        init_pop.append(r['x'])
    # Pad with random if needed
    rng = np.random.default_rng(123)
    while len(init_pop) < popsize:
        x = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        init_pop.append(x.tolist())
    init_pop = np.array(init_pop)

    t0 = time.time()
    all_evals = []

    def objective(x):
        c = cost_function(x, xml_base, TERRAIN_ROUGH, 0.04, duration)
        all_evals.append({
            'body_kp': round(float(x[0]), 4),
            'body_kv': round(float(x[1]), 4),
            'body_roll_k': round(10**x[2], 6),
            'leg_roll_k': round(10**x[3], 6),
            'cost': round(c, 2),
            'x': x.tolist(),
        })
        return c

    result = differential_evolution(
        objective,
        bounds=bounds,
        init=init_pop,
        maxiter=maxiter,
        popsize=popsize,
        seed=42,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,  # skip local refinement to save time
    )

    wall = time.time() - t0
    best_x = result.x
    best_cost = result.fun

    winner = {
        'body_kp': round(float(best_x[0]), 4),
        'body_kv': round(float(best_x[1]), 4),
        'body_roll_k': round(10**best_x[2], 6),
        'leg_roll_k': round(max(10**best_x[3], 10**best_x[2]), 6),
        'cost': round(best_cost, 2),
        'x': best_x.tolist(),
        'de_nfev': result.nfev,
        'de_nit': result.nit,
        'de_success': result.success,
    }

    all_evals.sort(key=lambda r: r['cost'])
    stage2_out = {'winner': winner, 'all_evals': all_evals[:20]}

    with open(os.path.join(OUT_DIR, "stage2_de.json"), 'w') as f:
        json.dump(stage2_out, f, indent=2)

    print(f"\n  Stage 2 done in {wall:.0f}s ({result.nfev} evals, {result.nit} iters)")
    print(f"  WINNER: kp={winner['body_kp']:.2f} kv={winner['body_kv']:.3f} "
          f"brk={winner['body_roll_k']:.1e} lrk={winner['leg_roll_k']:.1e} "
          f"cost={winner['cost']:.1f}")

    # Return top 3 unique candidates
    seen = set()
    top3 = []
    for r in all_evals:
        key = (round(r['body_kp'], 1), round(r['body_kv'], 2),
               round(np.log10(r['body_roll_k'])), round(np.log10(r['leg_roll_k'])))
        if key not in seen:
            seen.add(key)
            top3.append(r)
        if len(top3) >= 3:
            break

    return top3


# ═══════════════════════════════════════════════════════════════════════
#  STAGE 3: Extended verification
# ═══════════════════════════════════════════════════════════════════════

def stage3_verify(xml_base, top_candidates, duration=3.0):
    """Extended 3s runs on flat + rough for top candidates."""
    print("\n" + "="*72)
    print(f"  STAGE 3: Extended verification ({duration}s, flat + rough)")
    print("="*72)

    results = []
    for cand in top_candidates:
        kp = cand['body_kp']; kv = cand['body_kv']
        brk = cand['body_roll_k']; lrk = cand['leg_roll_k']

        for tname, tpng, zmax in [
            ("flat",  TERRAIN_FLAT,  0.001),
            ("rough", TERRAIN_ROUGH, 0.04),
        ]:
            t0 = time.time()
            res = simulate(kp, kv, brk, lrk, xml_base, tpng, zmax, duration)
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
                  f"{tname:>6s} | fwd={entry.get('fwd_mm','?'):>7}mm "
                  f"trq={entry.get('torque_rms','?')} "
                  f"trk={entry.get('trk_rms_deg','?')}° "
                  f"status={entry.get('status','?')} | {wall:.0f}s")

    with open(os.path.join(OUT_DIR, "stage3_verify.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0,
                        help="Run only this stage (1/2/3), 0=all")
    parser.add_argument("--lhs-samples", type=int, default=20)
    parser.add_argument("--de-maxiter", type=int, default=15)
    parser.add_argument("--de-popsize", type=int, default=8)
    args = parser.parse_args()

    with open(XML_PATH, 'r') as f:
        base_xml = f.read()

    if args.stage == 0 or args.stage == 1:
        lhs = stage1_lhs(base_xml, n_samples=args.lhs_samples, duration=1.0)
    else:
        with open(os.path.join(OUT_DIR, "stage1_lhs.json")) as f:
            lhs = json.load(f)

    if args.stage == 0 or args.stage == 2:
        top3 = stage2_de(base_xml, lhs, duration=1.5,
                         maxiter=args.de_maxiter, popsize=args.de_popsize)
    else:
        with open(os.path.join(OUT_DIR, "stage2_de.json")) as f:
            d = json.load(f)
            top3 = d['all_evals'][:3]

    if args.stage == 0 or args.stage == 3:
        stage3_verify(base_xml, top3, duration=3.0)

    # Restore XML
    with open(XML_PATH, 'w') as f:
        f.write(base_xml)
    print("\n  XML restored. All results in:", OUT_DIR)
