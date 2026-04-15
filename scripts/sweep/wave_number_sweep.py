#!/usr/bin/env python3
"""
Wave-number × wavelength sweep.

For every combination (wave_number k, terrain wavelength λ), runs N trials
with random initial yaw. Same terrain-gen / sensor / phase-lag pipeline as
wavelength_sweep.py, just with the body-wave wave-number as an extra axis.

Per-trial outputs
-----------------
  - Video (MP4, tracking camera) under `videos/k{k}/wl_{λ}mm/trial_{t:02d}.mp4`
  - Sensor time-series CSV (time, root xyz, heading, mean body-yaw q,
    mean body-pitch q, mean body-roll q, terrain slope at COM) under
    `sensors/k{k}/wl_{λ}mm/trial_{t:02d}.csv`

Per-run outputs
---------------
  - results.json                full sweep + per-trial metrics + config
  - results_aggregated.csv      one row per (k, λ) cell
  - results_all_trials.csv      one row per trial

Usage
-----
  python scripts/sweep/wave_number_sweep.py \\
      --wave-numbers 1.5,2,2.5,3,3.5 \\
      --wavelengths 350,130,90,65,36,30,24,18,16,15,14,13,12,10,8,7,6,5,4,3 \\
      --n-trials 20 --duration 5 --amplitude 0.01 --video

Notes
-----
  * `--wavelengths` is the default — matches the list the user requested
    (350 dedup'd to a single entry).
  * Per-wave-number body/leg yaw amplitudes can be supplied via
    `--body-amps "k1:a1,k2:a2,..."` and `--leg-amps "k1:(dof0|dof1),k2:..."`.
    If omitted the amplitudes in configs/farms_controller.yaml are used for
    every k.  Use flat_ground_tune.py first to pick those values.
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "terrain", "generator"))

from generate import resolve_morphology  # noqa: E402

# Reuse the battle-tested pieces from wavelength_sweep unchanged.
from wavelength_sweep import (  # noqa: E402
    generate_single_wavelength_terrain,
    save_wavelength_terrain,
    patch_xml_terrain,
    set_initial_yaw,
    get_body_heading,
    _try_make_renderer,
    _make_tracking_camera,
    _save_video,
    TerrainSampler,
    compute_phase_lag,
    aggregate_trials,
    VID_W, VID_H, VID_FPS,
    CAM_DISTANCE, CAM_AZIMUTH, CAM_ELEVATION,
    MAX_PITCH_DEG, MAX_ROLL_DEG,
    XML_PATH, CONFIG_PATH, TERRAIN_CFG,
)

import mujoco  # noqa: E402
from impedance_controller import ImpedanceTravelingWaveController  # noqa: E402
from kinematics import FARMSModelIndex  # noqa: E402

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "wave_number_sweep")


# ═══════════════════════════════════════════════════════════════════════════════
# Config override helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_base_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_config_override(base_cfg, wave_number, body_amp=None, leg_amps=None):
    """Return a deep-copied config dict with wave_number and (optional)
    body/leg amplitudes patched in."""
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("body_wave", {})["wave_number"] = float(wave_number)
    if body_amp is not None:
        cfg["body_wave"]["amplitude"] = float(body_amp)
    if leg_amps is not None:
        cfg.setdefault("leg_wave", {})["amplitudes"] = [float(a) for a in leg_amps]
    return cfg


def write_tmp_config(cfg, tag):
    """Write a cfg dict to a temp YAML alongside the base config."""
    tmp_path = os.path.join(os.path.dirname(CONFIG_PATH),
                            f".farms_controller_{tag}.tmp.yaml")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp_path


# ═══════════════════════════════════════════════════════════════════════════════
# Amplitude-table parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_body_amp_table(s, wave_numbers):
    """'1.5:0.6,2:0.55' → {1.5: 0.6, 2: 0.55}.  Missing k → None (use config default)."""
    if not s:
        return {k: None for k in wave_numbers}
    table = {}
    for chunk in s.split(","):
        k_str, a_str = chunk.split(":")
        table[float(k_str)] = float(a_str)
    out = {}
    for k in wave_numbers:
        out[k] = table.get(k, None)
    return out


def parse_leg_amp_table(s, wave_numbers):
    """'1.5:0.6|0.3|0|0,2:0.5|0.25|0|0' → {1.5:[..], 2:[..]}."""
    if not s:
        return {k: None for k in wave_numbers}
    table = {}
    for chunk in s.split(","):
        k_str, vec_str = chunk.split(":")
        table[float(k_str)] = [float(x) for x in vec_str.split("|")]
    out = {}
    for k in wave_numbers:
        out[k] = table.get(k, None)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation runner — like wavelength_sweep.run_simulation but dumps sensor CSV
# ═══════════════════════════════════════════════════════════════════════════════

SENSOR_DECIMATE = 20  # save one row every 20 physics steps (~100 Hz @ dt=0.5ms)


def run_simulation(xml_path, config_path, duration,
                   yaw_rad=0.0, video_path=None, sensor_csv_path=None):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    set_initial_yaw(model, data, yaw_rad)

    ctrl = ImpedanceTravelingWaveController(model, config_path)
    idx  = FARMSModelIndex(model)
    terrain = TerrainSampler(model)

    body_yaw_qposadr = []
    pitch_qposadr    = []
    roll_qposadr     = []
    for j in range(model.njnt):
        nm = model.joint(j).name or ""
        if nm.startswith("joint_body_"):
            body_yaw_qposadr.append(model.jnt_qposadr[j])
        if "joint_pitch_body" in nm:
            pitch_qposadr.append(model.jnt_qposadr[j])
        if "joint_roll_body" in nm:
            roll_qposadr.append(model.jnt_qposadr[j])

    n_actuators = model.nu
    total_mass = sum(model.body_mass[i] for i in range(model.nbody))
    gravity = abs(model.opt.gravity[2])
    n_steps = int(duration / model.opt.timestep)
    dt = model.opt.timestep
    SETTLE_STEPS = 200

    root_body = None
    for b in range(model.nbody):
        if model.body(b).name and "root" in model.body(b).name.lower():
            root_body = b; break
    if root_body is None:
        for b in range(model.nbody):
            jnt_start = model.body_jntadr[b]
            if jnt_start >= 0 and model.jnt_type[jnt_start] == mujoco.mjtJoint.mjJNT_FREE:
                root_body = b; break

    # Video
    renderer, frames, vid_cam = None, [], None
    vid_dt = 1.0 / VID_FPS
    last_frame_t = -1.0
    cam_azimuth_fixed = CAM_AZIMUTH + math.degrees(yaw_rad)
    if video_path:
        renderer, ok = _try_make_renderer(model)
        if ok:
            vid_cam = _make_tracking_camera(idx, data)
            vid_cam.azimuth = cam_azimuth_fixed
        else:
            video_path = None

    # Sensor CSV
    sensor_f = None
    if sensor_csv_path:
        os.makedirs(os.path.dirname(sensor_csv_path), exist_ok=True)
        sensor_f = open(sensor_csv_path, "w", encoding="utf-8")
        sensor_f.write("time,x,y,z,heading_rad,"
                       "body_yaw_mean_deg,body_yaw_max_deg,"
                       "pitch_mean_deg,pitch_max_deg,"
                       "roll_mean_deg,roll_max_deg,"
                       "terrain_slope\n")

    energy_sum = 0.0
    pitch_angles, roll_angles = [], []
    start_pos = None
    buckled, buckle_reason = False, ""

    PHASE_SAMPLE_INTERVAL = 50
    terrain_slope_ts, body_pitch_ts = [], []
    phase_sample_dt = PHASE_SAMPLE_INTERVAL * dt

    try:
        for step_i in range(n_steps):
            ctrl.step(model, data)
            mujoco.mj_step(model, data)

            if step_i == SETTLE_STEPS and root_body is not None:
                start_pos = data.xpos[root_body].copy()

            if step_i > SETTLE_STEPS:
                for a in range(n_actuators):
                    tau = abs(data.actuator_force[a])
                    jnt_id = model.actuator_trnid[a, 0]
                    if 0 <= jnt_id < model.njnt:
                        dof_adr = model.jnt_dofadr[jnt_id]
                        omega = abs(data.qvel[dof_adr])
                        energy_sum += tau * omega * dt

            if step_i > SETTLE_STEPS and step_i % PHASE_SAMPLE_INTERVAL == 0:
                com = idx.com_pos(data)
                heading = get_body_heading(model, data)
                slope = terrain.get_slope_along(com[0], com[1], heading)
                terrain_slope_ts.append(slope)
                mean_pitch = (np.mean([data.qpos[a] for a in pitch_qposadr])
                              if pitch_qposadr else 0.0)
                body_pitch_ts.append(mean_pitch)

            # Sensor dump (rare — keep file small)
            if sensor_f and step_i % SENSOR_DECIMATE == 0:
                com = data.xpos[root_body] if root_body is not None else np.zeros(3)
                heading = get_body_heading(model, data)
                yaws = np.array([abs(math.degrees(data.qpos[a]))
                                 for a in body_yaw_qposadr]) if body_yaw_qposadr else np.array([0.0])
                pit  = np.array([abs(math.degrees(data.qpos[a]))
                                 for a in pitch_qposadr])    if pitch_qposadr    else np.array([0.0])
                rol  = np.array([abs(math.degrees(data.qpos[a]))
                                 for a in roll_qposadr])     if roll_qposadr     else np.array([0.0])
                try:
                    slope = terrain.get_slope_along(com[0], com[1], heading)
                except Exception:
                    slope = float("nan")
                sensor_f.write(
                    f"{data.time:.4f},{com[0]:.5f},{com[1]:.5f},{com[2]:.5f},"
                    f"{heading:.4f},"
                    f"{yaws.mean():.3f},{yaws.max():.3f},"
                    f"{pit.mean():.3f},{pit.max():.3f},"
                    f"{rol.mean():.3f},{rol.max():.3f},"
                    f"{slope:.6f}\n"
                )

            if renderer and video_path and data.time - last_frame_t >= vid_dt - 1e-6:
                vid_cam.lookat[:] = idx.com_pos(data)
                renderer.update_scene(data, camera=vid_cam)
                frames.append(renderer.render().copy())
                last_frame_t = data.time

            if step_i % 200 == 0 and step_i > 0:
                for a in pitch_qposadr:
                    q_deg = abs(math.degrees(data.qpos[a]))
                    if q_deg > MAX_PITCH_DEG:
                        buckled = True
                        buckle_reason = f"pitch({q_deg:.1f} t={data.time:.1f}s)"
                        break
                if not buckled:
                    for a in roll_qposadr:
                        q_deg = abs(math.degrees(data.qpos[a]))
                        if q_deg > MAX_ROLL_DEG:
                            buckled = True
                            buckle_reason = f"roll({q_deg:.1f} t={data.time:.1f}s)"
                            break
                if buckled:
                    break

            if step_i % 100 == 0:
                for a in pitch_qposadr:
                    pitch_angles.append(abs(math.degrees(data.qpos[a])))
                for a in roll_qposadr:
                    roll_angles.append(abs(math.degrees(data.qpos[a])))
    finally:
        if sensor_f:
            sensor_f.close()

    if video_path and frames:
        _save_video(frames, video_path)

    end_pos = data.xpos[root_body].copy() if root_body is not None else None
    if start_pos is not None and end_pos is not None:
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
    else:
        distance = 0.0

    settle_time = SETTLE_STEPS * dt
    effective_time = max(data.time - settle_time, 0.01)
    forward_speed = distance / effective_time

    cot = (energy_sum / (total_mass * gravity * distance)
           if distance > 0.001 else float("inf"))
    phase_info = compute_phase_lag(
        np.array(terrain_slope_ts), np.array(body_pitch_ts), phase_sample_dt)

    return {
        "survived":        not buckled,
        "buckle_reason":   buckle_reason,
        "yaw_deg":         float(math.degrees(yaw_rad)),
        "sim_time":        float(data.time),
        "distance_m":      float(distance),
        "forward_speed":   float(forward_speed),
        "cot":             float(min(cot, 1e6)),
        "energy_J":        float(energy_sum),
        "max_pitch_deg":   float(max(pitch_angles)) if pitch_angles else 0.0,
        "mean_pitch_deg":  float(np.mean(pitch_angles)) if pitch_angles else 0.0,
        "max_roll_deg":    float(max(roll_angles)) if roll_angles else 0.0,
        "mean_roll_deg":   float(np.mean(roll_angles)) if roll_angles else 0.0,
        "total_mass_kg":   float(total_mass),
        "phase_lag_deg":   float(phase_info["phase_lag_deg"]),
        "phase_coherence": float(phase_info["coherence"]),
        "phase_freq_hz":   float(phase_info["dominant_freq_hz"]),
        "video_path":      video_path or "",
        "sensor_csv":      sensor_csv_path or "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_WAVELENGTHS_MM = [350, 130, 90, 65, 36, 30, 24, 18, 16, 15,
                          14, 13, 12, 10, 8, 7, 6, 5, 4, 3]
DEFAULT_WAVE_NUMBERS   = [1.5, 2.0, 2.5, 3.0, 3.5]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wave-numbers", type=str,
                   default=",".join(str(x) for x in DEFAULT_WAVE_NUMBERS),
                   help="Comma-separated list of body-wave wave_numbers")
    p.add_argument("--wavelengths", type=str,
                   default=",".join(str(x) for x in DEFAULT_WAVELENGTHS_MM),
                   help="Comma-separated terrain wavelengths in mm")
    p.add_argument("--n-trials",   type=int,   default=20)
    p.add_argument("--duration",   type=float, default=5.0)
    p.add_argument("--amplitude",  type=float, default=0.010,
                   help="Terrain peak amplitude in metres (current default: 10 mm)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--video",      action="store_true")
    p.add_argument("--no-sensors", action="store_true",
                   help="Disable per-trial sensor CSV dumping")
    p.add_argument("--body-amps",  type=str, default=None,
                   help="Per-k body yaw amplitudes, e.g. '1.5:0.6,2:0.55,...'")
    p.add_argument("--leg-amps",   type=str, default=None,
                   help="Per-k leg amplitudes (4 DOFs, | separated), "
                        "e.g. '1.5:0.6|0.3|0|0,2:0.5|0.25|0|0'")
    args = p.parse_args()

    wave_numbers = [float(x) for x in args.wave_numbers.split(",")]
    # Preserve user order but dedupe (350 is in the request twice)
    seen = set(); wls_mm = []
    for x in args.wavelengths.split(","):
        v = float(x)
        if v not in seen:
            seen.add(v); wls_mm.append(v)
    wavelengths = np.array(sorted(wls_mm, reverse=True)) / 1000.0   # → metres

    body_amp_table = parse_body_amp_table(args.body_amps, wave_numbers)
    leg_amp_table  = parse_leg_amp_table(args.leg_amps,  wave_numbers)

    with open(TERRAIN_CFG, encoding="utf-8") as f:
        t_cfg = yaml.safe_load(f)
    lengths = resolve_morphology(t_cfg)
    img_size = int(t_cfg["world"]["image_size"])
    world_half = float(t_cfg["world"]["size"])

    base_cfg = _load_base_config()

    rng = np.random.default_rng(args.seed + 9999)
    all_yaws = rng.uniform(0, 2 * math.pi,
                           size=(len(wave_numbers), len(wavelengths), args.n_trials))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"sweep_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    can_video = False
    if args.video:
        try:
            import mediapy  # noqa
            can_video = True
        except ImportError:
            print("  WARNING: mediapy not installed -- skipping video.")

    total_sims = len(wave_numbers) * len(wavelengths) * args.n_trials
    print("=" * 70)
    print("Wave-number × Wavelength Sweep")
    print("=" * 70)
    print(f"  wave_numbers  : {wave_numbers}")
    print(f"  wavelengths   : {[f'{w*1000:.0f}' for w in wavelengths]} mm")
    print(f"  trials per cell: {args.n_trials}  (random yaw 0-360°)")
    print(f"  duration      : {args.duration}s  amp: {args.amplitude*1000:.1f}mm")
    print(f"  total sims    : {total_sims}")
    print(f"  video         : {'ON' if can_video else 'OFF'}")
    print(f"  sensors       : {'ON' if not args.no_sensors else 'OFF'}")
    print(f"  output        : {run_dir}")
    print()

    cell_results = []
    all_trial_results = []
    t0 = time.time()
    sim_count = 0

    for ki, k in enumerate(wave_numbers):
        # Build a dedicated config for this wave-number once; reuse across all
        # wavelengths and trials.
        ovr_cfg = make_config_override(
            base_cfg, k,
            body_amp=body_amp_table[k],
            leg_amps=leg_amp_table[k],
        )
        cfg_tag = f"k{str(k).replace('.', 'p')}"
        cfg_tmp_path = write_tmp_config(ovr_cfg, cfg_tag)

        for wi, wl in enumerate(wavelengths):
            print(f"[k={k}  λ={wl*1000:6.1f}mm]")

            h_m, rms_m, peak_m = generate_single_wavelength_terrain(
                wavelength_m=wl, amplitude_m=args.amplitude,
                seed=args.seed + wi, image_size=img_size, world_half=world_half)
            png_path = save_wavelength_terrain(h_m, wl, args.seed + wi, run_dir)
            z_max = max(2.0 * peak_m, 0.005)
            tmp_xml = patch_xml_terrain(XML_PATH, png_path, z_max)

            trial_results = []
            for t in range(args.n_trials):
                sim_count += 1
                yaw = float(all_yaws[ki, wi, t])
                yaw_deg = math.degrees(yaw)

                vid_path = None
                if can_video:
                    vid_dir = os.path.join(run_dir, "videos",
                                           f"k{k}", f"wl_{wl*1000:.0f}mm")
                    os.makedirs(vid_dir, exist_ok=True)
                    vid_path = os.path.join(vid_dir, f"trial_{t:02d}_yaw{yaw_deg:.0f}.mp4")

                sensor_path = None
                if not args.no_sensors:
                    s_dir = os.path.join(run_dir, "sensors",
                                         f"k{k}", f"wl_{wl*1000:.0f}mm")
                    sensor_path = os.path.join(s_dir, f"trial_{t:02d}.csv")

                try:
                    metrics = run_simulation(
                        tmp_xml, cfg_tmp_path, args.duration,
                        yaw_rad=yaw, video_path=vid_path,
                        sensor_csv_path=sensor_path)
                except Exception as e:
                    metrics = {
                        "survived": False, "buckle_reason": str(e),
                        "yaw_deg": yaw_deg, "cot": 1e6, "forward_speed": 0,
                        "distance_m": 0, "max_pitch_deg": 0, "mean_pitch_deg": 0,
                        "max_roll_deg": 0, "mean_roll_deg": 0, "energy_J": 0,
                        "sim_time": 0, "total_mass_kg": 0,
                        "phase_lag_deg": float("nan"), "phase_coherence": 0,
                        "phase_freq_hz": 0, "video_path": "", "sensor_csv": "",
                    }
                metrics["wave_number"]  = float(k)
                metrics["wavelength_m"] = float(wl)
                metrics["wavelength_mm"] = float(wl * 1000)
                metrics["trial_idx"]   = t
                trial_results.append(metrics)
                all_trial_results.append(metrics)

                status = ("OK" if metrics["survived"]
                          else f"FAIL:{metrics['buckle_reason']}")
                eta = (time.time() - t0) / sim_count * (total_sims - sim_count)
                print(f"  [{sim_count:4d}/{total_sims}] "
                      f"yaw={yaw_deg:5.1f}  "
                      f"CoT={metrics['cot']:7.1f}  "
                      f"speed={metrics['forward_speed']*1000:5.1f}mm/s  "
                      f"{status}  (ETA {eta/60:.0f}min)", flush=True)

            if os.path.exists(tmp_xml):
                os.remove(tmp_xml)

            agg = aggregate_trials(trial_results)
            agg.update({
                "wave_number":  float(k),
                "wavelength_m": float(wl),
                "wavelength_mm": float(wl * 1000),
                "frequency":    float(1.0 / wl),
                "amplitude_m":  float(args.amplitude),
                "rms_m":        float(rms_m),
            })
            cell_results.append(agg)

            print(f"  => CoT {agg['cot_mean']:.1f}±{agg['cot_std']:.1f}  "
                  f"speed {agg['speed_mean']*1000:.1f}±{agg['speed_std']*1000:.1f}mm/s  "
                  f"survived {agg['n_survived']}/{agg['n_trials']}")
            print()

        # Leave the per-k tmp config in place until the run finishes; some OSes
        # hold file handles longer than expected. Clean up at the end.
    # Remove temp configs
    for k in wave_numbers:
        cfg_tag = f"k{str(k).replace('.', 'p')}"
        cfg_tmp_path = os.path.join(os.path.dirname(CONFIG_PATH),
                                    f".farms_controller_{cfg_tag}.tmp.yaml")
        if os.path.exists(cfg_tmp_path):
            os.remove(cfg_tmp_path)

    elapsed = time.time() - t0

    # ── Save results ──────────────────────────────────────────────────────
    out_json = os.path.join(run_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump({
            "timestamp":     timestamp,
            "wave_numbers":  wave_numbers,
            "wavelengths_mm": [float(w*1000) for w in wavelengths],
            "n_trials":      args.n_trials,
            "duration":      args.duration,
            "amplitude":     args.amplitude,
            "body_amp_table": body_amp_table,
            "leg_amp_table":  leg_amp_table,
            "morphology":    {k: float(v) for k, v in lengths.items()},
            "elapsed_s":     elapsed,
            "cell_results":  cell_results,
            "all_trials":    all_trial_results,
        }, f, indent=2)

    out_agg = os.path.join(run_dir, "results_aggregated.csv")
    with open(out_agg, "w") as f:
        hdr = ["wave_number", "wavelength_mm", "frequency",
               "n_survived", "survival_rate",
               "cot_mean", "cot_std", "cot_median", "cot_min", "cot_max",
               "speed_mean", "speed_std",
               "max_pitch_mean", "max_pitch_std",
               "max_roll_mean", "max_roll_std",
               "phase_lag_mean", "phase_lag_std"]
        f.write(",".join(hdr) + "\n")
        for r in cell_results:
            f.write(",".join(str(r.get(h, "")) for h in hdr) + "\n")

    out_raw = os.path.join(run_dir, "results_all_trials.csv")
    with open(out_raw, "w") as f:
        hdr = ["wave_number", "wavelength_mm", "trial_idx", "yaw_deg",
               "survived", "cot", "forward_speed", "distance_m",
               "max_pitch_deg", "mean_pitch_deg",
               "max_roll_deg", "mean_roll_deg",
               "energy_J", "phase_lag_deg", "phase_coherence",
               "video_path", "sensor_csv"]
        f.write(",".join(hdr) + "\n")
        for r in all_trial_results:
            f.write(",".join(str(r.get(h, "")) for h in hdr) + "\n")

    print(f"\n{'=' * 70}")
    print(f"DONE ({elapsed/60:.1f} min, {total_sims} sims)")
    print(f"{'=' * 70}")
    print(f"  JSON         : {out_json}")
    print(f"  Aggregated   : {out_agg}")
    print(f"  All trials   : {out_raw}")
    if can_video:
        print(f"  Videos       : {os.path.join(run_dir, 'videos')}/")
    if not args.no_sensors:
        print(f"  Sensors      : {os.path.join(run_dir, 'sensors')}/")


if __name__ == "__main__":
    main()
