# Centipede MuJoCo — Setup and Usage Guide

A from-scratch guide for installing MuJoCo, running the FARMS centipede locomotion simulation, executing sweep experiments, optimizing controller gains, and analyzing results.

This document is the canonical reference for new contributors and for setting up the project on a new machine. It covers Windows 10/11, Ubuntu 22.04/24.04, and macOS workflows. The shorter `README.md` covers project layout; **start here for installation and first-run.**

---

## Contents

1. [What This Project Is](#1-what-this-project-is)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation-from-scratch)
4. [Verifying the Install](#4-verifying-the-install)
5. [Project Structure](#5-project-structure)
6. [Quick Start — First Simulation](#6-quick-start--first-simulation)
7. [Running Simulations](#7-running-simulations)
8. [Configuration Files](#8-configuration-files)
9. [Terrain Generation](#9-terrain-generation)
10. [Sweep Experiments](#10-sweep-experiments)
11. [Optimization](#11-optimization)
12. [Analysis and Plotting](#12-analysis-and-plotting)
13. [Cross-Simulator Comparison](#13-cross-simulator-comparison-with-isaac-lab)
14. [Troubleshooting](#14-troubleshooting)
15. [Glossary](#15-glossary)

---

## 1. What This Project Is

A MuJoCo-based simulation of a 21-segment biological centipede that walks under an **impedance traveling-wave controller** with a **Kuramoto central-pattern-generator (CPG)** coordinating the body wave. The robot is FARMS-derived (originally an SDF model, converted to MJCF) with 19 body yaw joints, 20 body pitch joints, 20 body roll joints, and 4 × 2 × 19 = 152 leg DOFs, for ~287 total joints.

The simulator is built to study **morphology-anchored terrain resonance**: how robot locomotion performance depends on the spatial wavelengths present in the terrain (relative to body length, segment length, and leg length). The scientific output is a set of `(wave_number × terrain_wavelength)` response curves that reveal which terrain spatial frequencies the centipede couples to.

Companion repositories on the same project:
- **Centipede_IsaacLab** — PhysX/NVIDIA Isaac Lab port of the same robot for GPU-parallel RL experiments
- **Centipede_IsaacLab_RL** — RL-specific fork with `DirectRLEnv` wrapper

---

## 2. System Requirements

**OS:** Windows 10/11, Ubuntu 22.04+, macOS 12+

**Python:** 3.10 or 3.11 (3.11 recommended; 3.12 not yet tested with MuJoCo wheels we use).

**Hardware:**
- **CPU:** any modern x86_64 (4+ cores recommended for sweeps).
- **GPU:** **not required for MuJoCo**. MuJoCo runs on CPU. A GPU helps only if you also use the Isaac Lab side of the project.
- **RAM:** 8 GB minimum, 16 GB recommended for sweeps.
- **Disk:** 5 GB for code + dependencies, plus 1–10 GB per sweep run if you save videos.

**Display:** required only for the interactive MuJoCo viewer. Headless servers can run with `--headless` flag.

---

## 3. Installation From Scratch

### 3.1 Install Python

#### Windows

Download Python 3.11 from <https://www.python.org/downloads/windows/>. **During install, tick "Add python.exe to PATH."** Verify:

```cmd
python --version
```

Should print `Python 3.11.x`. If `python` isn't found, restart the terminal.

#### Ubuntu

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev git
python3.11 --version
```

#### macOS

Using Homebrew:

```bash
brew install python@3.11 git
python3.11 --version
```

### 3.2 Clone the Repository

If you have git:

```bash
git clone https://github.com/wxy108/Centipede_MuJoCo.git
cd Centipede_MuJoCo
```

If you don't have git (lab PC workflow):

```bash
cd ~
wget https://github.com/wxy108/Centipede_MuJoCo/archive/refs/heads/main.zip
unzip -o main.zip
mv Centipede_MuJoCo-main Centipede_MuJoCo
cd Centipede_MuJoCo
```

### 3.3 Create a Virtual Environment

Strongly recommended — keeps the project's dependencies isolated from other Python work.

**Option A: conda** (recommended if you also use the Isaac Lab side):

```bash
conda create -n centipede python=3.11 -y
conda activate centipede
```

**Option B: venv** (simpler if you only need MuJoCo):

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3.11 -m venv venv
source venv/bin/activate
```

You should see `(centipede)` or `(venv)` in your shell prompt afterward. If you don't, the activation didn't take.

### 3.4 Install MuJoCo

MuJoCo is installed as a Python package — no separate compilation needed since version 2.3+. The relevant package is just `mujoco`:

```bash
pip install --upgrade pip
pip install mujoco
```

That's the entire MuJoCo install. The `mujoco` pip wheel includes the simulator binary, the Python bindings, and the viewer. No additional system packages, no shared library setup.

Verify:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

Should print `3.x.x` (any 3.x is fine).

### 3.5 Install Project Dependencies

From the project root:

```bash
pip install numpy scipy matplotlib pyyaml lxml mediapy
pip install scikit-optimize optuna
pip install opencv-python imageio imageio-ffmpeg
```

These are the runtime requirements:

- `numpy`, `scipy` — array math, terrain spectral synthesis
- `matplotlib` — analysis plotting
- `pyyaml` — config file parsing
- `lxml` — MJCF XML patching for terrain
- `mediapy`, `imageio[-ffmpeg]`, `opencv-python` — video recording
- `optuna`, `scikit-optimize` — Bayesian gain optimization

If you want a single command from a future `requirements.txt`:

```bash
pip install -r requirements.txt
```

(Not committed yet; we'll add it in a follow-up.)

---

## 4. Verifying the Install

Run a 10-second flat-ground simulation in the viewer:

```bash
cd controllers/farms
python run.py --duration 10
```

A MuJoCo viewer window should open showing the centipede settling for ~1 second, then starting to walk forward as the CPG engages. Close the viewer to end the run.

If the viewer doesn't open (headless server, SSH without X-forwarding, etc.):

```bash
python run.py --duration 5 --headless
```

Should print progress and exit cleanly. No window required.

Both runs save trajectory data to `outputs/data/run_<timestamp>/results_FARMS.npz`.

If either of these errors out, see [Troubleshooting](#14-troubleshooting).

---

## 5. Project Structure

```
Centipede_MuJoCo/
├── README.md                            Original short README
├── SETUP_GUIDE.md                       This file
│
├── models/                              MuJoCo MJCF models + meshes
│   └── farms/
│       ├── centipede.xml                Main MJCF — 21 segments, ~287 joints
│       └── meshes/                      OBJ meshes per link (21 of them)
│
├── controllers/
│   └── farms/
│       ├── impedance_controller.py      MAIN controller (impedance + Kuramoto CPG)
│       ├── controller.py                Older position-control variant (legacy)
│       ├── passive_axial_controller.py  Passive-walk diagnostic (no active torques)
│       ├── kinematics.py                Joint/body index resolution
│       ├── run.py                       Main run entry point — viewer + headless + video
│       ├── run_passive.py               Run with passive controller
│       └── run_rough.py                 Run on rough terrain
│
├── configs/
│   ├── farms_controller.yaml            CONTROLLER gains + wave parameters
│   ├── farms_controller.yaml.backup_*   Manual backups of previous gain sets
│   ├── terrain.yaml                     Terrain generation parameters (morphology
│   │                                    scales + spectral bands)
│   ├── blender_controller.yaml          Legacy Blender model config
│   └── _robustness_test.yaml            Robustness sweep override config
│
├── terrain/
│   ├── generator/
│   │   ├── generate.py                  Multi-band spectral terrain synthesis
│   │   └── patch_xml.py                 Inject heightfield into MJCF + sensors
│   └── output/                          Generated terrain PNGs (gitignored)
│
├── scripts/
│   ├── sweep/
│   │   ├── wavelength_sweep.py          λ sweep at fixed wave_number
│   │   ├── wave_number_sweep.py         (k × λ) 2D sweep
│   │   ├── analyze_wave_number_sweep.py Analysis + plotting for wave-number sweeps
│   │   ├── terrain_sweep.py             Multi-band terrain sweep
│   │   ├── terrain_browse.py            Interactive terrain viewer
│   │   ├── flat_ground_tune.py          Per-k amplitude tuning on flat ground
│   │   ├── verify_gains.py              Standalone gain replay
│   │   └── sensor_recorder.py           Per-trial sensor capture utility
│   │
│   ├── optimization/farms/
│   │   ├── optimize_impedance_local.py     LOCAL impedance tuner (narrow ±0.5
│   │   │                                    decade search, current canonical)
│   │   ├── optimize_impedance_three_eval.py Three-evaluation impedance tuner
│   │   ├── optimize_impedance_biological.py Bio-constrained impedance tuner
│   │   ├── optimize_impedance_bayesian.py   Generic Bayesian impedance tuner
│   │   ├── optimize_pitch_v2.py             Pitch-only tuner (2 params)
│   │   ├── render_top_trials.py             Re-render top-N trials as MP4
│   │   ├── apply_optimal_gains.py           Copy a trial YAML into farms_controller.yaml
│   │   └── (various legacy optimization helpers)
│   │
│   └── (dataset, analysis, etc.)
│
├── analysis/                            Post-simulation analysis utilities
│   └── analyze_tracking_farms.py        Command vs actual tracking
│
├── outputs/                             ALL run artifacts (gitignored)
│   ├── data/                            .npz trajectories from single runs
│   ├── videos/                          MP4 recordings
│   ├── optimization/                    Optimizer studies + top_videos + top_yamls
│   └── wave_number_sweep/               2D sweep outputs (CSVs + videos + analysis)
│
└── archive/                             Legacy/deprecated files (kept for reference)
```

---

## 6. Quick Start — First Simulation

### Run in the viewer (default)

```bash
cd controllers/farms
python run.py
```

The MuJoCo viewer opens with the centipede on a flat plane. The simulation runs at real-time. Use the viewer's mouse to orbit, pan, and zoom.

Default duration is set in `configs/farms_controller.yaml` (`simulation.duration`). Press Ctrl+C to stop early.

### Run headless (no window)

```bash
python run.py --headless --duration 10
```

Saves trajectory data to `outputs/data/run_<timestamp>/results_FARMS.npz` and exits.

### Record a video

```bash
python run.py --headless --duration 10 --video walk.mp4
```

Writes `walk.mp4` to the current directory. To save into the project's video folder:

```bash
python run.py --headless --duration 10 --video ../../outputs/videos/walk_demo.mp4
```

### Use a custom config

```bash
python run.py --config ../../configs/farms_controller.yaml.backup_v3_20260506
```

Swaps in a backed-up controller config without modifying the live `farms_controller.yaml`.

---

## 7. Running Simulations

### CLI Arguments (run.py)

| Flag | Default | Purpose |
|------|---------|---------|
| `--model PATH` | `models/farms/centipede.xml` | MJCF file |
| `--config PATH` | `configs/farms_controller.yaml` | Controller config |
| `--headless` | off | No viewer, faster |
| `--duration SEC` | from config | Override simulation duration |
| `--video PATH` | none | Save video to path |
| `--output PATH` | auto | Override .npz output path |

### Variants

**Rough terrain** (use `run_rough.py`):

```bash
cd controllers/farms
python run_rough.py --duration 10 --headless
```

This runs with a heightfield terrain generated by the spectral synthesis pipeline.

**Passive walk** (no active torques, just passive body):

```bash
python run_passive.py --duration 5
```

Useful for verifying the robot's resting posture and contact behavior independently of the controller.

---

## 8. Configuration Files

### configs/farms_controller.yaml

The main controller config. Edited every time you change gains, frequency, amplitude.

```yaml
body_wave:
  amplitude:        0.4         # Body yaw wave amplitude (rad)
  frequency:        1.0         # Body wave frequency (Hz)
  wave_number:      3.0         # Number of full waves on body
  speed:            1.0         # Wave phase speed multiplier
  head_hold_joints: 1           # Joints held by heading servo
  head_ramp_joints: 2           # Joints in cosine head-end taper
  head_ramp_min:    0.0         # Minimum of the head-end taper

leg_wave:
  amplitudes:    [0.6, 0.2, 0.0, 0.0]   # Per-leg-DOF amplitudes
  phase_offsets: [-1.57, 3.14, 0.0, 3.14] # Per-DOF phase offsets (rad)
  dc_offsets:    [0.0, 0.0, 0.0, 0.524]   # Per-DOF resting positions (rad)
  active_dofs:   [0, 1]                   # Which DOFs carry the wave

cpg:
  enabled:            true
  coupling_omega:     -2.5      # Body Kuramoto coupling gain
  leg_coupling_omega: -2.5      # Leg coupling gain

simulation:
  duration:  10.0               # Seconds
  timestep:  0.0005             # Physics dt (0.5 ms — MuJoCo-grade fidelity)

impedance:
  body_kp:        0.0835        # Body yaw stiffness (per-joint)
  body_kv:        0.00654       # Body yaw damping
  body_kp_head:   2.0           # Head-joint heading servo stiffness
  body_kv_head:   0.20
  pitch_kp:       0.0603
  pitch_kv:       0.0000371
  roll_kp:        0.004
  roll_kv:        0.0016
  settle_time:    1.0           # Seconds of rest before wave starts
  ramp_time:      1.0           # Head→tail blend-in time
  leg:
    kp: [0.113, 0.083, 0.00106, 0.0733]   # hip_yaw, hip_pitch, tibia, tarsus
    kv: [2.56e-05, 1.42e-08, 1.09e-07, 0.00118]
```

The current values are from MuJoCo local optimizer rank 1, trial 197 (May 2026).

### configs/terrain.yaml

Controls the spectral-band terrain generator.

```yaml
morphology:
  L_b:    0.1025    # Body length (m)
  n_wave: 3
  L_ell:  0.0074    # Leg length (m)

world:
  size:                 0.5     # Arena half-extent (m)
  image_size:           1024
  terrain_world_height: 0.05    # Max heightfield range (m)
  smooth_sigma:         1.5

spectral_bands:
  world:    { anchor: "L_w",   amplitude: 0.006, ... }
  body:     { anchor: "L_b",   amplitude: ..., ... }
  segment:  { anchor: "L_s",   amplitude: ..., ... }   # L_s = L_b/n_wave
  leg:      { anchor: "L_ell", amplitude: ..., ... }
```

The four bands correspond to the four characteristic length scales of the robot. Override amplitudes via CLI: `--world-amp`, `--body-amp`, `--segment-amp`, `--leg-amp`.

---

## 9. Terrain Generation

### Flat ground

The default `run.py` already runs on flat ground — no terrain generation needed.

### Single-wavelength rough terrain

Generate a heightfield with one dominant wavelength:

```bash
cd terrain/generator
python generate.py --wavelength 18 --amplitude 0.01 --output ../output/rough_wl18.png
```

Then patch the MJCF and run:

```bash
python patch_xml.py --terrain ../output/rough_wl18.png \
    --xml ../../models/farms/centipede.xml \
    --output ../../models/farms/centipede_rough.xml
cd ../../controllers/farms
python run.py --model ../../models/farms/centipede_rough.xml --duration 10 --headless
```

### Multi-band terrain

Generate terrain combining all four morphology bands:

```bash
cd terrain/generator
python generate.py --all-bands --output ../output/multi_band.png
```

Override individual band amplitudes:

```bash
python generate.py --all-bands \
    --world-amp 0.006 --body-amp 0.003 \
    --segment-amp 0.002 --leg-amp 0.001 \
    --output ../output/custom.png
```

---

## 10. Sweep Experiments

The headline science of this project. Two scripts:

### 10.1 Wavelength sweep (1D)

For a fixed wave_number (from `configs/farms_controller.yaml`), sweep over many terrain wavelengths and measure response:

```bash
cd scripts/sweep
python wavelength_sweep.py --n-trials 10 --duration 8
```

Outputs to `outputs/wavelength_sweep_<timestamp>/`. Per-cell metrics: forward speed, CoT, body z RMS, roll/pitch peaks, phase lag.

### 10.2 Wave-number × wavelength sweep (2D)

The main experiment for the resonance hypothesis:

```bash
cd scripts/sweep

# Full default: 5 wave_numbers × 20 wavelengths × 20 trials = 2000 sims, ~65h
python wave_number_sweep.py --n-trials 20 --duration 8 --video

# Practical 800-sim version: 4 wave_numbers × 20 wavelengths × 10 trials = 800 sims, ~28h
python wave_number_sweep.py \
    --wave-numbers "2,2.5,3,3.5" \
    --n-trials 10 \
    --duration 8 \
    --video \
    --no-sensors
```

CLI flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--wave-numbers LIST` | `"1.5,2,2.5,3,3.5"` | Comma-separated body wave numbers |
| `--wavelengths LIST` | 20-element default list | Comma-separated terrain wavelengths (mm) |
| `--n-trials INT` | 20 | Trials per (k, λ) cell |
| `--duration FLOAT` | 8.0 | Seconds per trial |
| `--amplitude FLOAT` | 0.010 | Terrain peak amplitude (m) |
| `--seed INT` | 42 | Master RNG seed |
| `--video` | off | Record MP4 per trial (~1 GB total at full size) |
| `--no-sensors` | off | Skip per-trial sensor CSVs |

**Long sweeps must be run in `tmux` or `screen`** so an SSH disconnect doesn't kill the job:

```bash
tmux new -s sweep
python scripts/sweep/wave_number_sweep.py --wave-numbers "2,2.5,3,3.5" \
    --n-trials 10 --duration 8 --video --no-sensors \
    2>&1 | tee outputs/wave_number_sweep_run.log
# Detach: Ctrl-B then D
# Reattach later: tmux attach -t sweep
```

Output structure:

```
outputs/wave_number_sweep/sweep_<TIMESTAMP>/
├── results.json                full nested data
├── results_aggregated.csv      one row per (k, λ) cell
├── results_all_trials.csv      one row per trial
├── videos/                     MP4s if --video used
│   ├── k2/wl_3mm/trial_00.mp4
│   ├── ...
└── sensors/                    per-trial sensor CSVs if --sensor-data used
```

---

## 11. Optimization

Three optimizer flavors, picked based on what you're tuning.

### 11.1 Local impedance optimizer

The current canonical for polishing gains around a known-good seed (Isaac Lab rank 5 or MuJoCo trial 197). Narrow ±0.5 decade log search:

```bash
cd scripts/optimization/farms

# Quick smoke (5 trials, ~3 min)
python optimize_impedance_local.py --n-trials 5 --duration 6

# Full local search (200 trials, ~6h)
python optimize_impedance_local.py --n-trials 200 --duration 8 --run-tag local_v1
```

Outputs:

```
outputs/optimization/impedance_local_<run_tag>_<TIMESTAMP>/
├── results.json
├── best_params.yaml         drop-in replacement for farms_controller.yaml
├── all_trials.csv           per-trial cost + params + metrics
├── study.db                 Optuna SQLite study (resumable)
└── terrains/                generated heightmaps used
```

### 11.2 Render top trials

After an optimization run, replay the top N trials as MP4 for inspection:

```bash
python render_top_trials.py \
    outputs/optimization/impedance_local_local_v1_<TIMESTAMP>/all_trials.csv \
    --top 10 --duration 8
```

Outputs:

```
outputs/optimization/impedance_local_local_v1_<TIMESTAMP>/
├── top_videos/
│   ├── rank_01_trial_0197_cost_+008.08.mp4
│   ├── rank_02_trial_0129_cost_+008.62.mp4
│   ├── ...
└── top_yamls/
    ├── rank_01_trial_0197.yaml         drop-in replacement YAML
    ├── ...
```

### 11.3 Apply a winning gain set

```bash
cp outputs/optimization/impedance_local_local_v1_<TIMESTAMP>/top_yamls/rank_01_trial_0197.yaml \
   configs/farms_controller.yaml
```

After copy, all subsequent `run.py` and sweep scripts will use the new gains.

### 11.4 Other optimizers

- `optimize_impedance_three_eval.py` — averages cost across three (terrain, frequency) regimes for robustness
- `optimize_impedance_biological.py` — applies bio-anchored constraints (force ≤ 8× motor, velocity quality ≥ 0.3, roll ≤ 5°)
- `optimize_impedance_bayesian.py` — generic broad TPE search
- `optimize_pitch_v2.py` — pitch-gain-only sweep, useful when other gains are locked

---

## 12. Analysis and Plotting

### 12.1 Analyze a wave-number sweep

```bash
cd scripts/sweep

# Auto-finds the most recent sweep
python analyze_wave_number_sweep.py

# Specific sweep
python analyze_wave_number_sweep.py outputs/wave_number_sweep/sweep_<TIMESTAMP>
```

Default behavior: drops `k = 1.5` (very noisy), trims trial outliers beyond `±2σ` of cell median (using robust MAD-based sigma), plots `±0.6σ` error band on trimmed trials.

CLI options:

| Flag | Default | Purpose |
|------|---------|---------|
| `--exclude-k LIST` | `"1.5"` | Comma-sep `n` values to drop |
| `--trim-sigma FLOAT` | 2.0 | Outlier trim threshold (MAD-based) |
| `--band-sigma FLOAT` | 0.6 | Plot error-band width in trimmed σ units |
| `--body-length FLOAT` | 0.1025 | Body length in metres |
| `--world-length FLOAT` | 0.5 | L_w reference line position (m) |
| `--leg-length FLOAT` | 0.0074 | L_l reference line position (m) |

Output to `<sweep_dir>/analysis/`:

```
metrics_vs_wavelength.png      6-panel overview (speed, CoT, survival, pitch, roll, phase lag)
speed_heatmap.png              2D heatmap of forward speed across (n × λ)
resonance_forward_speed.png    Screenshot-style multi-curve resonance plot
resonance_cot.png              Same for CoT (log y-axis)
resonance_max_pitch_deg.png    Same for max pitch
resonance_max_roll_deg.png     Same for max roll
resonance_phase_lag_deg.png    Same for terrain→body phase lag
summary.md                     Markdown table: best (n, λ) per metric, marginals, all-buckled cells
```

Each `resonance_*.png` figure shows:

- **Multi-curve mean lines** — one per wave number `n`, color-coded
- **±σ shaded bands** — bulk of trimmed trials
- **Individual trial scatter** — light dots behind, color-matched
- **Fixed morphology lines** — `L_w` (world), `L_b` (body), `L_l` (leg) as vertical dashed lines
- **Per-`n` body wavelength lines** — `L_b/n` as additional vertical dashed lines, color-matched to curves (replaces the single `L_s` line)
- **Reversed log-x axis** — large λ (smooth terrain) on left, small λ (rough) on right

### 12.2 Single-run trajectory analysis

After a `run.py` execution, analyze the saved `.npz`:

```bash
python analysis/analyze_tracking_farms.py outputs/data/run_<TIMESTAMP>/results_FARMS.npz
```

Plots commanded vs actual joint angles, body trajectories, contact forces.

---

## 13. Cross-Simulator Comparison with Isaac Lab

This project lives alongside `Centipede_IsaacLab` (PhysX-based). When both stacks use the **same** `configs/farms_controller.yaml` gain set, you get a controlled comparison of MuJoCo vs PhysX dynamics on the same controller.

Standard cross-simulator workflow:

1. Run the wavelength sweep on MuJoCo (this repo)
2. Copy `configs/farms_controller.yaml` to the equivalent Isaac Lab file
3. Run the equivalent sweep on Isaac Lab
4. Compare resonance peaks between the two

The same RNG seed produces the **same terrain heightmaps** on both stacks (verified byte-faithful in the Isaac Lab port), so any difference in measured metrics reflects contact-solver dynamics, not terrain noise.

---

## 14. Troubleshooting

### `ModuleNotFoundError: No module named 'mujoco'`

The MuJoCo install didn't complete or your venv isn't activated. Re-activate the venv (or conda env) and run:

```bash
pip install --upgrade mujoco
python -c "import mujoco; print(mujoco.__version__)"
```

### `GLFW failed to create window` (Linux without X11)

You're on a headless server or SSH without X-forwarding. Use `--headless`:

```bash
python run.py --headless --duration 5
```

For video output without a display, MuJoCo's offscreen renderer is used automatically with `--video`.

### Viewer opens but the centipede explodes

Almost always a gain set that's destabilizing on the current terrain. Try:

1. Reverting to a known-good gain set:
   ```bash
   cp configs/farms_controller.yaml.backup_v3_20260506 configs/farms_controller.yaml
   ```
2. Or running on flat ground first (no `--rough` / `--terrain` flag):
   ```bash
   python run.py --duration 5
   ```

### `MujocoException: gain_prm[0] out of range`

The XML has an actuator with bad `gainprm`. Re-export the model via:

```bash
cd scripts/optimization/farms
python patch_forcelimits.py
```

This rewrites force-limits on all actuators to safe defaults.

### Sweep runs very slowly

Check:

- `--video` enabled? Recording adds 30–50 % overhead.
- `--sensor-data` enabled? Rich sensor capture adds disk I/O.
- Disk full? `df -h .` — sensor CSVs and videos eat a lot of space.
- For maximum speed: `--no-sensors` and skip `--video` for the production sweep, run a separate short sweep with `--video` for paper figures.

### Out-of-memory during long sweeps

Per-trial MuJoCo model is small (~10 MB) but the recorder accumulates trajectories. Restart the sweep with `--no-sensors`, or reduce `--duration`.

### git push rejected

Someone else (or you on another machine) pushed first:

```bash
git pull --rebase origin main
git push origin main
```

If there are conflicts, the rebase will tell you which files. Edit them to resolve, then `git rebase --continue`.

---

## 15. Glossary

- **CPG (Central Pattern Generator)** — A coupled-oscillator network producing rhythmic output. We use a Kuramoto chain with nearest-neighbor coupling for the body wave.
- **FARMS** — Framework for Animat Robot Modelling and Simulation, the SDF model origin
- **MJCF** — MuJoCo's XML model format
- **Wave number `n`** — number of full body waves on the centipede body. Higher `n` = more body undulations per cycle.
- **Terrain wavelength `λ`** — spatial period of the dominant terrain feature (mm).
- **Body wavelength `L_b/n`** — the body-wave-projected spatial wavelength on the ground. Hypothesized resonance with `λ` ≈ `L_b/n`.
- **L_w, L_b, L_s, L_l** — World (arena), Body, Segment (=`L_b/n_wave`), Leg characteristic length scales.
- **CoT (Cost of Transport)** — `Σ|τ·ω|·dt / (m·g·d)`. Dimensionless energy per unit distance per unit weight.
- **Impedance control** — Torque law `τ = kp·(target − q) + kv·(target_dot − qdot)`. The "PD-tracking" form (with `target_dot` rather than `−qdot`) lets the controller follow a moving reference without lag-induced fighting.
- **Survival rate** — Fraction of trials in a cell that didn't buckle (pitch < 45°, roll < 45°).

---

## License & Citation

See `LICENSE`. If you use this code in a paper, cite the project's preprint (TBD) and the underlying MuJoCo simulator:

> Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo: A physics engine for model-based control. IROS.

---

## Maintenance Notes

- All sweep outputs (`outputs/`) are gitignored. Don't `git add outputs/` — the artifacts can be many GB.
- All backup YAMLs (`configs/*.backup_*`) are gitignored.
- The sweep temp XML (`models/farms/*.sweep_tmp.xml`) is regenerated by the sweep scripts every run — gitignored.
- Per-trial videos from sweeps live under `outputs/wave_number_sweep/<run>/videos/` — also gitignored.

For new contributors: follow §3 to install, §6 to verify, then §10 to reproduce a sweep. The whole stack should be reproducible from scratch in about an hour of human time plus whatever the sweep takes to run.
