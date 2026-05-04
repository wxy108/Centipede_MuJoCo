# Centipede MuJoCo Locomotion Simulation

Biologically-inspired centipede locomotion simulation in MuJoCo, with two model variants (high-fidelity Blender and FARMS-derived), multi-frequency terrain generation, and automated terrain roughness sweep experiments.

## Project Structure

```
Centipede_MUJOCO-main/
│
├── models/                     # MuJoCo model definitions and mesh assets
│   ├── blender/                #   High-fidelity Blender-generated model (21 segments)
│   │   ├── centipede.xml       #     MJCF model definition
│   │   ├── meshes/             #     Visual + collision OBJ meshes
│   │   └── textures/           #     Body textures (PNG/JPEG)
│   └── farms/                  #   FARMS-derived model (19 segments, SDF origin)
│       ├── centipede.xml       #     MJCF model definition
│       └── meshes/             #     Link OBJ meshes
│
├── controllers/                # Simulation controllers and runners
│   ├── blender/                #   Blender model controller stack
│   │   ├── controller.py       #     PD servo + trajectory generators
│   │   ├── kinematics.py       #     Joint/actuator naming and indexing
│   │   ├── run.py              #     Simulation runner (viewer/headless/video)
│   │   └── test_joints.py      #     Individual joint verification tool
│   └── farms/                  #   FARMS model controller stack
│       ├── controller.py       #     Traveling-wave position controller
│       ├── kinematics.py       #     FARMS joint naming and indexing
│       └── run.py              #     Simulation runner (viewer/headless/video)
│
├── configs/                    # All configuration files (centralized)
│   ├── blender_controller.yaml #   Blender model wave parameters + servo gains
│   ├── farms_controller.yaml   #   FARMS model wave parameters
│   └── terrain.yaml            #   Spectral terrain generation parameters
│
├── terrain/                    # Terrain generation pipeline
│   ├── generator/              #   Terrain generation scripts
│   │   ├── generate.py         #     Multi-frequency spectral terrain generator
│   │   └── patch_xml.py        #     Patches FARMS XML with hfield + sensors
│   └── output/                 #   Generated terrain PNGs (gitignored)
│
├── scripts/                    # Experiment and automation scripts
│   ├── sweep/                  #   Terrain roughness sweep experiments
│   │   ├── terrain_sweep.py    #     Full L0-L3 sweep with per-level z_max
│   │   └── terrain_browse.py   #     Interactive terrain browser
│   ├── optimization/           #   Parameter tuning (Bayesian optimization)
│   │   ├── blender/            #     Blender model PD gain tuning
│   │   └── farms/              #     FARMS model gain tuning + XML patches
│   └── dataset/                #   ML dataset collection
│       ├── batch_terrain_sweep.py  # Batch terrain sweep for ML training
│       └── test_terrain_levels.py  # Quick terrain level verification
│
├── analysis/                   # Post-simulation analysis tools
│   ├── analyze_tracking_blender.py  # Command vs actual tracking (Blender)
│   └── analyze_tracking_farms.py    # Command vs actual tracking (FARMS)
│
├── outputs/                    # All generated outputs (gitignored)
│   ├── data/                   #   Simulation data (.npz files)
│   ├── videos/                 #   Rendered simulation videos
│   ├── logs/                   #   Sweep logs, MuJoCo logs
│   ├── optimization/           #   Bayesian optimization results
│   └── dataset/                #   ML training dataset
│
└── archive/                    # Legacy/deprecated files (gitignored)
    ├── blender_pipeline/       #   Original Blender export scripts
    ├── farms_sdf/              #   Original FARMS SDF model
    └── legacy_scripts/         #   Superseded scripts and dumps
```

## Quick Start

### Run FARMS model in viewer

```bash
cd controllers/farms
python run.py
```

### Run Blender model in viewer

```bash
cd controllers/blender
python run.py
```

### Generate terrain and run sweep

```bash
cd scripts/sweep
python terrain_sweep.py --test     # quick 4-trial test
python terrain_sweep.py            # full 151-trial sweep
```

## Terrain Pipeline

The terrain system uses spectral band synthesis to generate heightfields with independently controllable roughness characteristics:

- **Low band** (1-5 cyc/m): body-scale hills
- **Mid band** (5-20 cyc/m): segment-scale mounds
- **High band** (20-40 cyc/m): leg-scale texture

Each terrain level has unique band amplitudes (different spatial patterns) and a per-level `z_max` that controls physical height in MuJoCo. All PNGs are normalized to full 0-255 range to avoid MuJoCo hfield staircase artifacts.

## Configuration

All configs live in `configs/`:

| File | Controls |
|------|----------|
| `blender_controller.yaml` | Wave parameters, servo gains, appendage motion (Blender model) |
| `farms_controller.yaml` | Wave parameters, servo gains (FARMS model) |
| `terrain.yaml` | World size, spectral band frequencies/amplitudes, smoothing |

## Dependencies

- Python 3.10+
- MuJoCo (via `mujoco` package)
- NumPy, SciPy, PyYAML
- mediapy (for video recording)
- lxml (for XML patching)
- matplotlib (for analysis)
- scikit-optimize (for Bayesian optimization)
