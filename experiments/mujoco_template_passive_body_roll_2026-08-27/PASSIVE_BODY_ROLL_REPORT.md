# Coupled-wave passive body-roll audit

Date: 2026-08-27  
Scope: Centipede Model only.

## Frozen coordination rule

- Flat: `body wave = leg wave = 1.5`.
- Rough: `body wave = leg wave = 3.0`.
- Relative leg phase override: `0 deg`.
- Wave number and relative phase are not optimization variables in this experiment.

The previous decoupled `body 1.5 / leg 3.0 / +45 deg` candidate is explicitly superseded and retained only as historical causal evidence.

## Physical representation

The 20 body-roll active actuators are disabled. Their measured actuator-force maximum is exactly zero in every run.

Body-roll resistance is represented only by explicit XML linear torsional springs and dampers at zero rest angle:

- baseline scale: `k = 1.1597375518325183e-4 N m/rad`, `c = 4.02715557956593e-7 N m s/rad`;
- tested stiffness scales: `1x, 2x, 3x, 4x, 6x`;
- damping scales with the square root of the stiffness multiplier;
- body-roll range remains `+/-30 deg` and is never used as the stabilization mechanism;
- armature remains exactly zero.

The accepted `bounded_k20_c18` q2 plant, body mass, inertia, timestep, body/leg amplitudes, knee target, and first-body-yaw warm-up rule remain unchanged.

## Main sweep result

No passive stiffness candidate passes the combined flat/rough roll, q2, knee-support, contact, posture, torque, and performance gates. Therefore this experiment has **no selected candidate**.

| terrain | passive scale | displacement | knee deviation P95 | local roll P95 | cumulative roll P95 | world roll P95 | head roll P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat | 1x | 83.54 mm | 18.53 deg | 10.27 deg | 45.30 deg | 22.75 deg | 20.46 deg |
| flat | 2x | 120.05 mm | 26.48 deg | 7.13 deg | 33.83 deg | 22.25 deg | 17.42 deg |
| flat | 3x | 127.11 mm | 33.93 deg | 5.01 deg | 30.76 deg | 22.63 deg | 16.00 deg |
| flat | 4x | 131.91 mm | 34.79 deg | 3.92 deg | 28.97 deg | 21.81 deg | 15.94 deg |
| flat | 6x | 133.59 mm | 35.81 deg | 2.86 deg | 24.21 deg | 21.34 deg | 16.67 deg |
| rough | 1x | 84.42 mm | 19.70 deg | 7.78 deg | 35.29 deg | 15.79 deg | 19.74 deg |
| rough | 2x | 87.79 mm | 21.19 deg | 4.48 deg | 25.52 deg | 13.66 deg | 16.78 deg |
| rough | 3x | 76.94 mm | 23.60 deg | 3.02 deg | 19.45 deg | 12.41 deg | 11.86 deg |
| rough | 4x | 76.80 mm | 24.52 deg | 2.38 deg | 16.84 deg | 12.04 deg | 10.38 deg |
| rough | 6x | 79.41 mm | 24.59 deg | 1.73 deg | 13.17 deg | 11.81 deg | 9.60 deg |

The passive spring clearly suppresses **relative inter-segment roll**. It does not materially suppress the flat **common-mode/free-root roll**: flat world-roll P95 changes only from 22.75 deg to 21.34 deg over the full 1x-to-6x range.

## Knee-support causal screen

At passive body-roll `4x`, knee Kp was independently bracketed from `1x` to `4x`, with knee damping scaled by the square root. Body/leg waves remained coupled.

| knee gain | knee deviation P95 | world roll P95 | head roll P95 | displacement |
|---:|---:|---:|---:|---:|
| 1x | 34.79 deg | 21.81 deg | 15.94 deg | 131.91 mm |
| 1.5x | 24.82 deg | 21.68 deg | 16.68 deg | 135.88 mm |
| 2x | 19.60 deg | 21.65 deg | 16.89 deg | 141.44 mm |
| 3x | 14.31 deg | 21.69 deg | 16.26 deg | 143.95 mm |
| 4x | 10.76 deg | 21.56 deg | 16.22 deg | 145.86 mm |

Restoring knee support does not reduce the global roll mode. This rejects the hypothesis that the remaining flat roll is caused primarily by knee collapse.

## Interpretation

A passive spring between adjacent body segments can oppose relative joint rotation, but it cannot directly generate a restoring moment against a free-root rigid-body roll shared by the whole chain. Ground contact can provide that external moment, which explains why the same passive stiffness helps more on rough terrain than on flat terrain, but the flat contact pattern does not produce enough stabilizing roll moment.

This is a physical limitation of the tested passive topology, not evidence that the body-roll spring should simply be increased further. Higher stiffness increasingly transfers load to the legs and q2 stops while leaving flat world roll almost unchanged.

## Diagnostic Demo status

`passive_k4` was rendered in flat/rough side, oblique, and top views solely as a failed-boundary diagnostic. It is labeled `diagnostic_not_selected`; all Demo rows do not pass.

## Evidence

- Runner: `run_passive_body_roll.py`
- Model manifest: `outputs/data/passive_body_roll_model_manifest.json`
- Full sweep: `outputs/data/passive_body_roll_sweep.json`
- Knee causal screen: `outputs/data/flat_knee_support_screen.json`
- Diagnostic Demo audit: `outputs/data/diagnostic_demo_audit.json`
- Time-sequence QA: `outputs/figures/diagnostic_passive_k4_flat_timeline.png` and `diagnostic_passive_k4_rough_timeline.png`

## Isolation

Robot Step 18/19 files were used only as read-only hash sentinels and remained unchanged before and after model construction, the ten-run sweep, knee screen, and Demo rendering.

## Next decision

Do not continue increasing body-roll passive stiffness blindly. The next experiment should first measure the flat roll-moment budget: per-foot normal/tangential contact force, moment arm about the body longitudinal axis, stance asymmetry, and root angular momentum. That determines which physically grounded leg-ground pathway can supply the missing external restoring moment while keeping the body and leg waves coupled.
