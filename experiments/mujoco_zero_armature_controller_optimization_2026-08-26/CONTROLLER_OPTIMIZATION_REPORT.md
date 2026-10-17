# Centipede Model controller optimization report

Date: 2026-08-26  
Scope: biologically inspired Centipede **Model** only; not the physical Robot.

> Follow-up: the impedance table below remains current, but the single fixed
> gait has been superseded by measured flat and rough/random gait profiles.
> See `FLAT_LEG_LIFT_OPTIMIZATION_REPORT.md` and
> `config/optimized_controller.json`.

## Evidence contract

The current clean RL runtime, not historical prose, defines the comparison:

- `Centipede_IsaacLab_RL_Clean/centipede/env.py`: live impedance and the
  projected-gravity failure gate `abs(g_z) < 0.259`.
- `Centipede_IsaacLab_RL_Clean/config/controller.yaml`: 1 Hz target generator,
  fixed joint amplitudes and head-pitch target.
- `Centipede_IsaacLab_RL_Clean/config/gait_map.json`: wave-dependent retained
  gait map.
- `Centipede_IsaacLab_RL_Clean/outputs/demo/TERRAIN_DEMO_PANEL_20260818.json`:
  9.6 s reference values 0.3323 BL/cycle on flat/w1.5, 0.3317 on L90/A8/w2,
  and 0.2626 on L60/A12/w3.

The July compliance log was used only as a historical hypothesis. Its useful
claim was re-tested here: vertical-only compliance had previously reached a
body-terrain correlation near 0.67 while walking. Stale armature and explicit
torque conclusions from that document were not carried into this plant.

## Optimization method

1. Freeze mass, geometry, inertia, armature=0, 1 ms physics, gait amplitudes
   and 1 Hz frequency.
2. Run a 64-point scrambled Sobol screen over 14 dimensions: Kp and Kv for
   seven joint families, with head pitch locked.
3. Apply feasibility gates before locomotion comparison: finite state, no
   MuJoCo warning, `|g_z| >= 0.259`, spine tortuosity <= 1.30, and active-joint
   peak speed <= 100 rad/s.
4. Send only retained candidates to L90/A8 and L60/A12. The fastest flat
   candidate, S047 at 0.254 BL/cycle, was rejected because it diverged on both
   rough terrains.
5. Refine locomotion-driving yaw/leg gains separately from pitch/roll/knee
   compliance. Retain the shared candidate with the best balanced
   worst-condition performance, not a weighted score.
6. Calibrate the clean controller's existing head-yaw reference interface.
   Gain 8 with a 0.6 rad clip is stable; gain 12 diverges at 0.681 s and is
   rejected.

## Frozen controller

| Family | Kp | Kv |
|---|---:|---:|
| body yaw | 2.29946e-4 | 3.53560e-7 |
| head pitch | 7.06377e-5 | 2.81058e-7 |
| body pitch | 1.88221e-4 | 6.24088e-7 |
| body roll | 7.84045e-5 | 3.11962e-7 |
| legbase yaw | 8.89611e-4 | 1.23563e-6 |
| legbase pitch | 6.34950e-4 | 9.29035e-7 |
| legbase roll | 8.64658e-6 | 2.78426e-7 |
| knee | 9.20618e-5 | 2.44202e-7 |

Head-yaw world-reference gain is 8.0 with target clip 0.6 rad. The actuator
path remains MuJoCo native affine implicit PD; Python sends a composite
position/velocity reference, never explicit torque.

## 9.6 s results

| Condition | Old speed | Optimized speed | Efficiency | yaw drift | min `|g_z|` | peak qvel | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| flat, w1.5 | 0.0031 forward | 0.2000 forward | 0.981 | -0.55 deg | 0.845 | 29.1 | pass |
| L90/A8, w2 | 0.1746 path | 0.2532 path | 0.943 | 1.78 deg | 0.587 | 73.4 | pass |
| L60/A12, w3 | 0.1598 path | 0.1786 path | 0.902 | 0.05 deg | 0.744 | 45.6 | pass |

The flat low-wave failure is removed: forward performance rises by about
64.8x and the trajectory becomes nearly straight. Mild rough path speed rises
45%; severe rough rises 11.8% while the previous 199 rad/s peak is reduced to
45.6 rad/s.

This does **not** claim IsaacLab parity. Relative to the retained clean RL demo,
the MuJoCo result reaches about 60% of flat speed, 76% of L90/A8 speed and 68%
of L60/A12 speed. It is now stable and useful, but further morphology/contact
calibration could still close the remaining gap.

## Pitch/roll compliance

| Condition | pitch RMS | roll RMS | vertical spine/terrain corr | roll/local-slope corr |
|---|---:|---:|---:|---:|
| flat | 3.84 deg | 4.57 deg | n/a | n/a |
| L90/A8 | 3.35 deg | 4.38 deg | 0.673 | 0.306 |
| L60/A12 | 2.97 deg | 4.74 deg | 0.864 | 0.655 |

The vertical correlation increases with terrain severity while locomotion
survives, so the body is conforming rather than merely collapsing. The pitch
orientation correlation is negative because of the model's pitch sign
convention; its magnitude rises from 0.334 to 0.592. Roll response has the
expected positive sign and strengthens on L60/A12.

## Random-contact retention

Three flat w3, q=0.3 seeds all pass the same gates. Forward speed spans
0.1506-0.1584 BL/cycle, path efficiency 0.944-0.953, minimum `|g_z|`
0.821-0.867, and peak active-joint speed 29.1-34.7 rad/s. There are no MuJoCo
warnings.

## Demo and provenance

The four 9.6 s videos each contain 289 frames at 30 fps:

- `outputs/videos/optimized_flat.mp4`
- `outputs/videos/optimized_l90a8.mp4`
- `outputs/videos/optimized_l60a12.mp4`
- `outputs/videos/optimized_random_contact_w3_q030_seed0.mp4`

Mid-frame visual review confirms an extended, upright body with no coiling or
flip. Exact hashes are in `outputs/data/demo_audit.json`.

The physical Robot repository was read only. Its Step 18/19 sentinel hashes
remain unchanged:

- Step 18 run configuration: `847BEE7492D75CA05FFADC40B4664C8CE02EEDB1D5E16B6683FF2EA9BD6CCF85`
- Step 18 trials: `C46F9B1C1F932971CE37EA8AD6C5E9892D01E238E7839867F135D1A507A8E3CC`
- Step 19 run configuration: `FF007DEC8796DD0E8F0AF6D58EFFFDF0287C1474E171CD7E086CED88A38EFBC8`
- Step 19 trials: `1A4AD813787B68A7EA409C9C261F3A977453AB61035E44B479B8A9A2E7CCEC48`

## Remaining limitation

The rough validation uses the clean canonical seed 244949. The next research
step should be multi-seed rough-terrain replication plus phase-resolved
whole-spine shape comparison. That is needed before making a population-level
claim about compliance or terrain generalization.
