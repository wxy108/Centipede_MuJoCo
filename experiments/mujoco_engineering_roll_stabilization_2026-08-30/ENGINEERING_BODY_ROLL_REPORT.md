# MuJoCo engineering body-roll stabilization

> **REJECTED / NON-CANONICAL (PI instruction, 2026-08-31).** The projected-gravity leg-pitch feedback in this experiment is not eligible for the clean Centipede Model controller. Preserve the measurements and videos only as historical causal evidence. The locked-head source XML and original controller runner were never modified, so no plant rollback is required.

Date: 2026-08-30  
Scope: **abstract Centipede Model only**. Centipede Robot Step 18/19 files were read only as SHA-256 sentinels and remained unchanged.

## Result

The selected controller is `pg_p300_l125`. It removes the visibly excessive common-mode axial body roll on both the locked-head flat and rough models without changing the MJCF plant, body/leg wave architecture, gait amplitudes, or distal-tip compliance.

The fix follows the useful part of the IsaacLab engineering pattern: projected gravity is treated as an attitude observation and converted into bounded feedback. It does **not** copy IsaacLab's raw body-roll stiffness into MuJoCo. A direct Isaac-style active body-roll PD was tested, but rejected because its actuator torque remained saturated for at least 95% of evaluated samples.

## Root cause and control path

The prior force audit showed that equal left/right contact counts do not imply zero roll moment: foot force magnitude and lateral lever arm differ. A passive spring between body segments can suppress relative twist, but cannot by itself generate the external restoring moment required to stop free-root common-mode roll.

The selected controller therefore:

1. estimates common chassis roll from the circular mean of projected-gravity roll across all chassis links;
2. adds a local per-segment projected-gravity residual so the front cannot simply hand its tilt to the rear;
3. low-pass filters attitude and roll rate;
4. applies a bounded, opposite left/right additive offset to the existing leg-pitch references, allowing foot-ground reactions to generate the restoring moment; and
5. ramps this feedback smoothly from 0.7 to 1.5 s.

Selected parameters:

| parameter | value |
|---|---:|
| common roll gain | 3.0 rad/rad |
| roll-rate gain | 0.14 s |
| local segment residual gain | 1.25 rad/rad |
| attitude / rate filter time constants | 0.10 / 0.06 s |
| leg-pitch correction clip | 0.52 rad (29.79 deg) |
| feedback start / ramp duration | 0.70 / 0.80 s |
| additional active body-roll Kp / Kv / torque | 0 / 0 / 0 |

The observed correction P95 is 15.16 deg on flat ground and 9.39 deg on rough ground, well below the 29.79 deg clip. The result is therefore not a saturation artifact.

## Frozen plant and gait contract

- Source XML files remain the locked-head `m29p0` flat and rough baselines; no new selected XML was generated because the plant did not change.
- Total mass, every body inertia, contact geometry, timestep, joint ranges, and all armatures are unchanged; armature remains exactly zero.
- Existing passive body-roll stiffness/damping remain `6.958425311e-4 N m/rad` and `9.86447628474e-7 N m s/rad`.
- Existing passive distal-tip stiffness/damping remain `4.5e-6 N m/rad` and `1.15375965e-9 N m s/rad` with the same nominal +/-38 deg range.
- Flat body wave and leg wave remain coupled at `1.5`; relative leg phase remains `0 deg`; gait amplitudes are unchanged.
- The head and link 1 remain genuinely fixed at the selected `-29 deg` export.

## Six-second deterministic validation

Statistics are evaluated after 1.5 s, covering 4.5 complete gait cycles.

| terrain | metric | frozen baseline | selected | result |
|---|---:|---:|---:|---:|
| flat | coherent chassis roll P95 | 16.343 deg | 4.919 deg | -69.9% |
| flat | front-seven coherent roll P95 | 18.699 deg | 7.091 deg | -62.1% |
| flat | all-link world roll P95 | 20.960 deg | 12.274 deg | -41.4% |
| flat | world-roll span P95 | 40.105 deg | 26.635 deg | -33.6% |
| flat | local body-roll P95 | 3.265 deg | 2.886 deg | -11.6%; not locked |
| flat | displacement | 143.736 mm | 143.193 mm | 99.6% retained |
| flat | path efficiency | 0.9320 | 0.9685 | improved |
| rough | coherent chassis roll P95 | 8.681 deg | 2.518 deg | -71.0% |
| rough | front-seven coherent roll P95 | 13.377 deg | 7.493 deg | -44.0% |
| rough | all-link world roll P95 | 12.926 deg | 7.540 deg | -41.7% |
| rough | world-roll span P95 | 27.477 deg | 18.156 deg | -33.9% |
| rough | local body-roll P95 | 2.366 deg | 1.876 deg | -20.7%; not locked |
| rough | displacement | 91.140 mm | 91.502 mm | 100.4% retained |
| rough | path efficiency | 0.9910 | 0.9924 | retained |

Contact penetration P95 is essentially unchanged: `0.06072 -> 0.06002 mm` on flat and `0.01803 -> 0.01773 mm` on rough.

Tip motion is not suppressed: P95 changes from `27.43 -> 30.39 deg` on flat and `26.73 -> 27.98 deg` on rough. The flat stop-neighborhood fraction changes from `0.00719 -> 0.01687`, an absolute increase of `0.00968`, which remains inside the preregistered `+0.01` gate. This is increased compliant response under redistributed support load, not hidden tip stiffness. It remains a quantity to monitor in random-contact work.

## Rejected alternatives

- Negative projected-gravity feedback increased flat coherent roll from `16.07` to `24.74 deg`; sign was causally rejected.
- Passive body-roll stiffness alone reduced inter-segment twist but not the free-root common mode; this reproduces the earlier passive audit.
- Pure common-mode leg correction improved overall roll but left the front seven links above 12 deg; local residual allocation was necessary.
- Contact-gated and tip-authority-guard variants either kept the tip near its stop or lost too much roll authority; both were rejected.
- Isaac-style active body-roll PD reduced local roll and tip P95, but its torque P95 sat at the imposed `3e-5` or `4e-5 N m` caps and the front-roll gate was not robustly met. It is not in the selected controller.

## Demo and integrity

Six final videos were rendered at 1280x720, 30 fps. Every file has 181 frames and duration 6.03 s. Flat and rough each have front, side, and top views. Three independent reruns per terrain produced identical dynamics metrics.

The contact sheet was visually inspected: all six views contain the model, rough terrain is lit and smooth, and no black-screen or missing-terrain failure is present.

## Evidence

- Runner: `run_engineering_roll.py`
- Selected controller contract: `SELECTED_CONTROLLER.json`
- Final baseline/selected panel and all gates: `outputs/data/engineering_roll_screen.json`
- Six-view rerun metrics and Robot sentinels: `outputs/data/engineering_roll_demo_audit.json`
- Contact sheet: `outputs/figures/engineering_roll_pg_p300_l125_contact_sheet.png`
- Runner SHA-256: `C23953C657683F44A4FA4EE7BB0B9839209CCA590A85C830A3E70730B102F1F3`
- Flat source XML SHA-256: `3815AB6CC8F6BC60834A9A85C246C19DC6B9FB94BD18DF893E072A0345132520`
- Rough source XML SHA-256: `33F2750A751719090E8925489641AE7641F879F740CCECC380AEB8F48643BF2F`

## Limitation and next gate

This is an engineering controller result on one deterministic flat surface and one deterministic smooth rough terrain. It is not yet a species-level dynamic calibration and should not be treated as proof across arbitrary contact patterns. The next experiment should run the selected controller through seed-locked initial-roll and random-contact matrices while retaining the current plant/gait contract and explicitly monitoring tip stop fraction, front-seven roll, contact force moment arms, and displacement.
