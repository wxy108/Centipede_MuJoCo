# Centipede Model — distal tip compliance redesign

Date: 2026-08-27

Scope: biological/template **Centipede Model** only. This experiment is isolated under `Centipede_MUJOCO-main`; it does not use or overwrite Centipede Robot Step 18/19 data.

## Outcome

Selected candidate: `tip_k4p5e-6_r38_bodyk6`.

The previous per-leg chain

`base yaw -> base pitch -> long proximal roll (_2) -> knee pitch (_3)`

was replaced by

`base yaw -> base pitch -> rigid compound seg1 + seg2 -> short passive tip roll`.

This matches the modeling intent that compliance belongs at the foot tip, while avoiding an extra articulation joint. Seg1 and seg2 are separate capsule geometries in one rigid body; therefore no fixed joint is created and no STL is required. This representation is preferable for MuJoCo-to-Isaac migration because the rigid compound is explicit, collision remains primitive-based, and inertias remain geometry-derived.

## Geometry and articulation evidence

- Seg1 centerline: 3.5 mm
- Seg2 centerline: 2.7 mm
- Tip centerline: 0.8 mm
- Total leg centerline: 7.0 mm, unchanged
- Fixed knee bend: 30 degrees (150-degree interior angle)
- Old compliant lever: 3.875 mm
- New compliant lever: 1.175 mm
- Compliant lever reduction: 69.677%
- Removed old `_2` and `_3` joints: 76 total
- Added distal tip roll joints: 38 total
- Net generalized-coordinate reduction: 38 (`nq` 218 -> 180; `nv` 217 -> 179)
- Actuators: 211 -> 135
- Old q2/q3 sensors removed: 190
- Nonzero armatures: 0

No artificial mass or inertia was inserted. Splitting the old two-capsule leg into three capsules would otherwise add an extra pair of capsule endcaps, so per-geometry density was redistributed to preserve the original combined leg mass. MuJoCo computes the body inertias from those actual geometries and assigned masses.

- Old total model mass: 0.00253603109178 kg
- New total model mass: 0.002536031091780036 kg
- Median distal-tip effective inertia: 1.15552e-13 kg m^2
- Median distal-tip mass-matrix diagonal: 1.58569e-13 kg m^2

## Kinematic probe

For an equal +/-30-degree roll sweep:

- Old long proximal `_2` roll foot-height peak-to-peak change: 0.259576 mm
- New short distal tip roll foot-height peak-to-peak change: 0.078710 mm
- Reduction: 69.677%

The decrease matches the compliant-lever ratio. This is the direct evidence that the new roll no longer converts a long knee-below lever into excessive whole-body height modulation.

## Selected passive parameters

- Tip roll stiffness: 4.5e-6 N m/rad
- Tip roll damping: 1.15376e-9 N m s/rad
- Damping ratio used to derive damping: 0.8
- Tip roll soft range: +/-38 degrees
- Passive body-roll stiffness: `k6`, 1.5 times the previous accepted `k4` diagnostic plant
- Active body-roll actuator: disabled
- Flat body/leg wave numbers: 1.5 / 1.5
- Rough body/leg wave numbers: 3.0 / 3.0
- Relative body/leg phase: 0 degrees

The stiffer-tip branch was rejected rather than selected: at tip stiffness 1.0e-5 N m/rad, rough-terrain motion became unstable/backward (about -15.1 mm displacement and about 175 degrees world-roll P95). This indicates contact torque was being transferred into the free root instead of providing useful support.

## Six-second validation

| Terrain | Forward displacement | Path efficiency | Tip roll P95 | Tip stop fraction | Local body-roll P95 | World body-roll P95 | Contact penetration P95 | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Flat | 144.32 mm | 0.937 | 27.68 deg | 0.752% | 3.25 deg | 21.02 deg | 0.0608 mm | all gates pass |
| Rough | 94.87 mm | 0.994 | 25.51 deg | 0.779% | 2.28 deg | 13.25 deg | 0.0182 mm | all gates pass |

Relative to the prior passive-body-roll `k4` baseline, displacement improved from 131.91 to 144.32 mm on flat ground (+9.4%) and from 76.80 to 94.87 mm on rough terrain (+23.5%). A 20% tip-stiffness reduction (`4.0e-6`) was rejected because it exceeded the tip-angle/stop-use gates. The selected 10% reduction increases compliance while preserving the original gates.

## Flat low-wave roll and feedback-force audit

The large flat-ground roll is not caused by a new mass or inertia-matrix error:

- Old and new total mass: both 2.536031 g within floating-point precision.
- Old root-roll effective inertia: 5.553956e-10 kg m^2.
- New root-roll effective inertia: 5.553950e-10 kg m^2.
- New/old ratio: 0.9999989.
- Nonzero armatures: zero in both models.
- With actuation disabled, roll P95 is only 1.47 degrees.
- The old q2/q3 flat coupled-wave plant already had 21.81 degrees world-roll P95 before the distal-tip redesign.

The roll is a real controller/contact effect. With the required coupled flat wave numbers 1.5/1.5:

- Model weight: 24.878 mN.
- Total vertical support mean: 24.842 mN (correct time-average weight support).
- Total vertical support P95: 34.650 mN = 1.393 body weights.
- Total vertical support maximum: 57.999 mN.
- System COM vertical peak-to-peak motion: 0.828 mm.
- Left/right support imbalance absolute P95: 0.327, where imbalance is `(R-L)/(R+L)`.
- Contact roll moment about the head-forward axis P95: 119.5 micro-N m.
- Mean number of contacting feet: 13.91.
- Mean load carried through explicit foot geoms: 93.95%.

The selected model's loaded-foot normal feedback force is 1.310 mN median, 4.277 mN P95, 5.920 mN P99, and 8.019 mN maximum. Leg actuator torque is 13.41 micro-N m P95 and 35.83 micro-N m maximum.

For diagnosis only, changing both body and leg wave numbers together from 1.5 to 3 while retaining flat leg amplitudes reduced world-roll P95 from 21.02 to 11.76 degrees, left/right support imbalance P95 from 0.327 to 0.110, vertical COM motion from 0.828 to 0.293 mm, and contact roll moment P95 from 119.5 to 22.7 micro-N m. This ablation is not a selected controller and does not relax the requirement that body and leg wave numbers remain coupled. It identifies sparse low-wave support distribution as the dominant roll source.

These are deterministic six-second controller validations, not biological identification results. The selected candidate is a template-model plant for the next contact/terrain validation stage.

## Reproducible artifacts

- Runner: `run_tip_compliance.py`
- Full model manifest: `outputs/data/tip_model_manifest.json`
- Full sweep and gates: `outputs/data/tip_compliance_sweep.json`
- Demo audit: `outputs/data/tip_demo_audit.json`
- Roll/force audit runner: `run_roll_force_audit.py`
- Roll/force audit data: `outputs/data/flat_roll_force_audit.json`
- Roll/force plot: `outputs/figures/flat_roll_force_audit.png`
- Flat selected XML: `models/centipede_flat_tip_k4p5e-6_r38_bodyk6_distal_roll.xml`
- Rough selected XML: `models/centipede_rough_tip_k4p5e-6_r38_bodyk6_distal_roll.xml`
- Six videos: `outputs/videos/selected_tip_tip_k4p5e-6_r38_bodyk6_*.mp4`
- Contact sheet: `outputs/figures/selected_tip_tip_k4p5e-6_r38_bodyk6_contact_sheet.png`

All six videos were decoded successfully as H.264, 1280x720, 30 fps, 6.03 seconds.

## Model/Robot isolation check

The final run re-read the protected Robot files after all model builds, sweeps, and demos. Their SHA-256 hashes remained exactly unchanged:

- Step 18 run configuration: `847BEE7492D75CA05FFADC40B4664C8CE02EEDB1D5E16B6683FF2EA9BD6CCF85`
- Step 18 trials: `C46F9B1C1F932971CE37EA8AD6C5E9892D01E238E7839867F135D1A507A8E3CC`
- Step 19 run configuration: `FF007DEC8796DD0E8F0AF6D58EFFFDF0287C1474E171CD7E086CED88A38EFBC8`
- Step 19 trials: `1A4AD813787B68A7EA409C9C261F3A977453AB61035E44B479B8A9A2E7CCEC48`

The Robot cleanup task had independently moved these files under its archive tree; this Model experiment only follows their current locations and did not modify them.

## Remaining limitation / next gate

The new topology solves the specific long-lever compliance problem and passes the current flat/rough deterministic gates. It does not by itself establish species-level tip stiffness. The next high-value step is to reduce low-wave support-force imbalance while keeping body/leg wave numbers equal, for example through a zero-net-support leg-amplitude envelope or passive load sharing, and then run contact randomization around the selected plant without changing the gait wave-number constraint.
