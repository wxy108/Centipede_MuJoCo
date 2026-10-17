# Centipede Model body-roll wobble audit and controller refinement

> **SUPERSEDED / INELIGIBLE AFTER 2026-08-27 PI CONSTRAINT.** The candidate in this report decouples flat body wave `1.5` from leg wave `3.0` and applies a `+45 deg` relative leg phase. The PI has explicitly ruled out changing that coordination architecture at this stage. Preserve this report only as historical causal evidence; do not use its selected candidate as the canonical Centipede Model. The replacement audit is `../mujoco_template_passive_body_roll_2026-08-27/PASSIVE_BODY_ROLL_REPORT.md`.

Date: 2026-08-27  
Scope: **Centipede Model only** (`Centipede_MUJOCO-main`). This experiment does not use, rewrite, or reinterpret Centipede Robot Step 18/19 data.

## Result

The user's visual diagnosis was correct. The accepted q2-compliance baseline had two different roll problems:

1. large relative motion at the 20 serial body-roll joints; and
2. a whole-chain axial roll mode in which the body could behave like a torsionally stiffer rod but still rock about the head-to-tail direction.

The selected candidate is `lw3_php45_c3_g4_r8_q2g28p5`. It suppresses both modes on the deterministic 6 s flat and rough validation cases while preserving the previously accepted zero-armature mass/inertia plant and q2 interior compliance.

This is a **template-model engineering result**, not a claim that *Scolopendra subspinipes* has literally zero body roll or that the thresholds below are biological confidence intervals. On rough ground, a small bounded roll response to local terrain attitude remains physically appropriate.

## Selected change

### Body-roll plant and controller

- 20 body-roll joints remain present.
- Armature remains exactly `0` on every DOF.
- Passive body-roll stiffness and damping remain exactly `0`; no hidden XML spring was introduced.
- Explicit body-roll range is ±8°. The selected trajectories never touch this stop: contact fraction is `0` on both flat and rough tests, and observed local maxima are only 4.19° and 3.31°.
- Active zero-target roll impedance is increased from the accepted controller baseline:
  - `Kp = 4.638950207330073e-4 N·m/rad` (4× baseline);
  - `Kv = 8.05431115913186e-7 N·m·s/rad` (2× baseline, following square-root damping scaling).
- A filtered projected-gravity correction is distributed over the head and segment leg-pitch targets:
  - head gain `-2.0`;
  - segment gain `-1.5`;
  - filter time constant `0.12 s`;
  - correction clip `0.4 rad`.

This active correction matters because roll-gain-only candidates reduced inter-segment torsion but did not remove the global rigid-body rocking mode.

### Flat-ground gait

- Body yaw wave remains `1.5`.
- Leg spatial wave is decoupled to `3.0`.
- Leg phase is shifted by `+45°`.
- Leg-pitch amplitude is kept at its full accepted value, `0.644 rad` (36.9°); it is not reduced.

This preserves leg lift and propulsion better than the amplitude-reduction candidates. Rough-terrain gait settings are not replaced by the flat-only wave/phase change.

### q2 compliance

- All 38 q2 joints keep the accepted `bounded_k20_c18` interior plant:
  - stiffness `2e-5 N·m/rad`;
  - damping `1.8e-7 N·m·s/rad`;
  - armature `0`.
- The nominal anatomical envelope remains ±30°.
- An explicit guard stop begins at ±28.5°. This is recorded in the candidate name, generated XML, model manifest, static audit, sweep data, and Demo audit; it is not a hidden stiffness change.
- Flat maximum is 30.121° (0.121° soft-stop penetration); rough maximum is 29.245°.

## Quantitative comparison

Metrics are evaluated after the 1.5 s warm-up over the same 6 s rollouts.

| terrain | metric | accepted q2 baseline | selected | change |
|---|---:|---:|---:|---:|
| flat | local body-roll P95 | 10.270° | 2.638° | -74.3% |
| flat | cumulative chain-roll span P95 | 45.304° | 14.203° | -68.7% |
| flat | chassis world-roll P95 | 22.746° | 9.108° | -60.0% |
| flat | head world-roll P95 | 20.461° | 5.856° | -71.4% |
| flat | forward displacement | 83.543 mm | 78.953 mm | 94.5% retained |
| flat | path efficiency | 0.9861 | 0.9537 | 96.7% retained |
| rough | local body-roll P95 | 7.565° | 2.203° | -70.9% |
| rough | cumulative chain-roll span P95 | 32.159° | 12.754° | -60.3% |
| rough | chassis world-roll P95 | 16.069° | 8.571° | -46.7% |
| rough | head world-roll P95 | 18.325° | 6.325° | -65.5% |
| rough | forward displacement | 81.208 mm | 81.526 mm | 100.4% retained |
| rough | path efficiency | 0.9871 | 0.9789 | 99.2% retained |

Additional selected-candidate checks:

| check | flat | rough |
|---|---:|---:|
| minimum root upright cosine | 0.913 | 0.931 |
| maximum spine tortuosity | 1.067 | 1.153 |
| mean contacting feet | 12.07 | 14.45 |
| maximum body-roll control torque | 3.407e-5 N·m | 2.711e-5 N·m |
| q2 P95 / maximum | 12.10° / 30.12° | 14.71° / 29.24° |
| body-roll stop contact fraction | 0 | 0 |

The lower flat contact count is consistent with the intended restoration of leg lift; it remains inside the predeclared 8–26-feet feasibility band.

## Causal screens and rejected shortcuts

- **Tight stop alone:** ±8° with the old gain contacted the stop during 29.4% of evaluated frames. Rejected because the apparent stability came from repeated stop support.
- **Roll gain alone:** 2×–8× gain strongly reduced local joint roll but left world/head roll near 16–22°. This exposed the global axial rocking mode.
- **Head-only projected gravity:** reduced head roll but displaced the twist downstream into the chain. Rejected.
- **Leg-pitch amplitude reduction:** reduced roll, but 0.75× and 0.5× variants lost propulsion or returned to the earlier many-feet-dragging behavior. Rejected.
- **Leg wave change alone:** wave 3.0 was the correct local roll region, but initially retained only about 87% flat displacement and had q2 overshoot.
- **Combined controller and gait:** wave 3.0, +45° phase, full leg amplitude, distributed projected-gravity correction, and 4×/2× roll impedance gave the first candidate passing the complete flat/rough gate set.
- **5× upper bracket:** also passed and reduced remaining roll by only roughly 0.5–1.8°, while flat displacement retention fell from 94.5% to 91.3%. The selected 4× candidate is the lowest tested sufficient gain.

Raw causal-screen files are retained under `outputs/data/body_roll_screen_*.json`; rejected candidates were not deleted.

## Static model audit

Both selected XML files compile in MuJoCo with:

- total mass `0.00253603109178 kg`;
- 212 bodies including world;
- 217 velocity DOFs and 211 actuators;
- all DOF armatures exactly zero;
- all non-world principal inertias finite and positive;
- principal-inertia range `1.46394e-14` to `1.28664e-9 kg·m²`;
- timestep `0.00025 s`;
- no non-q2 passive joint stiffness or damping.

No body mass, body inertia, armature, q2 interior stiffness, or q2 interior damping was modified in this body-roll experiment. The improvement comes from an explicit controller/gait refinement plus a declared q2 guard stop.

## Demo and visual QA

Six 1280×720, 30 fps, 6.03 s videos were regenerated from the selected models: flat and rough, each in side, oblique, and top views. The Demo rerun produced six identical-view metric rows; every row passed the full body-roll, q2, posture, contact, and performance gates.

Frames at 0.75, 2.0, 3.5, and 5.25 s were also inspected in all three views. The prior periodic whole-chain axial wobble is absent from the sampled flat sequence. Rough-terrain residual tilt follows terrain attitude and stays bounded. It should not be reported as zero roll.

## Evidence and reproducibility

- Runner: `run_body_roll.py`
- Selection and baseline rows: `outputs/data/body_roll_sweep.json`
- Six-view rerun metrics and Robot sentinels: `outputs/data/demo_audit.json`
- Compiled-model and XML invariants: `outputs/data/static_model_audit.json`
- Video integrity and visual QA: `outputs/data/video_metadata.json`
- Generated model manifest: `outputs/data/body_roll_model_manifest.json`
- Selected flat XML SHA-256: `5C7F33F513281D948E2CC239FB380034E94F6796E9D46650D441BDC975C19067`
- Selected rough XML SHA-256: `C6A714F5658FDADABB006582255F51E6F6D6D5DEADED59430088D6FF1785EA9D`

## Model/Robot isolation

Robot Step 18/19 files were treated only as read-only sentinels. Their SHA-256 values were identical before and after the sweep and again before and after Demo rendering:

- Step 18 run configuration: `847BEE7492D75CA05FFADC40B4664C8CE02EEDB1D5E16B6683FF2EA9BD6CCF85`
- Step 18 trials: `C46F9B1C1F932971CE37EA8AD6C5E9892D01E238E7839867F135D1A507A8E3CC`
- Step 19 run configuration: `FF007DEC8796DD0E8F0AF6D58EFFFDF0287C1474E171CD7E086CED88A38EFBC8`
- Step 19 trials: `1A4AD813787B68A7EA409C9C261F3A977453AB61035E44B479B8A9A2E7CCEC48`

## Remaining limitation and next experiment

The result is validated on one deterministic flat case and one deterministic smooth rough-terrain case. The next evidence step should be a seed-locked terrain/initial-attitude matrix, followed by the separate random-contact study. Neither is silently folded into this result, and neither should use Centipede Robot Step 18/19 data.
