# Centipede Model — MuJoCo template-fidelity Stage 1

Date: 2026-08-26

## Scope and isolation

This experiment belongs only to the biologically inspired/abstract **Centipede Model**. It does not use Centipede Robot dynamics, controller gains, Step 18/19 trials, or Robot outputs as optimization inputs. The four Robot Step 18/19 files were read only as sentinels and retained their original SHA-256 hashes.

Model experiment root:

`C:\Users\wxy22\Documents\Centipede_MUJOCO-main\experiments\mujoco_template_fidelity_stage1_2026-08-26`

The Robot paths under `C:\Users\wxy22\Documents\Centipede_robot_MUJOCO\exp_data\reliability` were not written.

## Stage-1 accepted candidate

The candidate passes the current 6 s flat/rough feasibility gates. This is a numerical/contact/controller baseline, not yet a claim of species-level validation.

- Base plant: geometry-derived inertia, total mass 0.00253603109178 kg, armature exactly zero on every DOF.
- Compiled dimensions: `nq=218`, `nv=217`, `nu=211`.
- Timestep: 0.00025 s; `implicitfast`.
- Foot/ground contact: `solref="0.0005 1.5"`, `solimp="0.95 0.99 0.0001"`.
- `_2` effective distal/leg-plane roll: 38 passive joints, active `_2` servo disabled at runtime, explicit stiffness `3.0e-5 N m/rad`, damping `1.8e-7 N m s/rad`, nominal stop ±30 degrees.
- Every non-`_2` joint remains at zero passive stiffness and damping; no hidden passive regularization was introduced.
- Body-yaw traveling-wave amplitude: 0.55 times the inherited value, i.e. 0.3014522884407344 rad.
- First body-yaw joint: zero target for the whole trial. This keeps head and first body segment approximately straight; the first leg pair retains normal yaw and pitch targets.
- Flat model SHA-256: `DC011F3971DCA244EACB5BE51D39D7CA741384ABFCD0A9C2CDEE0E5438923059`.
- Rough model SHA-256: `F21E83819F736A46CD8BEFB85C9034899408C4AAC59ECD1DA7938A17A830D6D8`.

The positive body masses span `2.1508814645819043e-7` to `1.4609461373232713e-4 kg`; positive principal inertias span `1.4639436968310588e-14` to `1.286644999766112e-9 kg m^2`. These values compile without inertia-matrix warnings. They remain template estimates derived from geometry and assumed density, and have not yet been identified against a measured *Scolopendra subspinipes* specimen.

## Accepted 6 s results

Evaluation starts after the 1.5 s warm-up. Thresholds are feasibility gates, not a scalar reward.

| Metric | Flat | L60/A12 rough | Gate |
|---|---:|---:|---:|
| Forward displacement | 80.442 mm | 81.173 mm | ≥40 / ≥25 mm |
| Forward speed | 0.174 BL/cycle | 0.176 BL/cycle | positive |
| Path efficiency | 0.9845 | 0.9845 | diagnostic |
| Minimum root upright cosine | 0.8435 | 0.8935 | ≥0.75 |
| Maximum spine tortuosity | 1.0536 | 1.0858 | ≤1.30 |
| `_2` p95 absolute angle | 6.019° | 10.017° | ≤27° |
| `_2` maximum absolute angle | 30.509° | 31.391° | ≤35° transient |
| Samples outside nominal ±30° | 0.1936% | 0.0874% | ≤0.25% |
| `_2` maximum speed | 5.336 rad/s | 23.478 rad/s | ≤100 rad/s |
| Foot penetration p95 | 0.0411 mm | 0.0214 mm | ≤0.18 mm |
| Contact-normal single-step jump p95 | 0° | 8.536° | ≤15° |
| MuJoCo warnings / non-finite state | none | none | none |

Evidence: `outputs/data/head_first_body_hold_probe.json` and `outputs/data/demo_audit.json`.

## Contact and terrain evidence

The exact rough collision surface is a 0.30 m, 1536×1536 float32 heightfield. Cell spacing is 0.1954397394 mm, giving 4.605 cells across the 0.9 mm foot diameter. The prior 3 m / 1536 field had 1.95 mm cells, so one foot was smaller than one collision cell.

The selected 0.25 ms contact profile passed all three single-foot cases. With a supported mass of 0.1268016 g, the settled contact force was 1.2439233 mN in every case. Static settled penetration p95 was 0.0002996 mm; maximum penetration from a 1 mm drop was 0.0135010 mm. These are numerical calibration results, not claims about biological tarsal compliance.

The previous audit's contact-normal rate is downgraded: it retained a mutable view of `contact.frame` across `mj_step`, which can make a flat plane appear to have changing normals. Stage 1 copies the frame before the next step and uses the timestep-independent single-step normal angle for acceptance. The old values must not be reused.

## Controller evidence and rejected branches

1. Range-only `_2` soft limits failed: joints exceeded the nominal range by roughly 3–10 times and hard-stop impacts did not constitute a compliant plant.
2. Passive `_2` stiffness of 2.5–10 µN m/rad failed: maximum angles remained 68–94 degrees, although forward displacement was positive.
3. Increasing explicit `_2` recovery to 20–40 µN m/rad plus the calibrated contact reduced p95 penetration below 0.126 mm and p95 `_2` angle below 20 degrees. The 30 µN m/rad candidate converged at 0.25 ms.
4. The initially rendered 1.0× body-wave candidate remained finite but curled into a C/U shape; its 6 s rough tortuosity was 1.683 and it failed the inherited ≤1.30 gate.
5. Scaling body wave alone to 0.45–0.75× did not solve rough-terrain curvature. At 0.55×, flat tortuosity was 1.056 but rough tortuosity was 1.426.
6. Holding only the first body-yaw target at zero, while leaving the first leg pair active, reduced 0.55× rough tortuosity to 1.086, raised path efficiency to 0.9845, and increased rough forward displacement from 52.91 to 81.17 mm. This identifies the prior high-gain/clipped first-joint heading target as the dominant long-horizon curling mechanism in this terrain.

## Demos

All final videos are 30 fps and approximately 6.03 s. They use the exact collision heightfield and the same deterministic trajectory for the three rough views.

- `outputs/videos/stage1_flat_q2pm30_restoring_k30_c18_b0p55_headhold_side.mp4`
- `outputs/videos/stage1_l60a12_local_q2pm30_restoring_k30_c18_b0p55_headhold_side.mp4`
- `outputs/videos/stage1_l60a12_local_q2pm30_restoring_k30_c18_b0p55_headhold_oblique.mp4`
- `outputs/videos/stage1_l60a12_local_q2pm30_restoring_k30_c18_b0p55_headhold_top.mp4`

The fifth-second visual QA sheet is `outputs/figures/demo_contact_sheet_final_headhold.png`. It shows a near-straight flat trajectory and a shallow rough-terrain body arc rather than the rejected C/U configuration.

## Robot Step 18/19 sentinels

| File | SHA-256 |
|---|---|
| Step 18 `run_configuration.json` | `847BEE7492D75CA05FFADC40B4664C8CE02EEDB1D5E16B6683FF2EA9BD6CCF85` |
| Step 18 `trials.csv` | `C46F9B1C1F932971CE37EA8AD6C5E9892D01E238E7839867F135D1A507A8E3CC` |
| Step 19 `run_configuration.json` | `FF007DEC8796DD0E8F0AF6D58EFFFDF0287C1474E171CD7E086CED88A38EFBC8` |
| Step 19 `trials.csv` | `1A4AD813787B68A7EA409C9C261F3A977453AB61035E44B479B8A9A2E7CCEC48` |

Every generated manifest, sweep, probe, and Demo audit reports `robot_sentinels_unchanged=true`.

## What is not yet validated

- The current acceptance set contains one deterministic L60/A12 terrain seed. It is not a withheld-terrain or random-contact generalization result.
- The template mass and inertia are geometry/density consistent but not fitted to specimen-level segment, leg, and tarsus measurements.
- `_2` compliance is an explicit engineering bracket, not an identified biological material constant.
- The 0.25 ms verification step is expensive and should not automatically become the RL training timestep. A separate error-versus-throughput convergence study is required.
- The random-contact erasure definition has not been changed in Stage 1. Before using this candidate for that study, contact-event semantics and an unseen terrain/seed set must be frozen and audited independently.

Recommended Stage 2: freeze this candidate, create withheld flat/rough terrain seeds plus the existing random-contact schedule, and validate a vector of survival, directed displacement retention, upright/tortuosity, foot-contact duty factor, `_2` occupancy, penetration, and energy metrics. Do not optimize or report a single scalar winner.
