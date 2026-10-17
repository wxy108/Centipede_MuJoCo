# Historical gait audit and rejected-search diagnosis

Date: 2026-08-31  
Scope: abstract/biological Centipede Model only.  Centipede Robot Step 18/19 is
read-only history and remains outside the experiment.

## Authority order

1. Latest PI judgment: the 2026-08-31 displacement-grid gait is visually
   unnatural and rejected.
2. Frozen live MuJoCo model and replayed trajectories.
3. The accepted clean flat profile selected immediately before the rejected
   search.
4. Earlier Model and IsaacLab/RL target amplitudes as transfer anchors only.
5. Old reports and labels such as “winner” or “global optimum”.

## How the two project lines reached the current task

### Physical Centipede Robot

The Robot random-contact program varied physical segment number and used
fixed-window/discrete leg-pitch states.  Later work separated numerical root
and contact-order bias from real fore-aft polarity.  Its current physical
wave demos use step-function stance/swing groups defined over body-wave
half-wave caves.  That discrete controller is not the Centipede Model's
continuous sinusoidal leg controller and must not be imported into this task.

### Abstract Centipede Model

The first separated Model erasure experiments used the FARMS/collision model,
integer wave numbers 1–9, and approximate target amplitudes:

- body yaw: 31.40 deg;
- leg yaw: 18.75 deg;
- leg pitch: 26.36 deg.

The IsaacLab/RL controller instead carries YAML anchors of 0.4/0.6/0.4 rad
(22.92/34.38/22.92 deg), with 1 Hz clock and wave 3 by default.  Its verified
flat/rough map later found flat terrain favoring wave 1.5 and rough terrain
favoring a region near wave 3–4.5.  The RL action interface historically used
0.3–1.5 amplitude scales, but amplitude and frequency were normally fixed
when studying stiffness/damping; these bounds are context, not a MuJoCo
optimum.

After armature removal and mass/inertia correction, the MuJoCo Model initially
used an aggressive flat recovery profile: body 17.27 deg, leg yaw 41.24 deg,
leg pitch 36.90 deg.  That profile was explicitly superseded during clean
roll tuning.  The accepted flat profile retained body/leg wave coupling at
1.5 and reduced leg pitch to 19.37 deg while keeping the controller open loop.

## Why the rejected search failed

The rejected search used:

- integer waves 1–9, omitting the canonical flat wave 1.5 and all half-wave
  values from the verified RL flat/rough map;
- body yaw 8–32 deg;
- leg yaw 24–60 deg;
- leg pitch 12–48 deg;
- a coarse 7×7×7 grid;
- forward displacement after stability/contact gates as the sole objective.

This space was not merely “small”; it was biased.  It excluded the old Model
leg-yaw anchor at 18.75 deg and all near-zero amplitudes, while placing a large
fraction of samples in aggressive swing regions.  Several selected rows hit
the 60 deg leg-yaw boundary.  The accepted clean profile was not registered as
a mandatory anchor.

The gates detected flips, roll, jumping, penetration, and some contact
failures, but did not measure swing height, joint tracking distortion,
fundamental-vs-high-frequency motion, duty factor, or visually excessive leg
sweep.  This let simulator-exploiting high-amplitude gaits rank first.

## Matched trajectory evidence

All values below were replayed on the same frozen locked-head MuJoCo model for
6 s with the same 1 Hz sinusoidal CPG.

| Case | Leg yaw | Leg pitch | Displacement | Pitch speed P99 | Swing height P95 | Pitch tracking RMS |
|---|---:|---:|---:|---:|---:|---:|
| accepted clean, wave 1.5 | 41.24° | 19.37° | 133.0 mm | 4.09 rad/s | 2.83 mm | 7.51° |
| Isaac YAML anchor, wave 1.5 | 34.38° | 22.92° | 139.1 mm | 5.51 rad/s | 4.26 mm | 7.50° |
| old Model erasure anchor, wave 3 | 18.75° | 26.36° | 86.1 mm | 5.36 rad/s | 4.95 mm | 9.21° |
| rejected wave 3 | 60° | 42° | 156.9 mm | 16.23 rad/s | 8.68 mm | 10.89° |
| rejected wave 5 | 60° | 42° | 121.1 mm | 24.07 rad/s | 7.26 mm | 16.78° |
| rejected wave 7 | 54° | 48° | 111.2 mm | 28.54 rad/s | 7.61 mm | 21.18° |

Leg-yaw response gain for the rejected cases is approximately 0.99–1.03.
Therefore the visually excessive yaw sweep is commanded, not a mass-matrix or
damping overshoot.  Pitch behaves differently: high-wave rejected cases show
large contact-driven tracking errors, high speeds, and 8–11 velocity
reversals/cycle instead of the roughly 5–6 seen in the accepted/historical
anchors.  The bad gait is a combination of commanded over-sweep and
contact-distorted pitch motion.

## Demo failure

The rejected renderer enabled every geom group.  Collision group 0 and visual
group 2 contain coincident surfaces, so they were rendered simultaneously and
produced gray flicker/z-fighting, especially at the locked head.  It also
recolored only collision geoms during erasure.  Those are not the intended
visible surfaces, so the physical contact channel was disabled but the event
was visually unmarked.

The replacement renderer must:

- hide collision meshes while leaving the ground visible;
- render only the visual subtree for the animal;
- recolor the complete visible left/right leg subtree during erasure;
- show a persistent color legend and active segment IDs;
- use matched exposure, camera distance, resolution, and focal target across
  all candidates;
- provide front, side, oblique, and local leg/foot zoom views.

## Consequence for the new search

The old winners are invalid.  Search must be broad, but feasibility must be
defined in realized kinematics rather than by narrow input bounds.  A broad
screen will include near-zero amplitudes and values beyond every historical
anchor, followed by anchor-calibrated naturality gates and adaptive refinement.
Forward displacement is compared only after every gate passes.

