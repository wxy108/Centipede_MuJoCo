# Flat-ground leg-lift and warm-up refinement

Date: 2026-08-26  
Scope: biologically inspired Centipede **Model** only. The physical Centipede
Robot and its Step 18/19 outputs are out of scope and were not modified.

## Result

The former flat gait was moving its leg-pitch joints but was mostly scraping:
median foot apex clearance was 0.254 mm, only 1.28% of feet were above 0.5 mm
at a typical instant, chassis contact occurred during 31.3% of samples, and
forward speed was 0.199 BL/cycle.

The retained flat/w=1.5 profile scales the clean-RL leg-yaw amplitude by 2.2
and leg-pitch amplitude by 1.4. The resulting absolute amplitudes are 0.7198
rad yaw and 0.6440 rad pitch. Knee DC remains 30 degrees; impedance, mass,
geometry-derived inertia, exact armature=0 and the 1 ms physics timestep are
unchanged.

In the final 9.6 s run (evaluation from 1.5 s):

| Metric | Previous flat diagnostic | Final flat profile |
|---|---:|---:|
| forward speed | 0.199 BL/cycle | 0.293 BL/cycle |
| forward displacement | 0.0918 m / 4.5 s eval | 0.2434 m / 8.1 s eval |
| path efficiency | 0.981 | 0.992 |
| median foot apex clearance | 0.254 mm | 1.248 mm |
| feet above 0.5 mm, mean instantaneous fraction | 1.28% | 19.91% |
| mean foot contact duty | 85.91% | 63.91% |
| chassis-contact sample fraction | 31.33% | 0.86% |
| mean fraction of non-foot leg geoms touching | not recorded | 6.87% |
| maximum spine tortuosity | 1.245 | 1.299 |
| peak active-joint speed | 29.09 rad/s | 29.09 rad/s |

The feet therefore have a measured aerial phase; this is not inferred from
joint angles or video alone. With 38 legs, at least one non-foot leg geom is
still in contact at every sample, so the useful quantity is the per-geom
fraction (6.87%), not the misleading any-contact boolean.

## First-body warm-up

The flat profile uses a 1.0 s settle and the existing 0.125 s per-segment DCT
activation. `joint_body_1` does not receive the traveling body wave. Its
world-yaw correction is released with a raised-cosine blend from 1.0 to 1.25
s, while the first left/right leg-pitch joints retain their normal gait wave.

Measured startup behavior:

- During 0-1.0 s, maximum head-yaw error is 1.16 degrees and maximum absolute
  first-body yaw is 0.39 degrees.
- Through the first leg-pitch activation window, maximum head-yaw error is
  2.99 degrees and maximum absolute first-body yaw is 7.12 degrees.
- First left/right leg-pitch target range is 73.80 degrees peak-to-peak; actual
  ranges are 73.56 and 73.65 degrees.

This meets the intended startup semantics: the head/front body remain roughly
straight, but the first leg pair is not frozen.

## Terrain-specific retention

The aggressive flat profile is not applied globally. Directly applying it to
L90/A8 violated the tortuosity and qvel gates. Rough terrain and w=3 random
contact therefore use 0.95 of the clean-RL leg-yaw and leg-pitch amplitudes,
with the same optimized impedance and full head-yaw reference from reset.

| Condition | forward speed | path speed | efficiency | min upright cos | max tortuosity | peak qvel | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| L90/A8, w2 | 0.229 | 0.241 | 0.948 | 0.807 | 1.263 | 61.78 | pass |
| L60/A12, w3 | 0.155 | 0.170 | 0.912 | 0.678 | 1.147 | 65.46 | pass |
| random q=0.3 seed 0 | 0.151 | 0.158 | 0.955 | 0.833 | 1.096 | 29.09 | pass |
| random q=0.3 seed 1 | 0.148 | 0.156 | 0.953 | 0.844 | 1.091 | 29.09 | pass |
| random q=0.3 seed 2 | 0.145 | 0.152 | 0.951 | 0.861 | 1.091 | 29.09 | pass |

Every retained run is finite, has no MuJoCo warning, keeps minimum upright cos
above 0.259, maximum spine tortuosity at or below 1.30, and peak active-joint
speed below 100 rad/s.

## Reproducible artifacts

- Final parameters: `config/optimized_controller.json`
- Controller and diagnostic implementation: `controller_optimization.py`
- Final flat evidence: `outputs/data/optimized_flat_leg_lift_side.json`
- L90 selection: `outputs/data/rough_l90_y095_p095.json`
- L60 regression: `outputs/data/final_profile_l60a12_regression.json`
- Random-contact regressions: `outputs/data/final_profile_random_seed0.json`
  through `final_profile_random_seed2.json`
- True-side demo: `outputs/videos/optimized_flat_leg_lift_side.mp4`
- Oblique demo: `outputs/videos/optimized_flat_leg_lift_oblique.mp4`

