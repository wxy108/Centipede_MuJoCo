# FARMS Centipede: Independent Short Contact-Erasure Pipeline

## Outcome

This directory contains a refined version of the detailed FARMS MuJoCo
collision model and a complete flat-ground reliability experiment.  The
morphology is fixed at 19 segment-pairs.  The experiment varies only the
integer spatial wave number and the independent contact-erasure probability.
There is no correlated-noise condition and no terrain variation in this stage.

The main result is twofold:

1. the `w=3` gait, whose phase groups contain 3--4 homologous segment-pairs,
   retains high forward displacement under independent short erasures; and
2. redundancy count alone is not sufficient: `w=2` and `w=4` have identical
   phase-group sizes but very different locomotion robustness because their
   spatial phase ordering and baseline whole-body dynamics differ.

## Model refinement

Source model:

`C:\Users\wxy22\Documents\Centipede_MUJOCO-main\models\farms\centipede.xml`

Refined model:

`models\farms\centipede_reliability_refined.xml`

The source file is not overwritten.  `refine_model.py` makes and verifies the
refined artifact.

| Quantity | Source | Refined |
|---|---:|---:|
| total mass | 0.002536031092 kg | unchanged |
| bodies / joints / geoms / actuators | 232 / 288 / 425 / 249 | unchanged |
| active controlled joints | 211 | unchanged |
| global armature | `1e-4` kg m^2 | `1e-5` kg m^2 |
| active-joint passive stiffness/damping | mixed, including duplicate body-roll PD | zero |
| 38 passive leg-roll hinges | spring/damper present | preserved |
| 38 passive foot hinges | damper present | preserved |
| physics step | 0.5 ms | 1 ms |

Strict zero armature is not numerically admissible for this MuJoCo collision
chain.  The model contains extremely light serial coordinate bodies; zero
armature produces a rank-deficient Hessian, and values through `1e-6` become
unstable when the gait and contacts start.  `1e-5` was the smallest tested
stable order.  It is a solver regularizer, not a claimed physical motor rotor
inertia.  It is ten times smaller than the historical model value.

The refined controller uses the latest clean Isaac-Lab impedance families and
velocity-error tracking.  Gains, gait amplitudes, 1 Hz frequency, terrain,
evaluation duration, and morphology are locked across all wave conditions.
In particular, the wave-dependent gait map is deliberately disabled so that
it cannot confound the redundancy experiment.

## Phase groups and redundancy

For segment-pair index (i\in\{0,\ldots,18\}) and integer spatial wave number
(w), the prescribed phase is

\[
\phi_i(w)=\frac{2\pi w i}{18}\pmod{2\pi}.
\]

The phase group associated with phase label (g) is

\[
\mathcal G_g(w)=\{i:\phi_i(w)=g\},
\qquad m_g=|\mathcal G_g(w)|.
\]

The number of distinct phase groups is

\[
G(w)=\frac{18}{\gcd(18,w)}.
\]

Because the model contains both endpoints (i=0) and (i=18), group sizes
are not exactly uniform.

| wave (w) | group count (G) | group sizes (m_g) | mean (19/G) |
|---:|---:|---:|---:|
| 1 | 18 | 1--2 | 1.056 |
| 2 | 9 | 2--3 | 2.111 |
| 3 | 6 | 3--4 | 3.167 |
| 4 | 9 | 2--3 | 2.111 |

Therefore, wave number is not itself the redundancy count.  In particular,
`w=2` and `w=4` deliberately form a same-redundancy/different-ordering control.

## Independent short-erasure channel

A channel use is one nominal support opportunity of duration

\[
\tau_s=0.5\ \mathrm{s}.
\]

For segment-pair (i) and channel use (k),

\[
Z_{i,k}\sim\operatorname{Bernoulli}(q)
\]

independently over segments and uses.  When (Z_{i,k}=1), both complete leg
subtrees of segment (i) have their collision masks disabled for a random
short interval.  The visual geoms are simultaneously colored red.

The normalized burst duration (A=\Delta/\tau_s) obeys

\[
A\sim\operatorname{LogUniform}(0.02,1),
\qquad \mathbb E[A]=0.2505098,
\]

and, conditional on duration, the start offset is uniform over the remaining
part of the support window.  Thus a burst may be as short as 10 ms and never
exceeds a support window.  This is contact noise, not a mandatory full-window
signal dropout.

The event probability (q) and erased-time load are different quantities.
The marginal random-time occupancy is approximately

\[
\nu=q\,\mathbb E[A].
\]

Under an independent snapshot approximation, a complete blackout of phase
group (g) has probability (\nu^{m_g}), and

\[
P_{\mathrm{any}}^{\mathrm{snap}}
\approx 1-\prod_g\left(1-\nu^{m_g}\right).
\]

The measured blackout-time fractions closely track this formula; see
`outputs\figures\group_blackout_theory_vs_measured.png`.

## Experimental protocol

- flat ground only;
- wave numbers `1, 2, 3, 4`;
- (q\in\{0,0.05,0.1,0.2,0.3,0.4\});
- 8 independent seeds for each nonzero-(q) cell;
- one deterministic baseline for each wave;
- 8 s per trial, with performance measured from 2--8 s;
- 12 channel uses per segment-pair, 228 opportunities per trial;
- forward displacement is the 21 physical chassis-link centroid projected on
  the model's initial (+x) direction.

The task reliability reported in the figures is

\[
\widehat{\mathcal R}_{D,w}
=P\!\left(
|\widehat D-D_{0,w}|\le0.1|D_{0,w}|
\right),
\]

where each wave is compared with its own zero-noise baseline.

## Results

Baseline forward displacements over the 6 s evaluation window were:

| wave | baseline displacement |
|---:|---:|
| 1 | 0.0892 m |
| 2 | 0.0346 m |
| 3 | 0.1601 m |
| 4 | 0.0749 m |

Selected independent-noise results:

| wave | (q) | mean displacement | SD | mean retention | reliability |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.05 | 0.0006 m | 0.0618 m | 0.006 | 0.125 |
| 2 | 0.05 | 0.0651 m | 0.0315 m | 1.885 | 0.000 |
| 3 | 0.05 | 0.1550 m | 0.0074 m | 0.968 | 1.000 |
| 3 | 0.10 | 0.1517 m | 0.0112 m | 0.947 | 0.750 |
| 3 | 0.20 | 0.1444 m | 0.0148 m | 0.902 | 0.750 |
| 3 | 0.30 | 0.1299 m | 0.0456 m | 0.811 | 0.375 |
| 3 | 0.40 | 0.1269 m | 0.0349 m | 0.793 | 0.625 |
| 4 | 0.05 | 0.0112 m | 0.0395 m | 0.150 | 0.000 |

The nonmonotonic empirical reliability at high (q) is sampling uncertainty
with only 8 seeds, not evidence that higher noise improves reliability.

`w=2` can show more (+x) displacement after noise because its baseline gait
turns strongly; a perturbation may reduce that turn.  Such trials still fail
the matched-baseline reliability criterion when the change exceeds 10%.
Absolute displacement, relative retention, path length, and yaw drift are all
retained in the raw CSV so this projection effect is auditable.

Most importantly, phase-group blackout is well predicted by the independent
channel approximation, but locomotion performance is not determined by
blackout alone.  At low (q), `w=2` and `w=4` have almost no complete group
blackout yet respond very differently.  Partial contact losses propagate
through whole-body dynamics, making spatial phase ordering and baseline gait
quality essential state variables.

## Collision-restoration audit

Two `w=3, q=0.3` trials were logged at every 1 ms step.

| diagnostic | ordinary steps | restoration steps |
|---|---:|---:|
| root-linear speed jump, p95 | 0.0211 m/s | 0.0390 m/s |
| root-linear speed jump, max | 0.1284 m/s | 0.0927 m/s |
| active-joint speed jump, p95 | 0.2578 rad/s | 0.2580 rad/s |

18.4% of restorations occurred with at least one erased leg geom slightly
overlapping the floor.  Restoration increases the root-level p95 transient,
but neither the maximum root transient nor the joint-level p95 exceeds natural
contact events.  Therefore no additional lift-leg or smoothing variable was
introduced in this first short-noise pipeline.  The limitation remains
explicitly logged in `outputs\data\recovery_transition_summary.csv`.

## Output index

Core data:

- `outputs\data\short_erasure_trials.csv`: all 164 trials;
- `outputs\data\short_erasure_summary.csv`: cell statistics;
- `outputs\data\experiment_manifest.json`: frozen configuration and groups;
- `outputs\data\model_ab_audit.csv`: original/refined flat baseline;
- `outputs\data\recovery_transition_summary.csv`: restoration audit;
- `outputs\data\demo_manifest.csv`: video-linked trial metrics.

Core figures:

- `absolute_displacement_vs_q.png`;
- `retention_vs_q.png`;
- `reliability_vs_q.png`;
- `group_blackout_vs_q.png`;
- `group_blackout_theory_vs_measured.png`;
- `displacement_vs_realized_erased_load.png`;
- `wave_number_vs_redundancy.png`;
- `original_vs_refined_baseline.png`.

Demo videos:

- `baseline_w3_q000.mp4`;
- `short_erasure_w1_q030.mp4`;
- `short_erasure_w2_q030.mp4`;
- `short_erasure_w3_q030.mp4`;
- `short_erasure_w4_q030.mp4`.

## Reproduction

Use the `isaaclab_50` Python environment.

```powershell
python refine_model.py
python run_pipeline.py ab-audit
python run_pipeline.py sweep --waves 1,2,3,4 --q 0,0.05,0.1,0.2,0.3,0.4 --seeds 8 --workers 4
python diagnose_recovery.py
python run_pipeline.py demos
```
