# Independent short-contact-erasure rerun

## Status

This is the completed flat-ground rerun on the detailed FARMS MuJoCo centipede. It uses the validated local FARMS impedance/CPG controller, preserves the original `1e-4 kg m^2` joint armature, and varies only the integer spatial wave number and the independent short-erasure channel. Correlated erasure and rough terrain are outside this experiment.

The 164-trial sweep passed the preregistered dynamics gate: 0 unstable trials, 0 non-finite trials, and 0 robot self-contact trials. Across all runs, the largest absolute root roll was 8.55 deg, the largest absolute root pitch was 22.48 deg, the largest segment tilt was 23.05 deg, and the deepest dynamic floor contact was 2.30 mm.

## Controlled model delta

- Source: `C:\Users\wxy22\Documents\Centipede_MUJOCO-main\models\farms\centipede.xml`
- Final model SHA-256: `fa8de07bf49e878b2d1a66452f6afcf50ebf4afe674f3435931c00055c3ad5b4`
- Time step and integrator: 0.5 ms, RK4
- Armature: unchanged at `1e-4 kg m^2`
- Active-joint passive stiffness/damping: removed for 211 controller-driven joints
- Passive leg-roll/foot springs and dampers: preserved on 76 hinges
- Robot-floor contact: enabled
- Robot self-collision: disabled for the 211 robot collision geoms
- Mass, inertia, geometry, friction, actuator definitions, and topology: unchanged
- Floor texture, lighting, shadow quality, camera, and red erasure highlighting are visualization-only changes

`audit_model_delta.py` is the executable contract for these statements.

## Channel definition

Let segment-pair index \(i\) range from 0 to 18. Pair \(i\) contains the complete left and right leg subtrees attached to that segment. Each support opportunity \(e\) draws an independent trigger

\[
Z_{i,e}\sim\operatorname{Bernoulli}(q).
\]

If \(Z_{i,e}=1\), all collision geoms of both legs in pair \(i\) are disabled for

\[
\tau_{i,e}=A_{i,e}\tau_s,\qquad \tau_s=0.5\ \mathrm{s},
\]

where \(A\) is log-uniform on \([0.02,1]\). Thus bursts range from 10 to 500 ms and

\[
\mathbb E[A]=\frac{1-0.02}{\log(1/0.02)}=0.250510,
\qquad \nu=q\mathbb E[A].
\]

Here \(q\) is the event-trigger probability per support opportunity; it is not the fraction of wall-clock time erased. The realized erased-time fraction is recorded separately in every trial.

For phase group \(\mathcal G_g\) with \(m_g=|\mathcal G_g|\), the independent random-snapshot approximation is

\[
P(B_g=1)=\nu^{m_g},\qquad
P\!\left(\bigcup_g B_g\right)
=1-\prod_g\left(1-\nu^{m_g}\right).
\]

The mean absolute error between this prediction and measured any-group blackout-time fraction over the 20 nonzero-\(q\) cells is 0.0092; RMSE is 0.0155. This validates the event-level channel calculation, not a claim that displacement must equal blackout probability.

## Protocol

- Spatial wave number: \(w\in\{1,2,3,4\}\)
- Trigger probability: \(q\in\{0,0.05,0.1,0.2,0.3,0.4\}\)
- Trial duration: 8 s; measurement/noise interval: 2–8 s
- Replicates: one deterministic matched baseline per wave and eight seeds per nonzero \(q\)
- Total: 164 trials
- Terrain: flat and fixed
- Controller: fixed local FARMS configuration; only `body_wave.wave_number` changes
- Forward direction: native FARMS world \(-x\), reported as positive forward displacement

Wave number is not synonymous with redundancy. In this 19-pair/18-phase lattice, \(w=2\) and \(w=4\) both yield \(G=9\) and group sizes 2–3, whereas \(w=3\) yields \(G=6\) and sizes 3–4. The pair \(w=2,4\) is therefore an internal check that gait kinematics can change performance even when \(m_g\) is the same.

## Main results

Reliability at tolerance \(\epsilon\) is

\[
\widehat{\mathcal R}_{\epsilon,w}
=P\!\left(\text{stable}\ \land\ 
|\widehat D-D_{0,w}|\le \epsilon |D_{0,w}|\right),
\]

where \(D_{0,w}\) is the matched no-erasure baseline for wave number \(w\).

| w | G | group size \(m_g\) | baseline \(D_0\) (m) | \(D\) at \(q=0.4\) (m) | mean loss | \(\mathcal R_{2\%}\) | \(\mathcal R_{5\%}\) | \(\mathcal R_{10\%}\) |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | 18 | 1–2 | 0.2905 | 0.2862 ± 0.0093 | 1.47% | 0.500 | 0.875 | 1.000 |
| 2 | 9 | 2–3 | 0.3076 | 0.3039 ± 0.0017 | 1.23% | 1.000 | 1.000 | 1.000 |
| 3 | 6 | 3–4 | 0.2455 | 0.2448 ± 0.0006 | 0.27% | 1.000 | 1.000 | 1.000 |
| 4 | 9 | 2–3 | 0.2239 | 0.2191 ± 0.0011 | 2.16% | 0.375 | 1.000 | 1.000 |

The primary absolute-displacement result is that 10–500 ms independent local erasures are mild for this controller over the tested range. The largest mean loss at \(q=0.4\) is only a few percent. A ±10% task tolerance therefore saturates at reliability 1.0. The separate ±2%, ±5%, and ±10% figures expose this threshold dependence without selecting a tolerance after seeing the data. The non-monotonic empirical points for \(w=1\) at ±2% reflect only eight seeds per cell and should not be interpreted as noise-enhanced locomotion.

## Contact-restoration audit

In the two-seed \(q=0.3\) transition audit, 34.8% of contact restorations occurred while at least one restored geom had negative signed floor distance. The root linear-velocity increment at restoration had 95th percentile 0.003115 m/s versus 0.002460 m/s on ordinary steps. Its maximum was 0.006942 m/s versus 0.007697 m/s on ordinary steps. Restoration therefore produces a small measurable distribution shift, but not a larger worst-case impulse than the unperturbed contact dynamics in this audit. The raw transition samples are retained so a later soft-contact or delayed-reactivation variant can be compared directly.

## Figures

- `outputs/figures/absolute_displacement_vs_q.png` — primary performance plot
- `outputs/figures/displacement_vs_realized_erased_load.png` — performance versus realized time erasure
- `outputs/figures/reliability_02pct_vs_q.png` — strict task reliability
- `outputs/figures/reliability_05pct_vs_q.png`
- `outputs/figures/reliability_vs_q.png` — proposal's current ±10% criterion
- `outputs/figures/stability_vs_q.png`
- `outputs/figures/group_blackout_vs_q.png`
- `outputs/figures/group_blackout_theory_vs_measured.png` — information-theory bridge
- `outputs/figures/wave_number_vs_redundancy.png`

## Videos

- `outputs/videos/baseline_w3_q000.mp4`
- `outputs/videos/single_erasure_w3_segment09_100ms.mp4`
- `outputs/videos/short_erasure_w1_q030.mp4`
- `outputs/videos/short_erasure_w2_q030.mp4`
- `outputs/videos/short_erasure_w3_q030.mp4`
- `outputs/videos/short_erasure_w4_q030.mp4`

Red legs and a red status dot indicate active contact erasure. The single-event video forces one 100 ms erasure on segment-pair 9 at 3 s; the other erasure videos use the exact seeded stochastic protocol.

## Evidence files

- `outputs/data/short_erasure_trials.csv` — all 164 trial-level records
- `outputs/data/short_erasure_summary.csv` — cell summaries
- `outputs/data/experiment_manifest.json` — exact controller snapshot, hashes, groups, and protocol
- `outputs/data/recovery_transition_samples.csv`
- `outputs/data/recovery_transition_summary.csv`
- `audit_model_delta.py`, `stability_gate.py`, `diagnose_recovery.py`, and `run_pipeline.py`

## Interpretation and next experiment

The event-level result agrees with the independent-channel prediction: larger homologous group size strongly suppresses complete phase-group blackout. The locomotion result is subtler: the detailed robot is so tolerant of short local erasures that mean displacement changes little, and \(w=2\) versus \(w=4\) shows that gait dynamics remain a confound even at identical group-size structure. The next defensible step is not correlated noise yet. It is a calibrated severity extension under the same independent channel—longer burst distributions or increasing numbers of simultaneously erased segment-pairs—until the ±10% reliability curve leaves saturation, while retaining the same stability gate and matched baselines.
