# Strong Independent Contact-Erasure Experiment

## 1. Question

This experiment asks whether repeated, homologous segment-pairs protect centipede locomotion from independent loss of ground contact, and whether that protection follows an erasure-channel scaling law.

The experiment intentionally changes only the spatial gait wave number and the independent contact-erasure process. The refined MuJoCo model, controller amplitudes, actuator parameters, flat floor, timestep, and collision policy remain fixed. In particular, joint armature remains unchanged at $10^{-4}\,\mathrm{kg\,m^2}$.

## 2. Strong noise channel

The channel input is one segment-pair,

\[
\mathcal M_i=\{iL,iR\},
\]

where $i\in\{0,\ldots,18\}$. When an erasure is active, all ground-contact capability associated with that segment-pair is disabled; its visual geometry is colored red in the videos.

The support-window duration is

\[
T_s=0.5\ \mathrm{s}.
\]

For every segment-pair and every support opportunity, an erasure is triggered independently with probability $q$. For noise levels up to 0.75, conditional event duration is

\[
\Delta t=A T_s,
\qquad
A\sim\operatorname{Uniform}(0.5,1),
\]

so each burst lasts 250--500 ms and

\[
\mathbb E[A]=0.75.
\]

The plotted noise level is not the trigger probability. It is the expected erased-time fraction of one segment-pair:

\[
\boxed{\nu=q\,\mathbb E[A]=0.75q.}
\]

Thus $q=\nu/0.75$ in this regime. To extend the same erased-time definition beyond 0.75 without allowing a burst to exceed one support window, every opportunity triggers and the lower duration bound increases:

\[
q=1,
\qquad
A\sim\operatorname{Uniform}(2\nu-1,1),
\qquad
0.75<\nu\le1.
\]

Because $\mathbb E[A]=\nu$ in the high-noise regime, the horizontal axis retains the same meaning over the complete sweep:

\[
\nu\in\{0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0\}.
\]

At $\nu=1$, all 19 segment-pairs lose leg and foot collision continuously throughout evaluation. Chassis--floor collision remains active, preventing floor penetration and allowing residual body-drag to be observed rather than suppressed artificially. This definition makes the horizontal axis comparable across future burst-duration distributions: two channels with the same $\nu$ impose the same expected erased-time load, even if their event counts and burst lengths differ.

## 3. Wave number and multiplicity

Let $k$ denote the spatial wave number. Segment-pairs with identical phase constitute a phase group $\mathcal G_g$. Its multiplicity is

\[
m_g=|\mathcal G_g|.
\]

Because the model has 19 segment-pairs distributed over 18 spatial intervals, one endpoint phase is repeated. The tested templates are:

| Wave number $k$ | Phase groups $G$ | Multiplicity $m_g$ | Role |
|---:|---:|---:|---|
| 1 | 18 | 1--2 | Low-redundancy case |
| 2 | 9 | 2--3 | Intermediate redundancy |
| 3 | 6 | 3--4 | High redundancy |
| 4 | 9 | 2--3 | Same-multiplicity gait-ordering control |

The $N_{\mathrm{wave}}=4$ case is not treated as having more redundancy than $N_{\mathrm{wave}}=3$. It has the same phase-group multiplicities as $N_{\mathrm{wave}}=2$, but a different spatial ordering, and therefore controls for whole-body gait dynamics.

## 4. Information-theoretic prediction

At a randomly sampled time, let $Z_i=1$ mean that segment-pair $i$ is erased. Its marginal probability is approximately

\[
P(Z_i=1)=\nu.
\]

A phase group is completely unavailable only when all of its homologous copies are simultaneously erased:

\[
F_g=\prod_{i\in\mathcal G_g}Z_i.
\]

Under independent segment-pair erasures,

\[
\boxed{P(F_g=1)=\nu^{m_g}.}
\]

This is the redundancy law: multiplicity appears as an exponent. Increasing $m_g$ therefore suppresses group blackout rapidly when $\nu<1$, but the protection necessarily disappears as $\nu$ approaches one.

Two whole-template predictions are used:

\[
P_{\mathrm{mean}}=
\frac{1}{G}\sum_{g=1}^{G}\nu^{m_g},
\]

the expected blackout fraction averaged over phase groups, and

\[
P_{\mathrm{any}}=
1-\prod_{g=1}^{G}\left(1-\nu^{m_g}\right),
\]

the probability that at least one phase group is blacked out at a random time. These are random-time snapshot predictions; finite support windows and burst timing can cause small deviations.

## 5. Locomotion metrics

Every trial lasts 8 s. Noise and evaluation begin at 2 s, leaving a 6 s measurement interval. The primary performance variable is the absolute forward displacement

\[
D_{k,\nu}.
\]

Because changing $k$ changes the nominal gait speed, robustness is measured relative to the matched no-noise baseline $D_{0,k}$:

\[
L_{N_{\mathrm{wave}},\nu}
=1-\frac{D_{N_{\mathrm{wave}},\nu}}{D_{0,N_{\mathrm{wave}}}}.
\]

For tolerance $\epsilon$, task reliability is

\[
\widehat{\mathcal R}_{\epsilon}(k,\nu)
=
P\!\left(
\text{stable}
\;\land\;
|D_{k,\nu}-D_{0,k}|\le \epsilon D_{0,k}
\right).
\]

The report includes $\epsilon\in\{0.02,0.05,0.10\}$. Each stochastic nonzero $(k,\nu)$ cell contains eight random seeds; each wave number has one deterministic no-noise baseline and one deterministic complete-erasure endpoint. The complete extended dataset contains 296 trials.

## 6. Main results

### Matched-baseline performance loss

| $\nu$ | $N_{\mathrm{wave}}=1,\ m_g=1\text{--}2$ | $N_{\mathrm{wave}}=2,\ m_g=2\text{--}3$ | $N_{\mathrm{wave}}=3,\ m_g=3\text{--}4$ | $N_{\mathrm{wave}}=4,\ m_g=2\text{--}3$ control |
|---:|---:|---:|---:|---:|
| 0.1 | 3.5% | 1.0% | 0.0% | 1.9% |
| 0.3 | 13.5% | 6.4% | 3.7% | 7.3% |
| 0.5 | 23.6% | 20.1% | 15.8% | 20.1% |
| 0.6 | 31.3% | 32.3% | 28.1% | 30.4% |
| 0.7 | 46.7% | 48.2% | 47.3% | 47.4% |
| 0.8 | 64.2% | 66.1% | 69.0% | 65.8% |
| 0.9 | 72.9% | 79.2% | 84.9% | 80.4% |
| 1.0 | 86.3% | 99.5% | 98.3% | 99.1% |

At $\nu=0.3$, the 10% task reliabilities are 0.125, 1.0, 1.0, and 0.875 for $N_{\mathrm{wave}}=1,2,3,4$, respectively. This is the clearest useful regime: higher multiplicity substantially delays performance degradation without relying on falls or numerical failures.

At $\nu=0.5$ and above, the curves converge. This is expected from $\nu^{m_g}$: redundancy gives strong protection at moderate noise but cannot prevent simultaneous erasure when each constituent channel is unavailable for most of the time.

The relative-loss curves for $N_{\mathrm{wave}}=2$ and $N_{\mathrm{wave}}=4$ are close despite their different nominal speeds. At $\nu=0.4$, their losses are 12.0% and 11.8%; at $\nu=0.5$, both are 20.1%. This same-multiplicity control supports multiplicity as the leading probability-scale variable while still allowing gait dynamics to affect the exact displacement response.

At complete leg-contact erasure, the absolute displacements are 39.9, 1.7, 4.3, and 2.0 mm for $N_{\mathrm{wave}}=1,2,3,4$, respectively. The $N_{\mathrm{wave}}=2,3,4$ gaits are effectively immobile. The larger $N_{\mathrm{wave}}=1$ residual is a chassis-mediated body-drag mode: it is not evidence that leg redundancy survives $\nu=1$.

### Theory-to-simulation agreement

Across all nonzero-noise cells:

- mean phase-group blackout: correlation $r=0.994$, mean absolute error 0.0257;
- any-group blackout: correlation $r=0.996$, mean absolute error 0.0200.

Therefore, the independent erasure model predicts the internal channel-failure statistics accurately. The subsequent map from group blackout to forward displacement remains a whole-body dynamics problem; information theory predicts the occurrence of lost support information, not the exact mechanical cost of each loss.

## 7. Stability and restoration audit

All 296 trials pass the predefined finite-value, posture, floor-penetration, and self-contact gates:

- unstable trials: 0;
- nonfinite trials: 0;
- dynamic self-contact count: 0;
- maximum root roll: $12.70^\circ$;
- maximum root pitch: $25.76^\circ$;
- maximum segment tilt: $26.77^\circ$;
- minimum reported floor-contact distance: $-3.02$ mm.

Instantaneous collision restoration is a measurable but bounded numerical transient. In a separate four-trial, $\nu=0.5$ audit:

| Velocity increment | Ordinary-step p95 | Restore-step p95 | Ordinary maximum | Restore maximum |
|---|---:|---:|---:|---:|
| Root linear | 0.00598 m/s | 0.00731 m/s | 0.0812 m/s | 0.0495 m/s |
| Root angular | 0.2517 rad/s | 0.4277 rad/s | 6.76 rad/s | 1.97 rad/s |
| Maximum active-joint speed | 0.05933 rad/s | 0.05969 rad/s | 6.84 rad/s | 0.905 rad/s |

About 20.6% of restoration events occur while at least one restored geometry slightly overlaps the floor; the worst pre-restoration distance is $-2.80$ mm. Thus restoration modestly increases typical root angular increments, but it does not create the largest observed impulses, does not alter the active-joint p95 materially, and does not trigger instability. A future sensitivity test should compare instantaneous restoration against separation-gated or smoothly ramped contact restoration.

The high-noise extension requires a stronger qualification. In a separate $\nu=0.9$ audit, 90.1% of restore transitions occur while at least one disabled leg geometry still slightly overlaps the floor. Root-linear p95 increments are 0.0303 m/s at restoration versus 0.0151 m/s at ordinary steps, and root-angular p95 increments are 1.046 versus 0.651 rad/s. Active-joint-speed p95 does not increase (0.0682 versus 0.0987 rad/s), all trials remain stable, and restore maxima remain below ordinary-step maxima. Nevertheless, the $\nu=0.8$ and $0.9$ tail should be treated as exploratory until a separation-gated restoration sensitivity run is completed. The $\nu=1$ endpoint is unaffected by periodic restoration because collision remains disabled continuously throughout evaluation.

## 8. Interpretation

The experiment supports the following two-layer story:

1. **Channel layer:** independent segment-pair contact losses combine according to $P(F_g)=\nu^{m_g}$. Morphological multiplicity acts like repeated transmission of the same phase-level support function.
2. **Dynamics layer:** phase-group blackouts propagate through compliant whole-body mechanics to produce displacement loss, lateral motion, and energy changes. The $N_{\mathrm{wave}}=4$ control shows that multiplicity determines the leading reliability scale, while spatial gait ordering determines the mechanical conversion from blackout to task performance.

The defensible conclusion is therefore not merely that “more legs are better.” It is:

\[
\boxed{
\text{Under independent contact erasures, homologous multiplicity suppresses phase-group failure approximately as }\nu^{m_g},
}
\]

and this suppression produces a measurable locomotion-robustness advantage in the moderate-noise regime.

## 9. Artifacts

- Extended trials: `outputs/strong_independent_erasure/extended_full/trials.csv`
- Extended aggregated results: `outputs/strong_independent_erasure/extended_full/summary.csv`
- Extended figures: `outputs/strong_independent_erasure/extended_full/figures/`
- Original 0--0.6 results: `outputs/strong_independent_erasure/full/`
- Recovery audit: `outputs/strong_independent_erasure/recovery_audit/`
- High-noise recovery audit: `outputs/strong_independent_erasure/high_recovery_audit/`
- Videos and thumbnails: `outputs/strong_independent_erasure/videos/`
- Experiment driver: `run_severity_sweep.py`
- Restoration diagnostic: `diagnose_strong_recovery.py`
