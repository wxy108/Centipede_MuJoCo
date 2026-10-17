# Independent short-erasure sweep for wave numbers 5--9

## Scope

This extension changes only the integer spatial wave number and the independent
short-contact-erasure process.  The refined FARMS MuJoCo centipede, flat floor,
controller gains and amplitudes, actuator parameters, timestep, and joint
armature of `1e-4 kg m^2` are held fixed.  Correlated noise and rough terrain are
not included.

The channel noise coordinate is the expected erased-time fraction of one
bilateral segment-pair contact channel,

\[
\nu=\mathbb E[Z_i(t)]\in[0,1].
\]

For \(\nu\le0.75\), each support opportunity triggers independently with
\(q=\nu/0.75\), and the erased duration fraction is
\(A\sim\operatorname{Uniform}(0.5,1)\).  For \(\nu>0.75\), every opportunity
triggers and \(A\sim\operatorname{Uniform}(2\nu-1,1)\), so that
\(\mathbb E[A]=\nu\).  At \(\nu=1\), all leg--floor contact channels are
continuously disabled and there is no contact-restoration transition.

## Phase structure

The robot has 19 bilateral segment pairs distributed over 18 spatial phase
intervals.  Increasing wave number is therefore not equivalent to increasing
redundancy.

| Wave count \(N_{\mathrm{wave}}\) | Phase-group count \(G\) | Group sizes | Interpretation |
|---:|---:|---|---|
| 5 | 18 | one group of 2; seventeen of 1 | same-multiplicity control for \(N_{\mathrm{wave}}=1,7\) |
| 6 | 3 | 7, 6, 6 | high multiplicity |
| 7 | 18 | one group of 2; seventeen of 1 | same-multiplicity control for \(N_{\mathrm{wave}}=1,5\) |
| 8 | 9 | one group of 3; eight of 2 | same-multiplicity control for \(N_{\mathrm{wave}}=2,4\) |
| 9 | 2 | 10, 9 | highest multiplicity in this scan |

The ordered redundancy comparison is therefore

\[
N_{\mathrm{wave}}\in\{1,2,3,6,9\},\qquad
m_g\approx\{1\!\!\text{--}\!2,2\!\!\text{--}\!3,
3\!\!\text{--}\!4,6\!\!\text{--}\!7,9\!\!\text{--}\!10\}.
\]

The families \(\{1,5,7\}\) and \(\{2,4,8\}\) instead hold nominal
multiplicity fixed while changing spatial gait ordering.

## Experimental design and audit gate

- Noise levels: \(\nu=0,0.1,\ldots,1.0\).
- Duration: 8 s per trial; performance is measured after the first 2 s.
- Eight seeds at every stochastic level \(0.1\le\nu\le0.9\); one deterministic
  trial at \(\nu=0\) and \(\nu=1\).
- New trials for \(N_{\mathrm{wave}}=5\)--9: 370.
- Combined trials for \(N_{\mathrm{wave}}=1\)--9: 666.
- Dynamically valid combined trials: 663/666.

Invalid trials are never silently compressed into the primary displacement
mean.  `mean_displacement_m`, its standard deviation, and
\(D/D_{0,k}\) use dynamically stable trials only.  `stable_fraction`, the raw
trial table, and the all-trial displacement retain the failure evidence.

## Results

The baseline-normalized performance is

\[
\eta_{N_{\mathrm{wave}}}(\nu)
=\frac{\mathbb E[D_{N_{\mathrm{wave}}}(\nu)]}
{D_{0,N_{\mathrm{wave}}}}.
\]

| \(N_{\mathrm{wave}}\) | \(m_g\) | \(D_{0,N_{\mathrm{wave}}}\) (m) | \(\eta(0.3)\) | \(\eta(0.5)\) | \(\eta(0.6)\) | \(\eta(0.7)\) | \(\eta(0.8)\) | \(\eta(0.9)\) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1--2 | 0.290 | 0.86 | 0.76 | 0.69 | 0.53 | 0.36 | 0.27 |
| 2 | 2--3 | 0.307 | 0.94 | 0.80 | 0.68 | 0.52 | 0.34 | 0.21 |
| 3 | 3--4 | 0.246 | 0.96 | 0.84 | 0.72 | 0.53 | 0.31 | 0.15 |
| 6 | 6--7 | 0.149 | 0.99 | 0.91 | 0.78 | 0.50 | 0.17 | 0.09 |
| 9 | 9--10 | 0.097 | 1.00 | 0.98 | 0.87 | 0.54 | 0.31 | 0.12 |

The clearest redundancy regime is \(0\le\nu\lesssim0.6\).  At \(\nu=0.5\),
retention rises from 0.76, 0.80, and 0.84 for \(N_{\mathrm{wave}}=1,2,3\), to 0.91 for
\(N_{\mathrm{wave}}=6\) and 0.98 for \(N_{\mathrm{wave}}=9\).  The high-multiplicity templates therefore
delay performance loss under finite independent noise even though their clean
absolute displacement is lower.

At \(\nu\approx0.7\), the retained-performance curves converge near 0.5 and
the mechanical dynamics cease to follow a simple monotone ordering.  Three of
the eight \(N_{\mathrm{wave}}=9,\nu=0.7\) trials crossed the fixed dynamic floor-penetration
gate (minimum distances from -3.55 to -4.06 mm versus a -3.5 mm limit).  They
remained finite and did not flip, but are recorded as failed trials, giving
\(\widehat P(\mathrm{stable})=5/8\) for that cell.

For \(\nu\ge0.8\), all gaits lose most locomotion performance.  This regime
must be interpreted cautiously: contact-restoration impacts were previously
measured to be strongest near \(\nu=0.8\)--0.9.  The \(\nu=1\) endpoint is
cleaner because contact is never restored, and all waves approach zero forward
displacement there.

## Information-theoretic interpretation

Under the ideal independent erasure abstraction, complete loss of a phase
group containing \(m_g\) homologous segment pairs scales as

\[
P(\text{group blackout})=\nu^{m_g}.
\]

At \(\nu=0.6\), the predicted mean group-blackout fraction decreases from
0.587 for \(N_{\mathrm{wave}}=1,5,7\), to 0.344 for \(N_{\mathrm{wave}}=2,4,8\), 0.202 for \(N_{\mathrm{wave}}=3\), 0.040
for \(N_{\mathrm{wave}}=6\), and 0.008 for \(N_{\mathrm{wave}}=9\).  The measured blackout fractions preserve
this ordering, although the measured values are not identical to the ideal
formula because the implemented bursts have finite, asynchronous onset and
duration.

The data therefore support a deliberately limited claim: larger embodied
multiplicity suppresses phase-group erasure and improves locomotion retention
over a finite low-to-moderate noise range.  They do not support the stronger
claim that raw wave number or multiplicity alone determines absolute speed or
guarantees monotone superiority at every noise level.  The same-multiplicity
controls show that spatial gait ordering remains a separate mechanical design
variable.

## Output map

- `outputs/strong_independent_erasure/wave_5_to_9/trials.csv`: all 370 new raw
  trials.
- `outputs/strong_independent_erasure/wave_5_to_9/summary.csv`: aggregated
  \(N_{\mathrm{wave}}=5\)--9 results.
- `outputs/strong_independent_erasure/all_waves_1_to_9/`: merged 666-trial data,
  manifests, and figures.
- `figures_all/`: absolute displacement and all-wave comparisons.
- `figures_redundancy_scan/`: \(N_{\mathrm{wave}}=1,2,3,6,9\) ordered-multiplicity comparison.
- `figures_same_m_low/` and `figures_same_m_mid/`: spatial-ordering controls.
- `outputs/strong_independent_erasure/videos/strong_erasure_k[5-9]_nu0.60_seed0.mp4`:
  five 8 s, 1280x720, 30 fps demos with active erasures highlighted in red.
