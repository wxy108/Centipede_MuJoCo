# Body twitch and the proposed engineering PD route

Scope: Centipede Model only. This is a diagnostic replay, not a controller,
model, parameter, or winner-selection change. Robot Step 18/19 is untouched.

## Result

The twitch is not solely a rendering artifact. Full-rate trajectories contain
rapid local spine pitch/roll changes and sharp ground reaction forces. The
existing V2 gates did not certify local body smoothness. Their `certified`
label means only passage through the declared V2 finite-bank protocol, not
biological naturality or freedom from body impacts.

Frozen XML SHA-256:
`3815AB6CC8F6BC60834A9A85C246C19DC6B9FB94BD18DF893E072A0345132520`.
Actual timestep is **0.00025 s**, not 0.0005 s. Controller updates and logging
occur every physics step. Duration 6 s; steady-state summaries use 1.5–6 s.
Mass is 2.536031 g; armature remains zero. All reported forces are simulated
normal contact forces, not biological measurements.

| Case | Peak local spine pitch speed (rad/s) | Peak total ground normal force / bodyweight | Any loaded chassis-ground contact, fraction of time |
|---|---:|---:|---:|
| Previously accepted amplitudes, wave 1.5 | 15.82 | 2.58 | 5.83% |
| V2 selected amplitudes, wave 1.5 | 14.57 | 2.62 | 7.86% |
| V2 displacement winner, wave 2.5 | 19.74 | 4.59 | 4.29% |
| V2 selected amplitudes, wave 3 | 15.33 | 3.40 | 4.46% |

At t=1.69725 s in the wave-2.5 replay, total normal force is 114.07 mN,
including 94.88 mN on `link_body_1`; total weight is about 24.88 mN.
`link_body_1` belongs to the head/link1 fixed assembly. This is not motion
across a nonexistent head joint. Another large force event occurs after
2 s, so the finding is not confined to the controller startup interval.

In the separate 2–6 s analysis window, the maximum local pitch change over
10 ms is 2.49 degrees for the accepted wave-1.5 amplitudes, 2.54 degrees for
the new wave-1.5 selection, and 4.10 degrees for the wave-2.5 selection.
These were not included in the former body-smoothness gate.

This is partly a pre-existing baseline issue, not proof that the most recent
amplitude search introduced a new physics defect. The wave-2.5 candidate has
less chassis contact time than the accepted baseline but much stronger peak
contact force: minimizing contact occupancy alone does not bound impacts.

## What is established, and what is not

- Head-region and tail twitch events frequently coincide with chassis-ground
  contacts. Some large events occur without immediate chassis contact, so
  chassis contact is not a complete explanation; foot support changes and
  tip-limit reaction remain mechanisms to isolate.
- Existing body checks use the active body-yaw position waveform, averaged
  roll, and chassis-centroid height. They do not directly bound individual
  spine pitch/roll angular-rate changes. Most metrics were sampled at 100 Hz,
  versus the 4 kHz physics rate used here.
- Percentile tip-angle / per-tip occupancy gates do not prevent a small
  subset of tips from meeting their limits at a given instant. An aggregate
  `any limit active` fraction is not the same as per-tip stop occupancy.
- These replays are warning-free. Mass-matrix spot checks at nine post-startup
  states are positive definite. This is no evidence for a singular mass
  matrix causing the observed twitch, but it does not establish timestep
  convergence or exclude contact/solver sensitivity.
- The camera follows centroid height (or one segment for zoom) without
  smoothing. That can amplify perceived movement, but camera behavior cannot
  explain the measured joint/force events. Its visual contribution is not
  quantified here.
- The 125 ms per-segment startup ramp can also create target speeds above
  steady-state values; startup should be evaluated separately, not used to
  explain all later repeated events.

## Engineering PD proposal — not executed

Increasing appropriately selected joint impedance is a reasonable engineering
option and need not add state feedback or new CPG logic. It is not justified
to multiply every Kp/Kd indiscriminately.

Current representative values (Nm/rad and Nms/rad):

| Family | Kp or passive spring k | Kd or passive damping c |
|---|---:|---:|
| Body pitch, zero-target PD | 1.9168796e-4 | 6.1016173e-7 |
| Body roll, passive zero-rest spring/damper | 6.9584253e-4 | 9.8644763e-7 |
| Leg pitch, moving-target PD | 4.2673293e-5 | 5.0153838e-8 |

1. Freeze mass, inertia, armature, morphology, tip compliance, contact,
   frequency, phases and coupled body/leg wave number.
2. At fixed amplitudes, separately test body pitch and body roll damping,
   then paired stiffness/damping changes. Retain a baseline and compare
   several wave numbers. Do not quietly restore the previously rejected
   single-front-pitch modification as a new baseline.
3. Use effective-inertia/mode-aware damping as an initialization, not a
   universal proof: the single-DOF approximation is
   `Kd = 2*zeta*sqrt(Kp*I_eff)`. Thus Kp times four suggests Kd times two for
   unchanged approximate damping ratio, not both times four. The full
   contact-coupled chain still needs testing.
4. Leave the compliant tip passive. Stronger leg PD is not the first default:
   it can enforce the moving foot trajectory more aggressively against the
   ground. Body joint PD only stabilizes relative segment angles, not the
   floating base's world attitude by itself.
5. Treat body angular-rate changes, short-window pitch/roll excursions,
   contact impulses/force peaks, repeated chassis impacts, torque/work and
   forward displacement as separate acceptance measurements. New thresholds
   require explicit calibration; no threshold is silently fitted to retain
   a preferred winner.
6. Validate on independent durations and smaller timestep as diagnostics.
   Only after a new impedance is accepted should the three-amplitude bank
   be reranked/reoptimized. Old amplitude optima are conditional on old PD.

Official reference for the damping-ratio convention and implicit actuator
damping: https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#actuator-position

## Evidence and demos

`outputs/*_trace.npz` contains start-of-step kinematics, corresponding contact
forces and accelerations, every physics step. `*_summary.json` and
`comparison.json` define the numerical summaries. FFT diagnostics in the
comparison use 2–6 s and a Hann window; they are diagnostics, not new gates.

Existing unchanged baseline demos are in:
`../mujoco_wave_amplitude_natural_gait_optimization_2026-08-31/outputs/videos/final_certified/`.
Use `certified_w2p5_three_view_clear.mp4` and
`certified_best_w2p5_local_zoom_clear.mp4`. They are **not** post-PD-improvement
demos. Video labels from the prior run remain archival, with the certification
limitation explained above.
