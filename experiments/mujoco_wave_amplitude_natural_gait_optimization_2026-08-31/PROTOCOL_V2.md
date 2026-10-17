# Protocol V2 — natural-gait amplitude search

## Frozen items

- Model: `centipede_flat_locked_head_m29p0.xml`, SHA-256
  `3815AB6CC8F6BC60834A9A85C246C19DC6B9FB94BD18DF893E072A0345132520`.
- Mass, inertia, geometry, topology, armature, stiffness, damping, contact, and
  solver remain unchanged.
- Frequency remains 1 Hz.
- Body and leg wave numbers remain identical.
- Leg phase offsets and DC offsets remain unchanged.
- Controller remains the clean open-loop sinusoidal CPG; no feedback, gating,
  state-dependent target, or presentation-only controller is permitted.

## Wave set

Primary flat locomotion map:

`0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5, 6, 7, 8, 9`.

This restores the canonical flat 1.5 and the verified RL half-wave map while
retaining the integer 1–9 subset needed for phase-group/redundancy analysis.
Noninteger waves are not assigned an integer multiplicity claim.

## Broad amplitude domain

- body yaw: 0–36 deg (the complete positive half of the model's body-yaw
  joint range);
- leg yaw: 0–90 deg;
- leg pitch: 0–60 deg.

These are search bounds, not claims of biological plausibility.  They are
deliberately wider than the historical Model, Isaac YAML, clean, and rejected
grids.  Realized-motion gates, not an artificially tight input box, determine
admissibility.

Mandatory anchors are injected exactly into every relevant wave panel:

- accepted clean flat target;
- Isaac YAML target;
- old Model erasure target;
- zero-amplitude and one-axis ablations;
- rejected prior winner as a negative control.

## Sampling and globality claim

1. Scrambled Sobol broad screen over the full 3-D box.
2. Boundary/face/axis probes and all historical anchors.
3. Independent adaptive refinement of every feasible high-displacement basin,
   not just the current best basin.
4. Dense local certification grids around retained basin maxima.
5. Cold-instance, stride-1, six-second duplicate replay of every retained
   nominal winner.  The nominal flat model is deterministic at `q=0`, so seed
   variation is neither present nor claimed there.
6. Matched front/side/oblique and local-foot visual QA.  If this reveals a new
   repeatable pathology not represented by the declared gates, the pathology
   becomes an explicit measured gate and the candidate bank is reranked; a
   visually preferred point is never substituted by hand.
7. Ten-seed random-contact replay is reported separately for integer waves.

The final result may be called exact only on the explicitly enumerated
candidate set.  Continuous-domain mathematical global optimality is not
claimed for nonsmooth contact dynamics.

## Gate families

The numerical values below are provisional anchor envelopes and must be
rechecked during the broad pilot.  They are listed separately from the
objective.

### Numerical and stability

- finite, warning-free;
- positive forward displacement and path efficiency;
- no flip, excessive coherent roll, jump, ballistic interval, or excessive
  spine tortuosity;
- penetration bounded;
- active torque and joint speed bounded;
- passive tip angle/stop occupancy bounded using robust percentiles, not a
  single maximum-speed spike.

### Realized gait naturality

- leg-yaw speed P99 initially ≤ 8 rad/s;
- leg-pitch speed P99 initially ≤ 7 rad/s;
- leg-yaw velocity reversals P95 initially ≤ 3/cycle;
- leg-pitch velocity reversals P95 initially ≤ 6.25/cycle;
- leg-pitch tracking RMS initially ≤ 10 deg;
- swing-foot height P95 initially ≤ 5.5 mm;
- mean contact duty initially within 0.15–0.65;
- left/right contact-duty difference ≤ 0.10;
- no sustained chassis drag, evaluated by body-ground contact impulse/work as
  well as contact fraction;
- no extra high-frequency joint oscillation beyond the anchor envelope.

### Objective

After every gate passes, maximize forward displacement.  Ties are resolved by
higher path efficiency, lower high-frequency residual, then lower total
amplitude.  No weighted scalar reward is used for feasibility.

## Random contact

Nominal optimization is performed first.  Random contact erasure is a separate
validation, never folded into a hidden weighted reward.  For integer waves,
apply the established independent short-burst protocol to the selected
nominal gait.  Report both displacement retention and strict gate survival.
Every erased segment pair must be visibly recolored in demos.
