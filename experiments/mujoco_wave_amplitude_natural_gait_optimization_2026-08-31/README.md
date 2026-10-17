# Natural-gait amplitude re-optimization

Scope: abstract/biological Centipede Model only.  Centipede Robot Step 18/19 is
read-only historical context and must never be used as model data or overwritten.

This experiment replaces the rejected displacement-only finite grid.  The
frozen locked-head MuJoCo model, body/leg wave coupling, 1 Hz clock, phase
offsets, model dynamics, and clean open-loop controller architecture remain
unchanged.  Only body-yaw, leg-yaw, and leg-pitch amplitudes may vary.

The first stage is a matched audit of historical target anchors, the accepted
flat clean profile, and every rejected grid winner.  Its purpose is to measure
whether the visually strange swing is commanded excursion, dynamic overshoot,
contact-driven distortion, or a combination.

Status: complete.  The audit froze Protocol V2 before the new search.  The
completed bank contains 4,174 unique stride-1 candidates and three adaptive
refinement layers.  Nominal selections, cold-replay evidence, q=0.35 random
contact diagnostics, plots, and 15 labelled videos are summarized in
`FINAL_RESULTS.md`.
