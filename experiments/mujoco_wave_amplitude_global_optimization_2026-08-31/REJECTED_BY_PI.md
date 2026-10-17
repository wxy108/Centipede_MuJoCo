# Rejected by PI — 2026-08-31

This experiment is retained as historical diagnostic evidence only.

The finite-grid optimization is not an acceptable Centipede Model gait result.
It maximized forward displacement after coarse stability/contact gates but did
not gate biological/visual gait quality.  In particular, it admitted and often
selected unnaturally large swing excursions, omitted the previously selected
flat clean-controller anchor from the protocol, and used an amplitude domain
that was both coarsely sampled and biased toward large leg-yaw values.

The demo renderer is also rejected:

- it rendered collision group 0 and visual group 2 simultaneously, producing
  overlapping-surface flicker (especially at the locked head);
- it recolored collision geoms during erasure, not the visible leg subtree, so
  random-contact erasure was not visibly marked as required;
- lighting, scale, camera framing, and text size were insufficient for gait
  review.

Do not cite `final_selection_summary.json`, the old winner videos, or the old
report as a validated optimum.  Do not delete them; they are evidence of why a
naturality-aware protocol is required.

The frozen Centipede Model and Centipede Robot Step 18/19 files were not changed.

