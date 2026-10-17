# Irreversible chassis-contact rejection

Added while the existing `refine` workers were already running. Those workers retain their previously loaded evaluator; the new rule begins with `polish`. It does not change a stability threshold, a parameter bank, a physical model, or a controller setting.

The existing full-run chassis-contact gate permits at most 3% of the scoring steps to have summed chassis normal force greater than 1e-7 N. At the nominal 0.25 ms timestep, the 1.5-to-6 s scoring interval has 18,000 steps.

The prefix evaluator counts only steps where **one individual chassis contact** already exceeds 1e-7 N. Because the full summed force is nonnegative, this is a conservative lower bound on the full-run bad-step count; it avoids changes caused by a different floating-point summation order. It rejects only when the lower bound exceeds `ceil(0.03 * maximum_final_step_count) + 1`. For a six-second nominal rollout this means at least 542 bad steps, so the final fraction cannot fall back to 3% even if every remaining step has no chassis contact.

Verification compares the prefix decision with existing independent full rollouts at waves 2.5, 3, and 8. `polish` requires the verification file to exist and all checks to pass. Evidence is saved in `outputs/data/chassis_contact_pruning_verification.json`, separately from the earlier prefix-verification records. Feasible candidates still receive a complete cold six-second evaluation and the separate longer/half-step validation.

The wave-8 fixed-amplitude reference fails an existing force gate before the added contact-duration branch can trigger. A fourth independent full-run case (`flexible_flat_w8_4ce333d1c03fff`) therefore explicitly exercises the new branch; it has chassis contact throughout its scoring interval. The branch-exercise evidence is saved separately in `outputs/data/chassis_contact_pruning_branch_exercised.json` and is also required before `polish` runs.
