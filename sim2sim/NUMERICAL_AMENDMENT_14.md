# Diagnose the rejected TGS grid without changing the physical plant

Written after CPU 250→125 µs full-bank refinement failed: case 1 changed 12.2409%, case 3 changed 5.3720%. The 250 µs TGS candidate is not accepted even if nominal cross-engine scores are close. All original gates remain fixed.

Next independent checks:

1. Run TGS CPU at 62.5 µs, 16/4, the complete nine-environment validation batch, to measure continued refinement. This is a diagnostic of the limiting solution, not permission to choose whichever coarse step happens to match MuJoCo.
2. Run TGS CPU at 125 µs with 64/16 iterations, same complete batch and command clock as the existing 125 µs run. The earlier PGS iteration ablation does not establish TGS convergence.
3. Run TGS CPU at 125 µs while holding the common 250 µs commands, same complete batch. This separates command-clock refinement from the time integration change. No change to target values is fitted.
4. MuJoCo's documented regularized elliptic contact law can retain tangential drift inside the Coulomb cone. Test the documented NoSlip friction postprocessor at 20 iterations, tolerance 1e-12, on the sensitive 2.5/B holdout at 31.25 µs and the held 250 µs clock. This removes friction regularization in a numerical postprocessing sweep; it does not change friction coefficient 0.5. A locomotion improvement alone is insufficient: convergence, contact probes, and the full bank would be required for adoption. Preserve failed or unstable outcomes.

Reference: https://mujoco.readthedocs.io/en/latest/modeling.html#preventing-slip (accessed 2026-09-11). NoSlip is not guaranteed to enforce exact zero slip and can destabilize complex contact systems. It is a diagnostic here.

All new runners are copies with explicit receipts. Previously running scripts and physical assets remain frozen. Neither historical project defaults nor Clean are changed.
