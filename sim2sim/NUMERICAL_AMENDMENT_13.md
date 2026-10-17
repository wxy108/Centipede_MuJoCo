# TGS circular friction with external forces in every iteration

2026-09-11. This amendment is written before new full-gait results.

## Two independent installed-backend observations

1. The CPU TGS force-limit probe under the same 45° force that PGS held static moves (34.937, 34.971) µm. MuJoCo circular Coulomb moves (34.777, 34.789) µm. Therefore the installed TGS and PGS tangent-force behavior differs; the PGS square adapter is not appropriate for TGS.
2. Setting the documented scene field `enableExternalForcesEveryIteration=true` changes the TGS drop/slide probe final position from 114.098 to 127.498 µm, close to the rigid inelastic reference 127.71 µm. This addresses the specific mechanical failure that originally motivated moving away from TGS. Retain the full velocity traces to verify that the impact reversal is gone.

Evidence: `runs/tgs_friction_angle45_default_cpu_000`, `runs/tgs_drop_external_each_iter_cpu_000`. The setting is exposed by the installed 107.3.18 PhysX scene schema and described there as improving TGS solver stability. It affects external-force integration, not the physical gravity magnitude, PD gains or friction coefficients. This is a solver correction supported by probes, not locomotion tuning.

## New candidate to evaluate

Use native TGS 16/4, external forces each iteration, the original common physical plant, circular friction coefficient 0.5, original 20/40/50 µm contact settings and default CFM. Preserve v6 and all failed/unfinished PGS evidence. The complete GPU PGS half-step run is allowed to finish and will not be relabelled as TGS.

TGS uses internal temporal substeps, so try a 250 µs native outer step and independently test 125 µs over every frozen gait. Use a shared 250 µs position/velocity command clock for the nominal pair; MuJoCo uses the already justified coupled update at 31.25 µs and is independently checked at 15.625 µs while holding this command clock. The finer native run also refines target sampling and therefore checks the complete numerical scheme, as explicitly recorded in previous amendments.

All original eight-gait thresholds remain unchanged. Start with CPU validation and isolated drive checks. GPU acceptance requires its own complete runs and refinement. Do not promote this candidate from the two simple contact tests alone. The existing operating profile remains pending.
