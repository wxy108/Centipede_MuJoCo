# Objective clarification during the broad run

The PI requested maximum forward displacement **subject to stable motion and pose**, not retention of 90% of displacement of an unstable historical reference. The initial manifest inherited that 90% distance floor from the earlier fixed-amplitude model-parameter non-regression study. This is inappropriate as a safety feasibility gate in the current amplitude-only optimization, particularly for waves where the historical reference fails stability.

The floor is retained under `legacy_gates` / `legacy_feasible`, but not used for the primary feasibility set. Positive forward displacement remains required. No absolute stability, naturality, posture, tracking, force, impulse, slip, work-per-distance, or timestep/horizon criterion is relaxed. Objective ranking remains displacement only after all those gates pass.

This clarification was made during the broad bank (approximately 400 of 2100 candidates had finished), after wave 1.5 produced a 92.54 mm candidate passing every other gate but rejected by the inherited floor against an unstable 158.01 mm reference. The original manifest and old live-worker summary remain intact; the clarified re-assessment is saved separately as `broad_objective_summary.json`. It uses exactly the same raw records. No precheck rejects on displacement or the legacy floor, so this clarification does not remove any previously evaluated stable candidate or require a simulation replay. Refinement and validation use the clarified objective, and retain the old-floor result as a diagnostic.

Only three amplitudes are still variable. Physical parameters, PD, phase law, coupled wave numbers, contact definition and source XML files remain unchanged.
