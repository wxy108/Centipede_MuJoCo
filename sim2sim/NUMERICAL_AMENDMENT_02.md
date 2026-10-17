# Independent solver resolution and a shared command clock

Candidate 01 does **not** pass its declared full-bank convergence criterion. At 62.5 → 31.25 µs, MuJoCo displacement changes are 6.443%, 9.594%, 9.858%, and 16.912% for holdout cases 0–3. Cases 4–6 change 0.473%, 4.954%, and 1.183%; case 7 was still pending when this amendment was written. These failures remain part of the report.

The initial condition that both engines use the same *nominal* physics step is replaced by independent numerical-convergence requirements. This is an explicit protocol revision, not a claim that candidate 01 passed. PhysX TGS performs position iterations with internal substep semantics; MuJoCo implicitfast does not use the same algorithm. Equal nominal steps therefore do not establish equal numerical resolution. Forcing PhysX to use the tiny steps needed by this MuJoCo plant would add substantial cost without itself proving greater physical equivalence.

The revised paired comparison uses a common **62.5 µs command clock**. Both engines receive exactly the same float32 position and velocity targets on that clock. MuJoCo holds those targets through finer internal integration steps. Its initial candidate/fine pair is 15.625 / 7.8125 µs; further halving is required if the same 5% convergence limit fails. Physical Kp/Kd, force caps, mass, inertia, geometry, gravity, materials, frequency, amplitudes, initial pose and score remain fixed. No position-implicit adapter is used.

PhysX main runs use 62.5 µs, with the already-started 31.25 µs run checking refinement. That native refinement also refines the smooth analytic command samples; it is labelled as refinement of the complete discrete simulation/control scheme. It is not used as an exact single-factor decomposition of contact-solver error. Reported main pairs have the exact common command clock.

All original 10% displacement-gap, 5% refinement-change, nonzero-motion, full-bank completion and failure-reporting criteria remain. No physical or gait parameter is selected from holdout results. A failed holdout triggers resolution checks; it does not justify selecting a different PD or amplitude for that gait.

The CPU refinement runner explicitly requests eight PhysX worker threads using the installed PhysicsContext `worker_thread_count` setting, and records its actual value. This controls CPU scheduling, not physical parameters or solver iteration counts. Cross-environment collisions are disabled. Original default-thread results remain preserved.

Additional measured evidence: at the actual tip inertia scale, the first 5 ms isolated step-response error falls from 5.35% RMS at 250 µs to 1.21% at 62.5 µs. The peak angle difference falls from 0.146° to 0.032° for a 1° step. This shows a fast-transient discretization difference, but does not quantify its share of the full locomotion gap. The GPU/CPU calibration at 125 µs is 146.474 / 146.989 mm (0.351% symmetric gap).
