# Accelerated contact equation passes the active-ramp stress test

Written before new full-gait results. Anderson acceleration converges on all 1000 steps of the same active-ramp full-robot test. The 60- and 120-update budgets produce exactly the same recorded generalized positions, with mean 11.287 solves and max velocity residual 9.984e-9 m/s. This supports using the accelerated solver for a new diagnostic gait; it does not validate full locomotion yet.

Start 2.5/B at Mu dt125 and dt62.5 µs, held common 250 µs command clock, paired with the new TGS single-temporal-subdivision family. These matched time grids also make the physical implicit PD update easier to compare. The two steps are declared before results; they are not selected to match an already observed gait score. No old failed profile is relabelled as accepted. Use up to 120 corrections, same 1e-8 m/s residual target. Stop and save state on any unresolved contact equation rather than continuing to a misleading complete score. Save progress every 0.25 simulated seconds.

All prior physical inputs and full-bank acceptance gates remain fixed. Only the contact constitutive correction and its numerical solution are under test.
