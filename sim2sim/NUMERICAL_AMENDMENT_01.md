# Numerical convergence extension — before extra runs

The original 0.25 ms and 0.125 ms MuJoCo calibration runs yielded 99.335 and 126.993 mm respectively. Their 27.84% change exceeds the predeclared 5% convergence threshold. Neither result is eligible as a final comparison.

Extend the step-size ladder to 0.0625 and 0.03125 ms (and continue halving if the convergence criterion still fails). This is a numerical convergence study, with the physical model, materials, physical PD and gait bank unchanged. No holdout outcome has been inspected or used for selecting settings. Acceptance tolerances remain unchanged.

The seven isolated one-DOF tests at 0.25 ms have native-to-MuJoCo normalized response errors between 0.0224% and 0.4307%, below 2%. The optional position-implicit adapter made the tip probe worse (2.3502% error), so it is not adopted. This evidence does not justify attributing the full locomotion gap to PD implementation alone.

Isaac's tensor inertia getter returns inertia about the COM in link-aligned axes. The installed docstring's return section and the measured off-diagonal entries confirm this; applying the COM orientation a second time creates a spurious discrepancy. Correct comparison gives maximum full tensor relative error 1.2152e-7, with mass error 6.6806e-8 and zero-pose body-position error 4.6102e-8 m.

Two drive-probe setup failures are retained: loading MuJoCo DLL after Isaac boot failed (now run in separate processes), and all-integer zero gravity hit an installed binding type error (corrected to floats). The completed native readback worker hung only on app shutdown after writing all evidence; its verified owned PID was stopped. Subsequent workers flush evidence and exit directly.
