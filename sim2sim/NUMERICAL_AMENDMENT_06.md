# Articulation iteration-convergence check

The completed native PGS 16-position/4-velocity-iteration main bank still differs substantially from MuJoCo on large leg-yaw motions. An isolated drive or a single sphere contact cannot establish convergence of the coupled 173-joint articulated contact problem.

Run the same native CPU validation bank, physical definition, 62.5 µs step and command clock with 64 position / 16 velocity iterations. This fourfold iteration test is a numerical convergence diagnostic; its results must be retained even if agreement becomes worse. No physical PD, contact material or gait amplitudes are changed. Compare every gait against the original 16/4 bank. The current candidate is not accepted while this discrepancy remains unresolved.
