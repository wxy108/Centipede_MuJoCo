# Third-grid convergence check — 2026-09-11

The PGS-matched MuJoCo 62.5 → 31.25 µs pair failed the unchanged 5% displacement-convergence criterion in validation cases 1 and 3 (100.115 → 108.581 mm, 127.333 → 142.962 mm). Therefore 62.5 µs is not accepted as the MuJoCo production step. The single calibration case passing was insufficient.

Run the entire eight-case bank at 15.625 µs, holding the same 62.5 µs command clock, unchanged physical PD, amplitudes, geometry, mass, friction and scoring. Keep the mechanically derived backward-Euler adapter and numerical rigid-contact time constant 2h. Compare all cases against 31.25 µs; do not exclude failing gaits. No performance-based physical-parameter fitting is authorized by this protocol.

The native main, CPU refinement and GPU main banks remain under validation. This amendment changes only the resolution ladder; all performance and convergence acceptance thresholds remain unchanged.
