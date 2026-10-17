# Numerical mass-unit invariance diagnostic

At micromass and very small tip inertia, absolute numerical thresholds may matter even when float readback agrees. This is a hypothesis, not a measured cause.

Test the fixed w=2.5, amplitudes [16,72,24] degree case in a fresh native PGS CPU scene at the existing 62.5 µs, 16/4 iterations. Compare two representations of the same physical problem: numeric mass unit kg, and numeric mass unit mg (all masses, inertias, physical drive K/D and force caps multiplied by 10^6; physical output forces divided back by 10^6). Length and time units, gravity, friction, geometry, targets and starting pose remain unchanged. This is dimensional unit conversion, not making the physical centipede heavier or fitting different physical PD. The sole conversion factor is prescribed by kg→mg before its result.

The subset uses its original full-bank environment offset, so floating-point world-position scale is preserved. A scale-1 subset repetition checks the batch-size change. Both outcomes are diagnostic until physical readback after conversion, drive response and full-bank tests qualify. Do not adopt scaling merely because it moves the speed closer to another engine.
