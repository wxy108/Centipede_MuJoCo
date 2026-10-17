# Separate TGS temporal subdivision from velocity-solver convergence

Before this experiment: TGS 16/4 failed both CPU and GPU 250→125 µs checks, and the CPU 62.5 µs diagnostic also shifted some gaits substantially (3.5/B: 191.236→177.159 mm). A lower nominal gap at one grid is not a usable profile. The completed GPU 125 µs bank is retained, including three refinement failures.

Test TGS with one position iteration (one temporal position subdivision) and 32 velocity iterations, CPU, full nine-environment validation batch, dt=125 µs, held 250 µs command clock. Independently double velocity iterations to 64 at the identical time grid and commands, then halve the time grid to 62.5 µs with 32 velocity iterations. This separates time integration resolution from repeated velocity-constraint solution. It does not claim that single precision is a proven root cause or that one position iteration is inherently accurate. The measured iteration and timestep checks decide whether this numerical family is usable.

The physical plant, PD, materials, initialization, eight holdouts and all acceptance thresholds remain unchanged. No adoption before the full tests. CPU evidence cannot substitute for GPU evidence.
