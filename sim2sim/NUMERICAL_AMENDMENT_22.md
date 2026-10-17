# Control for CPU worker count in the later TGS diagnostics

The original TGS CPU250/125 runs used eight CPU worker threads. The later CPU62, CPU125 with 64/16 iterations, and CPU125 with held 250 µs commands used two workers to limit concurrent resource use. Their comparison with CPU125/eight-worker baseline therefore initially changes two numerical settings. Do not attribute those differences to time grid, iteration count or command clock alone until worker count is checked.

Run the identical full nine-environment TGS16/4 CPU125 configuration and 125 µs command clock with two workers. Compare full trajectories and all eight displacement values against the eight-worker run. This is a thread-count control, not a new physical configuration or fitting parameter. The separate one-position-iteration TGS family consistently uses two workers for all of its paired checks.
