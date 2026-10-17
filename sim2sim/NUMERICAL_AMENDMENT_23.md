# Equivalent execution of the frozen common-contact law

The reference shared contact function remains frozen. A separate vectorized implementation computes all mesh support z coordinates together, selects the same minimum vertex, then transforms only that selected xyz point. Normal and tangential laws, their SI constants, support geometry, memory update, time step and commands do not change. The 1000-state/nine-environment continuous randomized test returns bitwise identical forces, torques, memory and support indices. See audit/common_contact_vectorization.json.

In the common-contact branch, native ground collision was already disabled, self collision disabled, and different environments filtered. Disabling the remaining native collision shapes therefore removes no permitted contact pair. Both CPU and GPU nine-environment 0.02 s initialization trajectories remain bitwise identical to the reference. This is an execution check, not a full performance gate. The full CPU main bank continues on its frozen reference runner; GPU main and CPU refinement use the equivalent fast runner. Backend physical readbacks and input hashes pass independently in each.

A CPU partition timing probe preserves the original global Y offsets and selects validation cases 2, 4, 8. It is a bounded execution experiment, not permission to combine mismatching traces or replace the full validation bank. No production partition bank has been adopted.

The original 31.25→15.625 µs grids, shared 250 µs command clock, all eight fixed holdouts, 10% cross-engine and 5% refinement gates remain unchanged. No coarse step was selected to make nominal displacements match. Physical contact parameters were frozen before this branch's results.
