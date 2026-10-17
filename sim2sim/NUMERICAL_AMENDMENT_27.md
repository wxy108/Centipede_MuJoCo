# Per-step recording of the free-translation discrepancy

The complete zero-force translation probes established position dependence. Their dt31 far-origin translation did not match a simple root-position float32 repeated-addition model. Public PhysX `DyFeatherstoneForwardDynamic.cpp` currently shows root integration followed by link propagation and whole-articulation COM/momentum corrections; this is relevant context, not proof of the installed binary's exact version.

Before further diagnosis, record native body COM positions and COM velocities, root position and velocity at every physics step for 0.002 seconds, using the same two zero-gravity, zero-contact articulations and initial Y positions 0 and 1.75 m. Run both existing time steps. No physical parameter changes, no gait, no changes to the accepted-candidate workers. Compare whole-articulation sequential float32 weighted COM arithmetic against the actual per-step displacement. Report mismatches rather than forcing an explanation.

Primary source: https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/main/physx/source/lowleveldynamics/src/DyFeatherstoneForwardDynamic.cpp (root update, link propagation, and COM position correction in `updateBodies`).
