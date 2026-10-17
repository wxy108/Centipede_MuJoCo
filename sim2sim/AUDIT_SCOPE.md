# Sim-to-sim alignment: audit before changes

User authorization (2026-09-11): audit Centipede_MUJOCO-main and the original Centipede_IsaacLab_RL, then align simulation setup and shared physical parameters and validate similar performance. This resumes only the new sim-to-sim workstream. Old amplitude/contact queues remain stopped. Clean is outside the edit scope.

Research roots are selected by absolute path. Centipede_robot_MUJOCO is a separate robot scale and is not a dynamics source for this work.

The baseline pair is the actual plant/controller used by the completed 2026-09-09 common-PD study, not the historical files pointed to by generic READMEs. The original RL repository has pre-existing tracked changes; those are retained. Changes for this work are opt-in and dated, with original assets, sources and raw evidence preserved.

Audit sequence: entry points and dependencies; compiled/composed mass and inertia; actual collision geometry, joint axes and fixed groups; control units/targets/drive semantics; runtime contact and solver readbacks; initialization and matched scoring; saved result differences. No simulation setup is changed until the initial audit and proposed common physical contract are written.

Validation sequence: exact parameter/kinematic readback, small mechanical calibration, matched flat-ground gait bank, timestep/solver convergence and holdout gait checks. Similarity must be measured under one shared PD table and one shared gait command, not obtained by separately optimizing gains or changing scores.

Success targets will be declared before alignment outcomes are inspected. Zero movement in both engines is not acceptable evidence of locomotion agreement. A completed audit or a running experiment is not a claim that the gap is resolved.
