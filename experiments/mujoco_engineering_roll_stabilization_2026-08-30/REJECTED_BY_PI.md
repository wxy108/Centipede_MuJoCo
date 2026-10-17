# Rejected by PI

Date: 2026-08-31

The `pg_p300_l125` projected-gravity controller is non-canonical and must not be used by later Centipede Model experiments.

Reason: the controller must remain a simple open-loop CPG plus static impedance parameters. Projected-gravity feedback, contact gating, tip authority scheduling, and comparable state-dependent additions are out of scope.

The locked-head source XML files and `mujoco_template_locked_head_2026-08-27/run_locked_head.py` were not changed by this experiment. Existing outputs are retained only as causal/reference history. Centipede Robot Step 18/19 data remain untouched.
