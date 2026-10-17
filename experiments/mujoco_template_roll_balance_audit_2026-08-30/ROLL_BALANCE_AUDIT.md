# Roll-balance audit

This is an evidence-only experiment on the abstract Centipede Model. It does not change the selected model or controller.

## Baseline and rollback status

- The selected flat baseline remains coupled body/leg `wave=1.5`; `wave=1.0` below is diagnostic only.
- No first-pitch (`q1`) stiffness/damping override is applied.
- The locked-head model XML is read-only in this experiment.

## Main findings

- Target leg-pitch amplitude is spatially uniform after warm-up: front/middle/rear means are 36.898/36.898/36.898 deg.
- Actual amplitude front/middle/rear is 29.973/30.292/31.522 deg.
- The front-minus-middle difference is only -0.319 deg (about 1.1% of the actual amplitude), so this is not a front-half amplitude collapse. In the wave=1.0 ablation the ordering reverses: front/middle are 31.050/29.951 deg.
- In the selected gait, left/right contact counts are exactly equal in 27.8% of steady-state samples. Even within those samples, the P95 left/right vertical-force imbalance is 0.163 and the P95 contact roll moment is 130.372 uN m.
- Equal contact count is therefore not a zero-roll condition. Zero roll moment would require force magnitude, lever arm, friction force, timing, and local body orientation to cancel pairwise.
- The signed mean-across-links roll P95 falls from 16.295 deg at wave=1.5 to 10.858 deg at wave=1.0, but that statistic hides cancellation between differently tilted links. The standard production metrics do **not** improve: wave=1.0 has local-roll P95 4.086 deg and chassis-world-roll P95 21.800 deg versus 3.265/20.960 deg at wave=1.5. Forward displacement also falls from 143.736 mm to 128.187 mm. Therefore wave=1.0 is not a better selected gait.

## Mechanics

For the instantaneous body-forward axis, the contact contribution is `tau_roll = sum((r x F) dot e_forward)`. Its dominant vertical-force term is approximately `sum(y_i Fz_i)`, but lateral friction and vertical offsets also contribute. The user's link-plus-leg 'spindle' analogy is mechanically sound: leg-ground reaction can lift and rotate each compliant body link, and passive body-roll joints allow an imbalance to accumulate into a tilted support surface.

Figure: `C:\Users\wxy22\Documents\Centipede_MUJOCO-main\experiments\mujoco_template_roll_balance_audit_2026-08-30\outputs\figures\roll_balance_wave1p5_vs_wave1p0.png`
Data: `C:\Users\wxy22\Documents\Centipede_MUJOCO-main\experiments\mujoco_template_roll_balance_audit_2026-08-30\outputs\data\roll_balance_audit.json`
Wave=1.0 demo audit: `C:\Users\wxy22\Documents\Centipede_MUJOCO-main\experiments\mujoco_template_roll_balance_audit_2026-08-30\outputs\data\diagnostic_wave1p0_demo_audit.json`
