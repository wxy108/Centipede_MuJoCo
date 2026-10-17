# Selected gait stability summary

Nominal 6 s metrics, t >= 1.5 s. Selection also passed 12 s and half-timestep tests. Gates remain engineering criteria, not biological parameter calibration.

| Wave | D (mm) | Whole / front roll P95 (deg) | Height range (mm) | Peak ground force (BW) | Tip P95 / peak (deg) | Half-step D difference |
|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 156.999 | 7.84 / 7.90 | 0.822 | 1.136 | 11.93 / 28.93 | 3.61% |
| 2 | 185.973 | 3.43 / 5.11 | 0.408 | 1.126 | 9.72 / 19.03 | 0.27% |
| 2.5 | 142.824 | 3.76 / 5.79 | 0.627 | 1.175 | 7.16 / 19.13 | 1.97% |
| 3 | 130.941 | 2.33 / 5.84 | 0.466 | 1.244 | 11.03 / 17.88 | 0.54% |
| 3.5 | 113.803 | 2.69 / 3.59 | 0.186 | 1.069 | 5.62 / 10.50 | 0.27% |
| 4 | 113.429 | 1.54 / 2.98 | 0.330 | 1.146 | 6.72 / 13.67 | 0.22% |
| 4.5 | 105.711 | 1.43 / 2.95 | 0.579 | 1.249 | 4.52 / 10.32 | 0.56% |
| 5 | 97.391 | 0.94 / 2.34 | 0.233 | 1.146 | 4.98 / 10.19 | 0.24% |
| 6 | 59.413 | 2.70 / 3.62 | 0.357 | 1.322 | 4.02 / 11.14 | 0.03% |
| 7 | 55.144 | 1.10 / 2.29 | 0.280 | 1.055 | 1.53 / 4.65 | 0.11% |

## Individual-body roll envelope

Whole-body averaging can cancel opposite local rotations. These post-selection diagnostics do not change the gates or the selected parameters. Passing is not equivalent to zero roll.

| Wave | Worst individual body absolute P95 / peak (deg) | Worst relative roll joint peak (deg) | Mean geometric-center height (mm) |
|---:|---:|---:|---:|
| 1.5 | 12.84 / 13.60 | 4.05 | 4.635 |
| 2 | 10.09 / 10.52 | 0.56 | 5.756 |
| 2.5 | 14.54 / 15.42 | 2.99 | 5.454 |
| 3 | 14.44 / 15.79 | 5.20 | 5.933 |
| 3.5 | 6.80 / 7.03 | 2.32 | 5.220 |
| 4 | 8.58 / 9.02 | 2.92 | 4.981 |
| 4.5 | 5.40 / 6.18 | 2.64 | 5.031 |
| 5 | 4.84 / 5.61 | 2.60 | 4.983 |
| 6 | 5.92 / 6.15 | 2.70 | 4.648 |
| 7 | 4.93 / 5.26 | 2.22 | 5.260 |

Large single-step angular acceleration remains a diagnostic in the raw records; the jump gates operate on 10 ms angle/velocity changes and explicit velocity/force limits. Passing is not a claim that every instantaneous acceleration is biologically calibrated.

Full values and validation rollouts: outputs/data/stability_comparison.json. Actual per-joint motion: outputs/data/achieved_joint_motion.json and outputs/figures/joint_motion_w*.png.
