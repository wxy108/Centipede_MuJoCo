# First-pitch support audit

This experiment changes only `joint_pitch_body_1` effective Kp/Kd in the biological/template Centipede Model.

Selected candidate: **none; baseline retained**.

| Candidate | Kp/Kd scale | Flat q1 P95 | Flat link1 contact | Rough q1 P95 | Rough link1 contact | Rough displacement | Rough tip-stop | Both-terrain gates |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 1/1 | 4.19 deg | 4.26% | 7.51 deg | 8.64% | 91.14 mm | 0.987% | reference |
| k0p75_zeta | 0.75/0.866 | 5.62 deg | 4.63% | 7.61 deg | 7.53% | 53.99 mm | 1.017% | False |
| k0p75_d1 | 0.75/1 | 5.66 deg | 4.78% | 7.84 deg | 9.91% | 52.20 mm | 1.173% | False |
| k1p5_zeta | 1.5/1.22 | 2.87 deg | 3.72% | 4.78 deg | 9.81% | 91.29 mm | 1.188% | False |
| k2_zeta | 2/1.41 | 2.19 deg | 3.51% | 3.53 deg | 11.59% | 91.24 mm | 1.052% | False |
| k1p5_d1p5 | 1.5/1.5 | 2.80 deg | 3.61% | 4.93 deg | 10.40% | 90.50 mm | 1.191% | False |

Passing candidates: [].
Robot sentinels unchanged: True.

Selection uses explicit gates and no scalar reward.  Since no candidate passes both terrains, the production baseline is unchanged; k1p5_zeta demos are diagnostic only.
