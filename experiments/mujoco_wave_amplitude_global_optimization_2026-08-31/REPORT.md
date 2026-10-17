# Flat-ground wave/amplitude optimization

The body and legs share one wave number. Only body-yaw, leg-yaw, and leg-pitch amplitudes vary. The 343-point grid for every wave is complete and unique. A selected point must pass all 4 s gates and is then checked with the same gates at 6 s.

| wave | status | grid rank | body yaw | leg yaw | leg pitch | 4 s D | 6 s D | feasible | q=.3 retention | q=.3 gate survival |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 s global only; no 6 s-safe point | 1 | 24° | 42° | 18° | 82.1 mm | 149.0 mm | 2/343 | 91.4% | 0% |
| 2 | 4 s global only; no 6 s-safe point | 1 | 12° | 54° | 30° | 82.2 mm | 147.7 mm | 1/343 | 89.4% | 0% |
| 3 | 6 s-safe ranked runner-up | 2 | 16° | 60° | 42° | 86.2 mm | 155.7 mm | 114/343 | 83.0% | 0% |
| 4 | 4 s global + 6 s pass | 1 | 12° | 60° | 36° | 81.0 mm | 145.7 mm | 78/343 | 82.3% | 0% |
| 5 | 6 s-safe ranked runner-up | 2 | 12° | 60° | 42° | 67.3 mm | 120.9 mm | 154/343 | 90.4% | 0% |
| 6 | 4 s global + 6 s pass | 1 | 24° | 48° | 42° | 57.1 mm | 102.9 mm | 88/343 | 87.3% | 0% |
| 7 | 4 s global + 6 s pass | 1 | 20° | 54° | 48° | 61.7 mm | 111.3 mm | 46/343 | 86.4% | 0% |
| 8 | no 4 s feasible point | — | — | — | — | — | — | 0/343 | — | — |
| 9 | no 4 s feasible point | — | — | — | — | — | — | 0/343 | — | — |

Wave 1 and 2 have valid 4 s finite-grid optima but no 4 s-feasible candidate survives every 6 s gate. Wave 8 and 9 have no 4 s feasible point because every point violates the chassis-drag gate. Wave 3 and 5 therefore use rank-2 choices: all higher-ranked points were checked and failed the 6 s gate.

Random contact erasure is post-selection validation, never part of parameter selection. At q=0.3 no tested selected gait survives the entire strict gate set, even though forward displacement is partly retained.

Frozen model: `C:\Users\wxy22\Documents\Centipede_MUJOCO-main\experiments\mujoco_template_locked_head_2026-08-27\models\centipede_flat_locked_head_m29p0.xml`

SHA-256: `3815AB6CC8F6BC60834A9A85C246C19DC6B9FB94BD18DF893E072A0345132520`

Compiled mass: `0.002536031091780036 kg`; nonzero armatures: `0`; step: `0.00025 s`; integrator: `mjINT_IMPLICITFAST`.
