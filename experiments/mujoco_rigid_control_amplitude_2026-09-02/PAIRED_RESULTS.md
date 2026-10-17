# Fully rigid leg versus current compliant tip

Centipede Model only. Same amplitudes, wave-specific PD, mass, neutral leg geometry, frequency, phases, contact settings, initialization procedure and simulation step. No knee reinstated; all 38 distal tip joints removed by rigid aggregation, not by an artificially large spring.

Forward displacement is the world-forward displacement of the unweighted mean of the 21 chassis-body positions (geometric center, not mass-weighted COM), from 1.5 s to 5.999 s of a 6 s simulation. Values below are not full-duration displacement. Rough is diagnostic only.

| Terrain | Wave | Flexible D (mm) | Rigid D (mm) | Change | Absolute gates flexible / rigid |
|---|---:|---:|---:|---:|---|
| flat | 2.5 | 156.929 | 145.006 | -7.60% | PASS / FAIL |
| flat | 3 | 130.941 | 129.333 | -1.23% | PASS / FAIL |
| rough | 2.5 | 123.652 | 99.289 | -19.70% | FAIL / FAIL |
| rough | 3 | 128.232 | 104.041 | -18.86% | FAIL / FAIL |

At the tested identical PD and amplitudes, removing tip compliance did not improve either forward displacement or gate feasibility. This does not prove rigid legs can never work with a different controller; no rigid-specific tuning is mixed into this controlled comparison. The flexible model remains the primary plant.

The flat flexible cases pass absolute gates, which is not the same as passing every legacy non-regression / horizon / timestep check. Neither rough case is certified.

## Individual records and metrics

### flat — wave 2.5 — rigid

Record: [rigid_flat_w2p5_8ee11daeefcc0f](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/trials/ablation/rigid_flat_w2p5_8ee11daeefcc0f/rigid_flat_w2p5_8ee11daeefcc0f.json)

Failed applicable absolute gates: body_yaw_speed, body_velocity_jump, leg_yaw_speed, leg_pitch_speed, swing_height.

| Metric | Value |
|---|---:|
| root_roll_abs_p95_deg | 9.53822 |
| coherent_roll_p95_deg | 3.95065 |
| front_roll_p95_deg | 6.97559 |
| center_height_ptp_mm | 1.11161 |
| ground_force_peak_bw | 1.6617 |
| chassis_force_peak_bw | 0.473914 |
| body_deltaomega_10ms_peak | 12.2646 |
| loaded_foot_slip_p95_mm_s | 39.0427 |
| left_right_force_imbalance_rms_bw | 0.087034 |

### flat — wave 2.5 — flexible

Record: [flexible_flat_w2p5_bc2380a753982a](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/trials/ablation/flexible_flat_w2p5_bc2380a753982a/flexible_flat_w2p5_bc2380a753982a.json)

Failed applicable absolute gates: none.

| Metric | Value |
|---|---:|
| root_roll_abs_p95_deg | 9.58049 |
| coherent_roll_p95_deg | 3.74776 |
| front_roll_p95_deg | 5.1165 |
| center_height_ptp_mm | 0.398375 |
| ground_force_peak_bw | 1.47867 |
| chassis_force_peak_bw | 0.438542 |
| body_deltaomega_10ms_peak | 6.44684 |
| loaded_foot_slip_p95_mm_s | 29.6694 |
| left_right_force_imbalance_rms_bw | 0.047826 |
| tip_angle_p95_deg | 7.10151 |
| tip_angle_peak_deg | 18.6748 |

### flat — wave 3 — rigid

Record: [rigid_flat_w3_c4c9157a49cb30](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/trials/ablation/rigid_flat_w3_c4c9157a49cb30/rigid_flat_w3_c4c9157a49cb30.json)

Failed applicable absolute gates: pitch_10ms, active_speed, leg_yaw_speed, leg_pitch_speed.

| Metric | Value |
|---|---:|
| root_roll_abs_p95_deg | 9.91096 |
| coherent_roll_p95_deg | 2.8782 |
| front_roll_p95_deg | 5.75997 |
| center_height_ptp_mm | 0.604745 |
| ground_force_peak_bw | 1.55681 |
| chassis_force_peak_bw | 0 |
| body_deltaomega_10ms_peak | 8.39954 |
| loaded_foot_slip_p95_mm_s | 23.6878 |
| left_right_force_imbalance_rms_bw | 0.0593892 |

### flat — wave 3 — flexible

Record: [flexible_flat_w3_2d41279d135962](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/trials/ablation/flexible_flat_w3_2d41279d135962/flexible_flat_w3_2d41279d135962.json)

Failed applicable absolute gates: none.

| Metric | Value |
|---|---:|
| root_roll_abs_p95_deg | 11.6811 |
| coherent_roll_p95_deg | 2.32652 |
| front_roll_p95_deg | 5.83887 |
| center_height_ptp_mm | 0.465855 |
| ground_force_peak_bw | 1.24394 |
| chassis_force_peak_bw | 0 |
| body_deltaomega_10ms_peak | 5.87535 |
| loaded_foot_slip_p95_mm_s | 21.326 |
| left_right_force_imbalance_rms_bw | 0.0343733 |
| tip_angle_p95_deg | 11.0338 |
| tip_angle_peak_deg | 17.8767 |

### rough — wave 2.5 — rigid

Record: [rigid_rough_w2p5_7e0d77e745a63d](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/trials/ablation/rigid_rough_w2p5_7e0d77e745a63d/rigid_rough_w2p5_7e0d77e745a63d.json)

Failed applicable absolute gates: path_efficiency, spine, root_roll_peak, height, coherent_roll, front_roll, body_yaw_speed, body_pitch_speed, body_roll_speed, body_velocity_jump, pitch_10ms, total_force, chassis_force, foot_force, chassis_contact, cycle_consistency, body_wave, leg_yaw_wave, leg_yaw_speed, leg_pitch_speed, leg_yaw_reversals, swing_height.

| Metric | Value |
|---|---:|
| root_roll_abs_p95_deg | 22.42 |
| coherent_roll_p95_deg | 12.2248 |
| front_roll_p95_deg | 15.8767 |
| center_height_ptp_mm | 2.66719 |
| ground_force_peak_bw | 4.95626 |
| chassis_force_peak_bw | 4.31451 |
| body_deltaomega_10ms_peak | 19.8624 |
| loaded_foot_slip_p95_mm_s | 48.4648 |
| left_right_force_imbalance_rms_bw | 0.264815 |

### rough — wave 3 — rigid

Record: [rigid_rough_w3_69eaa380b581c5](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/trials/ablation/rigid_rough_w3_69eaa380b581c5/rigid_rough_w3_69eaa380b581c5.json)

Failed applicable absolute gates: root_roll_peak, height, coherent_roll, front_roll, body_pitch_speed, body_velocity_jump, pitch_10ms, roll_10ms, total_force, chassis_force, foot_force, chassis_contact, active_speed, leg_yaw_wave, leg_yaw_speed, leg_pitch_speed, swing_height.

| Metric | Value |
|---|---:|
| root_roll_abs_p95_deg | 22.5626 |
| coherent_roll_p95_deg | 11.8877 |
| front_roll_p95_deg | 18.1583 |
| center_height_ptp_mm | 3.29648 |
| ground_force_peak_bw | 6.9606 |
| chassis_force_peak_bw | 6.27866 |
| body_deltaomega_10ms_peak | 14.979 |
| loaded_foot_slip_p95_mm_s | 33.1799 |
| left_right_force_imbalance_rms_bw | 0.205235 |

### rough — wave 3 — flexible

Record: [flexible_rough_w3_aedba796d9fa0e](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/trials/ablation/flexible_rough_w3_aedba796d9fa0e/flexible_rough_w3_aedba796d9fa0e.json)

Failed applicable absolute gates: root_roll_peak, front_roll, body_velocity_jump, total_force, chassis_force, foot_force, swing_height.

| Metric | Value |
|---|---:|
| root_roll_abs_p95_deg | 21.1401 |
| coherent_roll_p95_deg | 9.4194 |
| front_roll_p95_deg | 14.0549 |
| center_height_ptp_mm | 1.35606 |
| ground_force_peak_bw | 9.01295 |
| chassis_force_peak_bw | 2.1758 |
| body_deltaomega_10ms_peak | 12.0488 |
| loaded_foot_slip_p95_mm_s | 22.1798 |
| left_right_force_imbalance_rms_bw | 0.169101 |
| tip_angle_p95_deg | 11.9165 |
| tip_angle_peak_deg | 31.0772 |

### rough — wave 2.5 — flexible

Record: [flexible_rough_w2p5_7d77ac431bf1d9](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/trials/ablation/flexible_rough_w2p5_7d77ac431bf1d9/flexible_rough_w2p5_7d77ac431bf1d9.json)

Failed applicable absolute gates: root_roll_peak, front_roll, body_yaw_speed, body_pitch_speed, body_roll_speed, body_velocity_jump, total_force, chassis_force, foot_force, tip_speed, cycle_consistency, leg_yaw_speed, swing_height.

| Metric | Value |
|---|---:|
| root_roll_abs_p95_deg | 23.0965 |
| coherent_roll_p95_deg | 8.94337 |
| front_roll_p95_deg | 16.3227 |
| center_height_ptp_mm | 1.95868 |
| ground_force_peak_bw | 7.22567 |
| chassis_force_peak_bw | 4.71308 |
| body_deltaomega_10ms_peak | 16.1424 |
| loaded_foot_slip_p95_mm_s | 67.6219 |
| left_right_force_imbalance_rms_bw | 0.172895 |
| tip_angle_p95_deg | 11.4593 |
| tip_angle_peak_deg | 37.243 |

## Demos

Every video contains three simultaneous views; top row flexible (orange tip), bottom row rigid (cyan tip). Raw saved poses, no motion smoothing. Local enlargement files show neutral/actual tip direction. Contact erasure is OFF and labelled.

- [paired_flat_w2p5_three_views.mp4](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_flat_w2p5_three_views.mp4)
- [paired_flat_w2p5_local.mp4](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_flat_w2p5_local.mp4)
- [paired_flat_w3_three_views.mp4](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_flat_w3_three_views.mp4)
- [paired_flat_w3_local.mp4](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_flat_w3_local.mp4)
- [paired_rough_w2p5_three_views.mp4](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_rough_w2p5_three_views.mp4)
- [paired_rough_w2p5_local.mp4](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_rough_w2p5_local.mp4)
- [paired_rough_w3_three_views.mp4](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_rough_w3_three_views.mp4)
- [paired_rough_w3_local.mp4](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_rough_w3_local.mp4)

## Resource-only implementation change

Search and rendering compile an in-memory XML copy with a 64 MiB arena reservation instead of the source 2.18 GB reservation. Only legacy size/nstack and size/njmax are replaced by size/memory; nconmax and all physical settings remain unchanged. The source model files are not edited. This is memory capacity, not solver accuracy or dynamics tuning.
The official [MuJoCo 3.3.7 size documentation](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#size) defines this setting. Exact model-array and dynamics checks are recorded in outputs/data/runtime_arena_equivalence.json.
