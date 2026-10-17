# 阶段性结果：跨波数参数挑战，后续搜索尚未完成

模型、控制器、每波数 PD、门槛、相位和频率不变；柔性 tip，平地，擦除关闭。
各波数 PD 仍然不同。仅改变 body yaw / leg yaw / leg pitch 三个幅值。
只有通过 6 s、12 s、半步长 6 s 和位移一致性检查的候选才进入下表。
本报告由程序生成；视频逐帧视觉审查状态见 delivery_audit，不应把文件存在等同于视觉通过。

已有条件 5036；累计不同条件 5235；新增 199。

| wave | body / leg yaw / leg pitch (deg) | D (mm) | 相对上轮 |
|---:|---|---:|---:|
| 0.5 | 未找到通过全部验证的候选 | — | — |
| 1 | 18.36 / 56.58 / 21.72 | 137.481 | +0.96% |
| 1.5 | 15.07 / 55.79 / 18.19 | 165.969 | +0.00% |
| 2 | 14.76 / 63.78 / 19.32 | 201.220 | +0.09% |
| 2.5 | 17.64 / 52.98 / 20.52 | 169.753 | +0.16% |
| 3 | 23.99 / 55.89 / 22.69 | 158.088 | +1.69% |
| 3.5 | 20.06 / 46.77 / 11.28 | 146.068 | +0.00% |
| 4 | 16.04 / 41.14 / 21.99 | 122.849 | +0.85% |
| 4.5 | 24.62 / 52.26 / 21.87 | 127.538 | +0.00% |
| 5 | 22.68 / 51.18 / 22.92 | 117.443 | +0.00% |
| 6 | 7.74 / 26.49 / 24.66 | 62.932 | +0.73% |
| 7 | 8.77 / 20.22 / 25.61 | 56.978 | +0.00% |
| 8 | 未找到通过全部验证的候选 | — | — |
| 9 | 未找到通过全部验证的候选 | — | — |

![幅值](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_amplitude_convergence_2026-09-03/outputs/figures/interim_amplitudes.png)

![性能与实际摆幅](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_amplitude_convergence_2026-09-03/outputs/figures/interim_performance_motion.png)

![body yaw 剖面](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_amplitude_convergence_2026-09-03/outputs/figures/interim_body_profiles.png)

近最佳点仅作为诊断，不增加平滑、单调或幅值差异的奖励。剖面是已测试点的最佳值，不是最优值上界。

搜索停止状态：`{}`。达到预算的起点仍记为未收敛。

![interim_w3_three_views](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_amplitude_convergence_2026-09-03/outputs/videos/interim_w3_three_views.mp4)

![interim_w3_local](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_amplitude_convergence_2026-09-03/outputs/videos/interim_w3_local.mp4)

![interim_w1_three_views](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_amplitude_convergence_2026-09-03/outputs/videos/interim_w1_three_views.mp4)

![interim_w1_local](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_amplitude_convergence_2026-09-03/outputs/videos/interim_w1_local.mp4)
