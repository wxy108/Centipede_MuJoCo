# Centipede Model：刚性腿对照与平地幅值优化

本轮仅处理生物启发的 Centipede Model。没有修改或混用 Centipede Robot 的 Step 18/19 数据，也没有覆盖原基线。

## 主要结果

- 检查了 3,298 组不同幅值组合，10/14 个波数获得通过全部复核的候选。
- 刚性对照采用真正的刚性合并：去掉 38 个 tip 被动自由度，保留髋部 leg yaw/pitch；不添加 knee，不使用超大 stiffness 代替刚性。总质量约 2.53603 g、腿长、几何和匹配惯性保持一致，armature 仍为零。
- 相同参数下，刚性腿没有表现出推进和稳定性的共同优势。因此幅值优化仍使用最新 local-Z 柔性 tip，刚性模型保留为独立对照。这个结果不等于刚性模型经另行调参后必然更差。
- 本轮只改变 body yaw、leg yaw、leg pitch 三个 amplitude；频率 1 Hz，body/leg wave number 一致，phase、tip 参数、质量、惯性、每个波数对应的 PD 均不变，contact erasure 关闭。
- 不同波数的历史 PD 并非完全相同，因此这里是各自固定 PD 条件下的幅值优化，不是单独隔离 wave number 的因果实验。

## 各波数结果

幅值单位为度。D 是 21 个身体节点几何中心在固定前向轴上的位移，不是质量加权质心位移；六秒试验统计区间为 t=1.5–5.999 s。原幅值的 D 仅作参考，并不代表原步态全部稳定。

| Wave | Body yaw / leg yaw / leg pitch | 合格 D (mm) | 原幅值 D (mm) | 状态 |
|---:|---|---:|---:|---|
| 0.5 | — | — | 53.744 | 未解决 |
| 1 | — | — | 123.950 | 未解决 |
| 1.5 | 19.078 / 43.977 / 16.918 | 156.999 | 158.007 | 通过复核 |
| 2 | 16.918 / 49.377 / 20.518 | 185.973 | 183.669 | 通过复核 |
| 2.5 | 19.954 / 40.865 / 24.150 | 142.824 | 156.929 | 通过复核 |
| 3 | 16.918 / 49.377 / 22.918 | 130.941 | 130.941 | 通过复核 |
| 3.5 | 22.655 / 36.382 / 24.184 | 113.803 | 125.018 | 通过复核 |
| 4 | 18.033 / 36.505 / 18.662 | 113.429 | 117.958 | 通过复核 |
| 4.5 | 30.280 / 40.066 / 21.058 | 105.711 | 116.037 | 通过复核 |
| 5 | 16.918 / 42.177 / 21.718 | 97.391 | 110.873 | 通过复核 |
| 6 | 7.739 / 26.489 / 17.459 | 59.413 | 94.516 | 通过复核 |
| 7 | 7.056 / 19.999 / 25.310 | 55.144 | 93.422 | 通过复核 |
| 8 | — | — | 61.260 | 未解决 |
| 9 | — | — | 45.321 | 未解决 |

筛选顺序是稳定性/动作质量/体态/接触与滑移约束先通过，再最大化前向位移。名义六秒候选按位移从高到低接受独立的 12 秒与半时间步（0.125 ms）测试；半时间步的六秒位移差异须不超过 10%。所有比最终入选者更快的名义合格候选均已检查。

连续参数空间的数学全局最优没有得到证明。准确表述是：全范围采样、两轮多起点细化及一轮有限耦合边界采样后的已测试集合内，通过声明验证条件的最高名义位移。未解决不代表不存在可行参数，也不代表所有失败都是翻倒；有的只未通过波形、滑移或跟踪约束。

逐节姿态核验：roll 并未被完全消除。wave=2 最差身体节点的瞬时绝对 roll 为 10.52°，wave=2.5 为 15.42°，wave=3 为 15.79°。不能只看整体平均 roll；逐节姿态包络已单独列入稳定性报告。这是结果诊断，没有新增或放宽门槛。

历史继承的“至少达到参考位移 90%”只作诊断，不作稳定性门槛；其他稳定性/姿态/动作/接触门槛没有因此放宽。细节见 [目标定义说明](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/OBJECTIVE_CLARIFICATION.md)。

## 刚性对照

| 地形 | Wave | 柔性 D (mm) | 刚性 D (mm) | 刚性相对变化 |
|---|---:|---:|---:|---:|
| 平地 | 2.5 | 156.929 | 145.006 | -7.60% |
| 平地 | 3 | 130.941 | 129.333 | -1.23% |
| 粗糙地形 | 2.5 | 123.652 | 99.289 | -19.70% |
| 粗糙地形 | 3 | 128.232 | 104.041 | -18.86% |

刚性平地两组均有稳定性/动作门槛未通过；粗糙地形的两种模型均未获得合格结论，相关视频只作对照诊断。幅值优化仅在平地进行。完整结果见 [刚性对照报告](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/PAIRED_RESULTS.md)。

## 全部 Demo

每个全景视频包含前视、侧视、斜视。优化视频上排是原幅值，下排是新候选；未解决波数明确标为 DIAGNOSTIC ONLY / NOT QUALIFIED。运动来自原始记录，没有对动作做平滑。

- wave=0.5（仅供诊断）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/unresolved_diagnostic_w0p5_three_views.mp4)
- wave=1（仅供诊断）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/unresolved_diagnostic_w1_three_views.mp4)
- wave=1.5（通过复核）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/optimized_w1p5_three_views.mp4)
- wave=2（通过复核）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/optimized_w2_three_views.mp4)
- wave=2.5（通过复核）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/optimized_w2p5_three_views.mp4)
- wave=3（通过复核）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/optimized_w3_three_views.mp4)
- wave=3.5（通过复核）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/optimized_w3p5_three_views.mp4)
- wave=4（通过复核）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/optimized_w4_three_views.mp4)
- wave=4.5（通过复核）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/optimized_w4p5_three_views.mp4)
- wave=5（通过复核）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/optimized_w5_three_views.mp4)
- wave=6（通过复核）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/optimized_w6_three_views.mp4)
- wave=7（通过复核）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/optimized_w7_three_views.mp4)
- wave=8（仅供诊断）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/unresolved_diagnostic_w8_three_views.mp4)
- wave=9（仅供诊断）：[三视图 Demo](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/unresolved_diagnostic_w9_three_views.mp4)

柔性/刚性对照：上排柔性、下排刚性；局部视频另有未弯曲参考线。

- 平地 wave=2.5：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_flat_w2p5_three_views.mp4)；[局部放大](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_flat_w2p5_local.mp4)
- 平地 wave=3：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_flat_w3_three_views.mp4)；[局部放大](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_flat_w3_local.mp4)
- 粗糙地形 wave=2.5：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_rough_w2p5_three_views.mp4)；[局部放大](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_rough_w2p5_local.mp4)
- 粗糙地形 wave=3：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_rough_w3_three_views.mp4)；[局部放大](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/videos/paired_rough_w3_local.mp4)

## 参数、图与审计

- [全精度合格幅值及固定 PD 参数表](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/data/selected_amplitude_profiles.json)
- [三个指令幅值随波数变化](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/figures/amplitudes_by_wave.png)
- [前向位移对比](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/figures/displacement_by_wave.png)
- [体态与接触指标图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/figures/stability_by_wave.png)
- [体态、tip 转角与时间步复核数值](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/STABILITY_RESULTS.md)
- [实际关节运动数据及各波数图路径](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/data/achieved_joint_motion.json)
- [代表性画面与逐节姿态核验说明](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/VISUAL_QA.md)
- [完整复核记录](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/data/validated_selection.json)
- [最终文件、参数与选择审计](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_rigid_control_amplitude_2026-09-02/outputs/data/final_delivery_audit.json)

这些通过条件是本轮工程化验证标准，不等于完成了生物学动态参数标定，也不保证任意地形、扰动或 random-contact 条件下稳定。
