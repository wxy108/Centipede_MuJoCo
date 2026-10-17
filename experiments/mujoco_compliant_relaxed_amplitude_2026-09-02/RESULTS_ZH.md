# Compliant tip：放宽门槛后的幅值优化

检查 5,036 个不同幅值组合；11/14 个波数通过六秒、12 秒和半时间步复核。
9 个合格波数出现明确幅值变化（任一幅值变化 ≥5°，或相对变化 ≥25%，分母至少 5°）。这仅是报告指标，没有用它强迫优化结果变化。

仍为同一个 local-Z compliant-tip Model；仅优化 body yaw、leg yaw、leg pitch 三个幅值。body/leg 波数相同、频率 1 Hz、每波数 PD 固定、相位与模型不变、contact erasure 关闭。不同波数的历史 PD 并非完全相同。Robot Step 18/19 和上一轮严格实验未修改。

参数文件中的 PD 配置值是相对参考模型的倍率，不是 SI 单位的绝对刚度/阻尼。roll 倍率作用于被动 body-roll 弹簧/阻尼，其余倍率作用于主动 PD。历史 PD 来源的原始状态标签予以保留；本轮验证的是相应 PD 与新幅值的完整组合，不把历史标签误作本轮验证结论。

## 新旧幅值直接对比

三个幅值顺序为 body yaw / leg yaw / leg pitch，单位为度。位移 D 是 21 个身体节点几何中心沿固定前向轴的位移（非质量加权质心），统计 t=1.5–5.999 s。带 * 的波数旧严格搜索未找到合格解，旧值仅为原固定幅值，不是旧合格最优。

图中的蓝色点线仅作原始固定幅值参照，不保证通过当前门槛，也不把这条参考曲线整体标为合格最优。表中原固定 D 带 †、位移图中的蓝色叉号，均表示该固定设置在当前六秒门槛下不合格；未标叉也不等于已通过本轮全部长时复核。

| Wave | 旧幅值（°） | 新幅值（°） | D 旧→新（mm） | 原固定 D（mm） | 最大幅值变化（°） | 状态 |
|---:|---|---|---|---:|---:|---|
| 0.5 * | 16.92 / 49.38 / 22.92 | — | 53.74 → — | 53.74 † | — | 未解决 |
| 1 * | 16.92 / 49.38 / 22.92 | 17.64 / 56.58 / 21.72 | 123.95 → 136.18 | 123.95 † | 7.20 | 合格，幅值明显变化 |
| 1.5 | 19.08 / 43.98 / 16.92 | 15.07 / 55.79 / 18.19 | 157.00 → 165.97 | 158.01 † | 11.81 | 合格，幅值明显变化 |
| 2 | 16.92 / 49.38 / 20.52 | 14.76 / 63.78 / 18.12 | 185.97 → 201.04 | 183.67 | 14.40 | 合格，幅值明显变化 |
| 2.5 | 19.95 / 40.86 / 24.15 | 17.64 / 52.98 / 19.32 | 142.82 → 169.48 | 156.93 | 12.11 | 合格，幅值明显变化 |
| 3 | 16.92 / 49.38 / 22.92 | 23.99 / 54.09 / 22.69 | 130.94 → 155.47 | 130.94 | 7.07 | 合格，幅值明显变化 |
| 3.5 | 22.65 / 36.38 / 24.18 | 20.06 / 46.77 / 11.28 | 113.80 → 146.07 | 125.02 | 12.90 | 合格，幅值明显变化 |
| 4 | 18.03 / 36.50 / 18.66 | 15.32 / 41.14 / 21.99 | 113.43 → 121.81 | 117.96 † | 4.64 | 合格 |
| 4.5 | 30.28 / 40.07 / 21.06 | 24.62 / 52.26 / 21.87 | 105.71 → 127.54 | 116.04 | 12.19 | 合格，幅值明显变化 |
| 5 | 16.92 / 42.18 / 21.72 | 22.68 / 51.18 / 22.92 | 97.39 → 117.44 | 110.87 † | 9.00 | 合格，幅值明显变化 |
| 6 | 7.74 / 26.49 / 17.46 | 7.74 / 26.49 / 23.46 | 59.41 → 62.47 | 94.52 † | 6.00 | 合格，幅值明显变化 |
| 7 | 7.06 / 20.00 / 25.31 | 8.77 / 20.22 / 25.61 | 55.14 → 56.98 | 93.42 † | 1.71 | 合格 |
| 8 * | 16.92 / 49.38 / 22.92 | — | 61.26 → — | 61.26 † | — | 未解决 |
| 9 * | 16.92 / 49.38 / 22.92 | — | 45.32 → — | 45.32 † | — | 未解决 |

[三幅值新旧对比图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/figures/amplitudes_strict_vs_relaxed.png)；[幅值变化热图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/figures/amplitude_differences.png)；[位移对比](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/figures/displacement_strict_vs_relaxed.png)

## 放宽了什么、保留了什么

抬腿高度 P95：5.5→7 mm；腿 yaw/pitch 速度 P99：8/7→10/9 rad/s；身体/腿 yaw/腿 pitch 波形残差：.40/.18/.40→.60/.25/.55。滑移和左右支撑力不平衡允许达到同模型参考的 1.35 倍，单位位移功耗 1.50 倍，10 ms 冲量 1.25 倍。旧 tip 轴参考与额外的参考相关姿态约束只作诊断。接触占空比等次级步态限制也适度放宽，完整阈值见 [冻结的实验协议](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/PROTOCOL.md)。

防翻、整体/前段/root roll、body 关节速度和跳变、严重撞地、穿透、力矩、tip 限位和跟踪误差等绝对底线保持不变。body yaw 最大搜索幅值仍为 36°，不跨模型关节限位。放宽门槛不代表改了刚度、阻尼、质量或惯性。

旧完整数据重新判定；不再成立的旧早停结论重新计算。加入每波数 64 个新 Sobol 样本，再做两轮多起点细化。合格候选按六秒位移从高到低复核，全部更快名义合格候选均已检查。所得是已测试集合内、放宽后的门槛下最高合格位移，不是连续空间的数学全局最优。

## 稳定性与代价

| Wave | 整体/前段 roll P95（°） | 最差身体节点 roll 峰值（°） | 高度波动（mm） | 峰值地面力（BW） | tip P95/峰值（°） | 半步长 D 差异 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.99/11.78 | 15.88 | 0.307 | 1.166 | 7.43/9.51 | 9.50% |
| 1.5 | 9.17/8.44 | 15.45 | 0.879 | 1.167 | 15.35/30.30 | 2.00% |
| 2 | 3.07/4.35 | 10.08 | 0.538 | 1.211 | 13.70/26.13 | 0.23% |
| 2.5 | 3.19/4.63 | 11.08 | 0.391 | 1.091 | 6.94/18.95 | 1.88% |
| 3 | 1.71/6.29 | 14.66 | 0.583 | 1.357 | 11.35/18.01 | 0.01% |
| 3.5 | 1.77/2.72 | 6.34 | 0.252 | 1.081 | 6.75/17.13 | 0.33% |
| 4 | 1.26/3.91 | 8.84 | 0.282 | 1.168 | 6.27/20.49 | 0.79% |
| 4.5 | 1.56/3.46 | 6.75 | 0.734 | 1.259 | 7.93/18.18 | 0.82% |
| 5 | 1.24/2.92 | 7.88 | 0.534 | 1.182 | 6.36/22.52 | 0.37% |
| 6 | 3.19/4.41 | 7.40 | 0.560 | 1.358 | 4.71/14.27 | 0.06% |
| 7 | 1.32/2.38 | 5.53 | 0.296 | 1.100 | 1.79/10.03 | 0.35% |

稳定不等于 roll 为零。这里同时报告逐节 world-roll 极值，避免平均值掩盖局部偏转；实际关节半幅也单独保存。单步角加速度仍作为诊断保留，合格并不意味着所有瞬时加速度已得到生物学校准。

[稳定性与接触新旧对比图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/figures/stability_strict_vs_relaxed.png)

接近门槛的情况需要单独注意：

- wave=1：整体/前段 roll P95 为 5.99°/11.78°（上限 10°/12°）；半步长位移差异 9.50%（上限 10%）。通过不等于余量充足，也不等于消除了低波数 wobble。
- wave=1.5：整体/前段 roll P95 为 9.17°/8.44°（上限 10°/12°）；半步长位移差异 2.00%（上限 10%）。通过不等于余量充足，也不等于消除了低波数 wobble。

具体收益与代价（相对旧严格最优，不是原固定幅值）：

- wave=1.5：位移 +5.71%；载荷足滑移 P95 141.74→146.43 mm/s；整体 roll P95 7.84°→9.17°。
- wave=2：位移 +8.10%；载荷足滑移 P95 33.78→40.55 mm/s；整体 roll P95 3.43°→3.07°。
- wave=3.5：位移 +28.35%；载荷足滑移 P95 16.93→26.89 mm/s；整体 roll P95 2.69°→1.77°。

新候选在旧严格规则下仍未通过的项目（六秒、12 秒、半步长分别检查）：

- wave=1: nominal_6s: body_wave, leg_yaw_speed, fixed_amp_loaded_foot_slip_p95_mm_s, fixed_amp_left_right_force_imbalance_rms_bw；long_12s: body_wave, leg_yaw_speed, fixed_amp_loaded_foot_slip_p95_mm_s, fixed_amp_left_right_force_imbalance_rms_bw；half_step_6s: body_wave, fixed_amp_loaded_foot_slip_p95_mm_s, fixed_amp_left_right_force_imbalance_rms_bw
- wave=1.5: nominal_6s: leg_yaw_speed；long_12s: leg_yaw_speed；half_step_6s: fixed_amp_work_per_distance
- wave=2: nominal_6s: leg_yaw_speed, fixed_amp_work_per_distance, fixed_amp_loaded_foot_slip_p95_mm_s；long_12s: leg_yaw_speed, fixed_amp_work_per_distance, fixed_amp_loaded_foot_slip_p95_mm_s；half_step_6s: leg_yaw_speed, fixed_amp_work_per_distance, fixed_amp_loaded_foot_slip_p95_mm_s
- wave=2.5: nominal_6s: fixed_amp_loaded_foot_slip_p95_mm_s, legacy_axis_left_right_force_imbalance_rms_bw；long_12s: fixed_amp_loaded_foot_slip_p95_mm_s, fixed_amp_left_right_force_imbalance_rms_bw, legacy_axis_left_right_force_imbalance_rms_bw；half_step_6s: fixed_amp_loaded_foot_slip_p95_mm_s, legacy_axis_left_right_force_imbalance_rms_bw
- wave=3: nominal_6s: leg_yaw_speed, swing_height, fixed_amp_loaded_foot_slip_p95_mm_s, fixed_amp_left_right_force_imbalance_rms_bw, legacy_axis_loaded_foot_slip_p95_mm_s, legacy_axis_left_right_force_imbalance_rms_bw；long_12s: leg_yaw_speed, swing_height, fixed_amp_loaded_foot_slip_p95_mm_s, fixed_amp_left_right_force_imbalance_rms_bw, legacy_axis_loaded_foot_slip_p95_mm_s, legacy_axis_left_right_force_imbalance_rms_bw；half_step_6s: leg_yaw_speed, swing_height, fixed_amp_loaded_foot_slip_p95_mm_s, fixed_amp_left_right_force_imbalance_rms_bw, legacy_axis_loaded_foot_slip_p95_mm_s, legacy_axis_left_right_force_imbalance_rms_bw
- wave=3.5: nominal_6s: leg_pitch_wave, leg_pitch_reversals, fixed_amp_work_per_distance, fixed_amp_loaded_foot_slip_p95_mm_s；long_12s: leg_pitch_wave, leg_yaw_reversals, leg_pitch_reversals, fixed_amp_work_per_distance, fixed_amp_loaded_foot_slip_p95_mm_s；half_step_6s: leg_pitch_wave, leg_pitch_reversals, fixed_amp_work_per_distance, fixed_amp_loaded_foot_slip_p95_mm_s
- wave=4: nominal_6s: leg_pitch_speed；long_12s: leg_pitch_speed；half_step_6s: leg_pitch_speed
- wave=4.5: nominal_6s: body_wave, leg_yaw_speed, leg_pitch_speed, fixed_amp_work_per_distance, fixed_amp_loaded_foot_slip_p95_mm_s；long_12s: body_wave, leg_yaw_speed, leg_pitch_speed, fixed_amp_work_per_distance, fixed_amp_loaded_foot_slip_p95_mm_s；half_step_6s: body_wave, leg_yaw_speed, fixed_amp_work_per_distance, fixed_amp_loaded_foot_slip_p95_mm_s, fixed_amp_left_right_force_imbalance_rms_bw
- wave=5: nominal_6s: body_wave, leg_yaw_speed, leg_pitch_speed, leg_pitch_reversals, fixed_amp_loaded_foot_slip_p95_mm_s；long_12s: body_wave, leg_yaw_speed, leg_pitch_speed, fixed_amp_loaded_foot_slip_p95_mm_s；half_step_6s: body_wave, leg_pitch_speed, fixed_amp_loaded_foot_slip_p95_mm_s
- wave=6: nominal_6s: leg_pitch_speed, leg_pitch_reversals；long_12s: leg_pitch_speed, leg_pitch_reversals；half_step_6s: leg_pitch_speed
- wave=7: nominal_6s: leg_pitch_speed；long_12s: leg_pitch_speed；half_step_6s: leg_pitch_speed

## 尚未解决的波数

以下仅列出用于 Demo 的完整诊断轨迹及其失败项，不是合格优化结果，也不代表该波数的所有候选都因同一原因失败。有限搜索未找到合格解，不等于证明不存在合格解。

| Wave | 诊断位移（mm，不计入合格成绩） | 该轨迹未通过的六秒检查 |
|---:|---:|---|
| 0.5 | 33.99 | feet_count, duty |
| 8 | 35.14 | chassis_contact, pitch_tracking |
| 9 | 24.87 | path_efficiency, chassis_contact |

## Demo

每个全景视频均有前视、侧视、斜视，上排为旧严格最优（旧搜索未找到合格解时显示原固定幅值），下排为新结果。未解决波数明确标为 NOT QUALIFIED。使用原始运动记录，不做动作平滑。

- wave=0.5，仅供诊断：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_unresolved_w0p5_three_views.mp4)
- wave=1，合格：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_optimized_w1_three_views.mp4)
- wave=1.5，合格：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_optimized_w1p5_three_views.mp4)
- wave=2，合格：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_optimized_w2_three_views.mp4)
- wave=2.5，合格：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_optimized_w2p5_three_views.mp4)
- wave=3，合格：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_optimized_w3_three_views.mp4)
- wave=3.5，合格：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_optimized_w3p5_three_views.mp4)
- wave=4，合格：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_optimized_w4_three_views.mp4)
- wave=4.5，合格：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_optimized_w4p5_three_views.mp4)
- wave=5，合格：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_optimized_w5_three_views.mp4)
- wave=6，合格：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_optimized_w6_three_views.mp4)
- wave=7，合格：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_optimized_w7_three_views.mp4)
- wave=8，仅供诊断：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_unresolved_w8_three_views.mp4)
- wave=9，仅供诊断：[三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_unresolved_w9_three_views.mp4)
- wave=2，合格：[局部放大](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_optimized_w2_local.mp4)
- wave=3.5，合格：[局部放大](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/relaxed_optimized_w3p5_local.mp4)

## 全精度结果与证据

- [合格参数及固定 PD](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/data/selected_amplitude_profiles.json)
- [新旧幅值、位移与严格规则复查](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/data/comparison_table.json)
- [实际关节运动及逐节 roll](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/data/achieved_joint_motion.json)
- [完整验证结果](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/data/validated_selection.json)
- [最终审计](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/data/final_delivery_audit.json)

- [画面抽查范围、局部偏转与边界提醒](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/VISUAL_QA.md)

本轮结论仅适用于所测试的平地工况、固定控制器和模型；不等于生物学参数标定，也不保证 rough terrain、扰动或 random-contact 条件下成立。
