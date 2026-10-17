# 阶段性结果：两个已经复核的候选

**这不是完整搜索的最终结果。** 初轮、新区域细化和全波数筛选仍在进行。
这里的两个候选已经通过六秒、12 秒、半时间步检查，但不称为最终最优。
它们不替换默认模型或默认参数。

仍为同一 local-Z compliant tip，仅优化三个幅值；PD、相位、频率、质量、
惯性、armature、接触模型不变。平地，擦除关闭，body/leg 波数相同。

三个幅值依次为 body yaw / leg yaw / leg pitch，单位为度。
位移统计 t=1.5–5.999 s，沿固定前向轴，使用 21 个身体节点的几何中心。

| Wave | 旧严格最优幅值 | 新候选幅值 | 六秒位移旧→新 | 半时间步位移差异 |
|---:|---|---|---|---:|
| 2 | 16.92 / 49.38 / 20.52 | 16.92 / 56.58 / 22.92 | 185.97 → 196.33 mm | 0.050% |
| 3.5 | 22.65 / 36.38 / 24.18 | 17.18 / 44.97 / 11.28 | 113.80 → 139.97 mm | 0.462% |

12 秒位移分别为 458.32 mm、326.58 mm；统计区间更长，不能直接当作
相对六秒的性能提升。两个候选在相应时长的同一套新门槛下均通过。

## 实际运动与代价

- wave=2：实际 leg-pitch 稳健半幅由 17.97–21.34° 变为 20.40–24.36°。
  整体 roll P95 为 3.51°，最差节瞬时 world roll 为 10.69°，旧值为
  10.52°。抬腿高度 P95 为 6.14 mm，超过旧的 5.5 mm 门槛，但低于新
  门槛 7 mm；不能说所有稳定性/运动学指标都改善了。
- wave=3.5：实际 leg-pitch 稳健半幅由 16.08–20.91° 变为 6.76–12.98°。
  整体 roll P95 为 1.78°，最差节瞬时 world roll 为 6.17°，旧值为
  7.03°。抬腿高度 P95 为 2.84 mm，并不是所有候选都需要更大的 pitch。
- “实际半幅”定义为各关节在 t≥1.5 s 的 (P99−P1)/2，不是正弦拟合幅值。
- 上述逐节极值、实际半幅和抬腿高度来自六秒记录。工程门槛通过不等于
  roll 为零，也不等于已得到生物学校准。

## Demo 和图

视频上排为旧严格最优，下排为新候选；前视、侧视、斜视同时显示。
六秒原始记录以 30 fps 重放，不做轨迹平滑。已核对帧数为 180，并抽样
核对重放位置误差小于 1e-7 m；抽查了 wave=2 的 t=5 s、wave=3.5 的 t=3 s
画面，标签和三个视角可辨。画面抽查不等于人工通看全部帧。

- [wave=2 三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/interim_w2_three_views.mp4)
- [wave=3.5 三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/videos/interim_w3p5_three_views.mp4)
- [实际关节半幅对比图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/figures/interim_actual_amplitude_comparison.png)
- [完整阶段性验证记录](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/data/early_demo_results.json)
- [实际运动和逐节 roll 数据](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/outputs/data/early_joint_motion.json)
- [本轮冻结协议](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_compliant_relaxed_amplitude_2026-09-02/PROTOCOL.md)

上一轮严格实验、Robot Step 18/19 均未修改。
