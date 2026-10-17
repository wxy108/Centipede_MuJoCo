# 历史振幅优化核对：统一 PD 的约束

2026-09-08。此次仅检查历史代码、输入和结果，并重新绘图；没有运行物理仿真，没有修改 PD、模型或筛选门槛。

用户记忆有对应的历史结果：RL 后期确实得到低波数 body 振幅总体较大的曲线，MuJoCo 8 月底固定同一套 PD 的优化也有类似总体趋势。这些数值就是 body-yaw 关节目标振幅，不是把横向位移误称为关节振幅。两组结果都存在非单调点，不能解释为任何模型下都必须严格递减。

![历史完整曲线](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/amplitude_history_audit_2026-09-08/historical_body_amplitudes.png)

**当前必须遵守的约束：所有 wave 使用同一份完整 PD 参数，优化过程中固定不变。** “同一套”指各关节及关节族的参数向量在不同波数之间完全一致，仍允许不同关节族本来具有不同增益。不能只在每个 wave 内固定、却让不同 wave 使用不同 PD。

上一轮 recovery 的 13 个入选波数实际用了 8 套不同 PD。其位移结果仍是那些具体控制器下的计算结果，但不满足现在要求的统一 PD 振幅研究，不能作为下一轮统一 PD random-contact 比较的最终基线。这里修正的是实验适用范围，未改写已封存的数据。

**RL 中找到的那组结果。** 7 月 21 日 v4 的全部 13 个波数如下。源配置的 body 基础振幅是 0.4 rad，表中角度由 `0.4 × body_amp` 换算。

| Wave | body_amp 倍率 | body-yaw 目标振幅 |
|---:|---:|---:|
| 1.50 | 1.407855 | 32.266° |
| 1.75 | 1.390455 | 31.867° |
| 2.00 | 1.408736 | 32.286° |
| 2.25 | 1.386049 | 31.766° |
| 2.50 | 1.370238 | 31.404° |
| 2.75 | 1.390706 | 31.873° |
| 3.00 | 1.398013 | 32.040° |
| 3.25 | 1.374389 | 31.499° |
| 3.50 | 1.350765 | 30.957° |
| 3.75 | 1.308550 | 29.990° |
| 4.00 | 1.297738 | 29.742° |
| 4.25 | 1.230982 | 28.212° |
| 4.50 | 1.164225 | 26.682° |

来源：[完整 v4 参数](C:/Users/wxy22/Documents/Centipede_IsaacLab_RL/outputs/report/p8_flat_roll_causal_summary_2026-07-21/data/flat_locked_gait_map_v4_13knot.json)、[当时报告](C:/Users/wxy22/Documents/Centipede_IsaacLab_RL/outputs/report/p8_flat_roll_causal_summary_2026-07-21/reports/FLAT_ROLL_CAUSAL_REPORT.md)、[0.4 rad 基础配置](C:/Users/wxy22/Documents/Centipede_IsaacLab_RL/outputs/analysis/p8_wave_amplitude_map_2026-07-20/cfg/farms_controller_head8_ramp025.yaml)。已逐一核对实际运行的 gains、impedance、环境分配与最终参数表；13 个波数对应 13 份不同的完整 PD 向量。

这一曲线不是单纯的固定 PD、最大平地速度优化：

- 最初 v1 搜索 2 个振幅和 4 个 yaw PD 参数；leg-pitch 倍率固定 1.15。经过可行性、Pareto、距每波最佳速度 3% 以内的筛选，再选择参数变化较平滑的组合。因此 v1 本身的 body 振幅在中间波数几乎持平。
- v2/v3 用多个粗糙地面种子的腿交叉指标重新选择高波数参数，要求平地性能保留。v3 还补充了中间波数的候选评估。这些步骤改变了高波数振幅，不能将最终下降全归因于平地速度目标。
- v4 固定新的 body-roll 参数，并重新细化 wave 1.5；它仍保留随波数变化的 body/leg yaw PD。

代码证据：[v1 搜索](C:/Users/wxy22/Documents/Centipede_IsaacLab_RL/scripts/analysis/p8_wave6d_coarse_2026-07-20.py)、[近最佳和平滑选择](C:/Users/wxy22/Documents/Centipede_IsaacLab_RL/scripts/analysis/p8_wave6d_refine_2026-07-20.py)、[高波数重新筛选](C:/Users/wxy22/Documents/Centipede_IsaacLab_RL/scripts/analysis/p8_highwave_cross_rescue_local2_2026-07-21.py)、[v4 确认](C:/Users/wxy22/Documents/Centipede_IsaacLab_RL/scripts/analysis/p8_flat_map_v4_13knot_confirmation_2026-07-21.py)。

当时还有[实际运动振幅审计](C:/Users/wxy22/Documents/Centipede_IsaacLab_RL/outputs/analysis/p8_realized_amplitude_audit_2026-07-21/data/realized_amplitude_by_wave.csv)：v3 平地 body-yaw 的 1 Hz 实现振幅/目标振幅约 0.9997–1.0003。这支持当时关节波确实实现了目标。该审计属于 v3，不能把其 wave 1.5 实测值配给后来修改过的 v4 wave 1.5；重复环境结果完全相同，也不是独立随机证据。

**MuJoCo 中更接近这次研究设计的历史结果。** 8 月 31 日 natural-gait 搜索固定一套 PD，仅搜索 body-yaw、leg-yaw、leg-pitch 三个振幅。全部入选点如下，包含与严格单调下降不符的点。

| Wave | Body yaw | Leg yaw | Leg pitch | 前进位移 |
|---:|---:|---:|---:|---:|
| 1.5 | 24.02° | 29.24° | 22.37° | 141.36 mm |
| 2.0 | 8.98° | 50.32° | 23.83° | 128.23 mm |
| 2.5 | 16.92° | 49.38° | 22.92° | 151.38 mm |
| 3.0 | 14.52° | 48.24° | 23.62° | 129.63 mm |
| 3.5 | 19.24° | 41.03° | 28.61° | 122.67 mm |
| 4.0 | 18.82° | 38.55° | 24.82° | 110.74 mm |
| 4.5 | 13.03° | 37.94° | 24.49° | 97.08 mm |
| 5.0 | 7.54° | 26.19° | 28.29° | 65.86 mm |
| 6.0 | 7.09° | 21.58° | 32.07° | 52.98 mm |

该候选库在 wave 0.5、1、7、8、9 没有通过所有门槛的点；这不是它们的速度为零或无解证明。9 个入选点均已匹配回保存的原始 trial 记录及当时的独立实例复跑记录，本次没有重新仿真。

来源：[原始选择结果](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_wave_amplitude_natural_gait_optimization_2026-08-31/outputs/data/final_certified_selection.json)、[原始报告](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_wave_amplitude_natural_gait_optimization_2026-08-31/FINAL_RESULTS.md)、[协议](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_wave_amplitude_natural_gait_optimization_2026-08-31/PROTOCOL_V2.md)。

固定 PD 的代码链为 `run_broad_sobol_pilot.py → FastEvaluator → LeanIndex → stage1.selected_impedance()`。最后一步读取同一份 [controller_selection.json](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_biophysical_terrain_contact_audit_2026-08-26/outputs/data/controller_selection.json)，不以 wave 为参数；波数及振幅只交给目标生成器。该历史运行实际使用的主动关节族值为：

| 主动关节族 | Kp (N·m/rad) | Kv (N·m·s/rad) |
|---|---:|---:|
| body_yaw | 2.265167036957e-4 | 6.489225555490e-7 |
| body_pitch | 1.916879593338e-4 | 6.101617251833e-7 |
| legbase_yaw | 4.288484228013e-4 | 3.640178470720e-7 |
| legbase_pitch | 4.267329290140e-5 | 5.015383756258e-8 |

此表用于历史溯源，没有应用到当前模型。旧模型的被动 body-roll/tip 系数属于其 XML；不能把选择文件里所有关节族都当作当时实际存在的主动执行器，也不能只复制上述四行就声称完整机械配置相同。

旧结果使用不同的足端模型，并把 tracking、波形、腿速、摆动足高等 naturality 条件用于筛选；当前 recovery 的模型和 displacement-first 门槛已不同。因此总体振幅趋势的差异还不能归因于某一个因素。它们提供历史候选和需要检查的问题，不提供当前模型的最优值。

其他历史分支也已检查：`Centipede_IsaacLab` 及早期 FARMS 的几个 `optimize_impedance_*` 搜索只优化 PD，振幅和 wave 固定，不是逐波振幅最优化；RL 5 月 one-factor 地形筛选的“body_amp 随粗糙度单调下降”还在[补齐数据后的报告](C:/Users/wxy22/Documents/Centipede_IsaacLab_RL/outputs/analysis/PHASE1_SWEEP_REPAIR_SUMMARY.md)中被撤回，不能用来支持本次逐波结论。

后续振幅实验必须先核对一份跨波数相同的完整参数清单，只允许三个振幅随 wave 优化。历史低波数约 30° 的 RL 候选及 MuJoCo 所有旧入选点可作为起点，在当前同一 PD、同一模型、同一步长下重测；旧分数不可混入。既不强制振幅单调，也不把期望的 random-contact 交叉点写进目标。若未出现预期趋势，再由同一实验配置下的结果判断振幅搜索、实际跟踪与筛选条件各自的影响。

本次核对的 79 份输入 SHA-256、所有提取值和匹配检查保存在 [audit.json](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/amplitude_history_audit_2026-09-08/audit.json)。此前 fine_step 的封存 receipt 哈希仍为 `DEC797624D36D81E08CDBE1496DD26901485DDB13433E2CB3897276212B84A57`。
