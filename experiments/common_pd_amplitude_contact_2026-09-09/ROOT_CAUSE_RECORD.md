# 统一 PD 对照：根因记录

状态：COMPLETE_MATCHED_DIAGNOSTIC；60/60 个完整对照。

所有对照使用同一组实际 Kp/Kv；只改变标明的振幅、被动足尖模型或腿相位。位移单位 mm。

| 波数 | 振幅来源 | 足尖模型 | 腿相位 | body 指令/实测一阶幅值 ° | 位移 | 工程失败项 |
|---:|---|---|---|---:|---:|---|
| 1.5 | august31 | august31 | legacy | 24.02 / 22.95 | 147.06 | 通过 |
| 1.5 | august31 | august31 | midpoint | 24.02 / 20.15 | 149.02 | 通过 |
| 1.5 | august31 | current | legacy | 24.02 / 22.95 | 148.92 | 通过 |
| 1.5 | august31 | current | midpoint | 24.02 / 20.38 | 151.24 | 通过 |
| 1.5 | recovery | august31 | legacy | 16.94 / 19.18 | 187.57 | root_roll_peak, tip_speed, tip_peak |
| 1.5 | recovery | august31 | midpoint | 16.94 / 17.54 | 176.18 | tip_speed, tip_peak |
| 1.5 | recovery | current | legacy | 16.94 / 18.82 | 189.56 | 通过 |
| 1.5 | recovery | current | midpoint | 16.94 / 17.39 | 176.36 | 通过 |
| 1.5 | rl_v4 | august31 | legacy | 32.27 / 28.31 | 144.18 | spine, tip_speed |
| 1.5 | rl_v4 | august31 | midpoint | 32.27 / 26.65 | 146.88 | spine, tip_speed |
| 1.5 | rl_v4 | current | legacy | 32.27 / 28.92 | 146.72 | spine |
| 1.5 | rl_v4 | current | midpoint | 32.27 / 26.88 | 148.29 | spine |
| 2 | august31 | august31 | legacy | 8.98 / 9.48 | 146.13 | 通过 |
| 2 | august31 | august31 | midpoint | 8.98 / 9.19 | 143.24 | 通过 |
| 2 | august31 | current | legacy | 8.98 / 9.40 | 149.48 | 通过 |
| 2 | august31 | current | midpoint | 8.98 / 9.29 | 143.76 | 通过 |
| 2 | recovery | august31 | legacy | 22.74 / 24.45 | 171.02 | spine, body_velocity_jump, pitch_10ms, roll_10ms, chassis_force, tip_speed, tip_peak |
| 2 | recovery | august31 | midpoint | 22.74 / 24.22 | 178.85 | spine, pitch_10ms, roll_10ms, total_force, chassis_force, tip_speed, tip_peak |
| 2 | recovery | current | legacy | 22.74 / 23.97 | 166.36 | spine, body_velocity_jump, pitch_10ms, roll_10ms, total_force, chassis_force, tip_speed, tip_peak |
| 2 | recovery | current | midpoint | 22.74 / 24.80 | 162.11 | spine, body_velocity_jump, pitch_10ms, roll_10ms, total_force, chassis_force, tip_speed, tip_peak |
| 2 | rl_v4 | august31 | legacy | 32.29 / 28.30 | 136.96 | tip_speed |
| 2 | rl_v4 | august31 | midpoint | 32.29 / 27.66 | 134.56 | 通过 |
| 2 | rl_v4 | current | legacy | 32.29 / 28.70 | 137.77 | 通过 |
| 2 | rl_v4 | current | midpoint | 32.29 / 27.65 | 136.80 | 通过 |
| 3 | august31 | august31 | legacy | 14.52 / 9.90 | 130.94 | tip_speed |
| 3 | august31 | august31 | midpoint | 14.52 / 9.28 | 132.73 | tip_speed, tip_peak |
| 3 | august31 | current | legacy | 14.52 / 10.16 | 136.25 | 通过 |
| 3 | august31 | current | midpoint | 14.52 / 9.26 | 136.87 | 通过 |
| 3 | recovery | august31 | legacy | 15.33 / 14.47 | 195.44 | tip_speed, tip_peak |
| 3 | recovery | august31 | midpoint | 15.33 / 14.68 | 186.88 | tip_speed, tip_peak |
| 3 | recovery | current | legacy | 15.33 / 14.79 | 201.40 | body_velocity_jump, tip_speed, tip_peak |
| 3 | recovery | current | midpoint | 15.33 / 14.18 | 194.69 | body_velocity_jump, tip_speed |
| 3 | rl_v4 | august31 | legacy | 32.04 / 29.67 | 108.87 | tip_speed |
| 3 | rl_v4 | august31 | midpoint | 32.04 / 28.11 | 112.98 | tip_speed, tip_peak |
| 3 | rl_v4 | current | legacy | 32.04 / 29.68 | 107.72 | 通过 |
| 3 | rl_v4 | current | midpoint | 32.04 / 27.79 | 113.76 | 通过 |
| 4.5 | august31 | august31 | legacy | 13.03 / 10.89 | 101.50 | 通过 |
| 4.5 | august31 | august31 | midpoint | 13.03 / 12.67 | 109.14 | tip_speed |
| 4.5 | august31 | current | legacy | 13.03 / 10.95 | 103.22 | 通过 |
| 4.5 | august31 | current | midpoint | 13.03 / 12.23 | 111.72 | 通过 |
| 4.5 | recovery | august31 | legacy | 24.64 / 11.83 | 138.29 | body_velocity_jump, chassis_force, tip_speed, tip_peak |
| 4.5 | recovery | august31 | midpoint | 24.64 / 20.52 | 129.65 | body_velocity_jump, active_speed, tip_speed, tip_peak, torque |
| 4.5 | recovery | current | legacy | 24.64 / 12.24 | 152.41 | body_velocity_jump, chassis_force, penetration, tip_speed, tip_peak |
| 4.5 | recovery | current | midpoint | 24.64 / 20.58 | 146.53 | body_velocity_jump, chassis_force, active_speed, tip_speed, tip_peak, torque |
| 4.5 | rl_v4 | august31 | legacy | 26.68 / 27.08 | 67.27 | tip_speed |
| 4.5 | rl_v4 | august31 | midpoint | 26.68 / 25.36 | 74.29 | tip_speed, tip_peak |
| 4.5 | rl_v4 | current | legacy | 26.68 / 26.99 | 74.11 | 通过 |
| 4.5 | rl_v4 | current | midpoint | 26.68 / 25.38 | 78.06 | 通过 |
| 6 | august31 | august31 | legacy | 7.09 / 6.98 | 55.75 | 通过 |
| 6 | august31 | august31 | midpoint | 7.09 / 8.29 | 63.38 | tip_speed |
| 6 | august31 | current | legacy | 7.09 / 7.00 | 56.08 | 通过 |
| 6 | august31 | current | midpoint | 7.09 / 8.05 | 61.23 | 通过 |
| 6 | recovery | august31 | legacy | 11.71 / 7.80 | 134.66 | tip_speed, tip_peak |
| 6 | recovery | august31 | midpoint | 11.71 / 12.37 | 142.97 | body_velocity_jump, tip_speed, tip_peak |
| 6 | recovery | current | legacy | 11.71 / 7.18 | 139.20 | 通过 |
| 6 | recovery | current | midpoint | 11.71 / 12.02 | 147.18 | tip_speed |
| 7 | recovery | august31 | legacy | 27.20 / 23.48 | 121.93 | body_velocity_jump, tip_speed, tip_peak |
| 7 | recovery | august31 | midpoint | 27.20 / 32.80 | 105.61 | body_velocity_jump, tip_speed, tip_peak, leg_pitch_speed |
| 7 | recovery | current | legacy | 27.20 / 23.92 | 124.33 | 通过 |
| 7 | recovery | current | midpoint | 27.20 / 32.41 | 110.18 | body_velocity_jump, tip_speed, leg_pitch_speed |

## 已确认与待验证

1. 历史图不是相同实验：RL v4 每个波数的 PD 不同，并含平滑选点/其他环境筛选；9 月 8 日恢复也用了 8 组 PD。8 月 31 日才是固定 PD 的振幅优化，但 PD、足尖、筛选与当前研究仍不同。
2. 几何证据：两个 MuJoCo 模型的腿根都在对应体关节前半节段。RL 的腿根相位补偿 +0.5Δ 与此相符，原 MuJoCo 没有这项。成对位移差见 bridge_analysis.json；修正不保证每个波数改善。
3. 原 spine≤1.2 限制是弧长/首尾距离。正常低波数大摆幅也会增加这个比值。RL 的 wave 1.5、body 32.27° 移入当前系统后，完整轨迹仅在此项失败；这证明筛选存在形状偏好，但不能证明解除它会得到速度最优趋势。
4. body 实测幅值使用 2–6 s 的 1 Hz 正弦/余弦最小二乘，排除锁定的首关节。它与形状的横向摆幅分别记录，避免混淆。
5. 历史自然性门槛按当前指标进行诊断复算，未用于更改本次主选择。前进位移、物理稳定和波形要求会偏好不同步态。

主优化仍独立最大化各波数通过工程门槛的位移；不按振幅单调性筛选。对照结果不是最终优化结果。

## 原版控制信号核验与离散波数边界

原版第一批固定PD对照包含14个波数×2种振幅组合（另4个槽位是确定性重复）。全部28个条件完整存活，第二次相同请求另行验证重置重复性。最后4秒的保存目标信号中，body一阶幅值与32°/16°指令一致；wave0.5–8的相位斜率拟合也与请求值一致。因此这批结果没有显示“参数写入却没有实际改变波形”的问题。低波0.5–1的实际身体幅值小于指令，需与目标幅值分开报告。第一批约666秒，不能把启动成功或这两组候选当作完成优化。

当前19个body关节的空间步长为Δ=2π·wave/18。wave9时Δ=π，理想离散信号为sin(ωt−πj)=(-1)^j·sin(ωt)，属于相邻关节交替的边界情形；仅凭空间相位unwrap不能唯一识别其传播方向。原版保存信号在wave9得到的普通斜率拟合不应被解释成控制器输出了另一个低波数。历史wave1.5–4.5的趋势与扩展到0.5–9的结果分别报告。

原版第二次相同请求已完成，但精确重复性失败：同一批次的4个重复槽位位置逐项完全相同；跨请求全身位置最大差12.397mm，正式前进位移最大差2.57957mm。PD读回保持不变，两批全部28个条件完整存活。另有补充10ms追踪的采样计数未逐请求重置（A从0ms采样，B从6ms采样）；这只是额外的观察时间网格问题，不能解释48ms正式原始姿态的差异。优化派发因此被拦住，原始失败保留。正在用冷启动后的同一批精确输入区分进程残留状态与跨启动变异。此发现仅适用于本次模型/控制配置，不能直接断言所有历史RL优化都有同样问题。

## 2026-09-09 物理筛选修正

新增 PHYSICAL_POLICY_AMENDMENT.md，保留旧分支。几何计算显示旧 R≤1.2 对理想 body 振幅的限制随 wave 从 0.5、1、1.5、2 到 ≥2.5 分别约为 7.76°、15.57°、23.34°、30.66°、≥36°。因此该门槛本身具有明显波数偏好。

新分支将 R 留作诊断，以每个物理步实际非相邻 chassis 穿透接触代替。5 条全时域轨迹逐项指标、位移和保存姿态完全一致。RL wave 1.5 的 32.27° 指令实际一阶幅值中位数 26.88°；R=1.25665，但所有物理步的非相邻 chassis 接触为 0，其他原工程门槛全通过。它原先被排除的原因已证实是形状门槛，不是碰撞。该步态速度仍低于另一小幅大腿摆动候选，所以这还不能证明最终最优振幅会下降。

源数据：outputs/data/shape_geometry_audit.json、outputs/physical_geometry_v2/data/physical_parity_check.json。mesh 距离查询出现不一致的 0 值，未当作穿透或间隙证明；最终使用实际碰撞计数。

## Contact erase 实现边界

旧 study.py 的 contype=0、conaffinity=0 会同时擦除腿的地面与机器人内部碰撞。新实现以单独 collision bits 只擦除指定腿与 ground 的兼容关系，保留 chassis-ground 和所有机器人内部兼容关系。18 组静态矩阵检查证明 p=0 全矩阵等同原模型，p>0 仅移除目标 ground 配对，且同 seed 下擦除集合随 p 嵌套。该静态检查没有推进物理仿真；还需要定稿后的 p=0 动态一致性核验。主通道采用一次抽取、保持到末尾，避免旧短时恢复深穿透的脉冲；它与旧 burst 通道单独报告。

## RL 评估质心核验（已更正）

必须区分三层：canonical cell 保存的 `spine_xy` / `spine_ids` 是21个物理身体link，用于姿态诊断；cell的 `fwd_signed_m` / `bl_per_cycle` 仍由root/head位移派生；正式原版分析通过 `scripts/analysis/locomotion_metrics_2026-07-17.py::from_alllink_npz` 从全link记录重新计算，使用61个spine link的无权几何中心。这是原版冻结的 `fwd_speed_bl_cycle`，不能改名或拿head字段替代。

本次先看到21-link数组时曾把它误认成最终计分口径；现已沿 `tip_softening_locomotion_20260907/code/analyse.py` 和历史 `p8_wave6d_coarse_2026-07-20.py` 追到正式分析函数，纠正该推断。原版fallback尚未运行，未污染新仿真/优化结果。未来原版主指标保持61-link，额外输出21-link供MuJoCo对照。详见 outputs/data/rl_metric_chain_audit.json。MuJoCo当前的body yaw实际限位为±36°，搜索范围没有随wave改变。

## 实际 MuJoCo PD 与最近原版 RL 配置的差距

下面仅比较 MuJoCo 实际读回与 RL wave 1.5 的配置输入；不是新的 PhysX 实测，也没有更改任何 PD。

| family | MuJoCo Kp / Kv | RL 配置 Kp / Kv | RL/MuJoCo Kp / Kv |
|---|---|---|---|
| body_yaw | 0.0002265167 / 1.297845e-06 | 0.01133665 / 1.136445e-07 | 50 / 0.0876 |
| body_pitch | 0.0007667518 / 2.440647e-06 | 0.0006 / 0.0009 | 0.783 / 369 |
| legbase_yaw | 0.0004288484 / 3.640178e-07 | 0.004665611 / 3.924185e-06 | 10.9 / 10.8 |
| legbase_pitch | 8.534659e-05 / 1.003077e-07 | 0.0007280325 / 9.855346e-07 | 8.53 / 9.83 |
| body_roll | 0.0006958425 / 3.945791e-06 | 0.07536 / 0.003606245 | 108 / 914 |

这说明原版 RL 的历史振幅图不能仅以仿真器标签与本次固定 PD 振幅图直接等同；尤其身体侧向与滚转的约束强度差别很大。需要把这些控制器差异与相位、头部、模型、筛选及求解方式逐项分开。

原版固定 PD 对照于 9 月 9 日约03:00启动成功，173关节、32个环境的实际 PhysX Kp/Kv读回一致。五个共享family及tip使用本次MuJoCo固定值；额外head-pitch使用同一个原版既有值。启动前两次分别停在默认缓存和默认ground资产加载，日志保留；正常主机访问重试通过，没有更改PD、模型或Clean。重复轨迹验证与正式振幅搜索状态须查原版 dated目录的记录，不能把启动成功当成优化完成。

同一USD的 authored约束另存于原版 `initial/authored_constraint_audit.json`：body yaw约±36°，body pitch约±90°，body roll约±30°，leg yaw/pitch约±180°，tip±38°。USD还含独立的limit damping字段。这些是模型约束输入，不能与自由运动区的Kp/Kv混为一谈，也不是新的PhysX约束读回。搜索命令范围保持body0–36°、leg yaw0–90°、leg pitch0–60°。

## 原版 RL 模型明确关闭内部自碰撞

原版实际引用的 `Centipede_IsaacLab_RL/outputs/analysis/mujoco_transfer_refinement_20260902/assets/centipede_rebuild_renamed_tip_experiment.usda` 第 10692 行明确写有 `bool physxArticulation:enabledSelfCollisions = 0`。原版配置没有覆盖此字段。MuJoCo 当前所有碰撞体的同一碰撞位同时允许 ground 与机器人内部碰撞，保存姿态中能观察到腿与身体的内部接触。这是已查实的模型碰撞语义差异；不是仅凭引擎名称推断。

新增 self_collision_diagnostic.py，只在独立小对照中关闭 MuJoCo 内部碰撞，并以完整配对矩阵断言保留全部原 ground 配对。统一 PD、振幅、相位、步长、初始姿态、指标均不变。主优化仍在原主模型运行；该对照不是新的振幅最优值。结果写入 outputs/self_collision_diagnostic/data/comparison.json。第一次预检查因错误地按 ground 名称查询平面而在物理步前停止；已改按唯一 plane 类型识别，错误记录保存在诊断子目录 audit 中。

## 自碰撞开关导致振幅组合排名反转

六组完整配对已经完成，PD 哈希全部相同。wave 1.5 的 recovery 小 body/大腿摆幅组合，内部碰撞开启时176.36mm，关闭后145.00mm；RL v4大 body/小腿摆幅组合则148.29→162.63mm。两组排名发生反转，证明碰撞语义能够直接改变优化器偏好的振幅组合。RL v4 w1.5 的实际 body 1Hz幅值中位数26.88→34.41°，同一指令32.27°。wave4.5 recovery组合146.53→198.71mm，其余共享工程门槛由多项失败变为全部通过。这个对照未搜索新振幅，不能称作最优趋势；主优化未混入其位移。

## 历史参数范围并不相同

原版 `scripts/analysis/p8_wave6d_coarse_2026-07-20.py` 明确把 body multiplier限定1.0–1.5（约22.92–34.38°），leg yaw multiplier限定0.4–0.6（约13.75–20.63°），leg pitch则固定为先前选定的1.15倍率（约26.36°）；同时优化四个PD参数。当前研究则固定PD、联合搜索body0–36°、leg yaw0–90°、leg pitch0–60°。因此当前允许的大腿摆幅推进模式，在历史粗搜中根本不在候选空间；历史后续局部refine/平滑路径还有各自规则，不能仅以历史平滑图推导当前宽范围速度最优值必须单调。任何恢复历史动作范围的实验都应单独声明，不能暗中缩小范围来制造想要的趋势。


## 原版 RL 同一进程重置偏差：冷启动已隔离

32-slot 相同请求 A/B 在同一进程内先后执行，canonical61 位移最大差 2.5795698166 mm，已对齐的48ms原始全身位置最大差12.397468 mm。每批内部4组重复槽的原始姿态均完全相同，173关节Kp/Kv读回也相同。补充10ms记录器的计数器未逐请求重置，A/B记录网格不同；这是独立的观测问题，不能解释已对齐原始姿态的差异。

另起进程 C，重复 A 的32-slot输入。raw.npz 和 tracking.npz 的每一个已保存数组均逐项完全一致（包括姿态、速度/力矩摘要、目标、关节位置、alive和采样时间），canonical61位移差为0。证据：原版 dated目录 cold_replay/repeatability_receipt.json。由此在此校准条件下将误差隔离到进程内保留的评估/仿真状态；尚未定位到具体PhysX缓存或控制器buffer。不得把它泛化为所有历史RL结果无效。

修复策略是每批只在全新原版进程中评估一次；保持runtime、原版cell、模型、PD、初始状态、步长、计分和搜索算法不变。search_cold独立保存新优化，旧持久进程失败复测与暂停状态保留。第一批启动前的GPU独占检查正确阻止了尚未退出的已完成复测进程；定向关闭并确认该PID消失后，才启动新批次。没有并发两个GPU物理进程。详见原版 COLD_SEARCH_PROTOCOL.md。

## 执行环境对照不改变 MuJoCo 数值结果

一个完全匹配的正常主机执行对照（wave1.5、固定PD与原振幅）复现原轨迹D=176.3599345877mm，全部共同指标完全一致。wall270.625s、CPU119.578s，未显示更换执行上下文能解决速度瓶颈。主优化没有因此迁移或改变求解器。证据：outputs/host_context_parity/data/receipt.json。


## 短时随机失联：恢复碰撞的几何保护

新增独立 transient 通道，0.5s刷新19个双侧腿对的期望Bernoulli失联，PD/振幅保持固定。恢复时逐条腿检查全部碰撞体对平面的最低支撑高度；每1ms检查一次，至少离开地面50µm才恢复。双侧腿交替摆动，若强制两腿同时抬起才恢复，会意外积累成永久失联，因此恢复按单腿进行，期望失联仍按双侧腿对抽取。

检测直接使用当前qpos更新后的几何位置，避免mj_step积分后的Cartesian缓存滞后一个物理步。sphere/capsule/box/cylinder/ellipsoid/compiled mesh的解析最低高度，在5条已保存轨迹共20个姿态上，与211个真实地面接触的最深距离完全相等；平移1mm的高度增量检查通过。独立静态状态转换证实穿地时不恢复、完全抬离后38腿均恢复，全部内部碰撞配对保留。检查没有推进任何物理步。源证据：outputs/data/transient_static_check.json。

该保护会让实际失联时间依赖步态，必须报告actual exposure和recovery delay，不得把相同期望p说成相同实际失联时间。与persistent结果单独作图、单独留出新随机种子确认；不能混合得到crossing。状态仍为准备完成，动态p=0一致性和所有擦除结果必须在振幅资格完成后执行。


## 头部合同与非单调选点代价的补充核验

MuJoCo当前XML通过custom numeric明确固定head pitch=-29°，且head pitch/roll关节已移除。原版配置文件名虽含head8，但实际head_pitch_offset=-0.57rad（-32.6586°），不应按文件名猜角度。已从原版控制目标数组确认各槽相同；实际角度受负载影响，与目标并不相等。静态证据和逐槽实测均保存在outputs/data/head_contract_audit.json。它是另一个明确的结构/目标差异，尚未独立测其因果效应；本次没有改变PD、目标或模型。

截至本次名义候选诊断，历史范围1.5–4.5的完全单调body振幅路径在已观测且通过连续身体碰撞检查的候选中存在，但最小化最大相对位移损失后，仍需至少一个波数损失约16.65%。这量化了独立速度最优与按期望形状选取路径的差别；当前非单调点不是简单3%近最优容差就能抹平。仅为有限候选集、6s名义条件的诊断，未通过完整长期资格的路径不得用于contact。主优化和后续冻结仍按各wave独立最优规则执行。证据：outputs/physical_geometry_v2/data/amplitude_identifiability.json及amplitude_identifiability.png。


## 身体—腿实际目标波形排查

对原版32槽校准的28个独立条件，读取末段实际发出的target数组，按关节名称逐一拟合1Hz幅值和相位，并与MuJoCo修正中点腿相位的解析目标公式比较。去除共同的身体时间相位后，所有身体/腿目标的最大相对相位误差约0.3754°，幅值最大相对误差约0.002061%。因此当前大幅度性能差异不能简单归咎于度/弧度倍率错、腿左右符号反转或仍遗漏半节段腿相位。此项仅排查目标输出，实际关节跟踪与碰撞动力学仍可不同。原始记录与代码哈希：outputs/data/cross_system_target_audit.json。

实际头部角度补充：原版28条件末段均值范围-30.0869°至-29.4557°，与MuJoCo固定-29°接近；不能仅凭-32.66°目标与-29°固定值之差，就把头部角度差夸大为已经证明的主要原因。

## 地面摩擦与原版双层地面：继续做匹配诊断

MuJoCo实际模型的滑动摩擦系数为1.6。原版USD没有authored PhysicsMaterialAPI/friction，环境未覆盖默认值；Isaac Lab的SimulationCfg→RigidBodyMaterialCfg→SimulationContext默认材料绑定链指向static/dynamic0.5。后者目前仍是源码/配置证据，尚未取得本次原版运行的实际材料读回，不可声称已证明两系统实际接触材料完全对应。

新增独立ground_friction_diagnostic，以wave1.5/3/4.5的recovery和RL v4六组固定振幅做摩擦×自碰撞对照。为只改变地面滑动摩擦，先检验几何和接触配对。首次零物理步预检查发现：机器人friction=[1.6,.02,.001]，地面=[1.6,.005,.0001]；优先级相同时有效摩擦取双方较大分量。因此简单降低地面stored friction并不会降低有效滑动摩擦。诊断给地面更高priority，并保存原max-combined尾部，仅替换滑动分量。原版预检查错误、源文件和协议完整归档；主优化文件未改。

修正后的24个姿态/处理组合中，接触几何、距离、坐标、维度、solref/solimp等逐项完全相同，仅地面接触的两个滑动分量改变。随后四组完整6s高摩擦priority重放的位移与既有结果差均为0，证明本对照的priority设置没有改动原条件数值轨迹。十二组低摩擦结果尚在运行，不能提前解释其效应。证据在outputs/ground_friction_diagnostic/data。

另一个源码审计发现，原版flat TransitionTerrain平面位于z=.0025m，另外的GroundPlane位于z=0，两者都由环境过滤器允许碰撞。未来原版erase必须覆盖这两个地面；只删除上层会让腿落到下层形成隐藏支撑。此项尚为配置/源码核对，非实际stage读回，记录在outputs/data/native_ground_contract_audit.json。

## 原版已完成仿真的退出延迟单独记账

冷启动优化前四批均保存完整response、raw与tracking，第三批的自然退出、第四批的空闲状态外部终止都在预设等待内未能完全卸载Windows GPU进程。退出超时不得被计为物理失败，也不重复计算已有完整响应。新独立监督器只调整进程生命周期等待，继续使用相同冻结的runtime、PD、模型、优化算法、评分、候选顺序和随机种子，确认旧进程真正退出后才启动下一批。没有把多进程重叠或持久进程状态差异引入新样本。

## 摩擦×自碰撞的完整对照已完成

十二组低摩擦结果于05:21完成，四组高摩擦priority复现此前均精确通过。wave1.5在内部碰撞开启时，recovery小body/大腿摆幅组合由176.36升至225.33mm，RL v4大body组合由148.29升至167.24mm，均通过共享工程门槛；降低摩擦仍未反转两组合的排名。内部碰撞关闭时，大body组合在两种摩擦下都更快：高摩擦162.63对145.00mm，低摩擦191.83对166.15mm；低摩擦小body组合有chassis_force失败。

因此这组配对证据表明：摩擦会改变性能，但单独把MuJoCo滑动系数改为0.5，不能解释这两个wave1.5候选中的原版大body偏好。内部碰撞开关在两种摩擦下都造成相同方向的组合排名变化。结论限定于这六组固定振幅，尚非重新优化后的最优趋势。wave3与4.5的失败结果同样保留并在图上标红叉，未当作合格成绩。

完整数值、源哈希与失败项：friction_collision_diagnostic.md，outputs/ground_friction_diagnostic/data/comparison.json及report_receipt.json。friction_collision_diagnostic.png/pdf已检查。主模型μ=1.6、自碰撞开启、统一PD及优化输入未改变。

## 原版后台物理属性已实际读回（06:35）

原版同配置独立场景的32环境、249机器人碰撞体实际PhysX张量读回一致：静/动摩擦0.5，恢复系数0，rest offset全0；contact offset为228个0.01962mm和21个约0.097865–0.111210mm。质量2.536051g，armature全0，完整PD仍相同。读取场景后退出，未提交振幅评估或contact erase；初始化可能推进物理。原版核心/资产、主搜索与Clean均未改变。

Live stage确认上层Collision Mesh路径/World/TransitionTerrain和下层/World/GroundPlane/GroundPlane/CollisionPlane均启用；下层实际绑定材质0.5，上层未独立绑定，沿原配置/源码使用场景默认0.5。内部碰撞在实际articulation prim上为false。地面绑定/selfcollision属于live-stage读回，机器人材质/offset属于张量后台读回，不能混称。

114个capsule原文件虽写有0.1mm contactOffset，但为custom属性且未应用PhysxCollisionAPI；后台没有按114个0.1mm生效。静态字段与实际输入不一致已确认，具体API解析因果尚未做单独启用API的启动对照。没有用这一发现修改运行中的模型。此更新替代前文关于机器人摩擦/offset“尚无实际读回”的状态。

原版详细记录：outputs/analysis/common_pd_amplitude_contact_20260909/NATIVE_PHYSICAL_CONTRACT.md；原始physical_contract_readback目录及collision_schema_audit.json。诊断成功快速退出引发监督器句柄查询竞态，完整读回保留；核对进程已消失及数据哈希后恢复原监督器，未重复评估。该监督器问题不影响物理读回。


## 2026-09-09 07:52 EDT — 联合方向精修与筛选审计

原来的28个坐标搜索起点全部达到归一化步长0.005、改进不超过0.2mm的停止条件，仍不能推出联合方向收敛。随后两个预定联合探查银行改善了wave1、2、2.5、3、3.5、5。相对之前各wave的最好坐标搜索终点，位移改进依次为0.236083、1.128125、2.456671、3.425837、0.653607、21.900320mm。wave5从152.082973提升到173.983293mm，约14.4%；不是通过修改PD、门槛或contact条件获得。

已经在第二个联合银行完成前写下的JOINT_POLISH_PROTOCOL.md规定：所有提高超过0.2mm的波数按同一规则，围绕新候选继续6个坐标方向加8个三参数角方向的小步探查，最多12轮。这个14方向有限搜索也不构成任意方向或全局最优证明。07:30:29完成旧进程到新进程的检查点交接；数据和源码哈希见outputs/physical_geometry_v2/data/joint_polish_handover.json，起点及增益见joint_polish_initial.json。没有非零contact结果参与这个决定。

第一轮之后wave1、2.5、3停止，wave2、3.5、5继续。第二轮后仅两个起点仍继续（后续实时检查点优先于本段快照）。当前尚未完成无损资格验证，也没有开始非零擦除。2673个名义评估的进度图显示，历史范围1.5–4.5的body幅度描述性Spearman rho=-0.642857，仍有2→2.5、2.5→3、4→4.5三处上升。此图已视觉检查，明确标为临时进度，不能代替最终选点。

另增加只读取结果的report_qualification.py。待资格验证完成后，逐wave报告最快名义候选与最终合格候选的幅度变化、位移代价、名义排名、所有长期/步长拒绝原因。它不运行物理、不选择替代曲线、不改变搜索。这样最终幅度趋势若因资格验证改变，可以追溯到具体原因。


## 2026-09-09 08:38 EDT — 振幅搜索结束，进入资格验证

28个原坐标起点达到对应停止条件。6个联合改进起点中，5个在原12轮联合精修内停止，wave5在达到上限时仍有2.089906mm的单轮改进，触发了08:15:25预先写下的条件续算。08:23:49仅交接了已完成物理银行的旧驱动；未修改PD、模型、门槛或资格代码。

wave5追加5轮后，最后一轮改进0.153037mm，小于0.2mm停止阈值。最终名义D=208.458299860mm，相对原坐标终点152.082972993mm提高37.0688%。幅度body/leg yaw/leg pitch为11.803158/72.551243/34.167417度。较大方向探查与原14方向小步探查都参与选点，所有被拒绝结果仍保留。joint_finish_audit.json记录converged=1、budget_limited=[]；原joint_polish_audit.json中的历史预算记录完整保留。

08:36:41开始qualification_000，共14个波数×6秒/12秒/6秒减半步长=42项，暂未完成资格判断。当前208.46mm不可称作最终合格成绩；没有非零擦除数据。新的审计还检查擦除前1.5秒所保存30Hz姿态是否在同波数/时长/步长的不同概率及种子间完全一致。该检查只验证保存的前缀，不声称观察未保存的中间状态。


## 2026-09-09T09:05:43.416184-04:00 — 短期成绩与资格结果分开记录

首轮14个候选全部通过6秒名义重跑，14个减半步长位移差也全部在10%以内，但只有7个通过完整资格条件。其余失败来自工程门槛，并非位移差阈值。wave1.5在12秒运行出现upright/root_roll_peak/tip_speed/tip_peak失败，在半步长6秒也出现姿态与tip速度失败；尽管位移仍较高，不能以此作为稳定无损步态。其余首次失败分别涉及0.5和4.5的半步长body_velocity_jump、3和6的长期tip_speed、4的长期及半步长tip_speed、5的半步长tip_speed。

后续按名义性能顺序、不改门槛继续验证。wave4的第二候选169.288367mm与wave5的第二候选208.305263mm通过；wave5的半步长位移差约0.2425%，仅比最快名义候选低0.153037mm，优化收益基本保留。wave6的第三候选139.443651mm也已通过。总资格流程仍未完成，禁止提前称作最终全波数结论或启动擦除。所有逐候选三条件原始记录保留在qualification_progress.json，完成后由资格审计统一汇总。

## 2026-09-09 09:16 EDT — 全波数资格完成及参数冻结

14个波数全部通过，未解决波数为空。wave0.5、1.5、3、4.5分别在第4、5、8、5个候选通过，最终body分别25.957765、16.667149、22.667324、7.349188度（全精度以validated_selection.json为准）。wave1.5相对最初名义最快候选损失约5.69%位移，wave3约2.01%；其余波数不超过0.5%。资格审计保存每次三条件结果、拒绝原因和有限候选排名，且已经完成图形检查。

2953个名义条件评估形成最终合格选点。历史范围1.5–4.5的body与wave的描述性Spearman相关系数-0.642857、线性斜率-4.017°/wave，仍有三处相邻上升；扩展0.5–9范围相关系数-0.591209。不能称为严格单调，也不把相关系数作为独立假设检验。最终三振幅与实际1Hz跟踪幅值见validated_amplitudes.md/png，图已检查。接触流程只在amplitude_validation_finished标记后读取冻结选点，先验证p=0精确复现，再开始非零擦除。

空间相位的结构诊断：控制器19个腿对位置，相邻空间相位增量为2πwave/18，中点修正pitch参考相位为(-wave*i/18+wave/36+1/2) mod 1。使用精确有理数计算得到各wave不同参考相位数为[19,18,12,9,19,6,19,9,4,18,3,18,9,2]，对应wave[0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9]。spatial_phase_groups.png/pdf及outputs/data/spatial_phase_groups.json保存结果，图已检查；没有物理步或模型修改。这仅说明参考相位的离散结构不随wave单调变化，并未证明实际支撑冗余或crossing机制。左右关节镜像符号不等于独立物理着地组，相位覆盖范围也不能由组数代替。

## 2026-09-09 09:23 EDT — p=0记录错误与透明修复

09:19首次p=0银行的14项均在保存阶段失败：Channel.finish通过dict(**task['erasure'],kind=...)重复传入kind。物理运行已经到达finish，但没有形成成功结果；失败不能解释为动力学失稳。旧静态测试只检查碰撞矩阵，没有调用finish，且测试任务缺少实际任务中的kind字段，故未提前发现。

确认旧工作进程全部退出后，将完整contact_erase输出目录、原10个封存文件、封存清单、静态检查与流程错误记录移动/复制到outputs/amendments/p0_metadata_20260909_0923。只删除finish中多余的显式kind参数，其值继续来自已经冻结的任务字段。新测试从归档源码提取原finish方法，复现原TypeError，再对修复方法执行真实JSON/轨迹保存回读，零物理步通过；18个碰撞过滤静态检查和211个短时恢复几何距离对照也通过。测试临时目录最初选在系统临时目录，被现有写入边界断言正确拒绝；随后移到本实验outputs目录，未改写入保护。

seal_contact_metadata_repair.py逐字断言仅一行变化，核实归档14项全部p=0且没有完成银行、没有非零或短时擦除评估，保存旧/新哈希和全部归档文件哈希后更新封存链。物理、PD、振幅、随机计划、统计判据均未改变。新流程session9843于09:23启动，重新生成相同最终幅度报告并重新执行p=0。修复不能代替后续动态精确复现证据。最终核验和报告包含此修复链。

09:26修复后p=0的14个波数全部通过，所有原评估指标精确复现、位移差均为0。随后逐数组比较归档失败运行与修复运行的t/center/upright/pose_t/qpos，14个波数全部精确相同；outputs/data/p0_metadata_repair_dynamic_parity.json保存输入哈希与结果。修复前的失败仅发生在物理循环完成后的保存步骤。

## 2026-09-09 10:12 EDT — 持续擦除探索完成，第二个保存错误及独立确认

10:05:23完成252个持续擦除条件，全部保留完整轨迹，没有不完整数值结果。p=1的14个波数在计分窗中的足部地面接触力峰值和平均接地足数均精确为0，证明该端点实际删除腿的地面支撑；身体地面接触有意保留，详见outputs/data/persistent_p1_dynamic_filter_check.json。

封存分析首先成功写下全部统计量、39个探索候选和选中的wave2.5/6，随后在写crossing_choice.json时因dict(**choice,interpretation=...)重复文字字段而失败。源错误已归档于outputs/amendments/discovery_metadata_20260909_1007。修复仅把record=dict(**choice,...)改成record=dict(choice,...)，允许使用源码原本指定的解释文字；均值、区间、候选排序和选点逐对象完全一致。新完整路径合成测试复现旧错误，并检验新选点文件与图成功保存；原配对/失败保留测试也通过。修复与252项输入哈希记录于repair_receipt.json，当前封存清单保留第一次修复链并增加此次修复；当时没有任何held-out确认运行。252项仿真没有重跑。

新流程session29158于10:10:25启动，复用完成银行，10:11:05开始20组新种子的确认。原9843已结束，不能同时重启。比较波数及区间仍为2.5/6、p=[0.4,0.6]，中点0.5也确认。冻结body目标幅度18.733338/17.076958度；探索低减高位移为p0:+65.709684mm、p0.2:+49.125159、p0.4:+10.837746、p0.6:-17.782263、p0.8:+15.124853、p1:+0.591693。高概率又出现探索性反向换位，故不能声称高波在首次交叉后始终更优。39个候选均不满足“两者全直立且平均身体支撑比例≤0.15”的探索分类；当前选点属于high-wave-remains-useful的后备类别，包含姿态/身体支撑问题。它仍需新种子及可能的步长/时长复验，尚未确认为最终crossing。

## 2026-09-09 10:31 EDT — 持续擦除未通过独立交叉确认

20组新配对种子100–119的120项结果全部完成。wave2.5减wave6的位移差在p=0.4、0.5、0.6分别为29.974891、18.012223、18.195993mm；对应97.5%配对区间为[19.590782,40.705269]、[5.040468,29.413382]、[5.321089,29.303266]mm。三个区间全部为正，没有复现探索阶段p=0.6的反超。confirmed_at_nominal_step=false，按照封存流程不触发减半步长/12秒交叉复验；这是条件分支正常结束，不是漏做已要求的复验。

在p=0.6，低/高波平均位移101.978527/83.782534mm，相对各自无损205.153335/139.443651mm约保留49.7%/60.1%。高波的相对保持率更好，但还不足以抵消无损基线差距。低波20条中60%始终满足直立门槛，高波100%；平均身体支撑时间比例44.5%/32.0%。所有120条均至少有一项工程门槛失败，完整失败位移仍计入均值。不能通过剔除失败、改为成功加权速度或另选掩码把未确认的交叉改写为成功。

数据：outputs/contact_erase/data/confirmation_analysis.json。final_crossing.png/pdf与crossing_mechanism.png/pdf已生成并视觉检查；后者用同一确认样本描述路径、方向效率与支撑变化，不是独立因果实验。短时通道于10:33:38通过14/14的p=0全部指标精确复现，10:33:39开始252项探索，沿用同一冻结振幅、PD和独立统计规则。两种擦除通道不合并。

## 2026-09-09 10:35 EDT — 振幅定义的事后只读核验

audit_amplitude_definitions.py从14个合格无损轨迹保存的qpos恢复21个身体中心位置，仅调用mj_kinematics，不推进物理。对2–6秒保存姿态逐帧去平移和主轴方向，计算横向半宽并取时间中位数。这是中心线几何半宽，不是空间正弦波拟合，也不是优化中的关节目标振幅。wave2.5/6的body目标为18.733/17.077°，实际关节1Hz幅值中位数19.034/16.671°，中心线横向半宽中位数4.089/1.556mm。三个定义分别记录，不能把较明显的几何幅度趋势替换成严格单调的关节目标趋势。

amplitude_definition_audit.png/pdf已视觉检查，原始输入和代码哈希见outputs/data/amplitude_definition_audit.json。wave7目标body=0°仍有实际1Hz幅值8.184°，说明目标值不能代替受力后的运动值。第一次离线读取错误假设motion.npz包含xpos，在任何输出前遇到KeyError；改为读取实际存在的qpos并做运动学重建。没有修改核心、冻结输入或统计分析。

另一次原版控制源码复核排除了文件名head8造成的误解：实际配置及cell均将head_yaw_gain设为0。两边都使用位置目标有限差分产生速度目标；该项为源码核验，没有新增实际速度目标数组的跨系统波形对照，不能声称速度目标逐样本完全一致。

## 2026-09-09 11:14 EDT — 短时失联探索完成，wave3.5/4进入独立确认

252项探索全部完成，42个探索性交叉候选。封存规则选中wave3.5与4，区间p=[0.2,0.4]，body目标10.093338/3.96°；无损位移201.430491/169.288367mm。p=0.2的探索均值161.388677/145.953803mm，低减高+15.434873mm；p=0.4为91.145063/104.078861mm，低减高-12.933798mm。两端四组样本的低减高符号各自一致，但仍只属于探索，不能代替20组新配对种子100–119的确认。

选中两波的探索轨迹均通过直立门槛；p=0.4身体支撑时间比例38.7069%/28.2424%，所以该候选不满足平均身体支撑≤15%的clean_exploratory分类，且擦除轨迹均未通过全部工程门槛。实际失联腿×时间比例在p=0.2为20.1702%/20.5355%，p=0.4为42.6506%/42.7306%；同一设定概率的实际失联时间没有被假定相等。p=0.8又有低波小幅领先（10.124634/8.849928mm），不能外推高波在首次反超后永远领先。

11:14:33开始120项确认，概率为0.2、0.3、0.4。outputs/contact_erase_transient/data/crossing_choice.json记录完整选择顺序与随机计划。contact_discovery.png与transient_exposure.png已视觉检查；全波数暴露图的默认颜色有重复，最终选中组合应使用独立清晰的配色图或数值表。统计内容和封存分析源码没有因排版问题修改。

独立保存计划审计audit_contact_schedules.py从种子重新生成uniforms，并从刷新/恢复事件重新计算实际失联序列与腿×时间比例。10:58的进度审计覆盖持续372条、短时149条完整轨迹，8,005次恢复和670,500个保存开关样本均符合记录；最小已记录恢复间隙50.0042µm。几何间隙是在运行中计算后记录的值，没有对应每个恢复时刻的独立qpos快照，不应说成逐事件独立重建了几何。最终流程还需运行覆盖全部结果的--final核验，包括两个通道p=0/p=1的28组确定性端点轨迹逐数组比较。

## 原版观测资格准备及一次记录层错误

原版dated目录新增NATIVE_QUALIFICATION_PROTOCOL.md和qualification_observer.py，只安装全身体法向接触力读取与运动学记录。它明确保留摩擦力、驱动力矩、逐接触穿透等未观测量的限制，不能称作与MuJoCo所有工程门槛相同。首先用原32槽校准请求验证增加观测后全部原始姿态/跟踪数组、完整PD和正式位移是否精确一致；未通过该检查前不得用于native资格。

observer_calibration_a在首次热身记录中遇到CPU质量张量与GPU力张量相除的设备错误，没有完成response，不能计为物理失败或合格运行。原源码、协议和协调器存于该目录audit_recording_device_failure；仅将观测用质量复制到相同设备，在observer_calibration_b另开记录。原版核心、资产、PD、搜索和Clean均未修改。协调器在已完成的原搜索批次边界暂停监督器，等待原GPU子进程完全退出后才运行校准；B于11:08:22开始评估，完成后再恢复原搜索。当前尚无合格的native振幅或非零native擦除结果。

11:20:00，observer_calibration_b通过完整校准：原raw.npz与tracking.npz的每个数组、全部PD读回和61-link正式位移均精确复现冷启动A/C。观察覆盖8,304个请求物理步，末6,000步对应6秒测量；每48步的保存姿态与原125个policy快照逐数组对齐。4个补齐槽的观察量、足部接地布尔数组和tip角度数组也全部精确一致。28个不同wave/振幅条件的平均法向合力模长和为0.999977–1.000716倍体重；这不是摩擦力测量，也不是动量平衡的独立完整证明。

这28个预先定义的基准条件中，21个通过当前6秒观测门槛，仍未完成长期与半步长资格。失败7个：wave0.5两种振幅组合、wave1大body组合、wave8和9两种组合。wave0.5大body组合足端速度峰值约1956rad/s、法向力模长和峰值4.09倍体重；wave8/9的身体支撑比例约11%–35%。原先的生存/有限状态初筛不能替代这些观察。原版observer_calibration.png/pdf已视觉检查，完整量和源哈希存于observer_calibration_b/report_receipt.json。观察器子进程完全退出后，11:20:30恢复原搜索监督器；没有并发两个GPU物理进程。
