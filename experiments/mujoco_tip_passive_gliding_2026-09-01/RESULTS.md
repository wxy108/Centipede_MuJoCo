# Tip passive-gliding 模型修正与验证结果

## 结论

**已完成轴向建模修正、25 组 tip 参数筛选及 100 个独立条件的记录；尚未找到全面通过验收、可直接替换旧基线的模型。**

本次交付的 `localz_k2em05_d3p2_r38` 是标准平地条件下、按既定门槛筛选后的有限参数集最优候选，不是全工况合格模型，更不是连续空间全局最优。它通过全部 6 个平地条件的绝对门槛；加入与匹配旧轴基准的非退化门槛后为 **5/6 通过**。粗糙地形 **0/2 通过**。原基线、Robot 项目和 Robot Step 18/19 没有被覆盖。

审计技能的 gate-first 要求直接影响了本次选择：较快的 `k=2.5e-5, d=4.2` 被非退化门槛淘汰，不能以更大位移抵消失败；当前候选也没有被冒充为正式通过。

## 模型改动

| 项目 | 旧模型 | 当前候选 |
|---|---|---|
| Tip 轴 | 与 tip 中心线相差 30°，扫过锥面一部分 | tip-local Z，垂直 tip 中心线，允许倾斜局部平面内前后弯曲 |
| Tip 弹簧 k | 4.5e-6 N m/rad | 2e-5 N m/rad |
| Tip 阻尼 c | 1.15375965086e-9 N m s/rad | 7.78348452079e-9 N m s/rad |
| 角度限制 | ±38° | 不变 |
| 总质量 | 2.53603109178 g | 完全相同 |
| Tip 自由度 / 全模型 nv / nu | 38 / 177 / 133 | 完全相同 |
| Armature | 0 | 0 |

未改变：刚性近端两段、腿长、足球、几何、质量、惯性张量、接触参数、控制器公式、波数耦合、三个幅值及两个波数对应的既有 PD 配置。保留 `_roll` 关节后缀仅为兼容索引，不再以名称解释物理轴。

新轴前后弯曲力臂约为旧轴两倍；同一足端线性刚度对应约四倍关节刚度。因此 `k=2e-5` 的中立姿态足端前后刚度只比旧值约高 11%，不是足端刚度增加 4.44 倍。局部有效惯量变化来自转轴投影，不是添加质量或惯量。依据、推导与文献边界见 [MECHANISM.md](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_tip_passive_gliding_2026-09-01/MECHANISM.md)。

## 平地结果

运行时长包含初始化。位移评估窗口是 `1.5 <= t < duration`；位移是各机身 body 原点平均位置沿固定前进轴的净变化，不是路径长度，也不等同于精确质量加权 COM 位移。名义物理步长为 0.25 ms；半步长为 0.125 ms。

| wave | 运行 / 步长 | 旧轴位移 mm | 新轴位移 mm | 绝对门槛 | 完整门槛 |
|---|---|---:|---:|---|---|
| 2.5 | 6 s / 0.25 ms | 149.962 | 156.929 | 通过 | 通过 |
| 3 | 6 s / 0.25 ms | 129.274 | 130.941 | 通过 | 通过 |
| 2.5 | 12 s / 0.25 ms | 351.067 | 366.486 | 通过 | 通过 |
| 3 | 12 s / 0.25 ms | 301.736 | 305.046 | 通过 | 通过 |
| 2.5 | 6 s / 0.125 ms | 149.737 | 154.282 | 通过 | 左右足端力非退化失败 |
| 3 | 6 s / 0.125 ms | 129.221 | 130.237 | 通过 | 通过 |

名义平地位移分别增加 4.65% 和 1.29%。半步长相对新模型名义位移分别改变约 -1.69% 和 -0.54%，这只是两种步长的一致性检查，不能称为完整收敛证明。

唯一平地完整门槛失败项：wave 2.5 半步长，`left_right_force_imbalance_rms_bw` 从 `0.0406891931` 增至 `0.0450538411`，增加约 10.73%；上限为旧值的 110%，即 `0.0447581124`。超限绝对量约 `0.000296 BW`。没有放宽阈值。此指标统计 38 个足球接触几何的左右法向力差，并非所有腿部接触的总力矩，也不是直接测得的净 body-roll torque。

| 名义平地诊断 | wave 2.5 | wave 3 |
|---|---:|---:|
| Tip 绝对角度 P95 | 7.10° | 11.03° |
| Tip 峰值角度 | 18.67° | 17.88° |
| Tip 限位占比 | 0 | 0 |
| 机身整体 roll P95 | 3.75° | 2.33° |
| 地面法向合力峰值 / BW | 1.479 | 1.244 |
| 接触穿透 P95 | 0.0717 mm | 0.0687 mm |
| Tip 速度峰值 | 301.65 rad/s | 418.39 rad/s |

仍不能宣称生物学高保真：tip 瞬时速度和关节加速度存在接触尖峰，既有工程上限很宽；约 0.07 mm 的穿透也不等于零 sink。短时 body 接触在名义 wave 2.5 中仍有发生，只是处于既有门槛内。视频没有对轨迹做平滑来隐藏这些现象。

## 粗糙地形：失败结果也保留

| wave | 位移 mm | Root roll 峰值 | 地面合力峰值 / BW | 结果 |
|---|---:|---:|---:|---|
| 2.5 | 123.652 | 26.25° | 7.226 | 未通过 |
| 3 | 128.232 | 23.73° | 9.013 | 未通过 |

失败涉及 body 角速度/10 ms 速度变化、身体/足端接触力等；wave 2.5 的 tip 速度峰值为 1008.94 rad/s，也略超原 1000 rad/s 上限。完整失败列表见门槛账本。

粗糙地形评估采用同一地形，初始化是保持相对关节不变的 root-only 接触起点二分定位，再上移 25 µm，所有初速度为零。旧轴 rough 对照也使用这一定位方法。原平地一次高度定位无法可靠处理 hfield，故此处改的是地形定位实现，不是身体/腿/控制器。

继承的 `swing_height` 是绝对 world-Z，不能在起伏地形中解释为相对局部地表的离地高度；root 高度/roll 等也有平地阈值适用性问题。但接触力与角速度的真实超限不能因此忽略。本轮不据此升级 rough 基线。

## 侧碰机构对照

相同 k/c、几何及质量，仅换轴；近端规定前进轨迹，重力关闭，无步态控制器。该测试检查弯曲机制，不测量真实动物受力或整机通过能力。

| 量 | 旧轴 | Tip-local Z |
|---|---:|---:|
| 最大向后足端偏移 | 0.0980 mm | 0.2413 mm |
| 接触期间平均向后偏移 | 0.0072 mm | 0.1529 mm |
| Tip 峰值弯曲 | 9.61° | 11.85° |
| 障碍法向力峰值 | 25.99 mN | 19.03 mN |

新轴通过让位方向、脱离障碍、回弹和未撞限位的机构检查。旧轴也并非完全不弯曲，不能将较小的后退位移描述成“没有自由度”。

## 文件与 Demo

- [平地候选 XML](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_tip_passive_gliding_2026-09-01/models/centipede_flat_localz_k2em05_d3p2_r38.xml)
- [粗糙地形候选 XML——未通过](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_tip_passive_gliding_2026-09-01/models/centipede_rough_localz_k2em05_d3p2_r38.xml)
- [冻结字段与全部门槛账本](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_tip_passive_gliding_2026-09-01/outputs/data/contract_and_selection.json)
- [交付视频与机构测试清单](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_tip_passive_gliding_2026-09-01/outputs/delivery/localz_k2em05_d3p2_r38/delivery_manifest.json)
- [平地三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_tip_passive_gliding_2026-09-01/outputs/delivery/localz_k2em05_d3p2_r38/flat_three_views.mp4)
- [平地 tip 局部三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_tip_passive_gliding_2026-09-01/outputs/delivery/localz_k2em05_d3p2_r38/flat_tip_local_three_views.mp4)
- [旧轴/新轴侧碰三视图](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_tip_passive_gliding_2026-09-01/outputs/delivery/localz_k2em05_d3p2_r38/side_contact/side_contact_axis_comparison_three_views.mp4)
- [粗糙地形三视图——失败诊断](C:/Users/wxy22/Documents/Centipede_MUJOCO-main/experiments/mujoco_tip_passive_gliding_2026-09-01/outputs/delivery/localz_k2em05_d3p2_r38/rough_three_views.mp4)

完整模型视频均明确标注 contact erasure OFF，上行为 wave 2.5、下行为 wave 3；平地及 rough 为前/侧/斜视，局部为俯/侧/斜视。侧碰视频上行为旧轴、下行为新轴，0.25 倍速；青色虚线为同一近端姿态下的未弯曲参考，橙色为实际 tip。平地视频来自 12 s 记录的前 6 s。每段视频核验帧数与尺寸，重建姿态抽查最大误差小于 3e-8 m。打光、颜色、隐藏重复碰撞几何仅用于渲染，不改变物理或滤波运动。

直接加载 XML 不会自动获得本轮既有的波数对应运行时 PD 配置；复现实验应使用本目录 `run_tip_flexion_search.py validate --candidate localz_k2em05_d3p2_r38`，它读取并应用冻结的 near-ground PD 配置及三个固定幅值。

## 下一步边界

这一轮不启动 random-contact 批量研究，也不替换正式模型。下一轮最小工作是先在 `k≈1.2e-5..2e-5` 之间补查平地支撑力/身体接触的可行区间，同时检查粗糙地形的真实接触尖峰和地形相对指标；保留控制器公式、幅值和波数关系不变。仅有几个通过的名义点，不构成“仅调 tip 参数一定能解决全部问题”的证明。
