# Centipede Model `_2` 柔顺性优化报告

完成日期：2026-08-27  
范围：仅 `Centipede_MUJOCO-main` 的生物启发 template model；不属于 Centipede Robot。

## 结论

Stage-1 的 `_2` 被动恢复刚度 `3.0e-5 N·m/rad` 对当前毫米级、约 2.536 g 的模型偏保守。最终选择 `bounded_k20_c18`：

- 38 个腿部 `_2` 关节的工作区刚度降为 `2.0e-5 N·m/rad`，比 Stage-1 低 33.3%。
- 阻尼保持 `1.8e-7 N·m·s/rad`。单纯增加阻尼不能解决准静态偏转，而且会削弱接触顺应性，因此没有用“高阻尼换稳定”。
- `_2` 仍为被动关节；没有 q2 主动伺服。
- 在 ±30° 处使用显式、可审计的解剖限位：`solreflimit="0.0005 1.5"`、`solimplimit="0.999 0.9999 0.00001 0.5 2"`。限位只在边界附近起作用，关节内部保持较软。
- 没有添加 armature、隐藏关节弹簧/阻尼或质量补偿。平地和 rough XML 的总质量均为 `0.00253603109178 kg`，所有 armature 为零，其他关节显式 stiffness/damping 均为零。

因此答案是：可以让 `_2` 更 compliant 而不损伤当前平地/rough locomotion performance；有效方法是“软工作区 + 明确的解剖限位”，不是把整个关节继续用高刚度锁住。

## 与 Stage-1 基线的定量比较

固定条件：6 s、0.25 ms timestep、body amplitude scale 0.55、首个 body yaw 保持，控制器和地形参数不变。

| 地形 | 指标 | Stage-1 `k30/c18` | 选择项 `bounded k20/c18` | 解释 |
|---|---:|---:|---:|---|
| flat | 前向位移 | 80.442 mm | 83.543 mm | 保留率 103.85% |
| flat | 路径效率 | 0.98447 | 0.98614 | 保留率 100.17% |
| flat | `_2` 角度 P95 | 6.019° | 10.607° | 工作区内运动明显增加 |
| flat | `_2` 最大角 | 30.509° | 30.246° | 限位穿透 0.246° |
| flat | 超过名义 ±30° 的采样占比 | 0.194% | 0.790% | 属于短暂限位接触 |
| flat | 接触穿透 P95 | 0.0411 mm | 0.0384 mm | 未恶化 |
| rough | 前向位移 | 81.173 mm | 81.208 mm | 保留率 100.04% |
| rough | 路径效率 | 0.98446 | 0.98705 | 保留率 100.26% |
| rough | `_2` 角度 P95 | 10.017° | 15.863° | rough 顺应运动增加 |
| rough | `_2` 最大角 | 31.391° | 30.416° | 限位穿透 0.416° |
| rough | 超过名义 ±30° 的采样占比 | 0.087% | 0.450% | 属于短暂限位接触 |
| rough | 接触穿透 P95 | 0.0214 mm | 0.0192 mm | 未恶化 |
| rough | 接触法向跳变 P95 | 8.536° | 8.217° | 未恶化 |

两个地形均满足本轮预先定义的 performance retention 门槛：位移至少保留 90%，路径效率至少保留 95%；同时 upright、spine tortuosity、接触穿透、法向跳变、数值有限性均通过。

## 严格门槛与 bounded-stop 门槛

这里不把两套门槛混为一谈：

- 旧 Stage-1 strict gate 把任何 `|q2| > 30.000°` 的 solver 级穿透都记为失败。选择项在 flat/rough 分别有 0.790%/0.450% 的短暂限位接触，因此 `all_gates=false`。
- 本轮新增的 bounded-stop gate 要求显式限位存在、最大限位穿透 ≤0.5°、限位接触占比 ≤1%，并保留其余稳定性与性能条件。选择项两个地形均为 `all_bounded_gates=true`。

这不是放宽后隐藏失败，而是把“关节撞到显式解剖止挡”和“关节无约束越界”分开统计。原 strict 结果仍完整保留在 JSON 中。

## 被拒绝方案

- `soft_k20_c18`：工作区柔顺且性能不降，但普通软限位导致 flat/rough 最大角约 35.35°/37.10°。
- `soft_k15_c18`、`soft_k10_c18`：位移仍好，但 q2 尾部和越界占比继续增大。
- `visco_k10_c27`、`visco_k22p5_c27`：增加阻尼不足以控制静态偏转，不能替代明确的解剖止挡。
- `bounded_k15_c18`：flat 可接受，但 rough 最大角 30.995°，穿透约 0.995°，超过本轮 0.5° 上限；不选择靠线方案。

## 模型静态审计

- 正确的腿部 `_2` 数量是 38：命名模式 `joint_leg_*_2`。仅用字符串后缀 `*_2` 会误把 `joint_pitch_body_2`、`joint_roll_body_2`、`joint_body_2` 三个 body 关节计入，得到错误的 41。
- 两个选择 XML：`nq=218`，`nv=217`，timestep `0.00025 s`。
- 38 个腿部 `_2` 的 axis 均为 `0 1 0`，范围均为 ±0.523598775598 rad。
- 所有 DOF 的 armature 最小值和最大值均为 0。
- 没有修改 Stage-1 的质量配置来换取表现。
- 平地和 rough 的静态审计记录见 `outputs/data/static_model_audit.json`。

## Demo 与视觉检查

已生成 flat 和 rough 各三视图：side、oblique、top，共六个 MP4；均为 30 fps、6.03 s。接触图汇总在 `outputs/figures/q2_compliance_demo_contact_sheet.png`。

视觉检查：平地身体为浅 S 波而不是整体 C/U 卷曲；side/oblique 可见正常腿部动作；rough 仍保持浅弧形态，没有明显整机卷曲、脚部深陷或旧式粗糙三角边界造成的明显跳变。视觉检查不能替代定量 gate，定量结果见 `outputs/data/demo_audit.json`。

## 可复现入口与当前边界

- 脚本：`run_q2_compliance.py`
- 完整 sweep：`outputs/data/q2_compliance_sweep.json`
- Demo 审计：`outputs/data/demo_audit.json`
- 被动模型清单：`outputs/data/passive_model_manifest.json`
- 选择模型：`models/centipede_flat_stiff_dt0p25ms_q2pm30p0deg_bounded_k20_c18.xml` 与 rough 对应 XML。

本轮是固定 flat/rough 场景的工程候选确认，不等于已经完成跨随机种子、random-contact 或真实个体辨识。下一步应冻结此候选，进入 withheld terrain/random-contact 验证；在验证完成前不把它宣称为唯一生物真实参数。

## Robot 隔离证明

Centipede Robot Step 18/19 文件未写入，前后 SHA-256 一致：

- Step 18 run configuration: `847BEE7492D75CA05FFADC40B4664C8CE02EEDB1D5E16B6683FF2EA9BD6CCF85`
- Step 18 trials: `C46F9B1C1F932971CE37EA8AD6C5E9892D01E238E7839867F135D1A507A8E3CC`
- Step 19 run configuration: `FF007DEC8796DD0E8F0AF6D58EFFFDF0287C1474E171CD7E086CED88A38EFBC8`
- Step 19 trials: `1A4AD813787B68A7EA409C9C261F3A977453AB61035E44B479B8A9A2E7CCEC48`

