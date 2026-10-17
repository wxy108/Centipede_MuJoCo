# Centipede Model 生物尺度、运动学与地形接触审计

日期：2026-08-26  
范围：仅 `Centipede_MUJOCO-main` 中的抽象/生物启发 Centipede Model。`Centipede_robot_MUJOCO` 的 Step 18/19 只作只读哈希哨兵，未参与参数选择，也未被写入。

## 结论

当前版本比上一版物理结构明显正确，但还不能称为“生物真实”或“fully stable”。

- 尺度合理：模型躯干长度 102.5 mm、总质量 2.536 g，与 2024 年实验中约 100 mm、3 g 的蜈蚣处于同一量级。
- 惯量来源合理：没有显式 `<inertial>`，没有单独质量覆盖；质量和惯量由封闭碰撞几何与统一密度生成，惯量三角不等式全部通过。
- 已消除明确的隐藏被动物理错误：最终审计变体 `armature=0`、joint `stiffness=0`、joint `damping=0`，并删除了 38 个位于球形脚中心、没有独立生物意义的 foot-spin DOF 及其 actuator/sensor。
- 质量矩阵仍不是“完全健康”：删除 foot-spin DOF 后保持正定、条件数从 `1.136e11` 改善到 `1.286e10`（约 8.83 倍），但最小有效惯量仍只有 `1.998e-13 kg·m²`。这是极细小 leg-roll 坐标的真实几何结果，意味着控制器不能按普通尺度随意加 Kp/Kd。
- 控制器仍存在隐藏高阻抗：plant 本身无刚度/阻尼，但 active PD 的 leg roll 为约 400 Hz、阻尼比约 5，knee 为约 325 Hz、阻尼比约 20；这与真实蜈蚣在障碍撞击时依赖被动腿弯曲的证据不一致。
- 运动方向与波传播定性合理：当前 1 Hz travelling wave 从头向尾传播，符合 scolopendromorph 前进时常见的 retrograde limb wave，以及轴向弯曲向后传播的实验结果。
- 运动速度偏低：平地 `0.128 BL/s`，L90/A8 `0.106 BL/s`，L60/A12 `0.070 BL/s`。1995 年 `Scolopendra heros` 的实验速度为 `0.5–1.5 BL/s`；跨物种不能直接定量等同，但当前模型至少仍明显“肌无力/低效”，不能靠重新加大刚度来掩盖。
- 三个 6 s 验证均 finite 且没有 MuJoCo warning，但三个 stability gate 全部失败，主要因为关节越限、峰值速度过高以及平地躯干曲折度过高。

## 与真实蜈蚣的证据对照

| 项目 | 生物证据 | 当前模型 | 判定 |
|---|---|---|---|
| 长度/质量 | 2024 TFM 个体约 10 cm、3 g、21 对腿 | 10.25 cm、2.536 g、19 对腿 | 尺度接近；腿对数是抽象化 |
| 脚端 | 真实末端是尖锐弯曲爪，软凝胶实验允许真实基底形变/刺入 | 半径 0.45 mm 的球形 foot geom，刚性地形 | 接触稳健但形态不真实 |
| 波传播 | `S. heros` 轴向弯曲向后传播；scolopendromorph 常用 retrograde leg wave | 头到尾 travelling wave | 定性一致 |
| 速度—躯干弯曲 | 真实动物速度升高时侧向位移和弯曲幅度显著增大 | body amplitude 固定，不随速度或负载自适应 | 未达到真实运动学 |
| 粗糙地形腿顺应性 | 真实蜈蚣腿撞到障碍后会被动弯曲并继续 stepping pattern | leg roll/knee 是不摆动的恒定目标，且 active PD 过阻尼 | 关键缺口 |
| 头部 warm-up | 生物资料不支持固定大俯仰角 | 第一 body-yaw 先保持直行，1.0 s 后用 0.25 s 释放；但第一 body-pitch 仍有固定 `-0.57 rad` 目标 | yaw 需求满足；sagittal 姿态需专门复核 |

说明：由几何计算惯量只能证明“模型内部质量—惯量一致”，不能证明密度分布就是生物真实。当前 94.85% 质量在 chassis、4.73% 在 legs、0.42% 在 feet；尚无逐节组织质量数据可用于确认这一分配。

原始资料：

- Anderson, Shultz & Jayne (1995), *Axial Kinematics and Muscle Activity During Terrestrial Locomotion of the Centipede Scolopendra heros*: https://journals.biologists.com/jeb/article/198/5/1185/6885/Axial-Kinematics-and-Muscle-Activity-During
- Rieu et al. (2024), *Dynamics of centipede locomotion revealed by large-scale traction force microscopy*: https://pmc.ncbi.nlm.nih.gov/articles/PMC11285478/
- Diaz et al. (2023), *Active and passive mechanics for rugose terrain traversal in centipedes*: https://pmc.ncbi.nlm.nih.gov/articles/PMC10215817/

## Plant 与控制器审计

### Plant 修正

基线 zero-armature 模型虽然没有 armature 和 stiffness，但保留了 38 个 foot-spin hinge，每个脚转轴 damping 为 `0.006117 N·m·s/rad`。单脚转动惯量约 `2.26e-14 kg·m²`，所以该阻尼相对这个自由度极端巨大，是此前不容易从绝对数值看出的隐藏 overdamping。

最终 Model-only 变体：

- 保留 foot body、质量和球形碰撞 geom；只去掉没有独立运动学意义的 foot-spin joint。
- 同步删除 38 个相关 actuator 和 76 个 joint position/velocity sensor。
- `nbody=212` 不变；`njnt: 250 → 212`；`nv: 255 → 217`。
- 总质量保持 `0.00253603109178 kg`。
- armature、被动 stiffness、被动 damping 均为 0。
- 质量矩阵满秩、正定；条件数仍高，不能通过重新添加 armature 来“修好”。

### Active PD 仍需降阶

几何归一化后的旧/审计控制器中位自然频率：

| family | 旧控制器 | 审计候选 | 变化 | 审计判定 |
|---|---:|---:|---:|---|
| body yaw | 100.8 Hz | 100 Hz | 近似不变 | 工程折中 |
| body pitch | 99.1 Hz | 100 Hz | 近似不变 | 工程折中 |
| body roll | 90.4 Hz | 110 Hz | 上升 | 需复核 |
| leg yaw | 648 Hz | 450 Hz | -31% | 仍高 |
| leg pitch | 1254 Hz | 325 Hz | -74% | 明显改善但仍高 |
| leg roll | 1047 Hz, ζ≈106 | 400 Hz, ζ≈5 | 大幅改善 | 仍非生物顺应 |
| knee | 1106 Hz, ζ≈9.2 | 325 Hz, ζ≈20 | 频率下降、过阻尼上升 | 不合格 |

粗糙地形完整摆幅下的静态能力估计：leg pitch 约 `2.66 mN`，leg yaw 约 `19.0 mN`。2024 TFM 文献给出的整只约 3 g 蜈蚣惯性力在 `0–10 mN` 范围，因此 yaw 单腿等效能力仍偏大。它没有转化为高效前进，说明下一步应调整 gait/contact/compliance，而不是增加 Kp。

审计控制器参数只保存在本实验的 `outputs/data/controller_selection.json` 并由 `run_audit.py` 使用；由于所有稳定性门槛尚未通过，没有将其晋升覆盖上游默认控制器。

## Terrain 与 contact 修正

### 已实施

- 旧地形：512×512 PNG、cell `5.87 mm`、8-bit 高度量化。
- 新地形：1536×1536 MuJoCo 自定义 float32 hfield、cell `1.954 mm`，不再有 PNG 高度量化。
- 接触采用 medium：`solref="0.003 1.2"`、`solimp="0.93 0.98 0.0003"`。
- 接触 sink 直接读取 `mjContact.dist`；负值才计为穿透。接触法线读取 `mjContact.frame[0:3]`，每只脚跟踪法向力最大的主接触，避免把极小辅助接触的出现/消失误计为跳变。
- 视频不再同时画碰撞 hfield 与低分辨率 visual mesh。最终 Demo 直接显示同一个 1536×1536 碰撞面，并关闭单面高密网格的 shadow mapping；因此没有视觉三角缝，也没有由另一张地面造成的假 sink。

MuJoCo 官方定义 heightfield 为 triangular-prism union，并提醒地形空间特征要明显大于碰撞 geom；`mjContact.dist < 0` 表示 penetration：

- https://mujoco.readthedocs.io/en/stable/XMLreference.html?highlight=solimpfix
- https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html

### 接触参数选择证据（L60/A12、同一审计控制器、4.5 s）

| contact | sink median / P95 | normal jump P95 | force P95 | peak qvel | forward | 结果 |
|---|---:|---:|---:|---:|---:|---|
| legacy | 0.262 / 0.666 mm | 180°/ms | 0.262 mN | 145 rad/s | 0.094 BL/s | sink/normal 不合格 |
| medium | 0.116 / 0.277 mm | 63.8°/ms | 1.068 mN | 169 rad/s | 0.059 BL/s | 三档中唯一通过 contact gate |
| firm | 0.053 / 0.480 mm | 135°/ms | 15.81 mN | 637 rad/s | 0.004 BL/s | 冲击与速度失控 |

与旧的 512 PNG + foot-spin + legacy end-to-end 基线相比，L60/A12 的 P95 sink 从 `1.718 mm` 降到 `0.273 mm`，约减少 84%；但这同时包含 plant、controller、resolution 和 contact 多项变化，不能解释成单因素因果。

### 尚未解决

- L60/A12 P95 sink `0.273 mm`，L90/A8 `0.336 mm`；分别为 foot radius 的 61% 和 75%，仍然可见。
- 非零主接触法线跳变 P95 仍为 `62.9°/ms` 和 `89.2°/ms`，triangular-prism 接触切换仍明显。
- 稀有最大穿透约 `0.92–0.97 mm`，接近/超过 0.9 mm foot diameter。
- 将 contact 继续调硬会显著放大冲击、关节速度和越限，不能作为修复。

因此，本轮修复了分辨率、量化、错误 foot DOF、contact 选择与视频伪影，但没有消除 MuJoCo heightfield 的分片平面接触本质。

## 6 s 最终验证

| 地面 | 速度 | 路径效率 | 最大 tortuosity | peak qvel | 最大 range fraction | sink P95 | stability gate |
|---|---:|---:|---:|---:|---:|---:|---|
| flat | 0.128 BL/s | 0.807 | 1.441 | 256.9 rad/s | 1.143 | 0.401 mm | FAIL |
| L90/A8 | 0.106 BL/s | 0.760 | 1.263 | 427.9 rad/s | 1.442 | 0.336 mm | FAIL |
| L60/A12 | 0.070 BL/s | 0.821 | 1.117 | 169.2 rad/s | 1.048 | 0.273 mm | FAIL |

三项均 finite，MuJoCo warning 为空；但“未崩溃”不等于 stable。L90/A8 的主要问题是 leg-roll 越限/高速；L60/A12 和 flat 也仍有轻重不等的 body-yaw 越限。

## 下一轮实验计划

1. **主动顺应性重构**：把不参与周期摆动的 leg roll/knee 从高阻尼 active hold 改为低带宽顺应控制或明确的被动 spring-damper；以 `peak qvel < 100 rad/s`、`range fraction ≤ 1.0`、无 warning 为硬门槛。
2. **头—第一节段复核**：保持现有 1.0 s yaw hold 和 0.25 s 平滑释放，单独 sweep 第一 body-pitch 的 `0 … -0.57 rad`；检查头/胸前端离地、drag、前进位移，避免把“直行”与“持续低头”混为一谈。
3. **运动学而非刚度优化**：共同 sweep leg yaw/pitch amplitude、phase offset、stance/recovery duty factor；第一目标是平地达到至少 `0.2 BL/s` 且路径效率 `≥0.85`，不允许通过提高 joint bandwidth 达成。
4. **平滑碰撞面实验**：对 1536/3072 hfield 与连续 SDF/procedural height surface 做同种子 A/B；硬门槛暂定 sink P95 `≤0.10 mm`、主法线跳变 P95 `≤15°/ms`、force P95 `≤5 mN`。
5. **random-contact 回归**：只有 clean flat、L90/A8、L60/A12 三个 nominal gate 全过后，再在相同 plant/controller 上恢复 random-contact；不得读取或覆盖 Robot Step 18/19 结果。

## 可复现实验与证据文件

- 主脚本：`run_audit.py`
- Plant 审计：`outputs/data/plant_audit.json`
- 模型清单与哈希：`outputs/data/model_manifest.json`
- Controller 选择：`outputs/data/controller_selection.json`
- Contact A/B：`outputs/data/selected_contact_sweep.json`
- 6 s flat/rough 验证：`outputs/data/final_validation.json`
- Demo 指标与 Robot 哨兵：`outputs/data/smooth_demo_audit.json`

最终 Demo：

- `outputs/videos/l60a12_final_exact_hfield_side.mp4`
- `outputs/videos/l60a12_final_exact_hfield_oblique.mp4`
- `outputs/videos/l90a8_final_exact_hfield_side.mp4`

三个视频均完成逐帧解码验证。

## Robot 隔离证明

运行前后 SHA256 完全一致：

- Step 18 `run_configuration.json`: `847BEE7492D75CA05FFADC40B4664C8CE02EEDB1D5E16B6683FF2EA9BD6CCF85`
- Step 18 `trials.csv`: `C46F9B1C1F932971CE37EA8AD6C5E9892D01E238E7839867F135D1A507A8E3CC`
- Step 19 `run_configuration.json`: `FF007DEC8796DD0E8F0AF6D58EFFFDF0287C1474E171CD7E086CED88A38EFBC8`
- Step 19 `trials.csv`: `1A4AD813787B68A7EA409C9C261F3A977453AB61035E44B479B8A9A2E7CCEC48`
