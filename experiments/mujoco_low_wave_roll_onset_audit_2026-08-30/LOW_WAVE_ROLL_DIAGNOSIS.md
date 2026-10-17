# Centipede Model 低波平地整体 roll 诊断书

日期：2026-08-30  
范围：仅 Centipede Model；平地；`wave=1.5`；只诊断，不调参、不改模型。

## 结论

当前看到的整体 roll **不是质量/惯量精确化或 q2 柔顺化之后新产生的失稳**，但它也不能用此前“所有 chassis/所有时刻合并后的绝对 world-roll P95”来解释。该旧指标把沿身体的局部扭转与整条身体共同侧倾混在了一起；用户从前视图判断“旧模型整体 roll 不大”是正确的。

真正发生了两件事：

1. 旧 A 模型的合并 absolute world-roll P95 虽为 `22.43 deg`，但逐帧全身有符号平均 roll 的 P95 只有 `4.60 deg`：主要是节间扭转，不是整条身体一起滚。
2. q2 k20 active-body-roll 阶段的共同模态仍只有 `5.06 deg`，说明 mass/inertia、接触重标定和 q2 柔顺化都没有制造当前视觉问题。
3. body-roll 改为 passive k4 后，全身共同模态跃升至 `12.47 deg`，为旧 A 的约 `2.7x`；同时局部关节 roll 从 `10.27 deg` 降至 `3.96 deg`，世界 roll 跨度从 `46.53 deg` 降至 `33.32 deg`。这正是从“分布式扭转”转成“更刚性一致的整体侧倾”。

因此，当前最符合证据的历史根因边界是：**active body-roll → passive k4 body-roll。较强的相对关节弹簧压低了节间变形，却没有绝对重力参考；腿—地面产生的非零 roll 力矩于是更多进入自由基座共同模态。** 新腿段、tip 和更清楚的视角增强了可见性，但问题在 tip/head 阶段之前已经出现。

## 因果交叉回放

所有回放使用历史 MJCF 原文件，只在运行时交换控制器表；统一仿真 6 s、评估 1.5–6.0 s、flat `wave=1.5`。

| 组合 | world-roll P95 | local body-roll P95 | 前进位移 | 解释 |
|---|---:|---:|---:|---|
| A 旧 plant + 旧 flat controller | 22.43° | 11.65° | 133.81 mm | 旧模型已存在大 world roll |
| B 旧 plant + 后继 contact-audit controller | 15.30° | 6.24° | 41.80 mm | 软参数降低 roll，但推进严重下降 |
| C Stage1 q2 k30 plant + 旧 controller | 25.60° | 15.04° | 95.17 mm | Stage1 只放大约 3.17° |
| D Stage1 q2 k30 plant + 后继 controller | 22.55° | 10.13° | 80.57 mm | 后继参数没有制造 roll |
| E q2 k20 plant + 后继 controller | 22.72° | 10.27° | 84.02 mm | q2 k30→k20 仅增加 0.17° |

上述表中的 world-roll P95 不是共同模态判据。重新按每一帧的有符号 chassis roll 平均后：

| 历史边界 | 全身共同 roll P95 | 前七节共同 roll P95 | 沿身体 world-roll 跨度 P95 | 局部关节 roll P95 |
|---|---:|---:|---:|---:|
| A 旧 active | 4.60° | 14.11° | 50.15° | 11.65° |
| E q2 k20 active | 5.06° | 13.84° | 46.53° | 10.27° |
| P passive k4 | 12.47° | 14.61° | 33.32° | 3.96° |

### 对各假设的判定

- **“质量/惯量重建制造共同 roll”——排除。** A 已是 geometry-derived inertia、armature=0 的历史精确模型，其共同模态只有 `4.60°`。
- **“contact-audit 的软 leg/knee 参数制造 roll”——排除为主因。** B 对 A、D 对 C 都使 roll 下降；它确实造成支撑与推进退化，但不是整体 roll 的来源。
- **“q2 k20 柔顺化制造 roll”——排除。** D→E 仅 `+0.17°`。
- **“passive k4 body-roll 产生模态转换”——高置信成立。** 共同 roll 从 `5.06°` 跃升至 `12.47°`，而局部 roll 同时下降。
- **“tip 重构或 head lock 首次制造 roll”——排除。** passive k4 阶段在 tip/head 改动之前已经出现 `12.47°` 共同模态。

## 对“前半段腿 pitchdown 不够”的检验

旧 A 回放的实际 leg-pitch 基频幅值：front/middle/rear = `36.55/36.49/36.46°`，几乎完全一致。当前谱系已测为约 `29.97/30.29/31.52°`；前后差约 `1.55°`，存在轻微梯度，但 active→passive 边界在 gait 不变时已经使全身共同 roll 增加约 `7.41°`，因此 pitch 梯度不是首要根因。

左右接触腿数量相同也不能保证净 roll 力矩为零。需要相消的是关于质心的有符号力矩：

`tau_roll = sum(y_i * Fz_i - z_i * Fy_i)`

接触位置、法向力幅值、切向力、落脚时刻和前后分布任一不对称，都会在接触数量相等时留下非零力矩。A 中非足端腿段接地在整个评估窗都曾发生，说明早期模型也不是“只有对称足尖支撑”的理想状态。

## Centipede RL 旧模型复核

这次使用的不是上一版 MuJoCo，而是 `Centipede_IsaacLab_RL` 中冻结的历史数据。

### 直接观测

- 2026-06-30 B1 在六个 fixed-gait flat 设计上得到：head 约 `10.3 deg p2p`、mid-body `1.6 deg p2p`、tail `3.1 deg p2p`。同一审计测得 body-roll joint torque 相对 body-yaw torque 约为 `0.0%`（图中精度），因此旧 RL 的平稳外观不能解释成 projected-gravity policy 或强绝对 roll servo 在在线纠偏。
- 2026-07-18 P8 `BASELINE / flat / wave=1.5` 的原始 251-body pose trace 仍在。用与 B1 相同的 yaw-independent up-vector roll 公式复算，21 节有符号平均的 coherent-roll absolute P95 为 `2.17 deg`，front-7 为 `9.26 deg`；沿身体的 world-roll span P95 却有 `39.80 deg`。也就是说，各段确实在按相位扭转，但没有整条身体同向滚倒。视频审计 PASS，HUD 末帧 roll 为 `2.12 deg`。
- 较晚的 `Centipede_IsaacLab_RL_Clean` open-loop `wave=1.5` 轨迹按同一公式复算，coherent-roll absolute P95 为 `3.59 deg`。它同样没有策略反馈，但控制器比 B1/P8 更强，只作为旁证，不与旧模型混成同一个版本。

### 与当前 MuJoCo 的关键结构差异

| 项目 | 历史 RL P8 baseline | 当前 MuJoCo locked lineage |
|---|---:|---:|
| total mass | 项目实测 `2.5361 g` | XML geometry mass sum `2.5360 g` |
| armature | `0.0` | `0.0` |
| foot topology | foot welded；无额外 passive foot DOF | 38 个 passive tip-roll，范围 `+-38 deg` |
| body roll | implicit-PD target 0；`Kp=0.00942`, `Kv=0.001275` | passive k6；`K=0.00069584`, `c=9.864e-7` |
| distal roll | legbase-roll target 0；`Kp=0.00024` | passive tip-roll；`K=4.5e-6`，无 actuator |
| knee / distal bend | knee 固定参考角；`Kp=0.00586`, `Kv=0.000433` | knee DOF 已移除；由短 tip-roll 提供柔顺 |
| feedback | fixed gait / open-loop；无 policy roll correction | scripted open-loop |

数值只用于识别量级，Isaac implicit PD 与 MuJoCo passive spring 不能当作完全等价元件。即便如此，差距足以定性：历史 P8 的 body-roll `Kp` 是当前 k6 的约 `13.5x`，`Kv` 约 `1292x`；历史 knee 相对当前 tip spring 的刚度量级约 `1302x`、阻尼约 `3.75e5x`。更重要的是，历史 RL 的脚端没有当前这个自由的接触端 roll DOF。

### 根因判断

旧 RL 没有大 coherent roll 的主要原因不是质量矩阵，也不是 projected gravity：它与当前 MuJoCo 总质量几乎相同、armature 同为 0，且上述平地证据是 fixed-gait/open-loop。它的左右支撑高度被“welded foot + 被阻抗保持的 knee/leg roll + 高得多的 body-roll damping”约束住；各段 roll 虽大，但相位错开，净支撑面仍接近水平。

当前 MuJoCo 则把相当一部分柔顺性放进了直接承载地面反力的自由 tip-roll，并把 body-roll 改为低阻尼被动相对弹簧。历史边界已经证明：在 tip 重构之前，active q2 k20 -> passive k4 就让 coherent roll 从 `5.06 deg` 升至 `12.47 deg`，同时 local joint roll 从 `10.27 deg` 降至 `3.96 deg`。所以 tip 不是首次起因；它是在同一条垂直支撑路径上继续放大左右有效腿高差。较强的相对 spine coupling 又让这些差异少表现为局部 twist，多进入自由基座的共同 roll 模态。

因此不能把旧 RL 的 PD 数值直接抄回 MuJoCo；那会重新引入此前要排除的工程性高刚度。它真正提供的线索是：**需要重新分配柔顺性与耗散的位置，尤其是“接触端允许多少有效腿长变化”，而不是重新做 mass/inertia。**

## 下一步仍然只诊断

若继续，建议做一项不改模型的 phase-resolved support audit：逐接触点记录 `r x F`，按左右、front/middle/rear 和 foot/non-foot 分类；同时记录左右接触端的有效高度、tip-roll 角与 leg-pitch 相位，并将净 roll 力矩与 root roll 角加速度做相位相关。只有该审计能区分“哪一簇腿在注入力矩”和“哪一侧 tip 先缩短支撑高度”。通过门槛应是：力矩求和可重构 root roll 响应、旧 RL/P8 与当前 MuJoCo 使用同一统计口径、且不依赖修改 stiffness 或 gait 来验证。

## 非干预声明

- 未修改任何 MJCF、mass、inertia、armature、joint stiffness/damping、controller 参数或 gait。
- 新增内容只有诊断脚本、JSON、图和相机一致的重放 Demo。
- Centipede Robot Step 18/19 四个哨兵文件前后 SHA-256 完全一致。
