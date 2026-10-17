# 最新 IsaacLab Centipede Model 代码与动力学审计

日期：2026-08-30  
范围：仅生物启发式 Centipede Model；只读审计，不修改模型、控制器或 gait。

## 结论先行

此前使用的 2026-07 P8 baseline 不是最新 IsaacLab 运行模型，只能保留为历史对照。当前唯一受支持的运行主线是：

`C:\Users\wxy22\Documents\Centipede_IsaacLab_RL_Clean`

它在 2026-08-17 从 active contract 建成 clean-room release，明确只保留一个 robot model、一个 controller 和一个 environment，并明确排除历史模型变体。旧仓库中的 `leaner`、`leaner_bent135`、`tarsus`、`tarsus_massfix` 文件虽然仍存在，但不被当前 Clean runtime 选择。

当前真正加载的物理资产是：

`assets/centipede_rebuild_v1_renamed.usd`  
SHA-256：`6dfe73a944edd3179dd792931ee13c134b68fda9818849f7780e56e235335db9`

这不是根据文件时间猜出的“最新”，而是从 `env_cfg.py -> CentipedeEnv -> ModelIndex -> Controller` 的实际调用链、Clean provenance/source lock、唯一保留的 canonical demo 和 USD composition 一层一层核对得到的。

对当前 MuJoCo 整体 roll 问题最重要的新结论是：**最新版 IsaacLab 不是靠质量矩阵或 projected-gravity 闭环把身体扶正，而是用更强的全关节 implicit-PD 支撑拓扑限制共同 roll。** 最新版 body-roll `Kp=0.07536`，是 P8 的 `0.00942` 的整整 `8x`；足端仍然焊接，legbase-roll 与 knee 也都被位置目标保持。它可作为机制对照，但不是可以直接移植到 MuJoCo 的生物学高保真目标。

## 当前运行权威的证据

- Clean `README.md` 声明这是 minimal runnable research mainline，只包含一个 robot model/controller/environment，且不含 alternate models。
- `provenance/MIGRATION_REPORT.md` 声明该 release 于 2026-08-17 从 active contract 建成，并明确排除 historical model variants 与 superseded controller branches。
- `provenance/SOURCE_ORIGINS.md` 指定上述 renamed USD 及四个 composition layer 为唯一物理运行资产；matching MJCF/OBJ 仅用于离线可视化，不定义第二套动力学。
- `centipede/env_cfg.py` 直接把 `ASSET_PATH` 指向上述 USD，没有模型选择分支。
- `scripts/check_project.py` 于本次审计重新运行，结果为 `CLEAN_PROJECT_CHECK: PASS`；本次审计前的 CPU suite 为 `5 passed`。

旧仓库 `C:\Users\wxy22\Documents\Centipede_IsaacLab_RL` 现在是 provenance/history 与 IsaacLab dependency checkout。它的 `demo/structure_reveal/README.md` 也明确写明 implementation authority 是 sibling Clean edition。因此旧仓库中更新日期更晚或名字更“新”的未纳入变体，不能越过实际加载链被称为当前模型。

## 通读的当前代码链

本次逐文件读完并交叉核对了：

- `README.md`、`PI_HANDOFF.md`
- `provenance/SOURCE_ORIGINS.md`、`MIGRATION_REPORT.md`、`SOURCE_LOCK.json`
- `centipede/model.py`、`gait.py`、`terrain.py`、`env_cfg.py`
- `centipede/controller.py` 全部 330 行
- `centipede/env.py` 全部 410 行
- `scripts/evaluate.py`、`open_loop_sweep.py`、`train.py`
- `config/controller.yaml`、`gait_map.json`、`rl_games.json`
- IsaacLab articulation 初始化中把 actuator armature/gain 写回 PhysX 的实现路径
- hash-locked USD 的 prim、mass、inertia、joint、drive 与 fixed-joint composition

## 最新 USD 的直接物理清单

只读打开 hash-locked USD 后得到：

| 项目 | 最新 USD |
|---|---:|
| rigid bodies | 252（运行时 deactive `worldBody` 后为 251 个 link pose） |
| revolute joints | 211 |
| fixed joints | 39 |
| body yaw / pitch / roll | 19 / 20 / 20 |
| legbase yaw / pitch / roll | 38 / 38 / 38 |
| knee | 38 |
| fixed foot joints | 38 |
| authored mass bodies | 251 |
| authored total mass | `0.0025360511 kg`（2.536051 g） |

质量分布：

| body family | 数量 | 总质量 | 占比 |
|---|---:|---:|---:|
| 21 main body links | 21 | 2.387127 g | 94.128% |
| 两组实体 leg segments | 76 | 0.134303 g | 5.296% |
| welded feet | 38 | 0.014505 g | 0.572% |
| pitch/roll 与 proximal-leg spacer bodies | 116 | 约 0.000116 g | 约 0.004% |

main body 的对角惯量范围为 `4.42e-10 ... 1.58e-9 kg m^2`；实体腿段为 `1.24e-13 ... 1.87e-12 kg m^2`；spacer bodies 使用 `1e-9 kg / 1e-15 kg m^2` 的数值占位。后者是多轴关节用串联刚体表达时的工程化 articulation spacer，不应解释成真实动物组织质量。

这份质量总量与当前 MuJoCo locked lineage 的约 `2.536031 g` 几乎相同。因此“最新版 Isaac 不整体滚，是因为它明显更重或质量矩阵完全不同”没有证据支持。两者真正不同的是关节保持方式与柔顺性放置位置。

### USD armature 与运行时 armature 必须分层

USD 的 211 个 revolute joint 都 authored 为约 `1e-4` armature。这只是资产层数值，不是 Clean 仿真最终生效值。

`centipede/env_cfg.py` 的唯一 `ImplicitActuatorCfg` 覆盖全部 211 个 active joints，并显式设置 `armature=0.0`。IsaacLab 的 articulation 初始化实现随后调用 `write_joint_armature_to_sim(actuator.armature, ...)`，并把同一值写回 `default_joint_armature`。所以当前运行 contract 的有效 armature 是 `0.0`；不能读取 USD 后就断言最新版仍用 `1e-4`。

本次还尝试在当前并发 Isaac 作业旁直接实例化一环境、从 PhysX view 二次读回。为避免干扰其他正在运行的 15 个 Isaac Python 进程，使用了独立 Warp cache；但在可用内存不足时初始化未在合理时间内完成，已停止本审计进程。故这里不伪造 live readback；有效值结论来自当前 config 覆盖范围与 IsaacLab 写回实现的闭合代码证据。

## 当前控制器实际在做什么

当前 active articulation 有 211 个 revolute DOF，全部接收 PhysX implicit-PD position 与 velocity targets：

| family | 数量 | target | Kp | Kv |
|---|---:|---|---:|---:|
| body yaw | 19 | CPG traveling wave | wave-conditioned；w1.5 为 `0.01133665` | `1.13645e-7` |
| head pitch | 1 | `-0.57 rad` | `0.0304180` | `0.00768830` |
| remaining body pitch | 19 | `0` | `0.0006` | `0.0009` |
| body roll | 20 | `0` | `0.07536` | `0.00360624` |
| legbase yaw | 38 | leg CPG | w1.5 为 `0.00466561` | `3.92418e-6` |
| legbase pitch | 38 | leg CPG | `0.000728032` | `9.85535e-7` |
| legbase roll | 38 | `0` | `0.00024` | `2.94e-8` |
| knee | 38 | `+0.523599 rad`（30°） | `0.00586` | `0.000433` |

控制目标进一步说明：

- body 基础 amplitude 为 `0.4 rad`、1 Hz；leg amplitude 为 `[0.6, 0.4, 0, 0]`。
- leg 只有 yaw 与 pitch 是波动 DOF；legbase-roll 始终保持 0，knee 始终保持 30°。
- 38 个 foot 通过 fixed joints 焊到 distal leg；没有 passive distal tip-roll DOF。
- controller 先 settle 1.0 s，再以 0.25 s ramp 进入波动。
- flat canonical demo 是 fixed `wave=1.5`；body 与 leg 使用同一个 realized wave number，没有拆成两个 wave number。
- `projected_gravity_b` 只进入 34-D observation 与 failure mask；fixed-wave open-loop demo 不调用 policy，因此没有 projected-gravity roll feedback。
- `worldBody` prim 在 scene setup 中被 deactive，root 是自由基座，不是隐藏 fixed-base。

## 最新版的 roll 直接测量

当前 Clean canonical flat demo 的 exact 251-link pose trace 使用 yaw-independent up-vector roll 公式重算，评估窗为 1.5--9.6 s：

| 指标 | 最新 Clean w1.5 | 历史 P8 w1.5 | 当前 Mu passive k4 |
|---|---:|---:|---:|
| 21 main links coherent roll abs P95 | **3.59°** | 2.17° | **12.47°** |
| front-7 coherent roll abs P95 | **10.05°** | 9.26° | 14.61° |
| 沿身体 world-roll span P95 | 49.23° | 39.80° | 33.32° |
| 单节 roll robust p2p（head/mid/tail） | 36.78° / 28.79° / 41.22° | 大量相位分布扭转 | 不同统计对象 |

因此最新版也不是“每节都不 roll”。相反，各节 world roll 很大且沿身体分相；它压低的是整条身体同向滚动的共同模态。前七节共同 roll 仍约 10°，所以只看正面前半身时会感觉有偏转；但全身共同模态明显小于当前 Mu passive k4。

最新版 flat open-loop 9.6 s 全程存活，前进速度为 `0.3323 body-length/cycle`。三视角视频均由同一 exact pose trace 渲染，front/side/top audit 均 PASS。

## 对当前 MuJoCo 根因判断的修正

最新版证据没有推翻 active-to-passive 边界，反而加强了它：

1. 质量总量几乎相同，且 current runtime armature 与 MuJoCo 都是 0；mass matrix 不是解释 coherent-roll 差异的首要变量。
2. 最新 Isaac 的 body-roll `Kp` 比历史 P8 高 `8x`、`Kv` 高约 `2.83x`，同时 38 个 legbase-roll、38 个 knee 与 38 个 welded foot 都限制支撑端有效腿长变化。
3. 当前 MuJoCo 把 body-roll 改为低阻尼 passive relative spring，并在直接承受地面反力的位置增加 38 个 passive tip-roll。这使左右接触端有效高度差更容易进入自由基座共同 roll。
4. Isaac 的相对零位 PD 也不是绝对重力伺服；但它把足端扰动分散成相位错开的节间 twist，而不是让整条链以低频共同侧倾。

所以最新版 Isaac 是一个“强工程支撑拓扑”的机制对照，不是要抄参数的 template-model anchor。若把 `Kp=0.07536` 直接复制到 MuJoCo，会重新引入本项目明确拒绝的隐性超大 stiffness。下一步仍应做 phase-resolved contact moment / effective-leg-height audit，定位哪一簇腿和哪一侧 tip 在向 root common-roll 注入力矩，再决定如何移动耗散，而不是先把模型硬起来。

## 产物与非干预声明

- USD 直接清单：`outputs/data/latest_isaaclab_usd_inventory.json`
- 最新轨迹 roll 复算：`outputs/data/centipede_rl_flat_roll_audit.json`
- USD 只读检查脚本：`inspect_latest_isaaclab_usd.py`
- 运行时读回脚本（未取得完整 live readback）：`probe_latest_isaaclab_runtime.py`
- 未修改任何 IsaacLab/MuJoCo physics asset、mass、inertia、controller、stiffness、damping、gait 或 terrain。
- 本审计未访问或写入 Centipede Robot Step 18/19 数据。
