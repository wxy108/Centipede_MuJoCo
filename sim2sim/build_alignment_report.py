"""Create a readable audit from preserved evidence, without promoting a pending candidate."""
import json
from pathlib import Path
import shared as s
summary=json.loads((s.ROOT/'audit/coupled_candidate_summary.json').read_text())
def val(x,precision=2):return '待完成' if x is None else f'{x:.{precision}f}'
lines=['# Centipede：MuJoCo 与 Isaac Lab 的 sim-to-sim 审计与验证', '',f"数据更新时间：{summary['updated_utc']}", '', '**当前状态：物理定义已对齐；整机性能仍在验证，尚未通过全组验收。**', '',
'本轮按“先审计、再改动、再验证”执行。原始审计写入 `AUDIT.md` 后，才建立新的共用物理模型和实验。所有调整位于独立的 2026-09-11 对齐分支；历史模型、优化结果及 Clean edition 保留。此报告的性能数据来自原生 MuJoCo 和 Isaac Lab / PhysX 的开放环路机械基准，不代表原 RL 训练、奖励、重置和策略管线已经全部迁移。', '',
'## 两个项目实际在做什么', '',
'MuJoCo 最近的幅值实验走的是 `common_pd_amplitude_contact_2026-09-09/core.py` → rigid-control amplitude engine → tip-flexion runner → 三幅值控制器与评价器。它固定一整套 PD，在平地 1 Hz 下搜索 body yaw、leg yaw、leg pitch 幅值。通用 README 中的 Blender/FARMS 流程并不等于这次实验的实际入口。', '',
'Isaac Lab 原研究项目由 `centipede_env.py` / `centipede_env_cfg.py` 组织物理场景、观测、奖励和重置，由 `impedance_controller_torch.py` / `kinematics.py` 生成关节位置及速度目标。九月实验实际通过 `stage_a_vec_sweep_cell_compliance_2026-07-13.py` 和对应 runtime 包装器运行，使用原生 PhysX 隐式关节驱动。它实际采用 1 ms，而通用环境默认配置是 4 ms；不能只读默认配置来解释已保存结果。', '',
'历史 RL v4 优化还随 wave number 调整阻抗，并做 near-best / smooth selection；九月 MuJoCo 则固定全表 PD、搜索更宽的三幅值范围。两者的优化变量、选择规则、启动过程与评分窗口不同。因此历史最优 PD、幅值和位移不能直接作为“同一个机器人在两个引擎中的差异”。', '',
'## 审计发现的真实差异', '',
'| 项目 | 原 MuJoCo 实验 | 原 Isaac Lab 九月实验 |', '|---|---|---|',
'| 总质量 | 2.536031092 g | 2.536051087 g |',
'| 尾部质量 | 0.146094614 g | 0.179572322 g，重约 22.915% |',
'| 一条完整腿质量 | 3.434946387 mg | 3.917995336 mg，重约 14.06% |',
'| 躯干 link_body_8 | 0.114060281 g | 0.109677720 g；主惯量最大相差约 14.765% |',
'| 身体与关节表示 | 173 个非 world 刚体，171 个 hinge | 251 个刚体，173 个 revolute，另有 77 个 fixed joint |',
'| 头部 | 固定相对角度 −29° | 头部 pitch / roll 可动 |',
'| 地面滑动摩擦 | 1.6 | 0.5 |',
'| 自碰撞 | 开启 | 关闭 |',
'| 地面 | 单平面 | 两个不同高度的有效表面 |',
'| 实际时间步长 | 0.125 ms | 1 ms，TGS |',
'| 驱动离散方式 | implicitfast 中速度项隐式 | 位置和速度驱动参与隐式约束求解 |',
'| 位移评分 | 21 个身体几何中心，1.500–5.999 s | formal 61-spine centroid，预热后 5.952 s |', '',
'总质量几乎相等掩盖了质量分布和惯量差异。比较 USD 中一个很轻的 carrier 与 MuJoCo 的整个焊接腿也会产生错误结论；本次按完整物理组成核算。历史成对自碰撞消融还会改变步态排名，说明它不是无关紧要的开关。', '',
'## 已建立的共用物理定义', '',
'共用模型以原 Isaac Lab 九月 rigid-tip 资产为明确来源，逐刚体导出质量、COM、完整惯量张量、关节坐标系和碰撞几何，再生成 MuJoCo 表示。它是新的对齐模型，不是把旧 MuJoCo 数据重新命名。', '',
'| 核对项 | 实测结果 |', '|---|---:|',
'| 刚体 / 转动关节数量 | 两边 251 / 173 |',
'| 共用总质量 | 2.536051087 g |',
'| 逐刚体质量最大相对误差 | 6.681 × 10⁻⁸ |',
'| 完整惯量张量最大相对误差 | 1.216 × 10⁻⁷ |',
'| COM 位置误差 | 0 |',
'| Kp / Kd 最大相对误差 | 3.710 × 10⁻⁸ / 5.670 × 10⁻⁸ |',
'| 零位姿身体位置误差 | ≤ 0.047 µm |',
'| 两个非零关节姿态的身体位置误差 | ≤ 0.066 µm |',
'| 两个非零姿态方向误差 | ≤ 6.180 × 10⁻⁷ rad |',
'| 249 个机器人碰撞体的材质回读 | 静 / 动摩擦 0.5 / 0.5，恢复系数 0 |',
'| PhysX contact / rest offset 回读 | 20 µm / 0 |',
'| 物理 armature / joint friction | 0 / 0；MuJoCo 另有明确标记的积分系数 |', '',
'惯量比较统一为“关于 COM、用 link 坐标轴表达的张量”。PhysX getter 已完成主惯量到 link 轴的旋转，不能再次乘 COM 旋转；否则会制造不存在的巨大差异。模型包括 114 个球、114 个 capsule 和 21 个 mesh 碰撞体；球半径为 0.4125、0.45、0.4875 mm，capsule 半径 0.375 mm。', '',
'两边使用单一 z=0 平面、重力 −9.81 m/s²、关闭自碰撞、零额外刚体阻尼、相同头部两个自由度与 −29° pitch 目标。所有关节从零位置和零速度开始，初始最小碰撞体离地 100 µm。', '',
'以下是共用的物理 PD，未按引擎、wave number 或性能重新拟合。单位是 N·m/rad 与 N·m·s/rad。', '',
'| 关节族 | Kp | Kd |', '|---|---:|---:|']
for name,(kp,kd) in s.C['impedance_Nm_per_rad_and_Nms_per_rad'].items():lines.append(f'| {name} | {kp:.10g} | {kd:.10g} |')
lines += ['', '## 为什么同样的 Kp / Kd 还需要对齐数值实现', '',
'MuJoCo 当前安装版本是 3.10.0，不能使用新在线文档中才出现的 `discrete` 积分器。现有 implicitfast 对速度导数隐式处理；PhysX PGS 驱动的后向 Euler 形式还包含步末位置。对于同一个物理力律 K(q_target−q_next)+D(v_target−v_next)，代入 q_next=q_old+h·v_next，可得到内部速度系数 D+hK。MuJoCo 对齐分支使用这个推导得到的系数，位置项仍为 K，目标速度前馈仍为 D·v_target。', '',
'因此两边的物理阻尼都是 D；MuJoCo 数值实现中的 D+hK 不是另行优化出来的物理阻尼。这个适配对 PGS 的实际惯量单关节测试成立，不能未经验证套在 TGS 上。七类关节、实际有效惯量尺度的单关节测试中，适配后的归一化响应 RMS 最大约 0.000396%。', '',
'公式依据：[MuJoCo 数值积分](https://mujoco.readthedocs.io/en/stable/computation/index.html#numerical-integration)、[PhysX 隐式驱动推导](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/_downloads/6acf3afb8f69452757e0e766b5a22978/implicitDrives.pdf)。整机是耦合、多接触系统，单关节误差很小不等于整机性能已经一致。', '',
'## 接触响应的独立证据', '',
'使用半径 0.45 mm、质量 1 mg 的球，锁定转动、释放水平与竖直平移，从 0.2 mm 高度以 20 mm/s 水平速度落向摩擦 0.5、恢复系数 0 的平面。刚性无弹性参考的停止位置约 127.71 µm。', '',
'| 接触设置 | 最终水平位置 | 观察 |', '|---|---:|---|',
'| 本机 PhysX TGS，16 / 4，h=62.5 µs | 114.10 µm | 碰撞时速度反向至约 −11.30 mm/s |',
'| 同一设置只将 TGS 改为 PGS | 126.85 µm | 无上述反向运动 |',
'| MuJoCo 固定 contact time constant=0.5 ms，细步长 | 约 132.50 µm | 仍有软接触差异 |',
'| MuJoCo rigid approximation，time constant=2h | 约 127–128 µm | 随步长减小接近刚性参考 |', '',
'这是这台机器、这个版本和这个探针中的实测差异，不是对所有 TGS 场景的泛化。轨迹表明反向运动发生在碰撞时；早先“空中提前摩擦”的猜测已被轨迹排除。降低 contact offset 或去掉 velocity iteration 并未消除反向；相关失败消融完整保留。', '',
'额外 30° 斜向滑动测试中，PGS h=62.5 µs 停止于 (218.17,124.15) µm，MuJoCo h=15.625 µs 为 (220.32,127.21) µm。它未显示可直接解释 20%–30% 整机差距的巨大方向摩擦异常。', '',
'![接触求解器独立测试](figures/05_contact_solver_comparison.png)', '',
'## 完整步态验证与验收状态', '',
'测试预先固定 4 个 wave number × 2 组幅值，共 8 组。A=[32,18,26]°；B=[16,72,24]°，依次为 body yaw / leg yaw / leg pitch。频率 1 Hz，启动 1 s，渐入至 1.25 s；统一取 link_body_0…20 的非质量加权几何中心，在 2–6 s 沿负世界 X 的前进位移。', '',
'主控制时钟为 62.5 µs。MuJoCo 细步长保持同一时钟的目标；PhysX CPU 半步长同时细化解析目标采样，因此是完整离散方案的收敛检查，不能说成只改变碰撞步长的单因素实验。', '',
'验收阈值保持不变：所有 8 组完成并前进至少 50 mm；两引擎对称相对位移差 ≤10%；各自进一步细化的位移变化 ≤5%。相对差分母为两者绝对位移均值，下限 10 mm。CPU / GPU 差异单独报告；CPU 步长检查不能冒充 GPU 步长检查。', '',
'| wave / 组 | Mu 31.25 µs (mm) | Mu 15.625 µs (mm) | PhysX CPU 62.5 µs (mm) | PhysX CPU 31.25 µs (mm) | PhysX GPU 62.5 µs (mm) |', '|---|---:|---:|---:|---:|---:|']
for r in summary['rows']:lines.append('| '+f"{r['wave']} / {r['family']}"+' | '+' | '.join(val(r[k]) for k in ['mu_dt31_mm','mu_dt15_mm','native_cpu62_mm','native_cpu31_mm','native_gpu62_mm'])+' |')
lines += ['', '![完整步态性能](figures/06_pgs_full_bank.png)', '', '![数值敏感性](figures/07_pgs_numerical_checks.png)', '',
'早期 common-v1 的单个校准步态曾通过 62.5→31.25 µs 的 5% 检查，但完整组中存在 6%–17% 的变化，因此没有采用。随后分步 PD / 接触方案仍对大摆腿步态敏感，相关整组细步长结果也保留。当前表格使用后续的耦合积分方案；MuJoCo 全部八组的 31.25→15.625 µs 检查已经通过。PhysX CPU 的半步长与 16/4→64/16 迭代检查也已完成，单独作为诊断。所有数值修订见 `NUMERICAL_AMENDMENT_01.md` 至 `_09.md`，保留原阈值、失败和变更依据。', '',
'没有用改变物理 PD、摩擦系数或幅值去压低误差；没有删除不接近的 wave number。也没有从当前有限范围推断所有速度、地形、随机失联或 RL 策略均可迁移。', '',
'## 证据与复现边界', '',
'- `AUDIT.md` / `audit/static_inventory.json`：修改前审计、实际调用链与源文件校验值。',
'- `COMMON_CONTRACT.json`：冻结的物理定义与原始验收标准；旧数值候选由单独 amendment 明确修订，未悄悄覆盖。',
'- `assets/common.xml`、`assets/common.usda`、`assets/mapping.json`：共用模型与逐刚体、逐关节映射。',
'- `runs/*/backend_validation.json`：新 PhysX CPU / GPU 场景的实际参数回读。',
'- `runs/*/trajectory.npz` / `results.json`：完整原始轨迹与评分；partial 文件不是最终成绩。',
'- `runs/*/quality_audit.json`：基于记录姿态的几何复核；脚部离地距离需结合 pose-fit 残差解释，不能当成精确 PhysX 接触穿透测量。',
'- `pgs_validation_table.csv` / `audit/pgs_candidate_summary.json`：全部固定步态的机器可读汇总。',
'- `audit/source_preservation.json`：41 个已审计原文件的保存核对。', '',
'正式采用配置与进入后续幅值优化之前，必须先处理完整验证表中的未通过项。当前报告仍是验证中的记录，不能据此宣布 gap 已完全解决。', '']
profile=json.loads((s.ROOT/'OPERATING_PROFILE.json').read_text())
maxgap=max(r['primary_gap_pct'] for r in summary['rows'])
mufine=[r['mu_refinement_pct'] for r in summary['rows'] if r['mu_refinement_pct'] is not None]
gpufine=[r['gpu_refinement_pct'] for r in summary['rows'] if r['gpu_refinement_pct'] is not None]
headline=(f"**物理定义已对齐。当前八组 MuJoCo / 原生 GPU 的最大位移差为 {maxgap:.2f}%；MuJoCo 半步长最大变化 {max(mufine):.2f}%。**" if len(mufine)==8 else '**物理定义已对齐，完整数值验证仍在进行。**')
if profile['status']=='validated_gpu_scope':headline+=' GPU 半步长检查和几何复核已通过，配置已限定在原生 GPU 基准范围内采用。'
elif summary['failed']:headline+=' 尚未通过验收，未通过项在下表后的记录中列出。'
else:headline+=' GPU 半步长检查仍在进行，当前尚未通过最终验收。'
lines=[headline if line.startswith('**当前状态：') else line for line in lines]
extra=['## 接触与驱动必须共享有效惯量', '',
'仅对齐 D+hK 还不够：旧 MuJoCo 流程先用物理质量矩阵 M 求接触力，再以 M+hD+h²K 更新速度；PhysX PGS 将驱动与接触一起求解。单关节无接触测试无法发现这个分步差异。', '',
'当前 MuJoCo 实现把 A=hD+h²K 加入数值质量矩阵，并以显式的 K(q_target−q_old)+D·v_target−(D+hK)·v_old 驱动力配合半隐式 Euler 更新，让接触和速度更新共享 M+A。这是同一物理 PD 的一阶后向 Euler 方程；没有重新拟合 Kp/Kd，也没有给真实机器人增加 rotor inertia。实现借用了数值 armature 槽，因此记录明确区分“物理 armature=0”和“积分所需的 A”。它也不等于新版本 MuJoCo 完整的 discrete 积分器。', '',
'原理与流程参见 [MuJoCo 计算流程](https://github.com/google-deepmind/mujoco/blob/main/doc/computation/index.rst) 和 [有效惯量共享的离散方法说明](https://mujoco.readthedocs.io/en/latest/computation.html)。具体推导、适用范围和修改前冻结记录见 `NUMERICAL_AMENDMENT_08.md`。', '',
'只驱动切向的球面接触测试中，两种 MuJoCo 实现本来都很接近 PhysX，未形成有效区分。加入法向 PD 后，归一化水平响应 RMS 从 0.602% 降到 0.075%；这支持了接触/驱动耦合机制，而不是仅凭整机位移挑一个更接近的结果。', '',
'同一物理模型、PD、幅值和 h=62.5 µs 下，w=2.5 / B 的 MuJoCo 位移由分步方案的 127.333 mm 提高到耦合方案的 155.066 mm；进一步减半为 160.372 mm。全部八组再用 15.625 µs 检查，最大变化为约 2.223%。', '',
'![接触与 PD 耦合测试](figures/10_coupled_contact_response.png)', '',
'## 额外核对与排除的解释', '',
'整机质量矩阵还需统一基座参考点：MuJoCo 自由基座的平移坐标在 link 原点，而原生根速度使用根 link 的 COM。未经换算会出现约 9.3% 的虚假根惯量差异；换算后，按对角惯量归一化的最大矩阵项误差，CPU 约 0.000354%，GPU 约 0.004461%。这不是原模型物理参数的 9.3% 变化。', '',
'进一步对七个关节族分别施加 1 µs 力矩脉冲，在无重力、无接触、无驱动时测量整机速度响应；两种原生后端相对共用模型 M 的逆响应，最大动能范数误差均低于 0.0026%。完整耦合惯性也能对应，不能把所有接地性能差异都归因于小惯量或质量读入错误。', '',
'严格把数值质量单位从 kg 改为 mg，同时缩放质量、惯量、PD 和力矩上限，CPU w=2.5 / B 仅由 194.838 mm 变为 194.973 mm，约 0.07%。原单位单环境重跑与原整组该环境完全一致。该单位变换没有被采用为性能修正。', '',
'CPU 增加四倍求解迭代后，八组最大位移变化不足 0.9%；CPU 步长减半最大变化约 1.1%。但 CPU / GPU 在部分大摆腿步态仍有明显差异，因此不能用 CPU 完成的收敛检查代替 GPU 的检查。当前配置只以原生 GPU 为候选使用范围，CPU 差异完整保留。', '',
'另一个直接写目标的加速尝试同时改变了环境批次大小和诊断读取，没有复现原批次轨迹，速度收益也很有限；不能从该混合试验单独断言批次大小就是根因。它未被采用，GPU 细步长验证继续使用原 v6 路径。', '',
'证据见 `audit/generalized_mass_comparison.json`、`audit/inverse_mass_response_comparison.json`、`runs/native_unit_probe_mg_case03_000` 和 `audit/fast_path_stop_receipt.json`。', '']
at=lines.index('## 接触响应的独立证据');lines[at:at]=extra
motion_file=s.ROOT/'audit/motion_quality_comparison.json'
if motion_file.exists():
    motion=json.loads(motion_file.read_text(encoding='utf-8'))
    motion_pairs=motion['paired']
    maxamp=max(p[k] for p in motion_pairs for k in p if k.endswith('_actual_amplitude_difference_deg'))
    maxlateral=max(p['lateral_drift_difference_mm'] for p in motion_pairs)
    maxheading=max(p['heading_change_difference_deg'] for p in motion_pairs)
    motion_lines=['## 动作相似程度与剩余轨迹差异','',
        f'对 2–6 s 实际关节轨迹作 1 Hz 正弦/余弦拟合，逐关节估计基频幅值，再按 body yaw、leg yaw、leg pitch 关节族取平均。MuJoCo / GPU 在八组中的最大关节族平均幅值差为 {maxamp:.3f}°。这是已执行动作的描述性统计，不是重新优化幅值，也不是逐关节最大误差。', '',
        f'前进距离接近仍不代表路径逐点重合。两边侧向净偏移差最大 {maxlateral:.2f} mm，出现在 w=1.5 / A；头尾连线的净朝向变化差最大 {maxheading:.2f}°。此基准关闭了航向反馈，报告保留这些差异，没有通过旋转坐标、事后对齐路径或修改控制器把它们消掉。', '',
        '![动作诊断](figures/11_motion_diagnostics.png)', '',
        '原始诊断表见 `motion_quality_table.csv` 与 `audit/motion_quality_comparison.json`。跟踪误差对比使用每个采样状态之前最后施加的位置目标；朝向采用头尾连线，含身体波动，不能当作刚体根 yaw 的同义量。', '']
    at=lines.index('## 证据与复现边界');lines[at:at]=motion_lines
lines=[line.replace('PhysX CPU 半步长同时细化','PhysX CPU 和 GPU 的半步长检查同时细化').replace('figures/06_pgs_full_bank.png','figures/08_coupled_full_bank.png').replace('figures/07_pgs_numerical_checks.png','figures/09_coupled_acceptance_checks.png').replace('`pgs_validation_table.csv` / `audit/pgs_candidate_summary.json`','`coupled_validation_table.csv` / `audit/coupled_candidate_summary.json`') for line in lines]
table_start=next(i for i,line in enumerate(lines) if line.startswith('| wave / 组 |'))
table=['| wave / 组 | Mu 31.25 µs (mm) | Mu 15.625 µs (mm) | GPU 62.5 µs (mm) | GPU 31.25 µs (mm) | CPU 62.5 µs，仅诊断 (mm) |','|---|---:|---:|---:|---:|---:|']
for r in summary['rows']:table.append('| '+f"{r['wave']} / {r['family']}"+' | '+' | '.join(val(r[k]) for k in ['mu_dt31_mm','mu_dt15_mm','native_gpu62_mm','native_gpu31_mm','native_cpu62_mm'])+' |')
lines[table_start:table_start+10]=table
if summary['failed']:lines[table_start+10:table_start+10]=['','未通过项：']+[f"- case {r['case']}：{r['criterion']} = {r.get('value_pct',r.get('value_mm'))}，限值 {r.get('limit_pct','见契约')}。" for r in summary['failed']]
if summary['missing']:lines[table_start+10:table_start+10]=['',f"尚缺 {len(summary['missing'])} 个验收值；GPU 细步长的中途轨迹不计作最终成绩。"]
elif not summary['failed']:lines[table_start+10:table_start+10]=['',f"八组数值门槛均通过。MuJoCo 半步长最大变化 {max(mufine):.3f}%，GPU 半步长最大变化 {max(gpufine):.3f}%；正式采用还以配置状态和几何复核收据为准。"]
lines += ['','## 对齐入口与演示','','MuJoCo：`Centipede_MUJOCO-main/experiments/sim2sim_alignment_2026-09-11/run_aligned.py`。原 Isaac Lab：`Centipede_IsaacLab_RL/outputs/analysis/sim2sim_alignment_20260911/run_aligned.py`。两者读取同一份 `OPERATING_PROFILE.json`，并检查冻结输入的校验值。当前状态由配置与验收收据决定，未通过时不会作为正式基准启动。', '',
'两入口运行开放环路共用物理基准。原 RL 训练、奖励、策略、历史优化入口及 Clean edition 没有被替换；以后比较优化结果需明确使用哪个入口和物理契约。', '',
'动作对照 `demos/sim2sim_w2p5_B_closeup_replay.mp4` 使用两边实际记录的姿态、同一几何模型与同一跟随视角，按真实采样时间播放，并显示 2–6 s 前进距离；渲染不重新推进物理。', '',
'![动作对照预览](demos/sim2sim_w2p5_B_closeup_preview.png)', '']
if profile['status']=='validated_gpu_scope':lines=[('本次通过范围以 `VALIDATION_RECEIPT.json` 为准。共用物理模型在所测平地、固定幅值、1 Hz、四个 wave number 的原生 GPU 基准中达到预先规定的接近程度；CPU 后端、其他步态范围与完整 RL 训练流程尚不在通过范围内。' if line.startswith('正式采用配置与进入后续幅值优化之前') else line) for line in lines]
report='\n'.join(lines)
(s.ROOT/'AUDIT_AND_STATUS.zh-CN.md').write_text(report,encoding='utf-8')
old=s.ROOT/'REPORT.md';archived=s.ROOT/'REPORT_INITIAL_COMMON_V1.md'
if old.exists() and not archived.exists():archived.write_bytes(old.read_bytes())
old.write_text(report,encoding='utf-8')
runtime_file=s.ROOT/'RUNTIME_ALIGNMENT.md'
runtime_prefix=runtime_file.read_text(encoding='utf-8').split('## Why a same-PD comparison still needs a time-step check')[0].split('## Current numerical profile')[0]
runtime_prefix=runtime_prefix.replace('| Joint armature and friction readback | 0 |','| Physical armature / joint friction | 0 / 0; the MuJoCo numerical inertia increment is separate |')
runtime_footer=f'''## Current numerical profile

Current status: **{profile['status']}**. The selected candidate is coupled MuJoCo at 31.25 µs and native PhysX GPU PGS 16/4 at 62.5 µs, with a shared 62.5 µs command clock. The eight nominal displacement gaps have maximum {maxgap:.3f}%. MuJoCo 31.25→15.625 µs refinement has maximum {max(mufine):.3f}%.

GPU refinement completion: {len(gpufine)}/8 gaits. See AUDIT_AND_STATUS.zh-CN.md and audit/coupled_candidate_summary.json for live acceptance results. The CPU backend retains a separate performance discrepancy and is not interchangeable with the GPU benchmark.

The older split-PD / fixed-soft-contact candidate failed full-bank convergence and was not adopted. In the current method, the contact and velocity solves share M+hD+h²K. The numeric armature slot carries hD+h²K; this is an integration coefficient, not a physical rotor inertia. The physical PD, body masses, COMs and tensors remain fixed. The contact time constant is the numerical rigid approximation 2h. Thus backend numerical fields are not claimed to be identical just because the physical definition is shared.

The complete-matrix comparison also transforms root-origin translation to root-COM velocity coordinates. After that transform, seven 1 µs force-to-velocity probes agree within 0.0026% in kinetic norm on both native CPU and GPU. This excludes a large hidden inverse-inertia loading error in those sampled directions; it does not prove equality of contact algorithms.

MuJoCo's numeric stage forces and native pre-step estimated PD forces are labelled separately. They are used as torque-cap diagnostics, not as interchangeable measurements of mechanical work.
'''
runtime_file.write_text(runtime_prefix+runtime_footer,encoding='utf-8')
print(str(s.ROOT/'AUDIT_AND_STATUS.zh-CN.md'))
