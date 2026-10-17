import json
from datetime import datetime,timezone
import shared as s
from common_contact_report_sections import sections
R=s.ROOT
current=sections()
old=(R/'REPORT_PGS_NOT_ADOPTED.md').read_text(encoding='utf-8')
static=old[old.index('## 两个项目实际在做什么'):old.index('## 为什么同样的 Kp / Kd')]
j=json.loads((R/'audit/tgs_candidate_summary.json').read_text())
c2=json.loads((R/('audit/common_contact_centered_summary.json' if (R/'CENTERED_COMMON_CONTACT_CANDIDATE_PROFILE.json').exists() else 'audit/common_contact_summary.json')).read_text())
def f(v):return '待完成' if v is None else f'{v:.3f}'
table=['| wave / 组 | Mu 31.25 µs | Mu 15.625 µs | CPU 250 µs | CPU 125 µs | GPU 250 µs | GPU 125 µs |','|---|---:|---:|---:|---:|---:|---:|']
for r in j['rows']:
    table.append('| '+f"{r['wave']} / {r['family']}"+' | '+' | '.join(f(r[k]) for k in ['mu_dt31_mm','mu_dt15_mm','native_cpu250_mm','native_cpu125_mm','native_gpu250_mm','native_gpu125_mm'])+' |')
fails='\n'.join(f"- 固定测试 {x['case']}：{x['criterion']} = {x.get('value_pct',x.get('value_mm')):.3f}，门槛 {x.get('limit_pct',50)}。" for x in j['failed']) or '尚无已计算的超限项；缺失测试不能视为通过。'
c2table=['| wave / 组 | Mu 31.25 µs | Mu 15.625 µs | CPU 31.25 µs | CPU 15.625 µs | GPU 31.25 µs | GPU 15.625 µs |','|---|---:|---:|---:|---:|---:|---:|']
for r in c2['rows']:c2table.append('| '+f"{r['wave']} / {r['family']}"+' | '+' | '.join(f(r[k]) for k in ['mu_dt31_mm','mu_dt15_mm','cpu_dt31_mm','cpu_dt15_mm','gpu_dt31_mm','gpu_dt15_mm'])+' |')
text=f'''# Centipede sim-to-sim 审计与当前验证结果

更新于 {datetime.now(timezone.utc).isoformat()}。

{current['intro']}

本轮在初始 `AUDIT.md` 完成后才建立新模型并运行仿真。改动集中于独立的日期目录和两个原项目内新增的可选入口；历史默认模型、优化结果和 Clean edition 未迁移。旧 amplitude 优化、contact erasure 和 RL 训练均未恢复。当前测量是原生 MuJoCo 与原 Isaac Lab / PhysX 的开环平地机械基准。

{current['overview']}

{static}
## 已完成的动力学核对

同样的 Kp/Kd 字段并不保证同样的离散力律。PhysX 隐式驱动使用步末位置/速度。MuJoCo 对齐实现由同一物理 PD 推导内部 D+hK 系数，并让接触和速度更新共享 M+hD+h²K。这里的 hD+h²K 借用数值 armature 槽，明确记录为积分系数；物理 rotor inertia 仍为零。安装版本 MuJoCo 3.10.0 没有在线新文档中的 discrete 积分器，此实现不冒充该积分器。

这个修改有独立证据：加入法向载荷的接触 PD 测试，水平归一化 RMS 从 0.602% 降到 0.075%；七个实际惯量尺度的单关节测试，PGS 最大约 0.000396%，TGS CPU/GPU 在共用指令时钟下最大约 0.0425%，均通过独立门槛。

整机无接触、无重力的 2 秒相同 PD / 步态测试中，PGS CPU 与 MuJoCo 最大关节角误差约 0.000037°。身体原点仍有最多 41.85 µm 差异，不能称完整位姿完全相同。根质量矩阵需换算根原点与 COM 的参考点；换算后的七组 1 µs 力矩脉冲，CPU/GPU 动能范数误差均低于 0.0026%。这些证据将剩余重点指向接触条件下的数值行为，但不证明所有接触误差的唯一来源。

推导参考：[MuJoCo 数值积分](https://mujoco.readthedocs.io/en/stable/computation/index.html#numerical-integration)、[PhysX 隐式驱动](https://nvidia-omniverse.github.io/PhysX/physx/5.5.0/_downloads/6acf3afb8f69452757e0e766b5a22978/implicitDrives.pdf)。引用文档的版本不代表已确认安装 PhysX SDK 恰为该版本；已知安装环境为 Isaac Sim 5.0 / OmniPhysics 107.3.18。

## 新发现的接触求解差异

在同一个 0.45 mm 球、同质量、μ=0.5、锁定转动的独立测试中，施加大小为 1.2μN、方向 45° 的切向力：安装版本的 PGS CPU/GPU 几乎保持静止；TGS 和 MuJoCo 椭圆锥都产生约 (34.8, 34.8) µm 位移。配合轴向测试，证实 PGS 在该测试中表现为分轴方形限幅，TGS 为圆形切向限幅。仅把 μ 都设为 0.5 不足以统一摩擦行为。

曾按解析几何构造 MuJoCo 方形锥适配器以诊断 PGS；没有拟合摩擦系数。它虽通过独立力界限测试，但整机 2.5/B 位移从圆形模型的 160.372 mm 降为 136.248 / 146.845 mm，离 PGS CPU 更远，故未采用。

TGS 默认外力积分还使落地滑行球产生 −11.30 mm/s 的瞬时反向速度。启用安装 schema 中有效的 `enableExternalForcesEveryIteration=true` 后，末位置由 114.098 µm 变为 CPU 127.498 / GPU 127.500 µm，接近独立刚性非弹性参考 127.71 µm。CPU 的明显反向速度消失；GPU 仍有 −0.0982 mm/s 的小残差，因此不能描述为完全精确。该改动改变求解流程，不改变重力、PD、质量或摩擦系数。

![独立接触证据]({(R/'figures/13_tgs_contact_evidence.png').as_posix()})

## 未采用的原生 TGS 接触：完整失败结果

四个 wave number：1.5、2.5、3.5、4.5。A 幅值为 [32,18,26]°，B 为 [16,72,24]°，顺序为 body yaw / leg yaw / leg pitch。没有按引擎改 PD 或幅值。所有位移使用 21 个身体原点的非加权中心，沿世界 −X，严格计算 2–6 s。

门槛在运行前冻结：每组完成且位移至少 50 mm；跨引擎对称相对差不超过 10%；每个引擎半步长变化不超过 5%。八组均需满足，另核查有限性、翻转与碰撞几何。指令使用共用 250 µs 时钟；MuJoCo 31.25→15.625 µs 保持指令，TGS 250→125 µs 同时细化指令，后者检查完整数值方案，不能称纯积分步长消融。

下表单位 mm。这是已完成但未通过的原生 TGS 候选：**{j['status']}**；缺失指标 {len(j['missing'])} 项。当前共用柔顺接触版本的结果见本报告开头及后面的独立分支章节。

{chr(10).join(table)}

已计算的超限项：

{fails}

![完整步态对照]({(R/'figures/14_tgs_full_bank.png').as_posix()})

![冻结验收门槛]({(R/'figures/15_tgs_acceptance_checks.png').as_posix()})

粗时间步下的接近不足以接受。后续诊断分别考察更细的 TGS、四倍求解迭代、固定指令时钟、单次位置迭代和 MuJoCo NoSlip 后处理。NoSlip20 使 Mu 2.5/B 从 160.110 变为 160.341 mm，仅约 0.145%，没有消除整机差异。CPU 线程数曾随诊断变化，现已补做唯一改动为 8→2 线程的完整对照，九组全部 601 帧的关节位置、速度、身体原点逐位相同；不再把这个因素与步长/迭代混为一谈。所有诊断均保留完整记录，未采用失败的修改。

## 持续滑动中的法向响应与未成功的修正

独立球体使用连续的物理水平 PD 驱动（K=100 N/m、D=0.01 N·s/m、目标速度 30 mm/s），竖直方向自由运动。同质量和 μ=0.5 下，MuJoCo 原接触模型的平均离地高度约 6.041 µm、竖直速度 RMS 8.121 mm/s；TGS 基本保持接触，约为数值零。规定水平速度的独立对照和更细步长也出现同类差异；NoSlip 和提高 impratio 未消除。这个证据排除了单纯轨迹复位伪影，符合关联与非关联摩擦法向关系的差异，但不证明整机差距只有这一项来源。

De Saxcé 法向修正在独立持续滑动、冲击滑行和斜向力测试中明显改善一致性。然而完整机器人产生难以收敛的接触非线性方程：强化后的求解器在 dt125 / dt62.5 µs 分别于 1.47625 / 1.367 s 停止，保存未收敛状态，不产生有效 2–6 s 位移。更换根求解方法和收紧内部容差仍不能解决保存帧；优化器的 success 字段不能替代接触残差验收。该方案没有采用，也没有声称证明了数学无解。

![持续滑动的法向响应]({(R/'figures/18_continuous_slide_normal_response.png').as_posix()})

## 当前独立分支：共用柔顺接触模型

`COMMON_CONTACT_V2.json` 在这一分支仿真前冻结；`CONTACT_V2_SCOPE.md` 同时建立，后来仅补充语义说明。两套原生动力学仍各自计算机器人惯性、关节约束、重力和同一物理 PD；原生地面响应关闭，接触外力来自同一个 float64 函数。固定参数：法向刚度 200 N/m、阻尼 0.01 N·s/m；切向刚度 20 N/m、阻尼 0.001 N·s/m；圆形 Coulomb 限幅 μ=0.5。1 mN 单点负荷对应 5 µm 法向压缩及摩擦极限处 25 µm 切向弹性位移。这是按几何尺度给定的工程基准，**不是实测材料参数，也不是拟合步态得到的参数**。

两边共用 114 个球支持点、228 个胶囊端点球支持点、21 个最低网格顶点，共 363 个候选点；切向位移在滑动时回映射，卸载或网格支持顶点切换时清零。这是明确的接触离散化选择，不冒充原生接触流形完全相同。每个刚体的全局力与关于 COM 的力矩通过原生外力接口施加；七组独立力/力矩脉冲的动能范数误差 CPU 约 0.002478%、GPU 约 0.002552%，均通过 2% 门槛。

资产中的原生摩擦、恢复系数和 contact offset 仍做回读，但在原生地面响应关闭后不再产生接触力。本分支的有效反弹由共用刚度、阻尼、有效惯量及积分决定，不能把材质属性 restitution=0 当作新模型的有效恢复系数。这里的刚体几何重叠表示等效接触变形，网格本身没有变形。

0.5 s 落地检查中，两边最小身体中心高度约 2.46284 / 2.46286 mm，最大关节角约 7.49083 / 7.49073°。这是启动检查，不是性能通过。首个短 CPU 检查继承了 PGS 不支持的 TGS 外力迭代开关，后端明确报错；正式完整组已改为 false，并保存回读。后续支持顶点向量化在 1000 个连续状态、9 个环境上得到逐位相同的力、力矩和摩擦记忆；GPU 的 0.02 s 初始化轨迹也逐位相同。原参考实现保留。

仍使用原八组步态、相同 PD、2–6 s 指标、10% 跨引擎差和 5% 半步长门槛；三种后端均检查 31.25→15.625 µs，指令固定 250 µs。柔顺模型允许物理压缩，压缩深度、身体倾斜和几何拟合误差单独报告，不将它与原生刚性模型零穿透要求混同。数值状态 **{c2['status']}**，缺失验收指标 {len(c2['missing'])} 项；采用状态：{current['state']}。

第一轮共用接触测试保留了 0.25 m 的环境间距，CPU/Mu、GPU/Mu 主步长最大前进差仅 0.026617% / 0.020956%，但细步长暴露横向坐标误差。因此它没有作为最终布局采用。当前独立副本全部放在数值原点附近；原生碰撞仍全部关闭，共用接触函数按环境分别施力，各机器人没有互相接触。没有运行中重置位置或改变速度，也没有改动物理量。第二轮额外检查完整 XY 路径和方向的跨引擎及半步长差，原有前进门槛保持不变。第一轮完整数字仍保存在 `audit/common_contact_summary.json`。

旧偏移布局的 GPU 半步长任务已停止，最后完整中间轨迹保存到 2.5 s，未生成 2–6 s 位移，也未计为通过。旧布局已经有完整 CPU 因果反证；停止它是为了把计算用于原点布局的完整 GPU 主步长与细步长。具体进程与保留文件记录在 `audit/common_contact_displaced_cancellation.json` 和 `audit/common_contact_displaced_closeout.json`。

{chr(10).join(c2table)}

{current['detail']}

## 已排除或尚不能支持的解释

- 完整惯性读入的大错误：逐体惯量、非零位姿和整机逆惯性脉冲均不支持。
- 单纯质量数量级：严格 kg→mg 的一致单位变换只使敏感测试位移改变约 0.07%。
- PGS CPU 迭代不足：四倍迭代，八组位移变化不到 0.9%；这不能外推为 TGS 已收敛。
- CFM、contact offset、friction correlation 单独造成主误差：两组敏感 B 测试的独立消融未消除差异。
- 球面自由滚动：独立测试末位置 464.707 对 465.088 µm，未出现整机程度的分歧。
- 更快目标传输路径：同时改变了传输、批量与诊断，未复现原整组结果；没有把其中某一项认定为根因。

MuJoCo 椭圆锥的正则化可产生持续切向滑移；NoSlip 的完整敏感步态消融已完成，未消除主要差异。[官方接触滑移说明](https://mujoco.readthedocs.io/en/latest/modeling.html#preventing-slip)。所有失败结果保留，没有只展示有利步态，也没有降低原验收标准。

## 使用边界与证据位置

{current['usage']}

原始审计见 `AUDIT.md`。当前数据见 `audit/common_contact_centered_summary.json`、`audit/common_contact_centered_motion_gates.json`、`common_contact_centered_validation_table.csv` 和各运行目录的轨迹、参数回读与完整性回执。旧失败 TGS 数据单独保留在 `audit/tgs_candidate_summary.json` 和 `tgs_validation_table.csv`。所有科学图均单独保存 PNG / PDF / SVG。现在不据此重新解释历史最优幅值，也不启动新的优化或 contact erase。
'''
if current['accepted']:text=text.replace('数值状态 **NUMERICAL_GATES_PASSED_QUALITY_REVIEW_REQUIRED**','数值门槛全部通过，质量复核已完成')
for name in ['REPORT.md','AUDIT_AND_STATUS.zh-CN.md']:(R/name).write_text(text,encoding='utf-8')
profile=json.loads((R/'TGS_CANDIDATE_PROFILE.json').read_text());profile['status']='rejected_refinement' if j['failed'] else 'validation_pending';profile['acceptance_summary']='audit/tgs_candidate_summary.json';profile['failed_criteria']=j['failed'];s.write(R/'TGS_CANDIDATE_PROFILE.json',profile)
rt=(R/'RUNTIME_ALIGNMENT_PGS_NOT_ADOPTED.md').read_text(encoding='utf-8');rt=rt[:rt.index('## Current numerical profile')]+current['runtime']
rt=rt.replace('Contact algorithms remain different and require dynamic verification.','Native contact algorithms differ; the current shared-force branch disables their contact responses. The readable native material attributes above do not define the active compliant contact law.')
(R/'RUNTIME_ALIGNMENT.md').write_text(rt,encoding='utf-8')
print(json.dumps(dict(report='REPORT.md',common_contact_status=c2['status'],missing=len(c2['missing']),accepted=current['accepted'])))
