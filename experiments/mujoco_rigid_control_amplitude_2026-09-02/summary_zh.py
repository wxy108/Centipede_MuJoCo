"""Chinese entry point for the completed, independently audited experiment."""
import json
from pathlib import Path
import engine as e


def link(label,path):
    return f'[{label}]({Path(path).as_posix()})'


if __name__=='__main__':
    audit=json.loads((e.DATA/'final_delivery_audit.json').read_text());assert audit['passed']
    table=json.loads((e.DATA/'amplitude_results_table.json').read_text())
    videos=json.loads((e.DATA/'delivery_video_manifest.json').read_text())
    paired=json.loads((e.DATA/'paired_ablation.json').read_text())
    qualified=[r for r in table if r['status']=='qualified']
    unresolved=[r['wave'] for r in table if r['status']!='qualified']
    motion=json.loads((e.DATA/'achieved_joint_motion.json').read_text())
    local_roll={r['wave']:max(r['world_roll_abs_peak_deg_per_body']) for r in motion if r['status']=='qualified'}
    lines=['# Centipede Model：刚性腿对照与平地幅值优化','',
        '本轮仅处理生物启发的 Centipede Model。没有修改或混用 Centipede Robot 的 Step 18/19 数据，也没有覆盖原基线。','',
        '## 主要结果','',
        f"- 检查了 {audit['distinct_amplitude_conditions']:,} 组不同幅值组合，{len(qualified)}/{len(table)} 个波数获得通过全部复核的候选。",
        '- 刚性对照采用真正的刚性合并：去掉 38 个 tip 被动自由度，保留髋部 leg yaw/pitch；不添加 knee，不使用超大 stiffness 代替刚性。总质量约 2.53603 g、腿长、几何和匹配惯性保持一致，armature 仍为零。',
        '- 相同参数下，刚性腿没有表现出推进和稳定性的共同优势。因此幅值优化仍使用最新 local-Z 柔性 tip，刚性模型保留为独立对照。这个结果不等于刚性模型经另行调参后必然更差。',
        '- 本轮只改变 body yaw、leg yaw、leg pitch 三个 amplitude；频率 1 Hz，body/leg wave number 一致，phase、tip 参数、质量、惯性、每个波数对应的 PD 均不变，contact erasure 关闭。',
        '- 不同波数的历史 PD 并非完全相同，因此这里是各自固定 PD 条件下的幅值优化，不是单独隔离 wave number 的因果实验。','',
        '## 各波数结果','',
        '幅值单位为度。D 是 21 个身体节点几何中心在固定前向轴上的位移，不是质量加权质心位移；六秒试验统计区间为 t=1.5–5.999 s。原幅值的 D 仅作参考，并不代表原步态全部稳定。','',
        '| Wave | Body yaw / leg yaw / leg pitch | 合格 D (mm) | 原幅值 D (mm) | 状态 |',
        '|---:|---|---:|---:|---|']
    for r in table:
        if r['amplitudes']:
            a=r['amplitudes'];angles=' / '.join(f'{a[k]:.3f}' for k in ('body','leg_yaw','leg_pitch'))
            distance=f"{r['D_mm']:.3f}";status='通过复核'
        else:angles=distance='—';status='未解决'
        lines.append(f"| {r['wave']:g} | {angles} | {distance} | {r['baseline_D_mm']:.3f} | {status} |")
    lines += ['', '筛选顺序是稳定性/动作质量/体态/接触与滑移约束先通过，再最大化前向位移。名义六秒候选按位移从高到低接受独立的 12 秒与半时间步（0.125 ms）测试；半时间步的六秒位移差异须不超过 10%。所有比最终入选者更快的名义合格候选均已检查。','',
        '连续参数空间的数学全局最优没有得到证明。准确表述是：全范围采样、两轮多起点细化及一轮有限耦合边界采样后的已测试集合内，通过声明验证条件的最高名义位移。未解决不代表不存在可行参数，也不代表所有失败都是翻倒；有的只未通过波形、滑移或跟踪约束。','',
        f"逐节姿态核验：roll 并未被完全消除。wave=2 最差身体节点的瞬时绝对 roll 为 {local_roll[2.]:.2f}°，wave=2.5 为 {local_roll[2.5]:.2f}°，wave=3 为 {local_roll[3.]:.2f}°。不能只看整体平均 roll；逐节姿态包络已单独列入稳定性报告。这是结果诊断，没有新增或放宽门槛。",'',
        '历史继承的“至少达到参考位移 90%”只作诊断，不作稳定性门槛；其他稳定性/姿态/动作/接触门槛没有因此放宽。细节见 '+link('目标定义说明',e.ROOT/'OBJECTIVE_CLARIFICATION.md')+'。','',
        '## 刚性对照','',
        '| 地形 | Wave | 柔性 D (mm) | 刚性 D (mm) | 刚性相对变化 |',
        '|---|---:|---:|---:|---:|']
    for terrain in ('flat','rough'):
        for wave in (2.5,3.):
            pair={r['topology']:r for r in paired if r['terrain']==terrain and r['task']['wave']==wave}
            flexible=pair['flexible']['metrics']['displacement_mm'];rigid=pair['rigid']['metrics']['displacement_mm']
            lines.append(f"| {'平地' if terrain=='flat' else '粗糙地形'} | {wave:g} | {flexible:.3f} | {rigid:.3f} | {100*(rigid/flexible-1):+.2f}% |")
    lines += ['', '刚性平地两组均有稳定性/动作门槛未通过；粗糙地形的两种模型均未获得合格结论，相关视频只作对照诊断。幅值优化仅在平地进行。完整结果见 '+link('刚性对照报告',e.ROOT/'PAIRED_RESULTS.md')+'。','',
        '## 全部 Demo','',
        '每个全景视频包含前视、侧视、斜视。优化视频上排是原幅值，下排是新候选；未解决波数明确标为 DIAGNOSTIC ONLY / NOT QUALIFIED。运动来自原始记录，没有对动作做平滑。','']
    for video in videos:
        status='通过复核' if video['selection_status']=='qualified' else '仅供诊断'
        lines.append(f"- wave={video['wave']:g}（{status}）："+link('三视图 Demo',video['video']))
    lines += ['', '柔性/刚性对照：上排柔性、下排刚性；局部视频另有未弯曲参考线。','']
    for terrain in ('flat','rough'):
        for wave in (2.5,3.):
            prefix=f'paired_{terrain}_w{wave:g}'.replace('.','p')
            lines.append(f"- {'平地' if terrain=='flat' else '粗糙地形'} wave={wave:g}："+
                link('三视图',e.OUT/'videos'/f'{prefix}_three_views.mp4')+'；'+link('局部放大',e.OUT/'videos'/f'{prefix}_local.mp4'))
    lines += ['', '## 参数、图与审计','',
        '- '+link('全精度合格幅值及固定 PD 参数表',e.DATA/'selected_amplitude_profiles.json'),
        '- '+link('三个指令幅值随波数变化',e.OUT/'figures'/'amplitudes_by_wave.png'),
        '- '+link('前向位移对比',e.OUT/'figures'/'displacement_by_wave.png'),
        '- '+link('体态与接触指标图',e.OUT/'figures'/'stability_by_wave.png'),
        '- '+link('体态、tip 转角与时间步复核数值',e.ROOT/'STABILITY_RESULTS.md'),
        '- '+link('实际关节运动数据及各波数图路径',e.DATA/'achieved_joint_motion.json'),
        '- '+link('代表性画面与逐节姿态核验说明',e.ROOT/'VISUAL_QA.md'),
        '- '+link('完整复核记录',e.DATA/'validated_selection.json'),
        '- '+link('最终文件、参数与选择审计',e.DATA/'final_delivery_audit.json'),
        '', '这些通过条件是本轮工程化验证标准，不等于完成了生物学动态参数标定，也不保证任意地形、扰动或 random-contact 条件下稳定。','']
    path=e.ROOT/'RESULTS_ZH.md';path.write_text('\n'.join(lines),encoding='utf-8')
    print(str(path))
