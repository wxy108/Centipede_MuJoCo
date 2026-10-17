"""Delivered report: common finer timestep, exact comparisons and honest limits."""
import collections,json
import core as c
from report import baseline_rows,vector,motion_measure
from diagnostics import failed_cases
np=c.np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    selection=c.read(c.DATA/'validated_selection.json');rows=c.nominal();indexed={r['id']:r for r in rows};baseline=baseline_rows();cells=[];export=[]
    checked=c.read(c.DATA/'validation_index.json')
    polish=c.read(c.DATA/'polish_complete.json') if (c.DATA/'polish_complete.json').exists() else None
    if (c.DATA/'polish_plan.json').exists():assert polish and polish['completed'] and selection['qualified_polish_complete']
    assert not [r for r in c.records() if 'error' in r]
    for w in c.WAVES:
        cell=selection['waves'][f'{w:g}'];sel=cell['selected'];r=sel['nominal'] if sel else None;old=baseline.get(w)
        ref=indexed[c.identity(c.task(w,[16.918311805232932,49.37746770784939,22.918311805232932],'reference'))]
        group=[a for a in rows if a['wave']==w]
        cells.append(dict(wave=w,qualified=bool(sel),D=r['D'] if r else None,amplitudes=vector(r) if r else None,
            old_D=old['D'] if old else None,old_nominal_feasible=old['feasible'] if old else None,old_failed=old['failed'] if old else None,
            gain_percent=100*(r['D']/old['D']-1) if r and old and old['feasible'] else None,
            reference_D=ref['D'],reference_nominal_feasible=ref['feasible'],reference_failed=ref['failed'],
            half_step_relative_difference=sel['half_step_relative_difference'] if sel else None,
            half_step_D_mm=sel['half']['D'] if sel else None,
            long_short_speed_ratio=(sel['long']['D']/10.499)/(r['D']/4.499) if sel else None,
            metrics=r['metrics'] if r else None,motion=motion_measure(r) if r else None,
            nominal_full=sum(a['full'] for a in group),nominal_prefix_rejections=sum(a.get('proven_infeasible',False) for a in group),
            nominal_unique=len(group),failed=collections.Counter(k for a in group for k in a['failed']),
            candidates_tested=cell['candidates_tested'],higher_untested=cell['higher_untested'],record=r['record'] if r else None,
            higher_untested_max_gain_mm=max([0.,*[a['D']-r['D'] for a in group if a['feasible'] and a['id'] not in checked]]) if r else None,
            unresolved_diagnostics=[dict(id=a['id'],kind=a['diagnostic_kind'],D=a['D'],amplitudes=a['amplitudes'],
                failed=a['failed'],max_ratio=a['diagnostic_max_ratio'],body_contact=a['metrics']['chassis_contact_fraction'],
                chassis_force=a['metrics']['chassis_force_peak_bw']) for a in failed_cases(group)] if not r else []))
        if r:
            assert cells[-1]['motion'] is not None
            export.append(dict(wave=w,amplitudes_deg=r['amplitudes'],pd_config=r['task']['config'],nominal_D_mm=r['D'],
                nominal_id=r['id'],long_id=sel['long']['id'],half_step_id=sel['half']['id'],half_step_relative_difference=sel['half_step_relative_difference']))
    local=c.read(c.DATA/'local_complete.json')
    c.write(c.DATA/'final_summary.json',dict(nominal_dt_s=.000125,half_dt_s=.0000625,nominal_window_s=[1.5,5.999],waves=cells,local_starts=local,
        fine_nominal_unique=len(rows),imported_exact_conditions=len(c.read(c.DATA/'imported_exact_conditions.json')['conditions']),global_optimum_proved=False,qualified_polish=polish))
    c.write(c.DATA/'selected_gaits.json',dict(model_path=c.history()['plant'],model_sha256=c.history()['plant_sha256'],nominal_dt_s=.000125,
        half_step_dt_s=.0000625,frequency_Hz=1.,contact_loss=False,per_wave_pd_preserved=True,gaits=export,
        unresolved_waves=[a['wave'] for a in cells if not a['qualified']],source_selection=str(c.DATA/'validated_selection.json'),source_selection_sha256=c.sha(c.DATA/'validated_selection.json')))
    figures(cells,rows)
    write_report(cells,local,rows,polish)
    print([{k:r[k] for k in ['wave','D','gain_percent','higher_untested']} for r in cells],flush=True)

def figures(cells,rows):
    folder=c.OUT/'figures';folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9));w=[r['wave'] for r in cells]
    axes[0,0].plot(w,[r['old_D'] if r['old_nominal_feasible'] else np.nan for r in cells],'o--',c='#777777',label='Previous provisional amplitudes, measured at 0.125 ms')
    axes[0,0].plot(w,[r['reference_D'] if r['reference_nominal_feasible'] else np.nan for r in cells],'.:',c='#b6a38e',label='Fixed-amplitude reference at 0.125 ms')
    axes[0,0].plot(w,[r['D'] if r['qualified'] else np.nan for r in cells],'o-',c='#176f9d',lw=2,label='New 6 s / 12 s / half-step qualified selection')
    for r in cells:
        if r['old_D'] is not None and not r['old_nominal_feasible']:axes[0,0].plot(r['wave'],r['old_D'],'x',c='#777777')
        if not r['qualified']:axes[0,0].annotate('NQ',(r['wave'],.025),xycoords=('data','axes fraction'),ha='center',fontsize=8,color='#9b4444')
    axes[0,0].set(title='Matched forward displacement',ylabel='D over 4.499 s (mm)');axes[0,0].legend(fontsize=7.4)
    for j,(name,color) in enumerate(zip(['Body yaw','Leg yaw','Leg pitch'],['#2166ac','#b35806','#38966d'])):
        axes[0,1].plot(w,[r['amplitudes'][j] if r['qualified'] else np.nan for r in cells],'o-',label=name,color=color)
        axes[1,0].plot(w,[r['motion']['actual_joint_half_P95_P05_deg'][j] if r['qualified'] else np.nan for r in cells],'o-',label=name,color=color)
    axes[0,1].set(title='Selected command amplitudes',ylabel='Degrees');axes[0,1].legend(fontsize=8)
    axes[1,0].set(title='Actual joint motion',ylabel='Median joint half (P95-P05), degrees');axes[1,0].legend(fontsize=8)
    full=[r['nominal_full'] for r in cells];prefix=[r['nominal_prefix_rejections'] for r in cells]
    axes[1,1].bar(w,full,width=.34,color='#658fa7',label='Full horizon');axes[1,1].bar(w,prefix,bottom=full,width=.34,color='#c8b69d',label='Proven gate violation prefix')
    axes[1,1].set(title='Exact finer-step nominal coverage',ylabel='Distinct amplitude conditions');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.set_xlabel('Wave number');ax.set_xlim(.25,9.25);ax.set_xticks(w);ax.tick_params(axis='x',labelsize=8,rotation=35);ax.grid(alpha=.15)
    fig.suptitle('Amplitude optimization — common 0.125 ms timestep, flat ground, 1 Hz',fontsize=15)
    cross_note=' Cross: previous amplitudes fail nominal gates.' if any(r['old_nominal_feasible'] is False for r in cells) else ''
    fig.text(.5,.008,'NQ: no selected qualified candidate, not zero performance.'+cross_note+'\nPer-wave PD is preserved and differs across waves. Half-step validation uses 0.0625 ms. No contact loss is used for selection.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95));fig.savefig(folder/'optimization_overview.png',dpi=170);fig.savefig(folder/'optimization_overview.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6));indexed={r['id']:r for r in rows};imported=c.read(c.DATA/'imported_exact_conditions.json')['conditions']
    stages=['warm','broad','feasibility_00','feasibility_01']+[f'adaptive_{i:02d}' for i in range(3)]+[name for i in range(6) for name in [f'local_{i:02d}',f'local_diagonal_{i:02d}']]
    stages += [p.stem for p in sorted((c.DATA/'banks').glob('polish_nominal_*.json')) if p.stem.count('_')==2 and p.stem[-2:].isdigit()]
    for j,wave in enumerate(c.WAVES):
        seen={a['id'] for a in imported if a['id'] in indexed and indexed[a['id']]['wave']==wave};best=max([indexed[k]['D'] for k in seen if indexed[k]['feasible']],default=0.);xx=[len(seen)];yy=[best or np.nan]
        for stage in stages:
            path=c.DATA/'banks'/f'{stage}.json'
            if not path.exists():continue
            for t in c.read(path):
                key=c.identity(t)
                if t['wave']!=wave or key in seen:continue
                seen.add(key);r=indexed[key]
                if r['feasible']:best=max(best,r['D'])
            xx.append(len(seen));yy.append(best or np.nan)
        ax.plot(xx,yy,'o-',ms=3,color=plt.cm.turbo(j/13),label=f'w{wave:g}')
    ax.set(xlabel='Cumulative distinct fine-step nominal conditions',ylabel='Best nominal feasible D (mm)',title='Search progress — a nominal best is not automatically qualified');ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1))
    fig.tight_layout();fig.savefig(folder/'search_progress.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    axes[0].plot(w,[r['D'] if r['qualified'] else np.nan for r in cells],'o-',c='#176f9d',label='Nominal dt = 0.125 ms')
    axes[0].plot(w,[r['half_step_D_mm'] if r['qualified'] else np.nan for r in cells],'s--',c='#b35806',label='Half dt = 0.0625 ms')
    axes[0].set(title='Selected amplitudes, two timesteps',ylabel='Forward D over 4.499 s (mm)');axes[0].legend(fontsize=9)
    axes[1].bar(w,[100*r['half_step_relative_difference'] if r['qualified'] else np.nan for r in cells],width=.34,color='#658fa7')
    axes[1].axhline(10,color='#9b4444',ls='--',label='Declared qualification limit: 10%')
    axes[1].set(title='Residual timestep sensitivity',ylabel='Absolute displacement difference (%)',ylim=(0,11.5));axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xlabel('Wave number');ax.set_xticks(w);ax.tick_params(axis='x',labelsize=8,rotation=35);ax.grid(alpha=.15)
    fig.text(.5,.012,'Same amplitudes and per-wave PD in both evaluations. These are deterministic timestep checks, not statistical confidence intervals.\nWave 8 remains unqualified; the gap does not represent zero displacement.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,1));fig.savefig(folder/'validation_overview.png',dpi=170);plt.close(fig)

def write_report(cells,local,rows,polish=None):
    accepted=[r for r in cells if r['qualified']]
    matched=[r for r in accepted if r['gain_percent'] is not None]
    unresolved_labels=', '.join(f"{r['wave']:g}" for r in cells if not r['qualified']) or '无'
    opening=f"14 个波数中，{len(accepted)} 个得到通过名义、长时和半步长检查的选定配置；未解决波数：{unresolved_labels}。"
    if matched:
        opening+=f" 有旧选定参数可同条件比较的 {len(matched)} 个波数，位移改善范围为 {min(r['gain_percent'] for r in matched):.2f}%–{max(r['gain_percent'] for r in matched):.2f}%。"
    lines=['# 各波数振幅优化：统一细时间步的结果','',
        opening,'','![优化结果概览](outputs/figures/optimization_overview.png)','',
        '本轮只调整 body yaw、leg yaw、leg pitch 三个振幅。所有波数的正式时间步统一为 0.125 ms；模型、1 Hz、相位关系、初始化与既定的每波数 PD 保持不变。最终选定配置还通过了 12 s 与 0.0625 ms 检查。','',
        '## 之前的问题及本轮修正','',
        '- 早期网格漏掉半波数、零附近振幅和小 leg-yaw 锚点；本轮包括所有 14 个波数和整个声明参数范围。',
        '- 曾加入过多自然度和参考相对限制；本轮沿用位移优先的工程门槛，滑移、能耗与外观不参与最终评分。',
        '- 上次搜索中途停止，临时结果过早成为后续 loss 对比的输入；本轮记录各阶段真实完成情况与局部搜索停止原因。',
        '- 旧完整记录覆盖不均，旧早停记录不能冒充完整分数。本轮核对了 5,282 个完整历史条件，旧数据只用于提出参数。',
        '- 本轮发现低波数的粗时间步偏差：wave 0.5 同一参数在 0.25 / 0.125 / 0.0625 ms 下为 107.46 / 77.56 / 74.88 mm；wave 1 为 154.86 / 125.92 / 120.99 mm。原粗步长搜索因此成为准备工作，其余计划由统一细步长方案取代。',
        '- 早期提案规则对不确定区域分配过多计算；细步长搜索分开使用预测性能、预期改善、不确定性和覆盖提案。所有最终成绩均来自实际仿真。','',
        '## 全波数结果','',
        'D 是 21 个身体节点的几何中心沿固定前向轴，在 t=1.5–5.999 s 的净位移；不是路径长度或质量加权 COM。旧参数也已在同一新时间步重新评估。旧参数未过名义门槛时，不报告相对它的有效性能增益。','',
        '| Wave | 旧参数 D(mm) | 新 D(mm) | 增益 | 新 body / leg yaw / leg pitch (°) | 半步长差异 | 更快未验证点（最大名义优势mm） |',
        '|---:|---:|---:|---:|---|---:|---:|']
    for r in cells:
        old=f"{r['old_D']:.2f}" if r['old_D'] is not None else '—'
        if r['old_nominal_feasible'] is False:old+='（未过门槛）'
        if not r['qualified']:lines.append(f"| {r['wave']:g} | {old} | 未解决 | — | — | — | {r['higher_untested']} |");continue
        gain=f"{r['gain_percent']:+.2f}%" if r['gain_percent'] is not None else '—'
        amp=' / '.join(f'{x:.4f}' for x in r['amplitudes'])
        pending=str(r['higher_untested'])+(f"（{r['higher_untested_max_gain_mm']:.3f}）" if r['higher_untested'] else '')
        lines.append(f"| {r['wave']:g} | {old} | {r['D']:.2f} | {gain} | {amp} | {100*r['half_step_relative_difference']:.2f}% | {pending} |")
    qualified=[r for r in cells if r['qualified']]
    if qualified:
        sensitive=max(qualified,key=lambda r:r['half_step_relative_difference'])
        lines += ['',f"入选配置中，步长差异最大的是 wave {sensitive['wave']:g}：名义位移 {sensitive['D']:.2f} mm，半步长位移 {sensitive['half_step_D_mm']:.2f} mm，相对差 {100*sensitive['half_step_relative_difference']:.2f}%。该指标描述数值步长敏感性；比较这一波数的性能时，应同时保留这项信息。",'']
        lines += ['![两种时间步下的选定配置](outputs/figures/validation_overview.png)','']
    unresolved=[r for r in cells if not r['qualified']]
    if unresolved:
        lines += ['','未解决波数的完整失败记录也保留在报告与视频中。下列“最接近门槛”按补充搜索的最大约束比例指标选择，仅用于诊断，不参与性能排名。','']
        for r in unresolved:
            for a in r['unresolved_diagnostics']:
                kind='最接近门槛的完整记录' if a['kind']=='least_violation' else '最快的完整失败记录'
                lines.append(f"- Wave {r['wave']:g}，{kind}：D={a['D']:.2f} mm，身体触地比例 {100*a['body_contact']:.2f}%（上限 15%），身体接触力峰值 {a['chassis_force']:.3f} BW（上限 2 BW）；失败项为 {', '.join(a['failed'])}。")
        lines += ['','身体触地比例沿用原指标：评估窗口内，21 个身体节点的总接触力指标超过 1e-7 N 的时步占比。尚未找到合格参数不证明参数空间中不存在解。','']
    lines += ['','## 搜索范围与停止原因','',f'当前共保留 {len(rows)} 个不同的细步长名义条件。精确匹配的本次会话旧运行按哈希复用；粗步长分数不进入这一排名。全域设计为每波数同一 16 个 Sobol 点、8 个边界锚点和振幅迁移，随后三轮每波数 8 个自适应提案。局部搜索从两个不同起点出发，最多六轮轴向细化，并在首次无实质改善时进行一次耦合对角挑战。','',
        '| 起点 | 最后归一化步长 | 停止原因 | 最近增益(mm) |','|---|---:|---|---:|']
    stop_labels={'no_material_improvement_at_declared_resolution':'达到声明分辨率，未见实质改善',
        'poll_budget_reached_not_converged':'达到轮数上限，未确认收敛',
        'no_fine_feasible_start':'没有通过细步长名义门槛的起点',
        'no_start_after_qualification_failure':'复核失败后没有可用起点'}
    for st in local:
        reason=stop_labels.get(st['stop'],st['stop'])
        gain=f"{st['history'][-1]['gain_mm']:.4f}" if st['history'] else '—'
        lines.append(f"| {st['key']} | {st['step']:.4f} | {reason} | {gain} |")
    lines += ['','归一化步长 0.04 对应 body / leg yaw / leg pitch 的 1.44° / 3.60° / 2.40°；最细声明步长 0.005 对应 0.18° / 0.45° / 0.30°。一轮改善不超过 0.2 mm 时缩小步长。这个停止标准描述已检查的局部方向，不表示邻域内每一点或整个参数空间都已排除。','',
        '对没有名义合格配置的波数，另做至多两轮、每轮八个针对现有约束的振幅提案。完整样本不足时直接挑战目前超限最少的参数邻域；样本足够后预测约束超限程度。负位移也纳入失败信息，不放宽任何物理门槛。补充搜索与主搜索有时间重叠，进展图按标示的阶段顺序累计。','',
        '达到预算上限不代表收敛；达到声明分辨率也不证明连续全局最优。更快未验证点的数量及最大名义优势明确保留，不能把当前选定值称为整个参数空间的最优值。','']
    if polish:
        plan=c.read(c.DATA/'polish_plan.json')
        lines += ['## 已验证配置的收尾细化','',f"初次排序复核之后，对 {len(polish['states'])} 个尚未完成选定点细步长检查的波数做了额外检查；其余 {len(polish['already_covered_waves'])} 个合格波数已完成相应检查。起点全部通过完整复核。初始归一化步长为 0.01 或 0.005，检查六个坐标方向；首次可用的延伸方向还要求此前两个端点均合格、且细时间步位移也有所改善。物理模型、评分及所有门槛保持一致。",'',
            '任何名义位移改善超过 0.2 mm 的候选，都会补齐长时和半步长检查；只有完整合格的改善才能更新起点。无实质合格改善时缩小步长，在 0.005 下仍无实质改善则停止。最多六轮，具体停止原因如下。该检查只覆盖所测试的方向，不证明连续全局最优。','',
            '| Wave | 最后步长 | 检查轮数 | 最后合格改善(mm) | 停止原因 |','|---:|---:|---:|---:|---|']
        for s in polish['states']:
            reason='声明分辨率下未见实质合格改善' if s['stop']=='no_material_qualified_improvement_at_declared_resolution' else '达到补充轮数上限，未确认收敛'
            lines.append(f"| {s['wave']:g} | {s['step']:.4f} | {len(s['history'])} | {s['history'][-1]['gain_mm']:.4f} | {reason} |")
        lines += ['','初次排序结果保留于 `selection_before_polish.json`；收尾方案、逐轮结果及完成记录分别保留于 `POLISH_PROTOCOL.md`、`polish_after_*.json` 和 `polish_complete.json`。','']
    lines += ['## 运动与物理诊断','',
        '足尖角速度峰值门槛为 1000 rad/s，力矩峰值门槛为 200 μNm；表中同时保留这些数值，便于判断通过配置距离限制还有多少余量。','',
        '| Wave | 身体触地比例 | 峰值地面力(BW) | Root roll峰值(°) | 足尖角速度峰值(rad/s) | 力矩峰值(μNm) | 长时/短时平均速度 | 实际 body / leg yaw / leg pitch 半幅(°) |','|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in cells:
        if not r['qualified']:continue
        m=r['metrics'];amp=' / '.join(f'{v:.2f}' for v in r['motion']['actual_joint_half_P95_P05_deg'])
        lines.append(f"| {r['wave']:g} | {m['chassis_contact_fraction']:.4f} | {m['ground_force_peak_bw']:.3f} | {m['root_roll_abs_peak_deg']:.2f} | {m['tip_speed_peak']:.2f} | {1e6*m['torque_peak_nm']:.2f} | {r['long_short_speed_ratio']:.3f} | {amp} |")
    lines += ['','实际半幅为每关节时间 P95–P05 的一半，再取相关关节的中位数；与正弦命令振幅的定义不同。视频重放已保存的真实 qpos，保持原时间轴，不做物理重算或姿态平滑。','',
        '## 复用与限制','',
        '- `outputs/data/selected_gaits.json`：可冻结复用的全精度振幅、PD、时间步与三项运行 ID。',
        '- `validated_selection.json`、`final_summary.json`、各阶段 banks/completed 与原始 trials：排名、失败、覆盖和来源证据。',
        '- `outputs/figures/`：性能、命令/实际振幅、覆盖、搜索进展及时间步诊断；`outputs/videos/`：动作对照与未解决案例。',
        '- 不同波数保留不同的既定 PD，因此结果比较的是各波数既定控制器下的优化，不能把全部差异归因于波数单独变化。',
        '- 数值与工程门槛通过不等于真实动物或实体机器人的物理验证；参数范围内的有限搜索也不等于全局最优证明。',
        '- 所有工作限定在本次 Model 实验目录；历史模型、Robot、IsaacLab/Clean 没有改动。',
        '- 本轮未重新运行 random contact。下一步应固定这里通过验证的配置，沿用相同时间步，并独立检查此前接触恢复/身体触地的问题，再判断是否有可靠交叉点。','']
    videos=sorted((c.OUT/'videos').glob('wave*_before_after.mp4'),key=lambda p:float(p.stem.split('_')[0][4:]))
    if videos:
        lines += ['## 动作对照','', '各视频包含前、侧、斜视；有旧参数基线的波数采用上下对照。', '',
            ' · '.join(f"[Wave {p.stem.split('_')[0][4:]}](outputs/videos/{p.name})" for p in videos),'']
    for p in sorted((c.OUT/'videos').glob('unresolved_wave*.mp4')):
        lines += [f'[未解决案例：{p.stem.replace("unresolved_wave","Wave ")}](outputs/videos/{p.name})（失败配置诊断，不是选定步态）。','']
    restarts=[c.read(p) for p in sorted((c.OUT/'audit/restarts').glob('restart_*.json'))]
    if restarts:
        lines += ['## 运行恢复记录','',f'本轮有 {len(restarts)} 次运行恢复。恢复前核对固定输入和所有已保存记录；按完全相同的条件 ID 复用完成结果，继续未完成阶段。恢复没有更改物理模型、评分或搜索规则。具体原因及运行维护修复逐次记录如下。','']
        for restart in restarts:
            a=restart['audit'];s=restart['last_status']
            lines.append(f"- {restart['time']}：核对 {a['assessment_count']} 条记录、{a['frozen_input_count']} 个固定输入；上次进度为 {s['stage']} 的 {s['done']}/{s['total']}。{restart.get('reason_zh','运行连接失效后原进程已不存在，具体终止原因未确定。')}")
        lines += ['','恢复记录在 `outputs/audit/restarts/`，持续日志在 `outputs/logs/`。最终阶段完成情况仍以各阶段完成记录和最终审计为准。','']
    (c.ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':main()
