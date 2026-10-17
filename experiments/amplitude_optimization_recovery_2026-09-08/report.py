"""All-wave evidence, current-runtime comparisons and search limitations."""
import collections,json
from pathlib import Path
import core as c
np=c.np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def vector(r):return [r['amplitudes'][k] for k in c.KEYS]
def baseline_rows():
    old=c.read(c.ROOT.parent/'mujoco_displacement_first_amplitude_2026-09-03/outputs/data/interim_validated_selection.json')
    fresh={r['id']:r for r in c.nominal()};out={}
    for w in c.WAVES:
        sel=old['waves'][f'{w:g}']['selected']
        if sel and sel['qualified']:
            raw=c.read(sel['nominal']['record']);key=c.identity(c.task(w,[raw['task']['amplitudes'][k] for k in c.KEYS],'comparison'))
            out[w]=fresh[key]
    return out

def motion_measure(row):
    folder=Path(row['record']).parent;path=folder/'motion.npz'
    if not path.exists():return None
    z=np.load(path);good=z['t']>=1.5
    b=z['bq'][good,1:,0];leg=z['lq'][good].reshape(-1,19,2,2)
    arrays=[b,leg[:,:,:,0].reshape(len(b),-1),leg[:,:,:,1].reshape(len(b),-1)]
    actual=[float(np.median(np.degrees((np.quantile(a,.95,axis=0)-np.quantile(a,.05,axis=0))/2))) for a in arrays]
    center=z['center'][good];D=float(c.e.fx.opt.base.FORWARD_SIGN*(center[-1,0]-center[0,0])*1000);assert D==row['D']
    return dict(actual_joint_half_P95_P05_deg=actual,D_recomputed_mm=D,lateral_displacement_mm=float((center[-1,1]-center[0,1])*1000),motion=str(path))

def main():
    selection=c.read(c.DATA/'validated_selection.json');rows=c.nominal();baseline=baseline_rows();cells=[]
    for w in c.WAVES:
        v=selection['waves'][f'{w:g}'];selected=v['selected'];group=[r for r in rows if r['wave']==w]
        row=selected['nominal'] if selected else None;old=baseline.get(w)
        cells.append(dict(wave=w,qualified=bool(row),D=row['D'] if row else None,amplitudes=vector(row) if row else None,
            old_D=old['D'] if old else None,old_amplitudes=vector(old) if old else None,
            gain_percent=100*(row['D']/old['D']-1) if row and old else None,
            nominal_full=sum(r['full'] for r in group),nominal_prefix_rejections=sum(r.get('proven_infeasible',False) for r in group),
            execution_errors=sum('error' in r for r in group),fresh_unique_conditions=len(group),
            half_step_disagreement=selected['half_step_relative_difference'] if selected else None,
            long_speed_mm_s=selected['long']['D']/10.499 if selected else None,
            short_speed_mm_s=row['D']/4.499 if row else None,
            metrics=row['metrics'] if row else None,record=row['record'] if row else None,
            motion=motion_measure(row) if row else None,validation_candidates_tested=v['candidates_tested'],higher_untested=v['higher_untested']))
    exported=[]
    for w,cell in selection['waves'].items():
        if not cell['qualified']:continue
        sel=cell['selected'];r=sel['nominal']
        exported.append(dict(wave=float(w),amplitudes_deg=r['amplitudes'],pd_config=r['task']['config'],
            nominal_id=r['id'],nominal_D_mm=r['D'],long_id=sel['long']['id'],half_step_id=sel['half']['id'],
            half_step_relative_difference=sel['half_step_relative_difference']))
    c.write(c.DATA/'selected_gaits.json',dict(model_path=c.history()['plant'],model_sha256=c.history()['plant_sha256'],
        frequency_Hz=1.,contact_loss=False,amplitude_only_search=True,per_wave_pd_preserved=True,
        gaits=exported,unresolved_waves=[r['wave'] for r in cells if not r['qualified']],
        source_selection=str(c.DATA/'validated_selection.json'),source_selection_sha256=c.sha(c.DATA/'validated_selection.json')))
    local=c.read(c.DATA/'local_complete.json');history=c.history()
    c.write(c.DATA/'final_summary.json',dict(waves=cells,local_starts=local,historical_full=len(history['full_rows']),
        old_incomplete_records=len(history['incomplete']),fresh_nominal_unique=len(rows),
        failed_gate_counts=dict(collections.Counter(k for r in rows for k in r['failed'])),global_optimum_proved=False))
    figdir=c.OUT/'figures';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,9));good=[r for r in cells if r['qualified']]
    xx=[r['wave'] for r in good]
    axes[0,0].plot(list(baseline),[r['D'] for r in baseline.values()],'o--',label='Previous provisional amplitudes, re-evaluated now',color='#777777')
    axes[0,0].plot(xx,[r['D'] for r in good],'o-',label='New long/half-step qualified selection',color='#176f9d',lw=2)
    axes[0,0].set(title='Forward displacement',ylabel='D over 4.5 s (mm)');axes[0,0].legend(fontsize=8)
    for j,(name,color) in enumerate(zip(['Body yaw','Leg yaw','Leg pitch'],['#2166ac','#b35806','#38966d'])):
        axes[0,1].plot(xx,[r['amplitudes'][j] for r in good],'o-',label=name,color=color)
        realized=[r for r in good if r['motion']]
        axes[1,0].plot([r['wave'] for r in realized],[r['motion']['actual_joint_half_P95_P05_deg'][j] for r in realized],'o-',label=name,color=color)
    axes[0,1].set(title='Selected command amplitudes',ylabel='Degrees');axes[0,1].legend(fontsize=8)
    axes[1,0].set(title='Realized joint motion',ylabel='Median joint half (P95-P05), degrees');axes[1,0].legend(fontsize=8)
    axes[1,1].bar([r['wave'] for r in cells],[r['fresh_unique_conditions'] for r in cells],width=.34,color='#658fa7')
    axes[1,1].set(title='Fresh nominal search coverage',ylabel='Unique evaluated amplitude conditions')
    for ax in axes.flat:ax.set_xlabel('Wave number');ax.set_xlim(.25,9.25);ax.grid(alpha=.18)
    fig.suptitle('Per-wave amplitude optimization — same plant, 1 Hz, no contact loss',fontsize=15)
    fig.text(.5,.01,'Only qualified waves receive a selected curve point. Prior selections use the same current runtime.\nPer-wave PD is frozen; different waves retain their established different PD. No monotonicity or crossover reward.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95));fig.savefig(figdir/'optimization_overview.png',dpi=170);fig.savefig(figdir/'optimization_overview.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6))
    stages=['warm','broad']+[f'feasibility_{i:02d}' for i in range(3)]+[f'adaptive_{i:02d}' for i in range(4)]+[f'local_{i:02d}' for i in range(8)]
    indexed={r['id']:r for r in rows}
    for i,w in enumerate(c.WAVES):
        seen=set();x=[];y=[];best=0.
        for stage in stages:
            path=c.DATA/'banks'/f'{stage}.json'
            if not path.exists():continue
            for t in c.read(path):
                key=c.identity(t)
                if t['wave']!=w or key in seen:continue
                seen.add(key);r=indexed.get(key)
                if r and r['feasible']:best=max(best,r['D'])
            x.append(len(seen));y.append(best if best else np.nan)
        ax.plot(x,y,'o-',ms=3,color=plt.cm.turbo(i/13),label=f'w{w:g}')
    ax.set(xlabel='Cumulative distinct nominal conditions',ylabel='Best nominal feasible D (mm)',title='Search progress: nominal best is not automatically validated');ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'search_progress.png',dpi=170);plt.close(fig)
    lines=['# 各波数振幅优化：复盘与当前结果','',
        '仅优化 body yaw、leg yaw、leg pitch 三个命令振幅。模型、1 Hz 频率、相位关系及既定的每波数 PD 均未改变；random contact loss 不参与优化。所有新评估均在同一当前运行环境中执行，旧数据仅用于提出候选。','',
        '## 之前哪里没有做好','',
        '1. 最早有限网格漏掉半波数、零附近振幅和历史小 leg-yaw 锚点，不能支持充分搜索的结论。',
        '2. 后续加入过多自然度和参考相对门槛，令大量更快步态在位移排名之前被排除。本轮保持已经授权的 9 月 3 日位移优先门槛，不增加外观或曲线形状约束。',
        '3. 最新搜索中途停止，第二套全域样本和多起点细化没有完成，临时结果被后续工作过早当作优化终点。',
        '4. 旧完整样本在各波数间覆盖不均，高波数样本少；旧早停前缀不能继续作为当前门槛下的不合格证明。',
        '5. 六秒高分不等于能通过长时/半步长验证；输入命令振幅不等于实际关节摆幅。',
        '',f"本轮从原始记录核对了 {len(history['full_rows'])} 个完整不同条件，并保留 {len(history['incomplete'])} 条旧早停记录的来源。`history.json` 提供逐条证据、历史覆盖和旧门槛失败原因。",'',
        '## 当前选定结果','',
        '位移为 21 个身体节点几何中心沿固定前向轴的净位移，统计 t=1.5–5.999 s；不是路径长度或质量加权 COM。旧基线是 9 月 3 日临时合格配置在当前环境重新测得的值。未曾选出合格配置的波数不虚构旧基线。','',
        '| Wave | 旧 D(mm) | 新 D(mm) | 增益 | 新 body / leg yaw / leg pitch (°) | 半步长差异 | 状态 |','|---:|---:|---:|---:|---|---:|---|']
    for r in cells:
        old=f"{r['old_D']:.2f}" if r['old_D'] is not None else '—'
        if r['qualified']:
            gain=f"{r['gain_percent']:+.2f}%" if r['gain_percent'] is not None else '新增合格候选'
            lines.append(f"| {r['wave']:g} | {old} | {r['D']:.2f} | {gain} | {' / '.join(f'{x:.3f}' for x in r['amplitudes'])} | {100*r['half_step_disagreement']:.2f}% | 已通过三项验证；更快未验证候选 {r['higher_untested']} |")
        else:lines.append(f"| {r['wave']:g} | {old} | — | — | — | — | 未解决 |")
    lines += ['','## 搜索是否充分','',
        f"新增 nominal 不同幅值条件共 {len(rows)} 个。每波数包含相同的 32 点全域设计、8 个边界/零附近锚点、跨波数迁移、四轮每波数 8 个自适应提案，以及两个多样起点的有限三维细化。每个条件保留完整结果或必要门槛已经被违反的前缀证据。",'',
        '搜索代理模型只提出下一个需要实际仿真的参数，不直接决定最终成绩。采用同一全域设计有利于波数间覆盖比较；各波数任务交错执行，避免再次只算完低波。早期长时与半步长验证的已知失败降低对应区域的提案优先级。','',
        '对始终没有通过六秒门槛的波数，另外执行至多三轮、每轮八个振幅提案，专门寻找满足现有约束的区域。该提案模型最小化约束超限的预测值；最终仍以完整验证后的实际位移排名，没有放宽通过标准。搜索进展图按标示的阶段顺序累计，补充探索与主搜索在实际执行中有时间重叠。','',
        '| 起点 | 最后步长（归一化） | 停止原因 | 最后一次增益(mm) |','|---|---:|---|---:|']
    for st in local:lines.append(f"| {st['key']} | {st['step']:.4f} | {st['stop']} | {st['history'][-1]['gain_mm']:.4f} |" if st['history'] else f"| {st['key']} | {st['step']:.4f} | {st['stop']} | — |")
    lines += ['','达到轮次上限仍属于未证明收敛；在声明步长下没有实质增益也不是连续全局最优证明。报告不把曲线单调、振幅差异或未来 loss 交叉作为优化成功标准。','',
        '## 稳定性与实际动作','',
        '| Wave | 身体触地比例 | 峰值地面力(BW) | Root roll峰值(°) | 长时平均速度 / 短时平均速度 | 实际 body / leg yaw / leg pitch 半幅(°) |','|---:|---:|---:|---:|---:|---|']
    for r in cells:
        if not r['qualified']:continue
        m=r['metrics'];motion=' / '.join(f'{v:.2f}' for v in r['motion']['actual_joint_half_P95_P05_deg']) if r['motion'] else '尚无已核对运动记录'
        lines.append(f"| {r['wave']:g} | {m['chassis_contact_fraction']:.4f} | {m['ground_force_peak_bw']:.3f} | {m['root_roll_abs_peak_deg']:.2f} | {r['long_speed_mm_s']/r['short_speed_mm_s']:.3f} | {motion} |")
    lines += ['','实际半幅采用每关节时间 P95–P05 的一半，再取相关关节的中位数；它与正弦输入振幅的定义不同。完整姿态、力、滑移、能耗与轨迹保留在原始记录。','',
        '## 文件与边界','',
        '- `outputs/data/manifest.json`：原始输入与门槛；`runtime.json`：解释器、NumPy、MuJoCo 版本与关键二进制指纹。',
        '- `outputs/data/banks/`：执行前固定的任务表及各版本代码指纹；`completed/`：实际阶段完成记录。',
        '- `outputs/data/validated_selection.json`：全精度参数、名义/长时/半步长结果及失败原因；`final_summary.json`：汇总。',
        '- `outputs/trials/`：原始数据；`outputs/figures/`：全波数曲线及搜索进展；`outputs/videos/`：实际记录的运动重放。',
        '- 所有修改只在本次新实验目录。模型、历史实验、Robot、IsaacLab/Clean 均未改动。',
        '- 本轮没有重新运行 random contact；后续应冻结这批经过验证的配置，再独立研究 loss 响应。','']
    lines += ['不同波数保留了各自此前确定的 PD，因此本报告比较的是每波数既定控制器下的振幅优化结果。跨波数的差异不能全部归因于 wave number 单独改变；这项限制也应跟随后续 loss 对照。','',
        '`outputs/data/selected_gaits.json` 提供全精度的已验证振幅、既定 PD 和三项评估 ID，可作为下一轮独立 loss 扫描的冻结输入。','']
    (c.ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print({r['wave']:dict(D=r['D'],gain_percent=r['gain_percent'],higher_untested=r['higher_untested']) for r in cells})

if __name__=='__main__':main()
