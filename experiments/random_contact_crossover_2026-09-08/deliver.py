"""Final evidence tables and probability-axis figures; includes all failed-gate runs."""
import collections,json
from pathlib import Path
import study as s
import analyze as a
np=s.np
plt=a.plt

def table(cells):
    waves=sorted({x['wave'] for x in cells});levels=sorted({x['nu'] for x in cells})
    lines=['| Wave | '+' | '.join(f'nu={v:g}' for v in levels)+' |','|---|'+'---:|'*len(levels)]
    for w in waves:
        values=[next((c for c in cells if c['wave']==w and c['nu']==v),None) for v in levels]
        lines.append(f'| {w:g} | '+' | '.join(f"{c['mean_D']:.2f}" if c and c['mean_D'] is not None else '—' for c in values)+' |')
    return '\n'.join(lines)

def main():
    for stage,count in [('main_accepted9',180),('independent_confirmation',48),('resolution_duration_check',8),('persistent_diagnostic',27)]:
        receipt=s.read(s.DATA/f'{stage}_complete.json');assert receipt['count']==count and receipt['completed']==count
    summary=s.read(s.DATA/'main_accepted9_summary.json');cells=summary['cells']
    plan=s.read(s.DATA/'confirmation_plan.json');conf=s.read(s.DATA/'independent_confirmation_summary.json')
    waves=plan['waves'];rows=a.records();primary=[r for r in rows if r['task']['wave'] in s.read(s.DATA/'accepted_screen_amendment.json')['accepted_waves']]
    checks=[r for r in primary if r['task']['duration']!=6 or r['task']['dt_scale']!=1]
    persistent=[s.read(p) for p in (s.OUT/'persistent_loss_diagnostic').glob('*/result.json')]
    baseline=[r for r in primary if r['task']['duration']==6 and r['task']['dt_scale']==1 and r['task']['nu']==0 and r['task']['wave'] in [2,3,7]]
    pcells=a.aggregate(persistent+baseline)
    for c in pcells:c['q']=c['nu']
    paudit=a.audit(persistent);all_audit=a.audit(primary)
    restored=[r for r in primary if r['completed'] and r['task']['duration']==6 and r['task']['dt_scale']==1 and r['task']['nu'] not in (0,1)]
    restoration=dict(events=sum(r['channel']['restoration_count'] for r in restored),
        floor_overlap_events=sum(r['channel']['restorations_overlapping_floor'] for r in restored),
        force_range_BW=[min(r['metrics']['ground_force_peak_bw'] for r in restored),max(r['metrics']['ground_force_peak_bw'] for r in restored)])
    failures=collections.Counter(k for r in restored for k in r.get('failed_viability',[]))
    extra=dict(primary_audit=all_audit,persistent_audit=paudit,persistent_cells=pcells,restoration=restoration,
        failed_gate_counts=dict(failures),validation_checks=[dict(task=r['task'],completed=r['completed'],D=r['D'],upright=r.get('upright_survived'),viable=r.get('viable'),error=r.get('error')) for r in checks])
    s.write(s.DATA/'final_audit.json',extra)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    a.FIG.mkdir(exist_ok=True,parents=True)
    colors={w:plt.cm.turbo(i/8) for i,w in enumerate(sorted({c['wave'] for c in cells}))}
    fig,ax=plt.subplots(figsize=(11,6.5))
    for w in colors:
        cc=[c for c in cells if c['wave']==w and c['nu']<=.75];xx=np.array([c['nu']/.75 for c in cc])
        ax.plot(xx,[c['mean_D'] for c in cc],'o-',color=colors[w],label=f'wave {w:g}',lw=2)
        ax.fill_between(xx,[c['ci_D'][0] for c in cc],[c['ci_D'][1] for c in cc],color=colors[w],alpha=.08)
    ax.set(xlabel='Loss-event probability per leg pair per 0.5 s window (q)',ylabel='Forward displacement over 4.5 s (mm)',
        title='Different waves lose performance at different rates',xlim=(0,1),ylim=(0,None))
    ax.set_xticks([0,1/3,2/3,1],['0','1/3','2/3','1']);ax.grid(alpha=.2);ax.legend(ncol=3,frameon=False)
    fig.text(.5,.02,'6 paired random seeds; shaded bands: descriptive 95% bootstrap intervals.\nBurst duration is uniform 0.25–0.5 s; expected erased-time fraction nu = 0.75 q.\nFrozen per-wave amplitudes and PD. All burst-loss trials fail at least one engineering gate.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,1));fig.savefig(a.FIG/'probability_curves.png',dpi=180);fig.savefig(a.FIG/'probability_curves.svg');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9));palette=['#2467a0','#c34825']
    for j,w in enumerate(waves):
        cc=[c for c in conf['cells'] if c['wave']==w];x=np.array([c['nu']/.75 for c in cc])
        axes[0,0].plot(x,[c['mean_D'] for c in cc],'o-',color=palette[j],label=f'wave {w:g}',lw=2)
        axes[0,0].fill_between(x,[c['ci_D'][0] for c in cc],[c['ci_D'][1] for c in cc],color=palette[j],alpha=.15)
        axes[0,1].plot(x,[c['mean_retention'] for c in cc],'o-',color=palette[j],lw=2)
        axes[1,1].plot(x,[c['upright_survival'] for c in cc],'o-',color=palette[j],lw=2,label=f'wave {w:g}')
    cr=[r for r in rows if r['task']['wave'] in waves and r['task']['duration']==6 and r['task']['dt_scale']==1 and (r['task']['seed']>=100 or r['task']['nu']==0)]
    levels=sorted({r['task']['nu'] for r in cr});dc=[]
    for nu in levels:
        rr=[r for r in cr if r['task']['nu']==nu];seeds=sorted({r['task']['seed'] for r in rr})
        ds=[]
        for se in seeds:
            pp=[next((r for r in rr if r['task']['wave']==w and r['task']['seed']==se and r['completed']),None) for w in waves]
            if all(pp):ds.append(pp[0]['D']-pp[1]['D'])
        ci=a.interval(ds);dc.append(dict(nu=nu,q=nu/.75,n=len(ds),mean=float(np.mean(ds)),ci=ci,high_win_fraction=float(np.mean(np.array(ds)<0)),differences=ds))
        if nu:axes[1,0].scatter([nu/.75]*len(ds),ds,color='#688995',s=18,alpha=.55)
    xx=[v['q'] for v in dc];yy=[v['mean'] for v in dc]
    axes[1,0].plot(xx,yy,'o-',color='#173b4d');axes[1,0].fill_between(xx,[v['ci'][0] for v in dc],[v['ci'][1] for v in dc],color='#688995',alpha=.18)
    axes[1,0].axhline(0,color='black',lw=1)
    for ax,title,ylabel in zip(axes.flat,['Absolute displacement','Retention of own baseline',f'Paired difference: wave {waves[0]:g} minus {waves[1]:g}','Upright survival (not all viability gates)'],['D (mm)','D / D0','Difference (mm)','Fraction of 8 seeds']):
        ax.set(title=title,ylabel=ylabel,xlabel='Loss-event probability q',xlim=(0,1));ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,loc='lower left');axes[1,1].set_ylim(-.03,1.05)
    zoom=axes[0,0].inset_axes([.55,.46,.42,.43])
    for j,w in enumerate(waves):
        cc=[c for c in conf['cells'] if c['wave']==w and c['nu']>=.625]
        zoom.plot([c['nu']/.75 for c in cc],[c['mean_D'] for c in cc],'o-',color=palette[j],lw=1.5,ms=4)
    zoom.set(xlim=(.81,1.01),ylim=(20,70),title='High-loss detail');zoom.title.set_fontsize(8);zoom.tick_params(labelsize=8);zoom.grid(alpha=.2)
    fig.suptitle('Independent confirmation: new seeds 100–107',fontsize=16)
    fig.text(.5,.012,'Pair selected from the full 9-wave screen. Every finite full-horizon outcome is included.\nShading is a seed bootstrap interval, not a precise crossover threshold; collision-restoration artifacts remain.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.95));fig.savefig(a.FIG/'independent_pair.png',dpi=180);fig.savefig(a.FIG/'independent_pair.svg');plt.close(fig)
    s.write(s.DATA/'independent_pair_differences.json',dc)
    checklines=[]
    for r in checks:
        t=r['task'];ref=next((v for v in checks if v['task']['wave']==t['wave'] and v['task']['duration']==t['duration'] and v['task']['dt_scale']==t['dt_scale'] and v['task']['nu']==0),None)
        retention=r['D']/ref['D'] if ref and ref['D'] and r['D'] is not None else None
        checklines.append(f"| {t['wave']:g} | {t['duration']:g} | {t['dt_scale']:g} | {t['nu']:g} | {t['seed']} | {r['D']:.3f} | {retention:.3f} | {r.get('upright_survived')} | {r.get('viable')} |" if retention is not None else f"| {t} | incomplete: {r.get('error')} |")
    lines=['# Random contact loss：波数与性能交叉实验','',
        '**当前结论：已有短时仿真中的交叉证据，但还不是干净、通用的高波鲁棒性结论。** 8 个新种子中，wave 1.5 / wave 7 的平均位移从无 loss 时 173.58 / 93.42 mm，变为 nu=.625 时 63.50 / 44.94 mm，再变为 nu=.75 时 26.66 / 31.05 mm。末点高波在 7/8 个种子获胜，配对“低减高”95% bootstrap 区间为 [-6.68, -2.20] mm。观察到的交叉区间是 nu=.625–.75（q=.833–1）；均值线性插值约 nu=.726（q=.968），只是估计，未在该精确点采样。','',
        '这个交叉的反超幅度约 4.39 mm；两种步态此时均约 71%–72% 的时间有身体触地，并存在恢复碰撞冲击，全部随机 burst 条件都违反至少一项原工程门限。wave 2 仍在主扫描 q=0、1/3、2/3、1 的平均绝对位移上全部领先，因此尚未出现“最佳波数由低波切换为高波”。后续应先校验无穿地恢复冲击、无固定窗口相位偏置的失支撑模型，再检验交叉。','',
        '12 s 与半时间步只检查了事先指定的首个新种子 100；该种子在原 6 s、半步长和 12 s 下均为低波略领先。它不是 8 个种子平均交叉的长期或数值收敛认证。','',
        '这是当前模型、每个波数各自冻结的振幅与 PD 下的对照。没有随 loss 重新优化；因此结论针对这些步态配置，不能单独归因于 wave number。旧振幅搜索尚未全部完成，这些配置不是已证明的全局最优。','',
        '## 试验与横轴','',
        '19 个左右腿对分别随机失去整个腿子树的碰撞能力，身体碰撞保留。每个随机种子的擦除时序在所有波数间逐时间步完全相同。控制命令在同一波数的不同 loss 条件下完全相同。平地、1 Hz、6 s 试验，使用 1.5–5.999 s 固定前进轴的 21 个身体节点几何中心净位移。','',
        '主图横轴 q 是每 0.5 s 窗口内每个腿对发生擦除事件的概率；事件持续 0.25–0.5 s，故预期擦除时间比例 nu=0.75q。nu=1 是全程所有腿擦除的单独极端对照，不能与 q=1、nu=.75 混为一谈。擦除概率不是实际足端触地概率。外部窗口与 1 Hz 步态有固定相位关系，结论不能推广到所有异步随机过程。','',
        '## 全部波数的发现结果','',
        '9 个波数 ×（无 loss 与全擦除各一次，加 3 个中间水平各 6 个配对种子）=180 个条件。下表单位 mm，包含所有完整试验，即使它违反稳定性或工程门限。', '',table(cells),'',
        '主图：`outputs/figures/probability_curves.png`；含全擦除、保持率、直立存活率及全部门限的图：`outputs/figures/main_accepted9_all_waves.png`。','',
        '## 独立种子复核','',
        f"按公开的发现阶段选对规则选取 wave {waves[0]:g} 与 {waves[1]:g}，随后才生成 100–107 共 8 个新种子。选择规则及所有候选见 `outputs/data/confirmation_plan.json`。",'',table(conf['cells']),'',
        '| nu | q | 低波减高波均值 (mm) | 配对 bootstrap 95% 区间 | 高波获胜比例 |','|---:|---:|---:|---:|---:|']
    lines += [f"| {v['nu']:g} | {v['q']:.3f} | {v['mean']:.3f} | [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | {v['high_win_fraction']:.3f} |" for v in dc]
    lines += ['','## 时间步与试验时长检查','','不是多种子稳定性认证；只检查选中配对在 seed 100 下的敏感性。12 s 使用自身 12 s 无 loss 基准计算保持率，不能与 6 s 总位移直接比较。','',
        '| Wave | 时长(s) | 时间步倍率 | nu | seed | D(mm) | 自身基准保持率 | 直立 | 全部门限 |','|---:|---:|---:|---:|---:|---:|---:|---|---|',*checklines,'',
        '## 恢复碰撞的物理限制','',
        f"主实验加独立确认共有 {restoration['events']} 次腿对恢复事件，其中 {restoration['floor_overlap_events']} 次至少一个恢复碰撞的几何体已经与地面重叠。随机 burst 条件的最大地面力峰值范围为 {restoration['force_range_BW'][0]:.2f}–{restoration['force_range_BW'][1]:.2f} 倍体重。全部 burst 条件至少一项原工程门限失败；位移交叉不能直接称为可靠生物力学优势。",'',
        '额外 27 个小型对照让随机选中的腿对从 t=1.5 s 到结束一直失去碰撞，不发生恢复。此处 nu 表示一次性选中腿对的概率，随机过程和主试验不同，结果不能合并或将差异全归因于恢复冲击。下表包含相同模型的无 loss 基准：','',table(pcells),'',
        '## 复现、排除与证据','',
        '- wave 2.5 当前无 loss 位移 178.4983 mm，历史 175.0807 mm，偏差 1.95%，未满足预先声明的 1% 复现界限，因此排除整条主曲线。原始试验完整保留；这是基准校准失败，不是数值发散。',
        '- wave .5、1、8、9 没有上一轮已合格的选定配置。其他 9 个当前无 loss 配置均通过原门限及 1% 复现界限。',
        '- 初始历史逐位复现未通过后，在查看任何 loss 结果之前完成无插桩复核：wave 2 与 6 的插桩/无插桩当前结果所有指标逐位相同。随后声明当前运行环境 1% 容差。历史差异原因尚未确定，旧数据未覆盖。',
        f"- 主试验与复核原始轨迹位移重算、跨波数擦除掩码、同波数控制命令审计：{json.dumps(all_audit,ensure_ascii=False)}。永久擦除对照审计：{json.dumps(paudit,ensure_ascii=False)}。",
        '- `outputs/data/manifest.json` 冻结模型、控制配置与源文件；`accepted_screen_amendment.json` 记录 wave 2.5 排除；`final_audit.json` 记录完整检查。',
        '- 每个条件保留 `result.json`、完整逐时间步擦除掩码、1 ms 几何中心轨迹、约 60 Hz qpos、事件表、恢复间隙和冲击诊断。所有证据只写入本实验目录。Robot、IsaacLab 和 Clean 未修改。',
        '- 视频从所保存 qpos 直接重放，前/侧/斜三视图，擦除腿为红色；30 fps，未改变物理或平滑轨迹。实际采样时间误差及视频校验记录位于对应 `.json`。','']
    (s.ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(waves=waves,paired=dc,audit=extra),ensure_ascii=False))

if __name__=='__main__':main()
