"""Read saved optimizer checkpoints; never select a gait or alter the search."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'outputs/physical_geometry_v2/data'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    checkpoints=sorted(p for p in DATA.glob('local_midpoint_*.json') if re.fullmatch(r'local_midpoint_\d\d\.json',p.name))
    assert checkpoints
    latest=read(checkpoints[-1]);old=read(DATA/'local_midpoint_07.json')
    prior={s['key']:s for s in old}
    completed=DATA/'local_convergence_audit.json'
    final=completed.exists()
    if final:latest=read(DATA/'local_midpoint_complete.json')
    lines=['# 振幅优化的收敛检查','',
           '状态：'+('局部搜索已结束；仍不等于连续全局最优。' if final else '搜索仍在运行；本页是最近完整检查点。'),'',
           '所有起点使用相同目标与停止条件：归一化步长0.005时，六个坐标方向均没有超过0.2mm的改进。对应body / leg yaw / leg pitch步幅0.18° / 0.45° / 0.30°。联动三振幅探查和后续长期、减半步长资格验证另外报告。',
           '', '| wave / 起点 | 已完成轮数 | 当前步长 | 搜索状态 | 相比原8轮终点提高 mm | 当前 D mm |',
           '|---|---:|---:|---|---:|---:|']
    items=[]
    for st in latest:
        gain=st['center']['D']-prior[st['key']]['center']['D']
        assert gain>=-1e-8
        status=st['stop'] or 'continuing'
        items.append(dict(key=st['key'],wave=st['wave'],polls=len(st['history']),step=st['step'],
                          status=status,extension_gain_mm=gain,D=st['center']['D']))
        label={'no_axis_improvement_at_resolution':'达到坐标探查停止条件','poll_budget_reached':'达到预算上限，未证明收敛','continuing':'继续搜索'}[status]
        lines.append(f"| {st['key']} | {len(st['history'])} | {st['step']:.3f} | {label} | {gain:.4f} | {st['center']['D']:.4f} |")
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
    waves=sorted({s['wave'] for s in latest})
    for i,w in enumerate(waves):
        color=plt.cm.viridis(i/max(1,len(waves)-1))
        starts=[s for s in latest if s['wave']==w]
        for j,st in enumerate(starts):
            history=st['history'];gains=np.cumsum([h['gain_mm'] for h in history])
            origin=st['center']['D']-gains[-1]
            axes[0].plot([0,*[h['poll']+1 for h in history]],[origin,*(origin+gains)],
                         color=color,ls='-' if j==0 else '--',alpha=.85,label=f'w={w:g}' if j==0 else None)
    axes[0].axvline(8,color='#a64640',lw=1,ls=':',label='Original poll budget')
    axes[0].set(xlabel='Completed coordinate polls',ylabel='Incumbent displacement (mm)',
                title='Two diverse starts per wave; no amplitude-trend reward')
    axes[0].legend(ncol=5,fontsize=8)
    x=np.arange(len(items))
    colors=['#337653' if r['status']=='no_axis_improvement_at_resolution' else '#b27638' for r in items]
    axes[1].bar(x,[r['extension_gain_mm'] for r in items],color=colors)
    axes[1].set_xticks(x,[r['key'] for r in items],rotation=60,ha='right',fontsize=7)
    axes[1].set(ylabel='Gain beyond original eight polls (mm)',title='Green: coordinate stopping criterion met; amber: unresolved or budget limited')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Local refinement '+('finished' if final else 'PROVISIONAL — still running'))
    fig.savefig(ROOT/'convergence_progress.png',dpi=180);fig.savefig(ROOT/'convergence_progress.pdf');plt.close(fig)
    sources=[*checkpoints,Path(__file__)]
    if final:sources += [completed,DATA/'local_midpoint_complete.json']
    result=dict(final_local_stage=final,latest_checkpoint=str(checkpoints[-1]),starts=items,
                converged=sum(r['status']=='no_axis_improvement_at_resolution' for r in items),
                unresolved=sum(r['status']!='no_axis_improvement_at_resolution' for r in items),
                no_gait_selection=True,source_hashes={str(p):sha(p) for p in sources})
    (DATA/'convergence_progress.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (ROOT/'convergence_progress.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('final_local_stage','latest_checkpoint','converged','unresolved')}))

if __name__=='__main__':main()
