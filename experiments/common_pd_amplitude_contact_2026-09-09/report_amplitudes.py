"""Scientific figures and a transparent finite-search result summary."""
from pathlib import Path
import argparse,collections,datetime,json,hashlib
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/physical_geometry_v2';DATA=OUT/'data'
WAVES=[.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,r):Path(p).write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf-8')

def trend(rows,lo,hi):
    chosen=[r for r in rows if lo<=r['wave']<=hi]
    if len(chosen)<3:return dict(n=len(chosen),available=False)
    w=np.array([r['wave'] for r in chosen]);a=np.array([r['amplitude']['body'] for r in chosen])
    return dict(n=len(chosen),available=True,spearman_rho=float(spearmanr(w,a).statistic),
        linear_slope_deg_per_wave=float(np.polyfit(w,a,1)[0]),
        adjacent_increases=[dict(low_wave=float(w[i]),high_wave=float(w[i+1]),increase_deg=float(a[i+1]-a[i])) for i in range(len(a)-1) if a[i+1]>a[i]+1e-8],
        interpretation='Descriptive association of independently selected optima, not an independent statistical sample or a monotonicity constraint.')

def main(final):
    allrows=[read(p) for p in (OUT/'assessments').glob('*.json')]
    nominal=[r for r in allrows if r['task']['phase']=='midpoint' and r['task']['plant']=='current' and r['task']['duration']==6 and r['task']['dt_scale']==.5]
    chosen=[]
    if final:
        selection=read(DATA/'validated_selection.json');assert selection['complete']
        for w,validation in selection['selected'].items():
            r=validation['nominal'];assert validation['qualified'] and 'nonadjacent_chassis_contact_steps' in r['metrics']
            chosen.append(dict(wave=float(w),amplitude=r['amplitudes'],D=r['D'],record=r['record'],
                long_D=validation['long']['D'],half_D=validation['half']['D'],
                half_relative_difference=validation['half_step_relative_difference'],motion=r['motion_path']))
    else:
        for w in WAVES:
            feasible=[r for r in nominal if r['wave']==w and r['feasible']]
            if feasible:
                r=max(feasible,key=lambda r:r['D'])
                chosen.append(dict(wave=w,amplitude=r['amplitudes'],D=r['D'],record=r['record'],provisional=True))
    chosen.sort(key=lambda r:r['wave'])
    for r in chosen:
        if r.get('motion'):
            z=np.load(r['motion']);good=(z['t']>=2)&(z['t']<6);t=z['t'][good]
            X=np.stack([np.sin(2*np.pi*t),np.cos(2*np.pi*t),np.ones_like(t)],axis=1)
            b=np.linalg.lstsq(X,z['bq'][good,1:,0],rcond=None)[0]
            r['realized_body_median_deg']=float(np.median(np.degrees(np.hypot(b[0],b[1]))))
    counts=[]
    for w in WAVES:
        rows=[r for r in nominal if r['wave']==w]
        counts.append(dict(wave=w,evaluated=len(rows),full=sum(r['full'] for r in rows),eligible=sum(r['feasible'] for r in rows),
            provisional_imports=sum(r.get('self_contact_verified') is False for r in rows),
            failure_counts=dict(collections.Counter(k for r in rows for k in r['failed']))))
    hashes=sorted({r['pd_readback']['active_pd_sha256'] for r in allrows if r.get('pd_readback')})
    assert len(hashes)==1
    info=dict(status='VALIDATED_FINITE_SEARCH' if final else 'PROVISIONAL_SEARCH_IN_PROGRESS',
        time=datetime.datetime.now().astimezone().isoformat(),selected=chosen,counts=counts,all_pd_sha256=hashes,
        historical_rl_range=trend(chosen,1.5,4.5),extended_range=trend(chosen,.5,9),
        unresolved_waves=[w for w in WAVES if w not in [r['wave'] for r in chosen]],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    label='validated_amplitudes' if final else 'amplitude_progress'
    write(DATA/f'{label}.json',info)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    x=[r['wave'] for r in chosen]
    for key,color in [('body','#24608b'),('leg_yaw','#ba6a2b'),('leg_pitch','#337c52')]:
        axes[0,0].plot(x,[r['amplitude'][key] for r in chosen],'o-',color=color,label=key)
    if final:axes[0,0].plot(x,[r['realized_body_median_deg'] for r in chosen],'s--',color='#579bbb',label='body realized 1 Hz median')
    axes[0,0].set(ylabel='Amplitude (deg)',title='Independent per-wave amplitude selection');axes[0,0].legend(fontsize=8)
    axes[0,1].plot(x,[r['D'] for r in chosen],'o-',label='6 s nominal')
    if final:axes[0,1].plot(x,[r['half_D'] for r in chosen],'s--',label='6 s half timestep')
    axes[0,1].set(ylabel='World-forward centroid displacement (mm)',title='Flat locomotion, no erasure');axes[0,1].legend(fontsize=8)
    for r in nominal:
        if r['full']:
            axes[1,0].scatter(r['wave'],r['amplitudes']['body'],s=9,c='#91bca3' if r['feasible'] else '#dedede',alpha=.4)
    axes[1,0].plot(x,[r['amplitude']['body'] for r in chosen],'o-',color='#1e5840',label='Best eligible / qualified')
    axes[1,0].set(ylabel='Body command amplitude (deg)',title='Evaluated space; grey = failed engineering gate');axes[1,0].legend(fontsize=8)
    axes[1,1].bar([r['wave'] for r in counts],[r['evaluated'] for r in counts],width=.32,label='All assessments',color='#cad6df')
    axes[1,1].bar([r['wave'] for r in counts],[r['eligible'] for r in counts],width=.32,label='Eligible',color='#49768d')
    axes[1,1].set(ylabel='Candidate count',title='Search coverage');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.set_xlabel('Coupled wave number');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('One fixed PD; physical geometry policy; '+('validated' if final else 'PROVISIONAL — optimization in progress'),fontsize=13)
    fig.savefig(ROOT/f'{label}.png',dpi=180);fig.savefig(ROOT/f'{label}.pdf');plt.close(fig)
    text=['# 固定 PD 振幅优化结果' if final else '# 固定 PD 优化进度（非最终结果）','',
      '状态：'+info['status'],'',f"共 {len(nominal)} 个名义条件评估，{len(chosen)} 个波数有当前选点。所有实际 PD 读回哈希一致。",
      '', '| wave | body ° | leg yaw ° | leg pitch ° | D mm |', '|---:|---:|---:|---:|---:|']
    for r in chosen:text.append(f"| {r['wave']:g} | {r['amplitude']['body']:.3f} | {r['amplitude']['leg_yaw']:.3f} | {r['amplitude']['leg_pitch']:.3f} | {r['D']:.3f} |")
    for name,key in [('历史 RL 范围 1.5–4.5','historical_rl_range'),('扩展范围 0.5–9','extended_range')]:
        t=info[key]
        if t['available']:text+=['',f"{name}：body amplitude 与 wave 的 Spearman ρ={t['spearman_rho']:.3f}，线性斜率 {t['linear_slope_deg_per_wave']:.3f} °/wave，相邻上升点 {len(t['adjacent_increases'])} 个。仅描述选点，不作为优化奖励或独立统计检验。"]
    text+=['','未解决波数：'+str(info['unresolved_waves']),
      '', '固定 PD 不代表不同仿真系统必然得到相同最优幅值。主结果使用修正的中点腿相位和真实身体碰撞门槛；历史 R 门槛分支完整保留。',
      '最终选点须通过完整 6 s、12 s 和减半步长验证；任何中途拒绝没有完整位移，不参与速度排名。' if not final else '此处选点已经通过完整 6 s、12 s 和减半步长核验；减半步长位移差不超过 10%。这是已测有限候选集的最优值，不声称连续全局最优。','']
    (ROOT/f'{label}.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(dict(status=info['status'],candidates=len(nominal),selected=len(chosen),trend=info['historical_rl_range']),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');main(p.parse_args().final)
