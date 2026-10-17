"""Descriptive curves and paired crossover discovery; never retunes the plant."""
from __future__ import annotations
import argparse,csv,itertools,json,math
from pathlib import Path
import study as s
np=s.np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG=s.OUT/'figures'
def records():return [s.read(p) for p in sorted(s.TRIALS.glob('*/result.json'))]
def interval(values,seed=42):
    a=np.asarray(values,dtype=float)
    if not len(a):return [None,None]
    if len(a)==1:return [float(a[0]),float(a[0])]
    rng=np.random.default_rng(seed)
    means=a[rng.integers(0,len(a),size=(4000,len(a)))].mean(axis=1)
    return np.quantile(means,[.025,.975]).tolist()
def aggregate(rows):
    keys=sorted(set((r['task']['wave'],r['task']['nu']) for r in rows));out=[]
    baseline={r['task']['wave']:r['D'] for r in rows if r['completed'] and r['task']['nu']==0}
    for w,nu in keys:
        group=[r for r in rows if r['task']['wave']==w and r['task']['nu']==nu]
        full=[r for r in group if r['completed']];d=[r['D'] for r in full]
        out.append(dict(wave=w,nu=nu,q=nu/.75 if nu<=.75 else 1.,n=len(group),completed=len(full),
            mean_D=float(np.mean(d)) if d else None,ci_D=interval(d),std_D=float(np.std(d,ddof=1)) if len(d)>1 else 0.,
            mean_retention=float(np.mean(d)/baseline[w]) if d and w in baseline else None,
            viability=sum(r.get('viable',False) for r in group)/len(group),
            upright_survival=sum(r.get('upright_survived',False) for r in group)/len(group),
            realized_erased_load=float(np.mean([r['channel']['realized_erased_load'] for r in full])) if full else None,
            max_force_BW=max([r['metrics']['ground_force_peak_bw'] for r in full],default=None),
            mean_chassis_contact=float(np.mean([r['metrics']['chassis_contact_fraction'] for r in full])) if full else None,
            seeds=[r['task']['seed'] for r in full]))
    return out
def paired(rows):
    waves=sorted(set(r['task']['wave'] for r in rows));levels=sorted(set(r['task']['nu'] for r in rows))
    data={(r['task']['wave'],r['task']['nu'],r['task']['seed']):r for r in rows};out=[]
    for lo,hi in itertools.combinations(waves,2):
        rlo=data.get((lo,0.,0));rhi=data.get((hi,0.,0))
        if not rlo or not rhi or not rlo['completed'] or not rhi['completed']:continue
        d0=rlo['D']-rhi['D']
        if d0<=0:continue
        curve=[]
        for nu in levels:
            seeds=sorted(set(r['task']['seed'] for r in rows if r['task']['nu']==nu))
            pairs=[(data.get((lo,nu,se)),data.get((hi,nu,se))) for se in seeds]
            pairs=[(a,b) for a,b in pairs if a and b and a['completed'] and b['completed']]
            if not pairs:continue
            delta=[a['D']-b['D'] for a,b in pairs]
            curve.append(dict(nu=nu,n=len(pairs),difference_mean=float(np.mean(delta)),difference_ci=interval(delta),
                low_mean=float(np.mean([a['D'] for a,b in pairs])),high_mean=float(np.mean([b['D'] for a,b in pairs])),
                both_upright=sum(a['upright_survived'] and b['upright_survived'] for a,b in pairs)/len(pairs),
                both_viable=sum(a['viable'] and b['viable'] for a,b in pairs)/len(pairs),
                high_win_fraction=float(np.mean(np.asarray(delta)<0))))
        crossings=[]
        for a,b in zip(curve,curve[1:]):
            if a['difference_mean']>0 and b['difference_mean']<0:
                estimate=a['nu']+(b['nu']-a['nu'])*a['difference_mean']/(a['difference_mean']-b['difference_mean'])
                crossings.append(dict(lower=a,upper=b,linear_estimate=estimate))
        if crossings:out.append(dict(low_wave=lo,high_wave=hi,no_loss_difference=d0,curve=curve,crossings=crossings))
    return out
def audit(rows):
    groups={};commands={};errors=[];rawchecked=0
    for r in rows:
        if not r['completed']:continue
        t=r['task'];k=(t['nu'],t['seed'],t['duration'],t['dt_scale']);groups.setdefault(k,set()).add(r['channel']['mask_sha256'])
        commands.setdefault((t['wave'],t['duration'],t['dt_scale']),set()).add(r['channel']['command_sha256'])
        z=np.load(Path(r['record_path']).with_name('trajectory.npz'));good=z['t']>=1.5
        d=float(r['forward_sign']*(z['center'][good][-1,0]-z['center'][good][0,0])*1000)
        if d!=r['D']:errors.append({'id':r['id'],'error':'raw displacement mismatch'})
        rawchecked+=1
    for k,v in groups.items():
        if len(v)!=1:errors.append({'condition':str(k),'error':'unmatched erasure masks across waves'})
    for k,v in commands.items():
        if len(v)!=1:errors.append({'condition':str(k),'error':'controller commands changed with noise'})
    return dict(passed=not errors,errors=errors,full_trajectories_recomputed=rawchecked,
        matched_mask_conditions=len(groups),command_groups=len(commands))

def plots(cells,rows,crossings,label):
    FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    waves=sorted(set(r['wave'] for r in cells));colors={w:plt.cm.turbo(i/max(1,len(waves)-1)) for i,w in enumerate(waves)}
    fig,axes=plt.subplots(2,2,figsize=(13,9),sharex=True)
    for w in waves:
        group=[r for r in cells if r['wave']==w and r['mean_D'] is not None];x=np.array([r['nu'] for r in group]);color=colors[w]
        y=np.array([r['mean_D'] for r in group]);axes[0,0].plot(x,y,'o-',color=color,label=f'wave {w:g}',lw=1.8,ms=4)
        axes[0,0].fill_between(x,[r['ci_D'][0] for r in group],[r['ci_D'][1] for r in group],color=color,alpha=.09)
        axes[0,1].plot(x,[r['mean_retention'] for r in group],'o-',color=color,ms=4)
        axes[1,0].plot(x,[r['upright_survival'] for r in group],'o-',color=color,ms=4)
        axes[1,1].plot(x,[r['viability'] for r in group],'o-',color=color,ms=4)
    for ax,title,ylabel in zip(axes.flat,['Absolute forward displacement','Retention of own no-loss baseline','Upright survival','All speed-first viability gates'],['Displacement (mm)','D / D0','Fraction of planned trials','Fraction of planned trials']):
        ax.set(title=title,ylabel=ylabel,xlim=(0,1));ax.grid(alpha=.2);ax.axhline(0,color='.6',lw=.6)
    axes[0,0].legend(ncol=2,fontsize=8,frameon=False)
    for ax in axes[1]:ax.set_xlabel('Expected erased-time fraction nu')
    axes[1,0].set_ylim(-.03,1.05);axes[1,1].set_ylim(-.03,1.05)
    fig.suptitle('Frozen per-wave gaits under matched bilateral contact loss',fontsize=16,y=.98)
    fig.text(.5,.015,f'{label} | Flat, 1 Hz, 6 s; measured t=1.5–5.999 s. Shading: descriptive 95% seed bootstrap.\nAll finite full-horizon outcomes included, even when viability fails. nu is time load; q is event probability.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95));fig.savefig(FIG/f'{label}_all_waves.png',dpi=170);fig.savefig(FIG/f'{label}_all_waves.svg');plt.close(fig)
    if crossings:
        fig,ax=plt.subplots(figsize=(10,6))
        for r in crossings:
            if r['low_wave']>3 or r['high_wave']<4:continue
            c=r['curve'];ax.plot([v['nu'] for v in c],[v['difference_mean'] for v in c],'.-',alpha=.65,label=f"{r['low_wave']:g} minus {r['high_wave']:g}")
        ax.axhline(0,color='black',lw=1);ax.set(xlabel='Expected erased-time fraction nu',ylabel='Paired low-wave minus high-wave displacement (mm)',title='Positive to negative indicates an observed crossing')
        ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(FIG/f'{label}_paired_differences.png',dpi=170);plt.close(fig)

def main(label,confirmation=False):
    allrows=records();accepted=s.read(s.DATA/'accepted_screen_amendment.json')['accepted_waves'] if (s.DATA/'accepted_screen_amendment.json').exists() else list(map(float,s.manifest()['profiles']))
    rows=[r for r in allrows if r['task']['duration']==6 and r['task']['dt_scale']==1 and r['task']['wave'] in accepted and ((r['task']['seed']>=100 or r['task']['nu']==0) if confirmation else r['task']['seed']<100)]
    if confirmation:
        selected=s.read(s.DATA/'confirmation_plan.json')['waves'];rows=[r for r in rows if r['task']['wave'] in selected]
    cells=aggregate(rows);crossings=paired(rows);checks=audit(rows)
    s.write(s.DATA/f'{label}_summary.json',dict(cells=cells,crossings=crossings,audit=checks,
        total=len(rows),noncompleted=[r for r in rows if not r['completed']],excluded_waves=[w for w in map(float,s.manifest()['profiles']) if w not in accepted],interpretation='Independent confirmation on newly drawn seeds' if confirmation else 'descriptive discovery, pending independent confirmation'))
    csvrows=[]
    for r in rows:
        t=r['task'];csvrows.append(dict(id=r['id'],**t,completed=r['completed'],D_mm=r['D'],viable=r.get('viable'),
            upright=r.get('upright_survived'),realized_nu=r.get('channel',{}).get('realized_erased_load'),
            failed=';'.join(r.get('failed_viability',[])),error=r.get('error','')))
    if csvrows:
        with (s.DATA/f'{label}_trials.csv').open('w',newline='',encoding='utf-8-sig') as f:
            wr=csv.DictWriter(f,fieldnames=list(csvrows[0]));wr.writeheader();wr.writerows(csvrows)
    plots(cells,rows,crossings,label)
    print(json.dumps(dict(label=label,count=len(rows),audit=checks,cells=cells,
        crossings=[dict(low=r['low_wave'],high=r['high_wave'],brackets=[dict(nu=[c['lower']['nu'],c['upper']['nu']],estimate=c['linear_estimate'],upper_difference=c['upper']['difference_mean'],upper_ci=c['upper']['difference_ci'],both_upright=c['upper']['both_upright'],both_viable=c['upper']['both_viable']) for c in r['crossings']]) for r in crossings]),ensure_ascii=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--label',default='screen');ap.add_argument('--confirmation',action='store_true');args=ap.parse_args();main(args.label,args.confirmation)
