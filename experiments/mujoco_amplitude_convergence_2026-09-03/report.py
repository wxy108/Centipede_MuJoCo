"""Honest progress/final figures and recorded-pose three-view comparisons."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import study as s
e,np=s.e,s.np


def previous_trace(wave):
    old=s.read(s.OLD/'outputs/data/validated_selection.json')['waves'][f'{wave:g}']
    chosen=old['selected'];row=chosen['nominal'] if chosen else old['diagnostic']
    stage='delivery' if chosen else 'delivery_diagnostic'
    path=s.OLD/'outputs/trials'/stage/row['id']/f"{row['id']}.json"
    assert path.exists() and path.with_suffix('.npz').exists()
    return path,bool(chosen)


def motion(path):
    raw=s.read(path)
    with np.load(Path(path).with_suffix('.npz')) as tr:
        mask=tr['t']>=1.5
        half=lambda x:(np.quantile(x,.99,axis=0)-np.quantile(x,.01,axis=0))/2
        b=half(np.degrees(tr['bq'][mask]))
        l=half(np.degrees(tr['lq'][mask])).reshape(19,2,2)
        tip=half(np.degrees(tr['tq'][mask])).reshape(19,2)
        world=tr['worldroll'][mask]
        return dict(command_deg=raw['task']['amplitudes'],
            actual_definition='(P99-P1)/2 over t>=1.5 s; not harmonic amplitude',
            mean_body_yaw_deg_excluding_zero_command_joint0=float(b[1:,0].mean()),
            body_yaw_pitch_roll_half_range_deg=b.tolist(),
            leg_yaw_pitch_half_range_deg=l.tolist(),tip_half_range_deg=tip.tolist(),
            individual_world_roll_abs_p95_deg=np.quantile(abs(world),.95,axis=0).tolist(),
            individual_world_roll_peak_deg=np.max(abs(world),axis=0).tolist())


def make_plots(label,table,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=s.OUT/'figures';out.mkdir(parents=True,exist_ok=True)
    status='INTERIM: transfer challenge only' if label=='interim' else 'BOUNDED SEARCH: validated finite-tested-set best'
    waves=[r['wave'] for r in table]
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),sharex=True)
    for ax,key,title in zip(axs,s.KEYS,('Body yaw','Leg yaw','Leg pitch')):
        ax.plot(waves,[r['old_amplitudes'][key] if r['old_amplitudes'] else np.nan for r in table],
                's--',color='.55',label='Previous validated')
        ax.plot(waves,[r['amplitudes'][key] if r['amplitudes'] else np.nan for r in table],
                'o-',color='#007c91',label='Current validated')
        ax.set(title=title,xlabel='Coupled wave number',ylabel='Command amplitude (deg)')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(status+'\nSame model, phase law, gates, and per-wave PD; no monotonicity constraint')
    fig.tight_layout();fig.savefig(out/f'{label}_amplitudes.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.3))
    for column,key,title in ((0,'D_mm','Nominal forward displacement (mm)'),(1,'actual_body_yaw','Actual mean body yaw half-range (deg)')):
        ax=axs[column]
        oldkey='old_'+key
        ax.plot(waves,[r.get(oldkey,np.nan) for r in table],'s--',color='.55',label='Previous validated')
        ax.plot(waves,[r.get(key,np.nan) for r in table],'o-',color='#007c91',label='Current validated')
        ax.set(xlabel='Coupled wave number',ylabel=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(status+'\nDisplacement: t=1.5..5.999 s | actual yaw excludes zero-command joint 0')
    fig.tight_layout();fig.savefig(out/f'{label}_performance_motion.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(4,4,figsize=(15,12),sharex=True)
    profile=[]
    for ax,w in zip(axs.flat,e.WAVES):
        allw=[r for r in rows if r['wave']==w and r['full']]
        feasible=[r for r in allw if r['feasible']]
        failed=[r for r in allw if not r['feasible']]
        ax.scatter([r['amplitudes']['body'] for r in failed],[r['D'] for r in failed],s=4,color='.75',alpha=.25,label='Failed gates')
        ax.scatter([r['amplitudes']['body'] for r in feasible],[r['D'] for r in feasible],s=7,color='#007c91',alpha=.35,label='Nominal feasible')
        nodes=[]
        for node in s.manifest()['body_profile_deg']:
            group=[r for r in feasible if abs(r['amplitudes']['body']-node)<1e-10]
            best=max(group,key=lambda r:r['D']) if group else None
            nodes.append(dict(body_deg=node,best=best))
        if label=='final':
            ax.plot([r['body_deg'] for r in nodes],[r['best']['D'] if r['best'] else np.nan for r in nodes],
                    'o-',color='#d06b00',ms=3,label='Exact-node best tested')
        maximum=max((r['D'] for r in feasible),default=None)
        near=[r for r in feasible if r['D']>=.97*maximum] if maximum is not None else []
        if near:
            ax.scatter([r['amplitudes']['body'] for r in near],[r['D'] for r in near],s=12,color='#a72b60',label='Within 3% of nominal best')
        profile.append(dict(wave=w,nodes=nodes,near_best_tested_ids=[r['id'] for r in near],
                            near_best_body_deg=[r['amplitudes']['body'] for r in near]))
        ax.set(title=f'wave={w:g}',xlabel='Body command (deg)',ylabel='D (mm)',xlim=(0,36));ax.grid(alpha=.15)
    for ax in list(axs.flat)[len(e.WAVES):]:ax.set_visible(False)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.962),ncol=4,fontsize=8)
    fig.suptitle(status+'\nProfile points are sampled LOWER estimates; gaps are unresolved, not proven infeasible')
    fig.tight_layout(rect=(0,0,1,.935));fig.savefig(out/f'{label}_body_profiles.png',dpi=150);plt.close(fig)
    s.write(s.DATA/f'{label}_profile_evidence.json',profile)


def report(label):
    s.verify_hashes()
    result=s.read(s.DATA/f'{label}_validated_selection.json')
    assert result['validation_complete']
    old=s.read(s.OLD/'outputs/data/validated_selection.json')
    rows=s.collect();table=[]
    for w in e.WAVES:
        key=f'{w:g}';st=result['waves'][key];before=old['waves'][key]['selected'];after=st['selected']
        item=dict(wave=w,status=st['status'],amplitudes=after['nominal']['amplitudes'] if after else None,
            old_amplitudes=before['nominal']['amplitudes'] if before else None,
            D_mm=after['nominal']['D'] if after else None,old_D_mm=before['nominal']['D'] if before else None,
            halfstep_D_relative_difference=after['halfstep_D_relative_difference'] if after else None)
        if st.get('trace_record'):
            item['motion']=motion(st['trace_record'])
            if after:item['actual_body_yaw']=item['motion']['mean_body_yaw_deg_excluding_zero_command_joint0']
        path,prior_ok=previous_trace(w)
        if prior_ok:item['old_actual_body_yaw']=motion(path)['mean_body_yaw_deg_excluding_zero_command_joint0']
        if after and before:
            item['gain_mm']=after['nominal']['D']-before['nominal']['D']
            assert item['gain_mm']>=-1e-8, (w,item['gain_mm'])
            item['gain_percent']=100*item['gain_mm']/before['nominal']['D']
        if after:item['metrics']=after['nominal']['metrics']
        table.append(item)
    s.write(s.DATA/f'{label}_comparison.json',table)
    make_plots(label,table,rows)
    renderer=e.load(s.OLD/'render_compliant_comparison.py','convergence_renderer')
    videos=[]
    if label=='interim':
        improvements=sorted((r for r in table if r.get('gain_mm',0)>1e-6),key=lambda r:-r['gain_percent'])
        view_waves=list(dict.fromkeys([r['wave'] for r in improvements[:2]]+[2.,3.5]))[:2]
    else:view_waves=list(e.WAVES)
    for w in view_waves:
        st=result['waves'][f'{w:g}'];before,old_ok=previous_trace(w)
        if not st.get('trace_record'):continue
        labels=['PREVIOUS VALIDATED' if old_ok else 'PREVIOUS DIAGNOSTIC / NOT QUALIFIED',
                ('INTERIM VALIDATED CANDIDATE' if label=='interim' else 'CURRENT VALIDATED CANDIDATE')
                if st['selected'] else 'DIAGNOSTIC / NOT QUALIFIED']
        paths=[str(before),st['trace_record']]
        tag=f'{w:g}'.replace('.','p')
        for local in ([False,True] if w in view_waves[:2] else [False]):
            name=f"{label}_w{tag}_{'local' if local else 'three_views'}"
            videos.append(renderer.make(paths,s.OUT/'videos'/f'{name}.mp4',local=local,labels=labels))
    convergence={}
    for kind in ('profile','three_d'):
        path=s.DATA/f'{kind}_convergence.json'
        if path.exists():convergence[kind]=dict(Counter(r['stop'] for r in s.read(path)))
    audit=dict(passed=True,label=label,previous_conditions=5036,
        distinct_conditions=len(rows),new_conditions=len(rows)-5036,
        global_optimum_proved=False,canonical_baseline_replaced=False,
        policy_sha256=s.gates.POLICY_SHA,frozen_sources_unchanged=s.verify_hashes(),
        convergence_stops=convergence,videos=videos,visual_review='pending human/model frame inspection')
    s.write(s.DATA/f'{label}_delivery_audit.json',audit)
    title='阶段性结果：跨波数参数挑战，后续搜索尚未完成' if label=='interim' else '三幅值收敛研究：有限测试集结果，不是连续全局最优证明'
    lines=[f'# {title}','',
        '模型、控制器、每波数 PD、门槛、相位和频率不变；柔性 tip，平地，擦除关闭。',
        '各波数 PD 仍然不同。仅改变 body yaw / leg yaw / leg pitch 三个幅值。',
        '只有通过 6 s、12 s、半步长 6 s 和位移一致性检查的候选才进入下表。',
        '本报告由程序生成；视频逐帧视觉审查状态见 delivery_audit，不应把文件存在等同于视觉通过。','',
        f'已有条件 5036；累计不同条件 {len(rows)}；新增 {len(rows)-5036}。','',
        '| wave | body / leg yaw / leg pitch (deg) | D (mm) | 相对上轮 |',
        '|---:|---|---:|---:|']
    for r in table:
        a=r['amplitudes']
        if not a:lines.append(f"| {r['wave']:g} | 未找到通过全部验证的候选 | — | — |")
        else:lines.append(f"| {r['wave']:g} | {a['body']:.2f} / {a['leg_yaw']:.2f} / {a['leg_pitch']:.2f} | {r['D_mm']:.3f} | {r.get('gain_percent',0):+.2f}% |")
    lines += ['',f'![幅值]({(s.OUT/"figures"/f"{label}_amplitudes.png").as_posix()})',
        '',f'![性能与实际摆幅]({(s.OUT/"figures"/f"{label}_performance_motion.png").as_posix()})',
        '',f'![body yaw 剖面]({(s.OUT/"figures"/f"{label}_body_profiles.png").as_posix()})','',
        '近最佳点仅作为诊断，不增加平滑、单调或幅值差异的奖励。剖面是已测试点的最佳值，不是最优值上界。',
        '',f'搜索停止状态：`{json.dumps(convergence)}`。达到预算的起点仍记为未收敛。','']
    for v in videos:lines += [f'![{Path(v["video"]).stem}]({Path(v["video"]).as_posix()})','']
    path=s.ROOT/('INTERIM_RESULTS_ZH.md' if label=='interim' else 'RESULTS_ZH.md')
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(report=str(path),audit=audit),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',choices=('interim','final'),default='final')
    a=p.parse_args();report(a.label)
