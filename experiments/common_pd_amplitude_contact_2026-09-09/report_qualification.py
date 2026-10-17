"""Explain qualification-induced changes; never selects or simulates a gait."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'outputs/physical_geometry_v2/data'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()


def failure_reasons(attempt):
    reasons=[]
    for condition in ('nominal','long','half'):
        row=attempt[condition]
        if not row['full']:reasons.append(condition+':incomplete')
        reasons.extend(condition+':'+gate for gate in row['failed'])
    if attempt['half_step_relative_difference']>.1:reasons.append('half:displacement_difference_over_10_percent')
    return sorted(set(reasons))


def main():
    source=DATA/'validated_selection.json';selection=read(source)
    assert selection['complete']
    rows=[];paths=[]
    for p in (DATA.parent/'assessments').glob('*.json'):
        r=read(p);t=r['task']
        if (t['phase']=='midpoint' and t['plant']=='current' and t['duration']==6
                and t['dt_scale']==.5 and t['initialize']=='near' and r['feasible']):
            rows.append(r);paths.append(p)
    waves=read(DATA/'manifest.json')['waves']
    report=[];counts=collections.Counter()
    for w in waves:
        attempts=[a for a in selection['attempts'] if a['wave']==w]
        # A fresh trace can invalidate an imported candidate whose continuous
        # self-contact check was previously unavailable. Its assessment file
        # is then overwritten. Preserve every attempted screening candidate
        # via the qualification record instead of hiding that lost incumbent.
        bank_by_id={r['id']:r for r in rows if r['wave']==w}
        for a in attempts:
            n=a['nominal'];assert n['full'] and n['D'] is not None and a['nominal_exact']
            bank_by_id[n['id']]=n
        bank=sorted(bank_by_id.values(),key=lambda r:r['D'],reverse=True)
        qualified=selection['selected'].get(str(float(w)))
        rejected=[]
        for a in attempts:
            if not a['qualified']:
                reasons=failure_reasons(a);assert reasons
                rejected.append(dict(id=a['nominal_id'],amplitude=a['amplitude'],D=a['nominal']['D'],
                    half_relative_difference=a['half_step_relative_difference'],reasons=reasons))
                counts.update(reasons)
        item=dict(wave=w,qualification_attempts=len(attempts),rejected=rejected,qualified=bool(qualified),
            nominal_best=None if not bank else dict(id=bank[0]['id'],D=bank[0]['D'],amplitude=bank[0]['amplitudes']))
        if qualified:
            q=qualified['nominal'];rank=1+sum(r['D']>q['D']+1e-8 for r in bank)
            assert q['D']<=bank[0]['D']+1e-8
            item.update(selected=dict(id=q['id'],D=q['D'],amplitude=q['amplitudes']),
                rank_in_eligible_nominal_bank=rank,
                D_loss_from_nominal_best_mm=bank[0]['D']-q['D'],
                D_loss_from_nominal_best_fraction=(bank[0]['D']-q['D'])/bank[0]['D'],
                body_amplitude_change_deg=q['amplitudes']['body']-bank[0]['amplitudes']['body'],
                half_relative_difference=qualified['half_step_relative_difference'])
        report.append(item)
    result=dict(time=datetime.datetime.now().astimezone().isoformat(),status='COMPLETED_QUALIFICATION_AUDIT',
        rows=report,rejection_counts=dict(counts),unresolved=selection['unresolved'],
        selection_changed=False,physics_steps=0,
        interpretation='Ranks use the union of currently eligible nominal candidates and every candidate submitted to qualification. The latter preserves screening incumbents whose first fresh nominal trace failed; this matters when a provisional imported self-contact assessment is overwritten. Ranks describe the sampled bank only. Rejection counts can overlap because one attempt can fail several conditions. No alternate amplitude path is selected.',
        input_hashes={str(source):sha(source),**{str(p):sha(p) for p in paths}},source_sha256=sha(__file__))
    (DATA/'qualification_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    for ax in axes:ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Coupled wave number')
    best=[r for r in report if r['nominal_best']];selected=[r for r in report if r['qualified']]
    axes[0].plot([r['wave'] for r in best],[r['nominal_best']['amplitude']['body'] for r in best],'o--',color='#a3a3a3',label='Best screening candidate')
    axes[0].plot([r['wave'] for r in selected],[r['selected']['amplitude']['body'] for r in selected],'s-',color='#245e80',label='Qualified selection')
    axes[0].set(ylabel='Body command amplitude (deg)',title='Effect of qualification on amplitude');axes[0].legend(fontsize=8)
    axes[1].bar([r['wave'] for r in selected],[100*r['D_loss_from_nominal_best_fraction'] for r in selected],width=.3,color='#a66c39')
    axes[1].set(ylabel='Displacement reduction (%)',title='Cost relative to nominal best')
    axes[2].bar([r['wave'] for r in report],[r['qualification_attempts'] for r in report],width=.3,color=['#327c64' if r['qualified'] else '#b46e39' for r in report])
    axes[2].set(ylabel='Attempted candidates',title='Green: qualified; amber: unresolved')
    fig.suptitle('Fixed PD; fresh 6 s, 12 s and half-timestep checks; finite candidate bank',fontsize=12)
    fig.savefig(ROOT/'qualification_audit.png',dpi=180);fig.savefig(ROOT/'qualification_audit.pdf');plt.close(fig)
    lines=['# 振幅资格验证对选点的影响','',
        '下表区分名义筛选阶段的最快候选与最终通过6秒、12秒及减半步长复验的候选。排名包含所有提交过资格验证的候选，保留第一次正式轨迹检查就失败的高分点，避免原评估文件更新后掩盖筛选代价。没有根据趋势或contact结果替换参数。排名只针对已经测量的有限候选集。','',
        '| wave | 最快候选body ° | 最终body ° | 最终名义排名 | 位移下降 % | 尝试次数 |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in report:
        b=r['nominal_best'];a='—' if b is None else f"{b['amplitude']['body']:.3f}"
        if r['qualified']:
            lines.append(f"| {r['wave']:g} | {a} | {r['selected']['amplitude']['body']:.3f} | {r['rank_in_eligible_nominal_bank']} | {100*r['D_loss_from_nominal_best_fraction']:.2f} | {r['qualification_attempts']} |")
        else:lines.append(f"| {r['wave']:g} | {a} | 未解决 | — | — | {r['qualification_attempts']} |")
    lines+=['','拒绝原因计数（同一候选可有多个原因）：']
    lines.extend(f'- {k}：{v}' for k,v in counts.most_common())
    if not counts:lines.append('所有已尝试候选均通过资格验证。')
    lines+=['','完整逐候选原因和输入哈希保存在 outputs/physical_geometry_v2/data/qualification_audit.json。','']
    (ROOT/'qualification_audit.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(qualified=len(selected),unresolved=selection['unresolved'],changed=sum(r['rank_in_eligible_nominal_bank']>1 for r in selected),physics_steps=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    if args.wait:
        print('Waiting for completed amplitude qualification; no simulation launched.',flush=True)
        while not (DATA/'validated_selection.json').exists():time.sleep(2)
    main()
