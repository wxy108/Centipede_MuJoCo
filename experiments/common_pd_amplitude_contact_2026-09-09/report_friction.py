"""Report a completed factorial diagnosis without changing selected gaits."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
DATA=ROOT/'outputs/ground_friction_diagnostic/data'

def main():
    rows=read(DATA/'comparison.json')['comparisons'];assert len(rows)==12
    assert read(DATA/'static_check.json')['passed'] and read(DATA/'priority_parity.json')['passed']
    old=read(ROOT/'outputs/self_collision_diagnostic/data/comparison.json')['comparisons']
    hashes={r['pd_readback']['active_pd_sha256'] for r in rows};assert len(hashes)==1
    for r in rows:
        high=next(x for x in old if x['wave']==r['wave'] and x['family']==r['family'])
        r['high_mu_failed']=high['shared_engineering_failures_self_off' if r['self_off'] else 'shared_engineering_failures_self_on']
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharey=True,layout='constrained')
    for i,off in enumerate((False,True)):
        for j,w in enumerate((1.5,3.,4.5)):
            ax=axes[i,j]
            for family,color,label in [('recovery','#24608b','Recovery amplitude combination'),('rl_v4','#c16a32','Historical RL amplitude combination')]:
                r=next(r for r in rows if r['wave']==w and r['self_off']==off and r['family']==family)
                yy=[r['high_mu_D_mm'],r['low_mu_D_mm']]
                ax.plot([0,1],yy,'o-',color=color,label=label)
                for k,key in enumerate(('high_mu_failed','low_mu_failed')):
                    ax.annotate(f'{yy[k]:.1f}',(k,yy[k]),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9,color=color)
                    if r[key]:ax.plot(k,yy[k],marker='x',markersize=13,markeredgewidth=2,color='#9c252a')
            ax.set(xticks=[0,1],xticklabels=['1.6','0.5'],xlabel='Ground sliding friction coefficient',
                   title=f'Wave {w:g} | internal collisions '+('OFF' if off else 'ON'),xlim=(-.25,1.25),ylim=(50,255))
            if j==0:ax.set_ylabel('Full-window forward displacement (mm)')
            ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Matched friction × self-collision diagnosis | same PD and fixed amplitudes\nRed × = an engineering gate fails; these points remain visible',fontsize=12)
    fig.savefig(ROOT/'friction_collision_diagnostic.png',dpi=180);fig.savefig(ROOT/'friction_collision_diagnostic.pdf');plt.close(fig)
    lines=['# 摩擦与内部碰撞匹配对照','',
           '六组固定振幅、两个自碰撞条件、两个地面滑动摩擦系数。所有PD读回相同。主优化没有采用此诊断的成绩或改变摩擦。',
           '', '| wave | 振幅来源 | 自碰撞 | μ=1.6 D mm | μ=0.5 D mm | μ=0.5 失败项 |','|---:|---|---|---:|---:|---|']
    for r in rows:lines.append(f"| {r['wave']:g} | {r['family']} | {'关闭' if r['self_off'] else '开启'} | {r['high_mu_D_mm']:.3f} | {r['low_mu_D_mm']:.3f} | {', '.join(r['low_mu_failed']) or '无'} |")
    lines+=['','wave1.5在保持内部碰撞开启时，降低摩擦使小body/大腿摆幅组合176.36→225.33mm，大body组合148.29→167.24mm；两者排名没有反转。因此在这两个固定组合中，摩擦差异本身不能解释原版的大body偏好。关闭内部碰撞后，大body组合在两种摩擦下都更快；但低摩擦小body组合有chassis_force失败，不能称作合格步态。',
            '', '这不是重新优化、不是contact erase结果，也不是两系统完全等价的证明。0.5是原版配置/默认材料源码链的参考值，尚无本次原版实际材料读回。24项零物理步接触字段检查与四组完整高摩擦复现通过，只有地面滑动分量发生变化。','']
    (ROOT/'friction_collision_diagnostic.md').write_text('\n'.join(lines),encoding='utf-8')
    receipt=dict(completed=True,diagnostic_only=True,selected_for_optimization=False,all_pd_sha256=list(hashes),
                 conditions=len(rows),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'comparison.json',DATA/'static_check.json',DATA/'priority_parity.json']})
    (DATA/'report_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
