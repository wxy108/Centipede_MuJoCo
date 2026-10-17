"""Two directly demonstrated causes, without pooling their experiments."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
caps=read(ROOT/'outputs/data/shape_geometry_audit.json')['ideal_caps']
pairs=read(ROOT/'outputs/self_collision_diagnostic/data/comparison.json')['comparisons']
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
rows=[r for r in caps if r['wave']<=4.5]
axes[0].plot([r['wave'] for r in rows],[r['ideal_body_amplitude_cap_deg'] for r in rows],'o-',color='#D55E00',lw=2,label='Old R <= 1.2 shape gate')
axes[0].axhline(36,color='#0072B2',ls='--',label='Shared physical command limit: 36 degrees')
axes[0].set(xlabel='Wave number',ylabel='Ideal body amplitude allowed (degrees)',ylim=(0,40),
            title='A. The old shape gate penalizes low waves')
axes[0].legend(loc='lower right',fontsize=8)
for i,r in enumerate(r for r in pairs if r['wave']==1.5):
    a=r['amplitude'];ys=[r['self_on_D'],r['self_off_D']]
    label=f"Body {a['body']:.2f} deg; legs {a['leg_yaw']:.2f}/{a['leg_pitch']:.2f} deg"
    axes[1].plot([0,1],ys,'o-',lw=2,ms=7,label=label,color=['#0072B2','#D55E00'][i])
    for x,y in zip([0,1],ys):axes[1].annotate(f'{y:.2f}',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
axes[1].set(xticks=[0,1],xticklabels=['Internal collisions ON','Internal collisions OFF'],xlim=(-.18,1.18),ylim=(135,190),
            ylabel='Full-window forward displacement (mm)',title='B. Collision semantics reverse the gait ranking')
axes[1].legend(loc='upper right',fontsize=7)
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
fig.suptitle('Documented causes of disagreement in amplitude optimization',fontsize=14)
fig.supxlabel('A: prescribed geometry only.  B: wave 1.5, identical PD and ground contacts; only internal collision filtering changes.\nNeither panel is an optimized amplitude trend or a contact-erasure result.',fontsize=9)
fig.savefig(ROOT/'root_cause_evidence.png',dpi=180);fig.savefig(ROOT/'root_cause_evidence.pdf');plt.close(fig)
