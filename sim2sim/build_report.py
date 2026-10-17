from pathlib import Path
import json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shared as s
F=s.ROOT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':9,'axes.edgecolor':'black','axes.linewidth':.8,'axes.axisbelow':True,'grid.color':'#b0b0b0','grid.linewidth':.8,'figure.facecolor':'white','axes.facecolor':'white','pdf.fonttype':42,'svg.fonttype':'path'})
colors=['#237f8d','#d95f02','#5b8c5a']
def read(name):
    p=s.ROOT/'runs'/name/'results.json'
    return json.loads(p.read_text()) if p.exists() else None
def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(F/(name+'.'+ext),dpi=240)
    plt.close(fig)
cal=[]
for name in ['mu_calibration_dt250_000','mu_calibration_dt125_000','mu_calibration_dt62_000','mu_calibration_dt31_000','native_calibration_cpu_dt250_000','native_calibration_cpu_dt125_fk_000','native_calibration_gpu_dt125_000','native_validation_cpu_dt62_000','native_validation_cpu_dt31_000']:
    r=read(name)
    if r:cal.append(dict(run=name,engine='MuJoCo' if r['engine']=='mujoco' else 'PhysX GPU' if 'cuda' in r['device'] else 'PhysX CPU',dt_us=r['arguments']['dt']*1e6,D_mm=r['rows'][0]['D_mm']))
fig,ax=plt.subplots(figsize=(7.3,4.4),layout='constrained')
steps=[250,125,62.5,31.25]
for j,engine in enumerate(['MuJoCo','PhysX CPU','PhysX GPU']):
    rr=sorted([r for r in cal if r['engine']==engine],key=lambda r:-r['dt_us'])
    if rr:ax.plot([steps.index(r['dt_us']) for r in rr],[r['D_mm'] for r in rr],marker=['o','s','^'][j],color=colors[j],label=engine)
ax.set_xticks(range(4),['250','125','62.5','31.25']);ax.set_ylim(0,200);ax.set_yticks(np.arange(0,201,25));ax.grid(axis='y',alpha=.18);ax.set_xlabel('Physics time step (µs)');ax.set_ylabel('Forward displacement, 2–6 s (mm)');ax.set_title('Same physical model, PD and gait | wave = 3.0');ax.legend(loc='lower right');save(fig,'01_calibration_convergence')
native={label:read(f'native_validation_cpu_{label}_000') for label in ['dt62','dt31']}
hold=[]
for i,c in enumerate(s.cases('holdout')):
    mu={label:read(f'mu_holdout_{label}_case{i:02d}_000') for label in ['dt62','dt31']}
    row=dict(index=i,**c)
    for label in ['dt62','dt31']:
        row['mu_'+label]=mu[label]['rows'][0]['D_mm'] if mu[label] else None
        row['native_'+label]=native[label]['rows'][i+1]['D_mm'] if native[label] else None
    def rel(a,b):return abs(a-b)/max((abs(a)+abs(b))/2,10) if a is not None and b is not None else None
    row['gap']=rel(row['mu_dt62'],row['native_dt62'])
    for eng in ['mu','native']:
        coarse,fine=row[eng+'_dt62'],row[eng+'_dt31'];row[eng+'_convergence']=abs(fine-coarse)/max(abs(coarse),10) if coarse is not None and fine is not None else None
    row['passed']=row['gap'] is not None and row['gap']<=.1 and all(row[x+'_convergence'] is not None and row[x+'_convergence']<=.05 for x in ['mu','native']) and min(row['mu_dt62'],row['native_dt62'])>=50
    hold.append(row)
if any(r['gap'] is not None for r in hold):
    fig,axs=plt.subplots(1,2,figsize=(10,4.3),layout='constrained',sharex=True,sharey=True)
    maximum=max(r[k] for r in hold for k in ['mu_dt62','native_dt62'] if r[k] is not None);top=max(200,math.ceil(maximum/50)*50)
    for family,ax in enumerate(axs):
        rr=hold[family::2]
        for j,key in enumerate(['mu_dt62','native_dt62']):ax.plot([r['wave'] for r in rr],[r[key] if r[key] is not None else np.nan for r in rr],marker=['o','s'][j],color=colors[j],label=['MuJoCo','PhysX CPU'][j])
        ax.set_title('Amplitudes = '+str(rr[0]['amplitudes'])+'°');ax.set_xlim(1.3,4.7);ax.set_xticks([1.5,2.5,3.5,4.5]);ax.set_ylim(0,top);ax.grid(axis='y',alpha=.18);ax.set_xlabel('Wave number');ax.legend()
    axs[0].set_ylabel('Forward displacement, 2–6 s (mm)');save(fig,'02_holdout_performance')
    fig,ax=plt.subplots(figsize=(8.5,4.2),layout='constrained');xx=np.arange(8);values=[100*r['gap'] if r['gap'] is not None else np.nan for r in hold]
    ax.bar(xx,values,color=[colors[0] if r['passed'] else '#b04b4b' for r in hold]);ax.axhline(10,color='black',ls='--',lw=1,label='10% gap threshold');ax.set_xticks(xx,[f"{r['wave']}\n{'A' if r['index']%2==0 else 'B'}" for r in hold]);ax.set_ylabel('Cross-engine relative gap (%)');ax.set_xlabel('Wave number / amplitude family');ax.grid(axis='y',alpha=.18);ax.legend();save(fig,'03_holdout_gap')
table=['| 波数 | 振幅组 | MuJoCo (mm) | PhysX CPU (mm) | 差距 | Mu 步长变化 | PhysX 步长变化 | 通过 |','|---|---|---:|---:|---:|---:|---:|---|']
def num(x,pct=False):return '待完成' if x is None else f'{x*100:.2f}%' if pct else f'{x:.2f}'
for r in hold:table.append(f"| {r['wave']} | {'A' if r['index']%2==0 else 'B'} | {num(r['mu_dt62'])} | {num(r['native_dt62'])} | {num(r['gap'],True)} | {num(r['mu_convergence'],True)} | {num(r['native_convergence'],True)} | {'是' if r['passed'] else '尚未通过'} |")
text='''# Centipede sim-to-sim 对齐审计与验证

物理参数、关节运动学及共享控制指令已对齐。性能验收状态以以下完整实验为准；未完成的测试不能计作通过。

本研究使用独立的共同模型：从原始 Isaac Lab 刚性足端资产生成 MuJoCo 模型，保留逐刚体质量、惯量、COM、关节轴与碰撞几何。原模型、历史优化结果及 Clean edition 均未修改。

## 已确认的原系统差异

总质量几乎相同掩盖了质量分布差异：原生模型尾部重约 22.9%，单条完整腿重约 14.1%；部分躯干惯量也明显不同。旧系统的摩擦系数为 1.6 对 0.5，自碰撞开关、头部自由度、接触偏移、地面、控制起始过程以及计分窗口也不同。因此历史 PD 最优值和位移数值不能直接比较。

共同模型的运行时质量、惯量和 PD 误差均在单精度舍入量级，非零姿态下的位置误差小于 0.07 微米。详见 [审计](AUDIT.md)、[运行时核对](RUNTIME_ALIGNMENT.md) 与 [共享物理合同](COMMON_CONTRACT.json)。

## 校准步态的步长收敛

'''
text+='| 引擎 | 步长 (µs) | 2–6 s 位移 (mm) |\n|---|---:|---:|\n'+'\n'.join(f"| {r['engine']} | {r['dt_us']} | {r['D_mm']:.3f} |" for r in cal)
text+='\n\n![收敛](figures/01_calibration_convergence.png)\n\n## 未用于选择设置的步态验证\n\nA = [32,18,26]°；B = [16,72,24]°，依次为 body yaw、leg yaw、leg pitch。所有波数共享 PD，固定 1 Hz；采用 62.5 µs 步长，以 31.25 µs 检查收敛。\n\n'+'\n'.join(table)
text+='\n\n差距 = |D₁−D₂| / max((|D₁|+|D₂|)/2, 10 mm)。通过还要求两边均前进至少 50 mm、步长减半变化均不超过 5%、仿真完整且无物理失败。当前表尚不单独构成存活/接触质量证明，需结合原始轨迹审查。\n\nGPU/CPU 等价性单独验证；CPU 结果不自动代表 GPU 结果。所有失败及初始不收敛结果保留，未调整 PD 或振幅以压低 gap。\n'
(s.ROOT/'REPORT.md').write_text(text,encoding='utf-8');s.write(s.ROOT/'audit/current_validation_summary.json',dict(calibration=cal,holdout=hold,all_holdout_passed=all(r['passed'] for r in hold)))
print(json.dumps(dict(calibration=cal,holdout=hold),indent=2))
