"""Explain low-wave timestep sensitivity; these runs do not enter optimization."""
import core as c
import validate as v

def main():
    path=c.DATA/'resolution_diagnostic_plan.json'
    if path.exists():plan=c.read(path)
    else:
        checked=c.read(c.DATA/'validation_index.json');plan=[]
        for w in [.5,1.]:
            rows=[r for r in checked.values() if r['nominal']['wave']==w and r['half']['full'] and not r['qualified']]
            if rows:plan.append(max(rows,key=lambda r:r['nominal']['D']))
        c.write(path,plan)
    tasks=[c.task(r['nominal']['wave'],v.vec(r['nominal']),'resolution_diagnostic',scale=.25) for r in plan]
    got={r['id']:r for r in c.bank(tasks,'resolution_diagnostic',2)};out=[]
    for prior,t in zip(plan,tasks):
        row=got[c.identity(t)];out.append(dict(wave=row['wave'],coarse=prior['nominal'],half=prior['half'],quarter=row,
            relative_half_quarter=abs(row['D']-prior['half']['D'])/abs(prior['half']['D']) if row['full'] else None,
            used_for_selection=False))
    c.write(c.DATA/'resolution_diagnostic.json',out)
    print([{k:r[k] for k in ['wave','relative_half_quarter']}|{'coarse_D':r['coarse']['D'],'half_D':r['half']['D'],'quarter_D':r['quarter']['D'],'quarter_failed':r['quarter']['failed']} for r in out],flush=True)

if __name__=='__main__':main()
