"""Record already-complete failed cases without prefix truncation, for diagnosis."""
from pathlib import Path
import core as c
import screening
from report import vector
from render import make

def main():
    rows=c.read(c.DATA/'unresolved_diagnostic_plan.json');checks=[]
    # This disables only the optional early exit. The unchanged full policy still
    # assesses every final record and failed gaits remain explicitly unqualified.
    screening.install(False);c._guard_installed=True
    for old in rows:
        assert old['full']
        r=c.run(c.task(old['wave'],vector(old),'unresolved_motion',trace=True))
        assert r['D']==old['D'] and r['metrics']==old['metrics'] and r['feasible']==old['feasible']
        assert (Path(r['record']).parent/'motion.npz').exists()
        make([(r,'UNRESOLVED WAVE / NOMINAL DIAGNOSTIC',False)],f"unresolved_wave{r['wave']:g}")
        checks.append(dict(id=r['id'],metrics_exact=True,record=r['record'],nominal_failed=r['failed']))
    c.write(c.DATA/'unresolved_motion_verification.json',dict(passed=True,cases=checks,never_promoted=True))

if __name__=='__main__':main()
