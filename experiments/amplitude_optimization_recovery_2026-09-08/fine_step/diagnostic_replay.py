"""Record already-complete failed cases without prefix truncation, for diagnosis."""
from pathlib import Path
import core as c
import screening
from report import vector
from render import make

def main():
    rows=c.read(c.DATA/'unresolved_diagnostic_plan.json');checks=[];groups={}
    # This disables only the optional early exit. The unchanged full policy still
    # assesses every final record and failed gaits remain explicitly unqualified.
    screening.install(False);c._guard_installed=True
    for old in rows:
        assert old['full']
        r=c.run(c.task(old['wave'],vector(old),'unresolved_motion',trace=True))
        assert r['D']==old['D'] and r['metrics']==old['metrics'] and r['feasible']==old['feasible']
        assert (Path(r['record']).parent/'motion.npz').exists()
        groups.setdefault(r['wave'],[]).append((r,old['diagnostic_label'],False))
        checks.append(dict(id=r['id'],metrics_exact=True,record=r['record'],nominal_failed=r['failed'],
            diagnostic_kind=old['diagnostic_kind'],diagnostic_max_ratio=old['diagnostic_max_ratio']))
    for wave,cases in groups.items():make(cases,f'unresolved_wave{wave:g}')
    c.write(c.DATA/'unresolved_motion_verification.json',dict(passed=True,cases=checks,never_promoted=True))

if __name__=='__main__':main()
