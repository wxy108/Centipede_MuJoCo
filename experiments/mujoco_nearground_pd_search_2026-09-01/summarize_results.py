"""Write a reproducible search ledger; never promote failed diagnostic candidates."""
import hashlib
import json
import platform
from pathlib import Path
from run_search import ROOT,OUT,OLD,pilot,opt,mujoco
from assess import assess_rows

WAVES=[.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9]

def main():
    rows=assess_rows();nominal=[r for r in rows if r['duration']==6 and abs(r['dt']-.00025)<1e-12]
    full=[r for r in nominal if r.get('metrics')];feasible=[r for r in full if r['feasible']]
    plans=[(None,WAVES),('broad_bank.json',WAVES),('coordinated_bank.json',WAVES),('basin15_bank.json',[1,2,2.5,3]),('basin18_bank.json',[2,2.5]),('basin2906_bank.json',[3.5,4.5]),('legyaw_boundary_bank.json',[2.5]),('legyaw_refine_bank.json',[2.5])]
    expected=set()
    for name,waves in plans:
        names=['baseline','p4_leg_k2_yaw_d2_roll_d4'] if name is None else [c['name'] for c in json.loads((ROOT/name).read_text())]
        for c in names:
            for w in waves:expected.add(f'{c}_w{w:g}_t6_dt0.00025'.replace('.','p'))
    completed={r['id'] for r in nominal}
    best={};diagnostics={}
    for w in WAVES:
        candidates=[r for r in feasible if r['case']['wave']==w]
        best[f'{w:g}']=max(candidates,key=lambda r:r['metrics']['displacement_mm']) if candidates else None
        group=[r for r in full if r['case']['wave']==w]
        diagnostics[f'{w:g}']=sorted(group,key=lambda r:(len(r['failed']),-r['metrics']['displacement_mm']))[:3]
    manifest=dict(status='registered_batch_complete' if expected<=completed else 'running_incomplete',
        robustness_certified=False,default_pd_replaced=False,
        planned_nominal=len(expected),completed_nominal=len(completed&expected),missing=sorted(expected-completed),
        full_nominal=len(full),proven_infeasible_nominal=len(nominal)-len(full),feasible_nominal=len(feasible),
        continuous_global_proof=False,fixed_amplitudes_deg={k:v for k,v in pilot.wave_case(2.5).items() if k!='wave'},
        wave_numbers=WAVES,best_feasible=best,diagnostic_candidates=diagnostics)
    (OUT/'final_selection.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'assessment.json').write_text(json.dumps(rows,indent=2))
    sources=[opt.MODEL,OLD/'run_pd_pilot.py',ROOT/'run_search.py',ROOT/'PROTOCOL.md',
        pilot.SOURCE/'outputs/data/final_certified_selection.json',opt.ROOT/'run_wave_amplitude_global.py']
    provenance=dict(python=platform.python_version(),mujoco=mujoco.__version__,
        files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        scope='Centipede Model only; no Robot writes',amplitude_certification='values reused; historical certification not inherited')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('best_feasible','diagnostic_candidates','missing')},indent=2))
    for w in WAVES:
        b=best[f'{w:g}'];d=diagnostics[f'{w:g}']
        if b:print(f'wave={w:g} FEASIBLE {b["config"]["name"]} D={b["metrics"]["displacement_mm"]:.3f}')
        elif d:print(f'wave={w:g} no feasible; diagnostic={d[0]["config"]["name"]} failures={d[0]["failed"]}')

if __name__=='__main__':main()
