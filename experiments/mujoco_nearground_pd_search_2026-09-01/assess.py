"""All-gates classification, separate displacement objective, no weighted reward."""
import json
from collections import Counter
from run_search import ROOT,OUT

def assess_rows(mode='common'):
    rows=[json.loads(p.read_text()) for p in (OUT/mode).glob('*.json')]
    rows=[r for r in rows if 'config' in r]
    bases={(r['case']['wave'],r['duration'],r['dt']):r for r in rows if r['config']['name']=='baseline' and r.get('metrics')}
    assessed=[]
    for r in rows:
        gates=dict(r['gates']);m=r.get('metrics')
        if m:
            # V2 assurance added transparently after the first all-V1-gates
            # feasible point exposed that yaw tracking was reported but not gated.
            gates['leg_yaw_tracking']=m['leg_yaw_tracking_rms_deg']<=10.
            b=bases.get((r['case']['wave'],r['duration'],r['dt']))
            if b:
                b=b['metrics']
                gates['displacement_retention']=m['displacement_mm']>=.9*b['displacement_mm']
                gates['work_per_distance']=m['work_per_distance_n']<=1.25*b['work_per_distance_n']
                for key in ('loaded_foot_slip_p95_mm_s','left_right_force_imbalance_rms_bw','normal_impulse_10ms_bw_s','chassis_impulse_10ms_bw_s'):
                    gates['nonregression_'+key]=m[key]<=1.1*b[key]+1e-12
            else:gates['matching_baseline_available']=False
        else:gates['full_evaluation_complete']=False
        assessed.append(dict(r,gates=gates,feasible=all(gates.values()),failed=[k for k,v in gates.items() if not v]))
    return assessed

def main():
    rows=assess_rows();summary={}
    for w in sorted({r['case']['wave'] for r in rows}):
        group=[r for r in rows if r['case']['wave']==w]
        full=[r for r in group if r.get('metrics')]
        feasible=sorted([r for r in full if r['feasible']],key=lambda r:-r['metrics']['displacement_mm'])
        summary[f'{w:g}']=dict(trials=len(group),full=len(full),proven_infeasible=len(group)-len(full),
            feasible=len(feasible),best_feasible=feasible[0]['id'] if feasible else None,
            diagnostic_full=sorted([dict(id=r['id'],D=r['metrics']['displacement_mm'],failed=r['failed']) for r in full],key=lambda r:(len(r['failed']),-r['D']))[:5],
            first_failure_counts=dict(Counter(g for r in group for g in r.get('gates',{}) if not r['gates'][g])))
    (OUT/'assessment.json').write_text(json.dumps(rows,indent=2))
    (OUT/'search_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
