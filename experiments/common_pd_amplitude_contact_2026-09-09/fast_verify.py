import json
import core as c
import fast_metrics
fast_metrics.install()
ORIGINAL_OUT=c.OUT
c.OUT=c.ROOT/'outputs/fast_verification'
c.e.TRIALS=c.OUT/'trials'

if __name__=='__main__':
    originals=[c.read(p) for p in (ORIGINAL_OUT/'assessments').glob('*.json') if c.read(p)['task']['stage']=='verify']
    assert len(originals)==3
    rows=c.bank([dict(r['task'],stage='fast_verification') for r in originals],'fast_verification',3)
    checks=[]
    for r in rows:
        old=next(x for x in originals if x['id']==r['id'])
        exact_D=r['D']==old['D']
        differences={k:float(c.np.max(c.np.abs(c.np.asarray(v)-c.np.asarray(r['metrics'][k])))) for k,v in old['metrics'].items()
            if not c.np.allclose(c.np.asarray(v),c.np.asarray(r['metrics'][k]),rtol=1e-12,atol=1e-12,equal_nan=True)}
        checks.append(dict(wave=r['wave'],exact_D=exact_D,material_metric_differences=differences,
            same_gates=r['failed']==old['failed'],old_wall_s=old['wall_s'],new_wall_s=r['wall_s']))
    passed=all(x['exact_D'] and not x['material_metric_differences'] and x['same_gates'] for x in checks)
    result=dict(passed=passed,checks=checks,observer_sha256=c.sha(c.ROOT/'fast_metrics.py'),evaluator_sha256=c.sha(c.ROOT/'fast_run_task.py'))
    c.write(c.DATA/'fast_trajectory_check.json',result);print(json.dumps(result,indent=2));assert passed
