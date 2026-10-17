"""A matched fixed-PD replay; execution context may change, physics may not."""
import time
import physical_search as ps
import core as c


if __name__=='__main__':
    expected=c.read(ps.SOURCE_DATA/'historical_amplitude_anchors.json')['recovery']['1.5']
    source=next(r for r in c.nominal('midpoint') if r['wave']==1.5 and all(abs(r['amplitudes'][k]-v)<1e-10 for k,v in zip(c.KEYS,expected)))
    reference_manifest=c.read(c.DATA/'manifest.json')
    c.OUT=c.ROOT/'outputs/host_context_parity';c.DATA=c.OUT/'data';c.e.TRIALS=c.OUT/'trials'
    c.write(c.DATA/'manifest.json',reference_manifest)
    t=c.task(1.5,expected,'host_context_parity',phase='midpoint',trace=True,prefix=False)
    wall=time.monotonic();cpu=time.process_time();result=c.run(t)
    checks={k:c.np.array_equal(c.np.asarray(v),c.np.asarray(result['metrics'][k]),equal_nan=True) for k,v in source['metrics'].items()}
    receipt=dict(passed=all(checks.values()) and result['D']==source['D'] and result['feasible'],
        reference=source['record'],result=result['record'],D_difference_mm=result['D']-source['D'],
        common_metrics_exact=all(checks.values()),wall_seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,
        context='Normal-host execution of unchanged same-environment source; diagnostic only, not a new optimizer candidate.')
    c.write(c.DATA/'receipt.json',receipt);print(receipt,flush=True);assert receipt['passed']
