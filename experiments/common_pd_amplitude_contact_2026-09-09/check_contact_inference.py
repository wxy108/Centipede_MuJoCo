"""Synthetic pre-outcome checks of pairing and failure accounting; zero physics."""
from pathlib import Path
import hashlib,json,tempfile
import numpy as np
import analyze_contact as ac

ROOT=Path(__file__).resolve().parent
def main():
    checks=[]
    # Shared mask variability must cancel in a paired difference. Independent
    # resampling of each gait would wrongly leave the common noise in its CI.
    common=np.linspace(-100.,100.,20)
    delta=(common+10.)-(common+7.)
    np.testing.assert_allclose(ac.boot(delta,.025),[3.,3.],atol=1e-12)
    assert ac.boot(common+10.,.025)[0]<0<ac.boot(common+10.,.025)[1]
    checks.append('paired difference cancels shared mask variation')
    parent=ROOT/'outputs/inference_static_checks';parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='synthetic_',dir=parent) as name:
        folder=Path(name);rows=[]
        for seed,D in enumerate([10.,20.,-5.,-25.]):
            path=folder/f'{seed}.json'
            path.write_text(json.dumps(dict(channel=dict(actual_erased_fraction=.25))),encoding='utf-8')
            rows.append(dict(id=f'synthetic_{seed}',wave=2.,task=dict(erasure=dict(probability=.4,seed=seed)),
                             full=True,D=D,feasible=seed<2,metrics=dict(min_upright=1. if seed<2 else .4,
                             chassis_contact_fraction=0. if seed<2 else .8),record=str(path)))
        result=ac.summarize(rows,2.,.4,[0,1,2,3])
        assert result['complete'] and result['mean_D']==0. and result['engineering_fraction']==.5
        assert result['upright_fraction']==.5 and result['values']==[10.,20.,-5.,-25.]
        checks.append('negative and failed full outcomes remain in unconditional mean')
        result=ac.summarize(rows[:-1],2.,.4,[0,1,2,3])
        assert not result['complete'] and 'mean_D' not in result and result['available']==3
        rows[-1]['full']=False;rows[-1]['D']=None
        result=ac.summarize(rows,2.,.4,[0,1,2,3])
        assert not result['complete'] and 'mean_D' not in result
        checks.append('missing and numerically incomplete outcomes cannot become survivor means')
        rows[0]['task']['erasure']['probability']=0.
        result=ac.summarize(rows,2.,0.,list(range(100,120)))
        assert result['complete'] and result['values']==[10.] and result['expected_seeds']==[0]
        checks.append('deterministic endpoint uses its sole seed-zero baseline')
    receipt=dict(passed=True,synthetic_only=True,physics_steps=0,checks=checks,
                 source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(ac.__file__)]})
    (parent/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__':main()
