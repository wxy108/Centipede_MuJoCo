"""Regression for the full selected-crossing write path; synthetic, no physics."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,tempfile
import analyze_contact as ac

ROOT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'outputs/amendments/discovery_metadata_20260909_1007'


def exercise(module,expect_failure):
    with tempfile.TemporaryDirectory(prefix='synthetic_discovery_',dir=ROOT/'outputs') as name:
        folder=Path(name);out=folder/'out';data=out/'data'
        data.mkdir(parents=True);(data/'completed').mkdir();(out/'assessments').mkdir()
        module.ROOT=folder;module.OUT=out;module.DATA=data
        manifest=dict(profiles={'1.0':dict(amplitude=dict(body=30.)), '2.0':dict(amplitude=dict(body=20.))},
            probabilities=[0.,.2],discovery_seeds=[0,1,2,3],confirmation_seeds=list(range(100,120)))
        module.write(data/'manifest.json',manifest);ids=[]
        for w in (1.,2.):
            for p in (0.,.2):
                for seed in ([0] if p==0 else [0,1,2,3]):
                    rid=f'w{w}_p{p}_s{seed}';ids.append(rid)
                    record=folder/f'{rid}.json';module.write(record,dict(channel=dict(actual_erased_fraction=p)))
                    score={(1.,0.):12.,(2.,0.):10.,(1.,.2):4.,(2.,.2):8.}[w,p]
                    row=dict(id=rid,wave=w,task=dict(erasure=dict(probability=p,seed=seed)),full=True,D=score,
                        feasible=True,metrics=dict(min_upright=1.,chassis_contact_fraction=0.),record=str(record))
                    module.write(out/'assessments'/f'{rid}.json',row)
        module.write(data/'completed/discovery.json',dict(ids=ids))
        try:module.discovery()
        except TypeError as error:
            assert expect_failure and "multiple values for keyword argument 'interpretation'" in str(error)
            result=module.read(data/'discovery_analysis.json')
            assert not (data/'crossing_choice.json').exists()
            return result,str(error)
        assert not expect_failure
        result=module.read(data/'discovery_analysis.json');choice=module.read(data/'crossing_choice.json')
        assert choice['low_wave']==1 and choice['high_wave']==2 and choice['bracket']==[0.,.2]
        assert choice['discovery_differences_mm']==[2.,-4.]
        assert choice['confirmation_seeds']==list(range(100,120))
        assert choice['interpretation']=='Frozen before new seeds; no guarantee of confirmation.'
        assert (folder/'contact_discovery.png').is_file()
        return result,None


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expect-original-failure',action='store_true');args=parser.parse_args()
    original=ARCHIVE/'analyze_contact.py'
    spec=importlib.util.spec_from_file_location('archived_contact_analysis',original)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    before,error=exercise(old,True)
    if args.expect_original_failure:print('Original selected-crossing writer reproduced:',error)
    else:
        after,_=exercise(ac,False)
        before.pop('code_sha256');after.pop('code_sha256');assert before==after
        receipt=dict(passed=True,physics_steps=0,synthetic_only=True,original_error=error,
            all_summary_candidates_and_selection_identical=True,successful_choice_and_figure_write=True,
            source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [original,Path(ac.__file__),Path(__file__)]})
        (ARCHIVE/'serialization_regression.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print('PASS: old failure reproduced; new full write path and figure passed; all computed statistics and selection identical.')
