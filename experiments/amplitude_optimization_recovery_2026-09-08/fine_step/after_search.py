"""Continue the active experiment through initial ranked final validation."""
import time,psutil
import core as c
import validate

def main():
    owner=c.read(c.DATA/'search_process.json')
    while not (c.DATA/'local_complete.json').exists():
        try:
            p=psutil.Process(owner['pid'])
            assert p.create_time()==owner['create_time'] and p.cmdline()==owner['cmdline'] and p.cwd()==str(c.ROOT)
        except (psutil.NoSuchProcess,AssertionError) as ex:
            c.write(c.DATA/'continuation_error.json',dict(error='Owned search ended before its completion receipt',detail=repr(ex)))
            raise RuntimeError('Search stopped before local completion') from ex
        time.sleep(20)
    for name in ['warm','warm_validation','broad','adaptive_00','adaptive_01','adaptive_02']:
        assert (c.DATA/'completed'/f'{name}.json').exists(),name
    validate.final(12)
    c.write(c.DATA/'initial_final_validation_complete.json',dict(completed=True,requires_final_evidence_review=True))
    print('Initial ranked final validation complete; ready for evidence review and trajectory delivery.',flush=True)

if __name__=='__main__':main()
