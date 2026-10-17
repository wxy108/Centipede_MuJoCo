"""Held-out paired transient confirmation, separate from persistent loss."""
import argparse
import contact_erase_transient as tr
c=tr.c

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['confirm','step_duration']);p.add_argument('--workers',type=int,default=24);a=p.parse_args()
    choice=c.read(c.DATA/'crossing_choice.json');waves=[choice['low_wave'],choice['high_wave']]
    declaration=dict(choice_sha256=c.sha(c.DATA/'crossing_choice.json'),source_sha256=c.sha(__file__))
    if a.action=='confirm':
        declaration.update(seeds=choice['confirmation_seeds'],probabilities=choice['confirmation_probabilities'])
        path=c.DATA/'confirmation_declaration.json';bank='confirmation'
        tasks=[tr.task(w,p,seed,bank) for p in declaration['probabilities'] for seed in declaration['seeds'] for w in waves]
    else:
        assert c.read(c.DATA/'confirmation_analysis.json')['confirmed_at_nominal_step']
        declaration.update(analysis_sha256=c.sha(c.ROOT/'assess_contact_robustness.py'),wrapper_sha256=c.sha(c.ROOT/'transient_analysis.py'),
            seeds=list(range(100,108)),probabilities=choice['bracket'],conditions=[dict(duration=6,scale=.25),dict(duration=12,scale=.5)],
            interpretation='Same first eight held-out masks; long schedule shares the nominal prefix. No selection on individual held-out results.')
        path=c.DATA/'step_duration_declaration.json';bank='step_duration'
        tasks=[tr.task(w,p,seed,bank,**cond) for cond in declaration['conditions'] for p in declaration['probabilities'] for seed in declaration['seeds'] for w in waves]
    if path.exists():assert c.read(path)==declaration
    else:c.write(path,declaration)
    c.bank(tasks,bank,a.workers)
