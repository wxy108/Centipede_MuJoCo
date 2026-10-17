"""Held-out paired confirmation and separately declared robustness checks."""
import argparse
import contact_erase as ce
c=ce.c

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['confirm','step_duration']);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    choice=c.read(c.DATA/'crossing_choice.json');waves=[choice['low_wave'],choice['high_wave']]
    if a.action=='confirm':
        declaration=dict(choice_sha256=c.sha(c.DATA/'crossing_choice.json'),source_sha256=c.sha(__file__),
            seeds=choice['confirmation_seeds'],probabilities=choice['confirmation_probabilities'])
        path=c.DATA/'confirmation_declaration.json'
        if path.exists():assert c.read(path)==declaration
        else:c.write(path,declaration)
        tasks=[ce.task(w,p,seed,'confirmation') for p in declaration['probabilities'] for seed in declaration['seeds'] for w in waves]
        c.bank(tasks,'confirmation',a.workers)
    else:
        analysis=c.read(c.DATA/'confirmation_analysis.json');assert analysis['confirmed_at_nominal_step']
        declaration=dict(choice_sha256=c.sha(c.DATA/'crossing_choice.json'),source_sha256=c.sha(__file__),
            analysis_sha256=c.sha(c.ROOT/'assess_contact_robustness.py'),
            seeds=list(range(100,108)),probabilities=choice['bracket'],conditions=[dict(duration=6,scale=.25),dict(duration=12,scale=.5)],
            interpretation='Same first eight held-out masks, declared before robustness runs; no selection using their individual outcomes.')
        path=c.DATA/'step_duration_declaration.json'
        if path.exists():assert c.read(path)==declaration
        else:c.write(path,declaration)
        tasks=[ce.task(w,p,seed,'step_duration',**condition) for condition in declaration['conditions'] for p in declaration['probabilities'] for seed in declaration['seeds'] for w in waves]
        c.bank(tasks,'step_duration',a.workers)
