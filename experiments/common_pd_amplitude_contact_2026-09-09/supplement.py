"""Complete 32 shared Sobol points per wave in the physical-policy branch."""
import argparse
import physical_search as ps
import core as c
import search as s

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    assert c.read(c.DATA/'physical_parity_check.json')['passed']
    points=s.qmc.Sobol(3,scramble=True,seed=90920261).random_base2(5)[16:]*c.HIGH
    declaration=dict(code_sha256=c.sha(__file__),seed=90920261,sobol_indices=list(range(16,32)),
        scope='Identical additional coverage at every wave; joint three-amplitude space; no curve-dependent sampling.')
    path=c.DATA/'supplement_sobol_declaration.json'
    if path.exists():assert c.read(path)==declaration
    else:c.write(path,declaration)
    tasks=[c.task(w,x,'supplement_sobol_midpoint',phase='midpoint') for w in c.WAVES for x in points]
    c.bank(tasks,'supplement_sobol_midpoint',a.workers);s.snapshot('midpoint')
