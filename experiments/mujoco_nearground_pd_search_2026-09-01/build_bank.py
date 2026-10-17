"""Two independent scrambled Sobol replicates, in log2 physical-gain multipliers."""
import argparse
import json
from run_search import ROOT,np,cfg,PREVIOUS
from scipy.stats import qmc

KEYS=('yaw_k','yaw_d','pitch_k','pitch_d','leg_yaw_k','leg_yaw_d','leg_pitch_k','leg_pitch_d','roll_k','roll_d')

def main():
    p=argparse.ArgumentParser();p.add_argument('--power',type=int,default=5);a=p.parse_args()
    lower=np.full(10,-2.);upper=np.array([4. if k.endswith('_k') else 5. for k in KEYS])
    bank=[]
    for seed in (1901,2901):
        points=qmc.Sobol(d=10,scramble=True,seed=seed).random_base2(a.power)
        for i,point in enumerate(points):
            values=2**(lower+point*(upper-lower))
            bank.append(cfg(f'sobol_s{seed}_{i:04d}',**dict(zip(KEYS,map(float,values)))))
    for key in KEYS:
        for endpoint in (-2.,4. if key.endswith('_k') else 5.):
            c=dict(PREVIOUS,name=f'axis_{key}_{2**endpoint:g}')
            c[key]=2**endpoint;bank.append(c)
    for factor in (2.,4.,8.,16.):
        bank.append(cfg(f'uniform_{factor:g}',**{key:factor for key in KEYS}))
    (ROOT/'broad_bank.json').write_text(json.dumps(bank,indent=2))
    (ROOT/'search_domain.json').write_text(json.dumps(dict(keys=KEYS,seeds=[1901,2901],sobol_per_seed=2**a.power,
        bounds={key:[.25,2**float(hi)] for key,hi in zip(KEYS,upper)},continuous_global_proof=False),indent=2))
    print(f'{len(bank)} configurations, 10 independent gain multipliers')

if __name__=='__main__':main()
