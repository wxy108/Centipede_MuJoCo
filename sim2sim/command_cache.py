"""Bounded command blocks, mathematically identical to shared.targets."""
import numpy as np
import shared as s
class CommandCache:
    def __init__(self,names,dt,rows,block=512):self.names=names;self.dt=dt;self.rows=rows;self.block=block;self.start=-block
    def positions(self,t):
        t=np.asarray(t);out=np.zeros((len(t),len(self.rows),len(self.names)),float)
        for e,row in enumerate(self.rows):
            w=row['wave'];amps=np.radians(row['amplitudes']);start=1+np.arange(19)*.125/18
            blend=.5*(1-np.cos(np.pi*np.clip(t[:,None]-start[None,:],0,.125)/.125));phase=2*np.pi*t[:,None]-2*np.pi*w*np.arange(19)[None,:]/18
            for k,n in enumerate(self.names):
                if n=='body_pitch_0':out[:,e,k]=np.radians(s.C['head']['pitch_target_deg'])*blend[:,0]
                elif n.startswith('body_yaw_'):
                    seg=int(n.rsplit('_',1)[1])-1
                    if seg:out[:,e,k]=blend[:,seg]*amps[0]*np.sin(phase[:,seg])
                elif n.startswith('legbase_'):
                    _,typ,seg,side=n.split('_');seg=int(seg);j=0 if typ=='yaw' else 1
                    out[:,e,k]=blend[:,seg]*amps[j+1]*(1 if side=='L' else -1)*np.sin(phase[:,seg]+np.pi*w/18+[-np.pi/2,np.pi][j])
        return out
    def get(self,k):
        if not self.start<=k<self.start+self.block:
            self.start=k//self.block*self.block;t=np.arange(self.start,self.start+self.block)*self.dt
            q=self.positions(t);v=(q-self.positions(t-self.dt))/self.dt
            if self.start==0:v[0]=0
            self.q=q.astype(np.float32);self.v=v.astype(np.float32)
        return self.q[k-self.start],self.v[k-self.start]
if __name__=='__main__':
    names=[x['name'] for x in s.M['joints']];worst=[0.,0.]
    for dt in [.00025,.000125,.0000625,.00003125]:
        for bank in ['calibration','holdout']:
            rows=s.cases(bank);c=CommandCache(names,dt,rows)
            for k in np.unique(np.r_[0,1,np.arange(round(.99/dt),round(1.26/dt),113),np.linspace(0,6/dt,100,dtype=int)]):
                old=s.targets(names,k*dt,dt,rows);new=c.get(k)
                for i in range(2):worst[i]=max(worst[i],float(np.abs(old[i]-new[i]).max()))
    assert max(worst)<1e-6,worst
    s.write(s.ROOT/'audit/command_cache_equivalence.json',dict(max_position_error_rad=worst[0],max_velocity_error_rad_s=worst[1],source_sha=s.sha(__file__)))
    print(worst)
