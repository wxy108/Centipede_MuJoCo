from pathlib import Path
R=Path(__file__).resolve().parent
src=(R/'common_contact_v2.py').read_text()
start=src.index('        for j,verts in enumerate(self.mesh_verts):')
end=src.index('        arm=point-com[:,self.ids]',start)
src=src[:start]+'''        # Only the support vertex needs a full xyz transform. Compute the same
        # z dot products for all vertices together, then transform selected xyz.
        z=np.einsum('nbj,bvj->nbv',rotation[:,self.mesh_body,2,:],self.padded_verts)+position[:,self.mesh_body,2,None]
        z=np.where(self.vertex_valid[None],z,np.inf)
        index=np.argmin(z,axis=2)
        selected=self.padded_verts[np.arange(len(self.meshes))[None],index]
        point[:,self.nr:]=np.einsum('nbij,nbj->nbi',rotation[:,self.mesh_body],selected)+position[:,self.mesh_body]
        self.xi[:,self.nr:][index!=self.mesh_index]=0.
        self.mesh_index[:]=index
'''+src[end:]
line='        self.xi=np.zeros((num_envs,self.np,2));self.mesh_index=np.full((num_envs,len(self.meshes)),-1,int)'
src=src.replace(line,line+'''
        self.mesh_body=self.ids[self.nr:];nv=max(len(v) for v in self.mesh_verts)
        self.padded_verts=np.zeros((len(self.meshes),nv,3));self.vertex_valid=np.zeros((len(self.meshes),nv),bool)
        for i,v in enumerate(self.mesh_verts):self.padded_verts[i,:len(v)]=v;self.vertex_valid[i,:len(v)]=True
''')
(R/'common_contact_fast.py').write_text(src)
