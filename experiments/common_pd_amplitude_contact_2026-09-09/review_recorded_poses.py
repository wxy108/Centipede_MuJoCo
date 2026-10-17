"""CPU-only projected geometry snapshots from saved poses; no physics steps."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection,LineCollection
from scipy.spatial import ConvexHull
import physical_search as ps
import core as c


def review(assessment, name):
    assert name.replace('_','').isalnum()
    row=c.read(assessment);assert row['full']
    record=c.read(row['record']);folder=Path(row['record']).parent
    with np.load(folder/'contact_trajectory.npz') as z: saved={k:z[k].copy() for k in z.files}
    model=c.e.runtime_model(c.PLANTS['current']);data=c.mujoco.MjData(model)
    index=c.e.fx.opt.tip.LeanIndex(model)
    body_ids=np.asarray(index.chassis_body_ids)
    body_set=set(body_ids)
    leg_of_body={}
    for pair in range(19):
        for side_index,side in enumerate(('L','R')):
            root=c.mujoco.mj_name2id(model,c.mujoco.mjtObj.mjOBJ_BODY,f'link_leg_{pair}_{side}_0')
            for b in range(1,model.nbody):
                parent=b
                while parent>0:
                    if parent==root:leg_of_body[b]=2*pair+side_index;break
                    parent=int(model.body_parentid[parent])
    pairs=[(int(model.body_parentid[b]),b,leg) for b,leg in leg_of_body.items()]
    meshes=[]
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in body_set or model.geom_group[g]!=0:continue
        assert model.geom_type[g]==c.mujoco.mjtGeom.mjGEOM_MESH
        mid=model.geom_dataid[g];start=model.mesh_vertadr[mid];n=model.mesh_vertnum[mid]
        meshes.append((g,model.mesh_vert[start:start+n].copy()))
    requests=np.array([0.,1.5,1.75,2.,2.25,row['task']['duration']-1/30])
    fig,axes=plt.subplots(2,len(requests),figsize=(16,3.6),layout='constrained',sharex=True,sharey='row',
                         gridspec_kw=dict(height_ratios=[70,23]))
    chosen=[]
    for j,t in enumerate(requests):
        k=int(np.argmin(abs(saved['pose_t']-t)));actual=float(saved['pose_t'][k]);chosen.append(actual)
        data.qpos[:]=saved['qpos'][k];c.mujoco.mj_kinematics(model,data)
        np.testing.assert_array_equal(data.qpos,saved['qpos'][k])
        center=data.xpos[body_ids].mean(0);origin=np.array([center[0],center[1],0.])
        active=np.zeros(38,bool)
        if actual>=1.5-1e-10:
            if 'erased_legs' in saved:
                mask_index=np.searchsorted(saved['mask_t'],actual+1e-10,side='right')-1
                assert mask_index>=0;active=saved['erased_legs'][mask_index]
            else:active=np.repeat(saved['erased_pairs'],2)
        geometry=[(vertices@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]-origin)*1000 for g,vertices in meshes]
        for axis_number,dims in enumerate(((0,1),(0,2))):
            ax=axes[axis_number,j]
            polygons=[]
            for vertices in geometry:
                points=vertices[:,dims];hull=ConvexHull(points);polygons.append(points[hull.vertices])
            ax.add_collection(PolyCollection(polygons,facecolor='#607a87',edgecolor='#263c45',linewidth=.45,alpha=.9))
            segments=[];colors=[]
            for a,b,leg in pairs:
                points=(data.xpos[[a,b]]-origin)*1000
                segments.append(points[:,dims]);colors.append('#df3632' if active[leg] else '#af714b')
            ax.add_collection(LineCollection(segments,colors=colors,linewidth=1))
            ax.set_aspect('equal',adjustable='box');ax.set_xlim(-65,65)
            ax.set_ylim((-35,35) if axis_number==0 else (-3,20))
            ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
            if axis_number==1:ax.axhline(0,color='black',lw=.7);ax.set_xlabel('Relative x (mm)')
            else:ax.set_title(f't={actual:.3f} s\n{active.sum()}/38 erased',fontsize=10)
    axes[0,0].set_ylabel('TOP: relative y (mm)');axes[1,0].set_ylabel('SIDE: height z (mm)')
    ch=record['channel'];a=row['amplitudes']
    fig.suptitle(f"Saved trajectory | wave {row['wave']:g}, p={ch['probability']:g}, seed={ch['seed']} | D={row['D']:.2f} mm\n"
        f"body / leg yaw / pitch={a['body']:.2f} / {a['leg_yaw']:.2f} / {a['leg_pitch']:.2f} deg | failed gates: {', '.join(row['failed']) or 'none'}",fontsize=12)
    fig.supxlabel('Projected chassis collision-mesh hulls and leg skeletons; camera follows center. Red = actual erased legs.\nSaved qpos only, no physics steps; skeletons do not certify collider clearance or contact force.',fontsize=8)
    dest=c.ROOT/'outputs/pose_reviews';dest.mkdir(exist_ok=True)
    path=dest/name;fig.savefig(path.with_suffix('.png'),dpi=180);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)
    c.write(path.with_suffix('.json'),dict(assessment=str(Path(assessment).resolve()),assessment_sha256=c.sha(assessment),
        trajectory_sha256=c.sha(folder/'contact_trajectory.npz'),model_sha256=c.sha(c.PLANTS['current']),
        code_sha256=c.sha(__file__),physics_steps=0,exact_saved_qpos=True,snapshot_times=chosen,
        projection='chassis collision mesh hulls and leg skeletons; not a contact-force or leg-clearance measurement'))
    print(path.with_suffix('.png'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('assessment');parser.add_argument('--name',required=True)
    args=parser.parse_args();review(args.assessment,args.name)
