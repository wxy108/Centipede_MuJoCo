"""Post-search definition audit; never substitutes for the optimization variables."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import physical_search as ps
import core as c

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))

if __name__=='__main__':
    path=ROOT/'outputs/physical_geometry_v2/data/validated_amplitudes.json'
    chosen=read(path);assert chosen['status']=='VALIDATED_FINITE_SEARCH'
    rows=[];hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
    model=c.e.runtime_model(c.PLANTS['current']);data=c.mujoco.MjData(model)
    body_ids=c.e.fx.opt.tip.LeanIndex(model).chassis_body_ids;assert len(body_ids)==21
    for selected in chosen['selected']:
        motion=Path(selected['motion'])
        with np.load(motion) as z:
            good=(z['pose_t']>=2)&(z['pose_t']<6)
            positions=[]
            for qpos in z['qpos'][good]:
                data.qpos[:]=qpos;c.mujoco.mj_kinematics(model,data)
                np.testing.assert_array_equal(data.qpos,qpos)
                positions.append(data.xpos[body_ids,:2].copy())
            points=np.asarray(positions);assert points.shape[1]==21
            centered=points-points.mean(axis=1,keepdims=True)
            covariance=np.einsum('tni,tnj->tij',centered,centered)/21
            eigenvalues,axes=np.linalg.eigh(covariance)
            lateral=np.einsum('tni,ti->tn',centered,axes[:,:,0])
            halfwidth=np.ptp(lateral,axis=1)*500
        rows.append(dict(wave=selected['wave'],body_target_deg=selected['amplitude']['body'],
            realized_joint_1hz_median_deg=selected['realized_body_median_deg'],
            centerline_pca_halfwidth_median_mm=float(np.median(halfwidth)),
            halfwidth_p05_p95_mm=np.quantile(halfwidth,[.05,.95]).tolist(),
            minimum_planar_eigenvalue_ratio=float(np.min(eigenvalues[:,1]/np.maximum(eigenvalues[:,0],1e-30)))))
        hashes[str(motion)]=hashlib.sha256(motion.read_bytes()).hexdigest()
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2),layout='constrained')
    wave=[r['wave'] for r in rows]
    axes[0].plot(wave,[r['body_target_deg'] for r in rows],'o-',label='Commanded yaw amplitude',color='#28618d')
    axes[0].plot(wave,[r['realized_joint_1hz_median_deg'] for r in rows],'s--',label='Actual yaw, 1 Hz median',color='#849faa')
    axes[0].set(ylabel='Joint angle amplitude (deg)',title='The optimized variable and its tracking')
    axes[0].legend(fontsize=8)
    axes[1].plot(wave,[r['centerline_pca_halfwidth_median_mm'] for r in rows],'o-',color='#bd6b36')
    axes[1].set(ylabel='Median lateral half-width (mm)',title='A different observable: body shape')
    for ax in axes:ax.set_xlabel('Coupled wave number');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('No-loss qualified traces, 2–6 s. Shape reconstructed from saved poses: 21 body centers, planar PCA, half lateral range.\nPost-search definition diagnostic; not a sinusoidal shape fit, a new optimization objective, or contact-crossing evidence.',fontsize=8)
    figure=ROOT/'amplitude_definition_audit';fig.savefig(figure.with_suffix('.png'),dpi=180);fig.savefig(figure.with_suffix('.pdf'));plt.close(fig)
    hashes[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    hashes[str(c.PLANTS['current'])]=hashlib.sha256(c.PLANTS['current'].read_bytes()).hexdigest()
    receipt=dict(physics_steps=0,post_search_diagnostic=True,changed_optimization=False,rows=rows,
        definition='Median over2–6s of half the lateral range of21 chassis centers in their per-frame planar PCA frame; rigid translation/heading removed. This is not the commanded or realized joint angle and is not a spatial sinusoidal fit.',source_hashes=hashes)
    (ROOT/'outputs/data/amplitude_definition_audit.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    for r in rows:print(r)
