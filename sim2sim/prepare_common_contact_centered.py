import difflib,json
from pathlib import Path
import shared as s
R=s.ROOT;source=R/'run_native_common_contact_fast_recorded.py';target=R/'run_native_common_contact_centered.py';src=source.read_text()
assert src.count('Gf.Vec3d(0.,e*.25,0.)')==1 and src.count(';initial[:,1]+=torch.arange(N,device=sim.device)*.25')==1
new=src.replace('Gf.Vec3d(0.,e*.25,0.)','Gf.Vec3d(0.,0.,0.)').replace(';initial[:,1]+=torch.arange(N,device=sim.device)*.25','')
new=new.replace("s.write(folder/'execution_backend.json'", "s.write(folder/'coordinate_layout.json',dict(global_origins_m=[[0.,0.,0.]]*N,independent_articulations=True,native_collisions_disabled=True,periodic_rebase=False,amendment='NUMERICAL_AMENDMENT_28.md'))\n    s.write(folder/'execution_backend.json'")
target.write_text(new)
s.write(R/'audit/common_contact_centered_preparation.json',dict(source_sha256=s.sha(source),target_sha256=s.sha(target),diff=''.join(difflib.unified_diff(src.splitlines(True),new.splitlines(True),fromfile=source.name,tofile=target.name))))
profile=json.loads((R/'COMMON_CONTACT_CANDIDATE_PROFILE.json').read_text());profile.update(name='common_plant_pd_and_compliant_contact_v2_centered',native_runner=target.name,layout='Nine independent articulations, all static environment translations and initial Y offsets zero. Native collisions disabled; per-environment contact forces independent. No dynamic rebasing.',validation_summary='audit/common_contact_centered_summary.json',additional_motion_limits=dict(max_relative_centroid_xy_path_difference_mm=.5,max_head_tail_heading_difference_deg=.25))
s.write(R/'CENTERED_COMMON_CONTACT_CANDIDATE_PROFILE.json',profile)
