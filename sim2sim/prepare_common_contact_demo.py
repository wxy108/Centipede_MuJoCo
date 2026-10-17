"""Prepare a renderer of saved trajectories, without generating physical data."""
from pathlib import Path
R=Path(__file__).resolve().parent
src=(R/'render_pose_demo_closeup.py').read_text()
src=src.replace('mu_coupled_rigid_dt31_case03_000','mu_common_contact_dt31_case03_000')
src=src.replace('native_pgs_validation_gpu_dt62_000','native_common_contact_validation_gpu_dt31_000').replace('PhysX GPU | 62.5 us','PhysX GPU | 31.25 us')
src=src.replace("'Shared physical model and PD'","'Same plant, PD and shared compliant contact'")
src=src.replace('np.arange(-.26,.151,.02)','np.arange(-.50,.201,.02)').replace('[0,50,100,150,200]','[0,50,100,150,200,250]').replace('value/200*gh','value/250*gh').replace('yy/200*gh','yy/250*gh')
src=src.replace('sim2sim_w2p5_B_closeup','common_contact_w2p5_B_closeup')
src=src.replace("physics_recomputed=False,rendering=", "physical_contact_model='COMMON_CONTACT_V2.json',native_isaac_renderer=False,physics_recomputed=False,rendering=")
(R/'render_common_contact_demo.py').write_text(src)
