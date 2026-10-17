from pathlib import Path
import ast
R=Path(__file__).resolve().parent
src=(R/'desaxce_contact.py').read_text(encoding='utf-8')
src=src.replace('corr=np.zeros(len(contacts));jacacc=np.zeros(d.nefc)', 'corr=np.zeros(len(contacts));jacacc=np.zeros(d.nefc)\n    if len(contacts) and max_iterations:\n        mujoco.mj_mulJacVec(m,d,jacacc,d.qacc_smooth)\n        corr=mu*np.linalg.norm((jacacc-base)[tangent],axis=1)')
src=src.replace('        corr=update','        corr=update\n        d.qacc_warmstart[:]=d.qacc')
p=R/'desaxce_contact_warm.py';assert not p.exists();ast.parse(src);p.write_text(src,encoding='utf-8')
probe=(R/'check_desaxce_pipeline_and_mechanics.py').read_text(encoding='utf-8').replace('from desaxce_contact import step','from desaxce_contact_warm import step').replace('desaxce20.npz','desaxce20_warm.npz').replace('desaxce_mechanical_suite.json','desaxce_warm_mechanical_suite.json').replace("'desaxce_contact.py'","'desaxce_contact_warm.py'")
(R/'check_desaxce_warm_mechanics.py').write_text(probe,encoding='utf-8')
