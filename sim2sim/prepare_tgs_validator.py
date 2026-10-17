from pathlib import Path
root=Path(__file__).resolve().parent;src=(root/'validate_backend_bank.py').read_text(encoding='utf-8')
src=src.replace("b=np.load(f/'backend.npz')","b={k:v for k,v in np.load(f/'backend.npz').items()}")
src=src.replace("checks['solver_type']==0","checks['solver_type']==1 and json.loads((f/'tgs_external_iteration.json').read_text())['enabled'] is True")
(root/'validate_backend_tgs.py').write_text(src,encoding='utf-8')
print('Prepared TGS backend validator, with unchanged physical parameter limits.')
