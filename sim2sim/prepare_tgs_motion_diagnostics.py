from pathlib import Path
R=Path(__file__).resolve().parent
src=(R/'summarize_motion_quality.py').read_text(encoding='utf-8')
src=src.replace('mu_coupled_rigid_dt31_', 'mu_coupled_cmd250_dt31_').replace('native_pgs_validation_gpu_dt62_000','native_tgs_ext_validation_gpu_dt250_000').replace('native_pgs_validation_cpu_dt62_000','native_tgs_ext_validation_cpu_dt250_000')
src=src.replace('motion_quality_comparison.json','tgs_motion_quality_comparison.json').replace('motion_quality_table.csv','tgs_motion_quality_table.csv').replace('11_motion_diagnostics','16_tgs_motion_diagnostics')
src=src.replace('Recorded motion diagnostics — same physical PD and target commands','TGS 250 µs diagnostics — candidate failed refinement; not adopted')
p=R/'summarize_tgs_motion_quality.py';assert not p.exists();p.write_text(src,encoding='utf-8')
