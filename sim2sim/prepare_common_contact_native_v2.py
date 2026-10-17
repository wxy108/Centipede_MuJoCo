from pathlib import Path
R=Path(__file__).resolve().parent
src=(R/'run_native_common_contact.py').read_text(encoding='utf-8').replace('CreateEnableExternalForcesEveryIterationAttr(True)','CreateEnableExternalForcesEveryIterationAttr(False)').replace("folder/'tgs_external_iteration.json'","folder/'external_iteration_setting.json'")
p=R/'run_native_common_contact_v2.py';assert not p.exists();p.write_text(src,encoding='utf-8')
print('PGS-incompatible TGS-only flag explicitly disabled in the new runner.')
