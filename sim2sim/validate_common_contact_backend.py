import json,subprocess,sys
import shared as s
name=sys.argv[1];folder=s.ROOT/'runs'/name
result=subprocess.run([sys.executable,'-B',str(s.ROOT/'validate_backend_bank.py'),name],capture_output=True,text=True)
assert result.returncode==0,result.stderr
base=json.loads((folder/'backend_validation.json').read_text());meta=json.loads((folder/'readback.json').read_text());setup=json.loads((folder/'common_contact_setup.json').read_text());ext=json.loads((folder/'external_iteration_setting.json').read_text())
checks=dict(physical_backend=base['passed'],ground_response_disabled=setup['native_ground_collision'] is False,pgs_external_iteration_flag_false=ext['enabled'] is False,world_com_wrench=setup['body_force_frame']=='world, torque about COM',same_input_hashes=all(s.sha(p)==h for p,h in meta['source_hashes'].items()),solver_iterations=meta['pos_iters']==16 and meta['vel_iters']==4)
s.write(folder/'common_contact_backend_validation.json',dict(passed=all(checks.values()),checks=checks,physical_readback='backend_validation.json',physics_dt_s=meta['physics_dt_s'],device=meta['device']));print(json.dumps(dict(run=name,passed=all(checks.values()),checks=checks)));assert all(checks.values())
