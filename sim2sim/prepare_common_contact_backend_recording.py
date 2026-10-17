from pathlib import Path
R=Path(__file__).resolve().parent
src=(R/'run_native_common_contact_fast.py').read_text()
src=src.replace("view=robot.root_physx_view", """view=robot.root_physx_view
    s.write(folder/'execution_backend.json',dict(scene_attributes={n:scene.GetAttribute(n).Get() for n in ['physxScene:enableGPUDynamics','physxScene:broadphaseType','physxScene:solverType']},torch_device=sim.device,torch_cuda_name=torch.cuda.get_device_name(sim.device) if 'cuda' in sim.device else None))""")
(R/'run_native_common_contact_fast_recorded.py').write_text(src)
