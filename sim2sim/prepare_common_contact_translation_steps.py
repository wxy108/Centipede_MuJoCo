from pathlib import Path
R=Path(__file__).resolve().parent
src=(R/'run_native_common_contact_translation_probe.py').read_text()
src=src.replace('stride=round(.01/a.dt);steps=','stride=1;steps=')
src=src.replace("root_positions=[arr(view.get_root_transforms())[:,:3]];", "com_positions=[arr(robot.data.body_com_pose_w)[...,:3]];com_velocities=[arr(robot.data.body_com_vel_w)]\n    root_positions=[arr(view.get_root_transforms())[:,:3]];")
src=src.replace("root_positions.append(arr(view.get_root_transforms())[:,:3]);", "com_positions.append(arr(robot.data.body_com_pose_w)[...,:3]);com_velocities.append(arr(robot.data.body_com_vel_w))\n            root_positions.append(arr(view.get_root_transforms())[:,:3]);")
src=src.replace("position=root_positions,velocity=root_velocities)","position=root_positions,velocity=root_velocities,com_position=com_positions,com_velocity=com_velocities)")
(R/'run_native_common_contact_translation_steps.py').write_text(src)
