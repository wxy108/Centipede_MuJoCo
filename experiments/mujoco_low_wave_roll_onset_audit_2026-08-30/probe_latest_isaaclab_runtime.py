"""Read back effective joint properties from the current Clean IsaacLab runtime."""

from pathlib import Path
import json
import sys

from isaaclab.app import AppLauncher


launcher = AppLauncher(headless=True)
simulation_app = launcher.app

ROOT = Path(r"C:\Users\wxy22\Documents\Centipede_IsaacLab_RL_Clean")
OUT = Path(__file__).resolve().parent / "outputs/data/latest_isaaclab_runtime_joint_properties.json"
sys.path.insert(0, str(ROOT))

from centipede.env import CentipedeEnv  # noqa: E402
from centipede.env_cfg import CentipedeEnvCfg  # noqa: E402


def family(name):
    return next(
        (
            prefix
            for prefix in (
                "body_yaw", "body_pitch", "body_roll", "legbase_yaw",
                "legbase_pitch", "legbase_roll", "knee",
            )
            if name.startswith(prefix)
        ),
        "other",
    )


def main():
    cfg = CentipedeEnvCfg()
    cfg.scene.num_envs = 1
    cfg.terrain_mode = "homogeneous"
    cfg.terrain_label = "flat"
    cfg.terrain_lambda_mm = 1000.0
    cfg.terrain_amplitude_mm = 0.0
    cfg.fixed_wave = 1.5
    env = CentipedeEnv(cfg=cfg, render_mode=None)
    try:
        env.reset()
        view = env.robot.root_physx_view
        armature = view.get_dof_armatures()[0].detach().cpu().tolist()
        stiffness = view.get_dof_stiffnesses()[0].detach().cpu().tolist()
        damping = view.get_dof_dampings()[0].detach().cpu().tolist()
        records = [
            {
                "joint": name,
                "family": family(name),
                "armature": float(armature[index]),
                "stiffness": float(stiffness[index]),
                "damping": float(damping[index]),
            }
            for index, name in enumerate(env.robot.joint_names)
        ]
        aggregate = {}
        for group in sorted({record["family"] for record in records}):
            subset = [record for record in records if record["family"] == group]
            aggregate[group] = {
                "count": len(subset),
                "armature_unique": sorted({record["armature"] for record in subset}),
                "stiffness_unique": sorted({record["stiffness"] for record in subset}),
                "damping_unique": sorted({record["damping"] for record in subset}),
            }
        output = {
            "asset": str(cfg.scene.robot.spawn.usd_path),
            "fixed_wave": cfg.fixed_wave,
            "joint_count": len(records),
            "family_properties": aggregate,
            "joints": records,
            "model_or_controller_modified": False,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps({key: output[key] for key in ("asset", "fixed_wave", "joint_count", "family_properties")}, indent=2))
        print(f"[runtime-probe] {OUT}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
