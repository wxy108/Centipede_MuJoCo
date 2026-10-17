"""Reconstruct recorded rigid-tip poses and recheck topology invariants."""
import json

import run_ablation as ab


np, mujoco = ab.np, ab.mujoco
records = (
    "rigid_tip_legyaw_refine_k0p29_w2p5_t6_dt0p00025",
    "rigid_tip_basin15_yk2_pd4_rk1_w3_t6_dt0p00025",
    "rigid_tip_legyaw_refine_k0p29_w2p5_t6_dt0p000125",
    "rigid_tip_basin15_yk2_pd4_rk1_w3_t6_dt0p000125",
)
results = []
for stem in records:
    trace = np.load(ab.OUT / f"{stem}.npz")
    model = mujoco.MjModel.from_xml_path(str(ab.MODEL))
    data = mujoco.MjData(model)
    index = ab.RigidTipIndex(model)
    sites = np.asarray(
        [
            ab.base.name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"site_foot_{segment}_{side}"
            )
            for segment in range(19)
            for side in range(2)
        ],
        dtype=int,
    )
    body_error = foot_error = 0.0
    for sample in np.linspace(0, len(trace["t"]) - 1, 20, dtype=int):
        data.qpos[:] = trace["qpos"][sample]
        mujoco.mj_forward(model, data)
        body_error = max(
            body_error,
            float(
                np.abs(data.xpos[index.chassis_body_ids] - trace["xpos"][sample]).max()
            ),
        )
        foot_error = max(
            foot_error,
            float(np.abs(data.site_xpos[sites] - trace["fq"][sample]).max()),
        )
    if max(body_error, foot_error) >= 1e-7:
        raise RuntimeError("recorded pose reconstruction failed")
    results.append(
        dict(
            id=stem,
            checked_poses=20,
            max_body_error_m=body_error,
            max_foot_error_m=foot_error,
        )
    )

topology = json.loads((ab.ROOT / "outputs" / "topology_audit.json").read_text())
assert topology["removed_tip_joint_count"] == 38
assert topology["tip_roll_dof_count"] == 0
assert topology["knee_joint_count"] == 0
assert topology["removed_tip_body_count"] == 38
assert topology["removed_foot_body_count"] == 38
assert topology["distal_fixed_child_body_count"] == 0
assert topology["nominal_leg_geometry_unchanged"]
assert topology["foot_site_max_error_m"] < 1e-11
assert topology["projected_active_mass_matrix_max_error"] < 1e-14
assert topology["source_total_mass_kg"] == topology["rigid_total_mass_kg"]

output = {
    "topology_checks_pass": True,
    "pose_checks": results,
    "max_pose_error_m": max(
        max(row["max_body_error_m"], row["max_foot_error_m"]) for row in results
    ),
}
(ab.ROOT / "outputs" / "verification.json").write_text(
    json.dumps(output, indent=2), encoding="utf-8"
)
print(json.dumps(output, indent=2))
