"""Render matched front-view demos from historical models without modifying them."""

import json

import diagnose_roll_onset as audit


def main():
    videos = audit.OUTPUTS / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    old_impedance = audit.normalized_impedance(audit.ctrl.final_impedance())
    selection = json.loads(audit.BIO_SELECTION.read_text(encoding="utf-8"))
    inherited_impedance = audit.normalized_impedance(selection["selected_impedance"])

    views = {
        "front": {"azimuth": 0.0, "elevation": -4.0, "distance": 0.19},
        "side": {"azimuth": 90.0, "elevation": -4.0, "distance": 0.19},
        "top": {"azimuth": 90.0, "elevation": -89.0, "distance": 0.19},
    }
    rendered = {"old": {}, "q2_onset": {}}
    old = onset = None
    for view, camera in views.items():
        old_video = videos / f"diagnostic_A_old_plant_old_controller_{view}.mp4"
        onset_video = videos / f"diagnostic_E_q2_k20_inherited_controller_{view}.mp4"
        if not old_video.exists():
            print(f"[demo] rendering historical A {view} view", flush=True)
            old = audit.replay_original_plant(
                "A old plant + old flat controller",
                old_impedance,
                video=old_video,
                camera_azimuth=camera["azimuth"],
                camera_elevation=camera["elevation"],
                camera_distance=camera["distance"],
            )
        if not onset_video.exists():
            print(f"[demo] rendering historical E {view} view", flush=True)
            onset = audit.replay_stage1_plant(
                "E q2 k20 plant + inherited controller",
                audit.Q2_MODEL,
                inherited_impedance,
                "bounded_k20_c18",
                video=onset_video,
                camera_azimuth=camera["azimuth"],
                camera_elevation=camera["elevation"],
                camera_distance=camera["distance"],
            )
        rendered["old"][view] = str(old_video)
        rendered["q2_onset"][view] = str(onset_video)

    previous_manifest = audit.DATA / "diagnostic_demo_manifest.json"
    previous = json.loads(previous_manifest.read_text(encoding="utf-8")) if previous_manifest.exists() else {}
    old_metrics = old or previous.get("old", {})
    onset_metrics = onset or previous.get("q2_onset", {})
    manifest = {
        "models_modified": False,
        "controller_files_modified": False,
        "camera_only": views,
        "old": {
            "videos": rendered["old"],
            "world_roll_p95_deg": old_metrics.get("chassis_world_roll_abs_p95_deg", old_metrics.get("world_roll_p95_deg")),
            "local_roll_p95_deg": old_metrics.get("body_roll_local_abs_p95_deg", old_metrics.get("local_roll_p95_deg")),
        },
        "q2_onset": {
            "videos": rendered["q2_onset"],
            "world_roll_p95_deg": onset_metrics.get("chassis_world_roll_abs_p95_deg", onset_metrics.get("world_roll_p95_deg")),
            "local_roll_p95_deg": onset_metrics.get("body_roll_local_abs_p95_deg", onset_metrics.get("local_roll_p95_deg")),
        },
    }
    path = audit.DATA / "diagnostic_demo_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
