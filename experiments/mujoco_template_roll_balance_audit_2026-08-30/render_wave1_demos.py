"""Render three diagnostic views of the coupled wave=1.0 flat ablation."""

from __future__ import annotations

import json
from pathlib import Path

import run_roll_balance_audit as audit


VIDEOS = audit.OUTPUTS / "videos"
DATA = audit.DATA


def main() -> None:
    VIDEOS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    before = audit.tip.robot_sentinels()
    original_flat_wave = audit.tip.FLAT_WAVE
    rows = []
    paths = []
    try:
        # run_case chooses the flat gait from this module constant.  Temporarily
        # setting it to 1.0 makes both the controller wave and its amplitude
        # branch explicit, without editing any model or production source.
        audit.tip.FLAT_WAVE = 1.0
        for view, azimuth, elevation in (
            ("side", 90.0, -12.0),
            ("oblique", 135.0, -25.0),
            ("top", 90.0, -80.0),
        ):
            video = VIDEOS / f"diagnostic_coupled_wave1p0_flat_{view}.mp4"
            print(f"[demo] START wave1.0 {view}", flush=True)
            row = audit.tip.run_case(
                "diagnostic_coupled_wave1p0",
                "flat",
                audit.MODEL,
                6.0,
                video=video,
                camera_azimuth=azimuth,
                camera_elevation=elevation,
            )
            rows.append(row)
            paths.append(video)
            print(f"[demo] DONE wave1.0 {view}", flush=True)
    finally:
        audit.tip.FLAT_WAVE = original_flat_wave
    after = audit.tip.robot_sentinels()
    output = {
        "selection_status": "diagnostic_not_selected",
        "production_controller_changed": False,
        "body_wave": 1.0,
        "leg_wave": 1.0,
        "wave_numbers_coupled": True,
        "videos": [str(path) for path in paths],
        "rows": rows,
        "robot_sentinels_before": before,
        "robot_sentinels_after": after,
        "robot_sentinels_unchanged": before == after,
    }
    path = DATA / "diagnostic_wave1p0_demo_audit.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"audit": str(path), "videos": output["videos"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
