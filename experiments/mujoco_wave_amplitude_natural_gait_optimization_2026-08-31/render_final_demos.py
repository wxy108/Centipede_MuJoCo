"""Render clear, labelled three-view demos for certified finite-bank winners.

Rendering is deliberately visual-only: physical collision meshes remain active
in MuJoCo but are hidden from the camera to prevent coplanar z-fighting.  During
random contact erasure the complete visible left/right leg subtree is bright red.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "data"
VIDEOS = ROOT / "outputs" / "videos" / "final_certified"
FIGURES = ROOT / "outputs" / "figures"
SELECTION = DATA / "final_certified_selection.json"
AUDIT_RENDERER = ROOT / "render_audit_demos.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


demo = load_module(AUDIT_RENDERER, "centipede_final_demo_renderer")


def wave_slug(wave: float) -> str:
    return f"{wave:g}".replace(".", "p")


def case_from_row(row: dict) -> dict:
    wave = float(row["wave"])
    return {
        "label": f"CERTIFIED finite-bank winner | wave {wave:g}",
        "wave": wave,
        "body": float(row["body_yaw_amplitude_deg"]),
        "leg_yaw": float(row["leg_yaw_amplitude_deg"]),
        "leg_pitch": float(row["leg_pitch_amplitude_deg"]),
    }


def snapshot(state, erasure_label: bool = False):
    panels = [
        state.render(180.0, -32.0, 0.16),
        state.render(90.0, -28.0, 0.16),
        state.render(135.0, -32.0, 0.16),
    ]
    canvas = demo.Image.new("RGB", (3 * demo.WIDTH, demo.HEIGHT + 120), (20, 25, 30))
    draw = demo.ImageDraw.Draw(canvas)
    for index, (name, panel) in enumerate(zip(("FRONT", "SIDE", "OBLIQUE"), panels)):
        canvas.paste(panel, (index * demo.WIDTH, 120))
        draw.text((index * demo.WIDTH + 22, 88), name, font=demo.FONT_SMALL, fill=(160, 219, 246))
    demo.title(draw, state.case, 12)
    if erasure_label:
        active = state.active.nonzero()[0].tolist()
        draw.rounded_rectangle((1960, 18, 2850, 100), radius=16, fill=(70, 8, 8), outline=(255, 70, 70), width=3)
        draw.rectangle((1990, 39, 2030, 79), fill=(255, 5, 5))
        draw.text((2050, 33), f"RED = CONTACT ERASED | active pairs: {active or 'none'}", font=demo.FONT, fill=(255, 235, 235))
    return canvas


def make_contact_sheet(images: list[tuple[str, object]], path: Path) -> None:
    if not images:
        return
    thumb_w, thumb_h = 1440, 330
    rows = math.ceil(len(images) / 2)
    sheet = demo.Image.new("RGB", (2 * thumb_w, rows * (thumb_h + 52)), (15, 20, 25))
    draw = demo.ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        x = (index % 2) * thumb_w
        y = (index // 2) * (thumb_h + 52)
        fitted = image.resize((thumb_w, thumb_h), demo.Image.Resampling.LANCZOS)
        sheet.paste(fitted, (x, y + 52))
        draw.text((x + 18, y + 13), label, font=demo.FONT_SMALL, fill=(218, 234, 244))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=95)


def render_case(case: dict, q: float, seed: int, path: Path, marked: bool):
    state = demo.DemoState(case, q=q, seed=seed)
    try:
        frames = demo.write_video(path, demo.three_view_frames(state, erasure_label=marked))
    finally:
        state.close()
    # Use an independent replay for a true mid-gait contact-sheet sample.
    state = demo.DemoState(case, q=q, seed=seed)
    try:
        state.advance(2.25)
        image = snapshot(state, erasure_label=marked)
        event_count = len(state.events)
        opportunities = state.opportunities
    finally:
        state.close()
    return frames, image, event_count, opportunities


def local_zoom_frames(state):
    for frame_id in range(int(round(demo.DURATION_S * demo.FPS)) + 1):
        t = frame_id / demo.FPS
        state.advance(t)
        panels = [
            state.render(90.0, -22.0, 0.040, focus_segment=9),
            state.render(135.0, -28.0, 0.040, focus_segment=9),
        ]
        canvas = demo.Image.new("RGB", (2 * demo.WIDTH, demo.HEIGHT + 120), (20, 25, 30))
        draw = demo.ImageDraw.Draw(canvas)
        for index, (name, panel) in enumerate(zip(("SIDE LOCAL ZOOM", "OBLIQUE LOCAL ZOOM"), panels)):
            canvas.paste(panel, (index * demo.WIDTH, 120))
            draw.text((index * demo.WIDTH + 22, 88), name, font=demo.FONT_SMALL, fill=(160, 219, 246))
        demo.title(draw, state.case, 12)
        draw.text((1780, 88), f"t={t:4.2f}s", font=demo.FONT_SMALL, fill=(230, 230, 230))
        yield canvas


def main() -> None:
    payload = json.loads(SELECTION.read_text(encoding="utf-8"))
    VIDEOS.mkdir(parents=True, exist_ok=True)
    manifest = []
    contact_images = []
    best_case = None
    best_row = None
    for wave_text, item in payload["waves"].items():
        selected = item.get("selected")
        if selected is None:
            continue
        case = case_from_row(selected)
        if best_row is None or float(selected["forward_displacement_m"]) > float(best_row["forward_displacement_m"]):
            best_case, best_row = dict(case), selected
        wave = float(case["wave"])
        slug = wave_slug(wave)
        clean_path = VIDEOS / f"certified_w{slug}_three_view_clear.mp4"
        frames, image, events, opportunities = render_case(case, 0.0, 0, clean_path, False)
        manifest.append(
            {
                "wave": wave,
                "mode": "clean flat",
                "video": str(clean_path),
                "frames": frames,
                "candidate_id": selected["candidate_id"],
                "amplitudes_deg": [case["body"], case["leg_yaw"], case["leg_pitch"]],
                "render_contract": "visual group 2 plus ground only; physical collision geometry hidden from camera",
            }
        )
        contact_images.append((f"wave {wave:g} | clean", image))

        if math.isclose(wave, round(wave)):
            erased_case = dict(case)
            erased_case["label"] = f"NOMINAL winner under random erasure | DIAGNOSTIC, strict gates failed | wave {wave:g}"
            erased_path = VIDEOS / f"certified_w{slug}_q035_erasure_marked_three_view_clear.mp4"
            frames, image, events, opportunities = render_case(erased_case, 0.35, 2, erased_path, True)
            manifest.append(
                {
                    "wave": wave,
                    "mode": "random contact erasure q=0.35 seed=2",
                    "video": str(erased_path),
                    "frames": frames,
                    "candidate_id": selected["candidate_id"],
                    "events": events,
                    "opportunities": opportunities,
                    "color_contract": "bright red complete visible leg pair = collision disabled during this interval",
                }
            )
            contact_images.append((f"wave {wave:g} | random erasure; RED=erased", image))

    if best_case is not None:
        best_case["label"] = f"BEST certified nominal displacement | wave {float(best_case['wave']):g} | local leg/foot motion"
        state = demo.DemoState(best_case)
        try:
            zoom_path = VIDEOS / f"certified_best_w{wave_slug(float(best_case['wave']))}_local_zoom_clear.mp4"
            frames = demo.write_video(zoom_path, local_zoom_frames(state))
        finally:
            state.close()
        manifest.append(
            {
                "wave": float(best_case["wave"]),
                "mode": "clean flat local side/oblique zoom",
                "video": str(zoom_path),
                "frames": frames,
                "candidate_id": best_row["candidate_id"],
                "purpose": "visual audit of leg swing, foot clearance, and passive tip contact",
            }
        )

    manifest_path = DATA / "final_demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sheet_path = FIGURES / "final_certified_demo_contact_sheet.png"
    make_contact_sheet(contact_images, sheet_path)
    print(json.dumps({"videos": len(manifest), "manifest": str(manifest_path), "contact_sheet": str(sheet_path)}, indent=2))


if __name__ == "__main__":
    main()
