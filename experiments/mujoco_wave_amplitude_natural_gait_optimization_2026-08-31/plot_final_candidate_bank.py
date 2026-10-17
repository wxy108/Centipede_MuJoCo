"""Plot wide-domain coverage and certified displacement using Pillow only."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "data"
FIGURES = ROOT / "outputs" / "figures"
TRUSTED_SITE = Path(sys.executable).resolve().parent / "Lib" / "site-packages"
if TRUSTED_SITE.exists():
    sys.path.insert(0, str(TRUSTED_SITE))

from PIL import Image, ImageDraw, ImageFont


def font(size: int, bold: bool = False):
    path = Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


F18 = font(18)
F20 = font(20)
F22B = font(22, True)
F30B = font(30, True)


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def color_for_displacement(mm: float):
    value = min(1.0, max(0.0, (mm - 35.0) / 125.0))
    if value < 0.5:
        u = value / 0.5
        return (int(48 + 20 * u), int(120 + 110 * u), int(245 - 85 * u))
    u = (value - 0.5) / 0.5
    return (int(68 + 187 * u), int(230 - 95 * u), int(160 - 95 * u))


def star_points(cx: float, cy: float, outer: float = 12.0, inner: float = 5.0):
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        radius = outer if index % 2 == 0 else inner
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def domain_figure(rows, selection, waves, output: Path) -> None:
    panel_w, panel_h = 560, 370
    header_h = 125
    image = Image.new("RGB", (4 * panel_w, header_h + 4 * panel_h), (15, 19, 24))
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), "Wide three-amplitude candidate bank and V2 gates", font=F30B, fill=(238, 245, 250))
    draw.text((28, 63), "Declared domain: body yaw 0–36°, leg yaw 0–90°, leg pitch 0–60°", font=F20, fill=(170, 209, 230))
    draw.text((28, 91), "Gray = hard-gate fail | color = natural/stable | yellow star = maximum displacement among feasible bank points", font=F18, fill=(190, 198, 207))
    for index, wave in enumerate(waves):
        col, row_index = index % 4, index // 4
        ox, oy = col * panel_w, header_h + row_index * panel_h
        left, top, right, bottom = ox + 58, oy + 62, ox + panel_w - 24, oy + panel_h - 48
        group = [item for item in rows if math.isclose(float(item["wave"]), wave)]
        selected = selection["waves"][f"{wave:g}"]["selected"]
        status = f"winner {1000*float(selected['forward_displacement_m']):.1f} mm" if selected else "NO V2-FEASIBLE POINT"
        title_color = (236, 241, 245) if selected else (255, 135, 135)
        draw.text((ox + 22, oy + 17), f"wave {wave:g} | {status}", font=F22B, fill=title_color)
        draw.rectangle((left, top, right, bottom), fill=(22, 28, 34), outline=(95, 108, 120), width=1)
        for tick in (0, 30, 60, 90):
            x = left + tick / 90.0 * (right - left)
            draw.line((x, top, x, bottom), fill=(46, 55, 64), width=1)
            draw.text((x - 10, bottom + 7), str(tick), font=F18, fill=(145, 157, 169))
        for tick in (0, 20, 40, 60):
            y = bottom - tick / 60.0 * (bottom - top)
            draw.line((left, y, right, y), fill=(46, 55, 64), width=1)
            draw.text((left - 38, y - 10), str(tick), font=F18, fill=(145, 157, 169))
        for item in group:
            x = left + float(item["leg_yaw_amplitude_deg"]) / 90.0 * (right - left)
            y = bottom - float(item["leg_pitch_amplitude_deg"]) / 60.0 * (bottom - top)
            if item.get("feasible_v2_pilot"):
                radius = 2.5 + 0.09 * float(item["body_yaw_amplitude_deg"])
                fill = color_for_displacement(1000.0 * float(item["forward_displacement_m"]))
            else:
                radius, fill = 1.6, (83, 94, 105)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
        if selected:
            x = left + float(selected["leg_yaw_amplitude_deg"]) / 90.0 * (right - left)
            y = bottom - float(selected["leg_pitch_amplitude_deg"]) / 60.0 * (bottom - top)
            draw.polygon(star_points(x, y), fill=(255, 224, 88), outline=(10, 10, 10))
        draw.text((left + 125, bottom + 29), "leg yaw amplitude (deg)", font=F18, fill=(170, 180, 190))
        draw.text((ox + 5, top + 90), "leg\npitch\n(deg)", font=F18, fill=(170, 180, 190), spacing=0)
    image.save(output, quality=95)


def displacement_figure(selection, waves, output: Path) -> None:
    image = Image.new("RGB", (1700, 720), (15, 19, 24))
    draw = ImageDraw.Draw(image)
    draw.text((45, 22), "Certified finite-bank winner per coupled wave", font=F30B, fill=(238, 245, 250))
    draw.text((45, 66), "Hard-gate feasibility precedes displacement ranking; red columns have no feasible candidate", font=F20, fill=(180, 203, 216))
    left, top, right, bottom = 95, 125, 1650, 625
    draw.rectangle((left, top, right, bottom), outline=(105, 118, 130), width=1)
    for tick in (0, 40, 80, 120, 160):
        y = bottom - tick / 160.0 * (bottom - top)
        draw.line((left, y, right, y), fill=(45, 55, 64), width=1)
        draw.text((45, y - 11), str(tick), font=F18, fill=(155, 168, 180))
    points = []
    for index, wave in enumerate(waves):
        x = left + index / (len(waves) - 1) * (right - left)
        selected = selection["waves"][f"{wave:g}"]["selected"]
        draw.text((x - 13, bottom + 18), f"{wave:g}", font=F18, fill=(175, 186, 197))
        if selected:
            value = 1000.0 * float(selected["forward_displacement_m"])
            y = bottom - value / 160.0 * (bottom - top)
            points.append((index, x, y))
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(91, 211, 255), outline=(240, 250, 255), width=1)
            draw.text((x - 24, y - 34), f"{value:.1f}", font=F18, fill=(210, 236, 248))
        else:
            draw.line((x, top, x, bottom), fill=(138, 45, 53), width=5)
            draw.text((x - 11, bottom - 112), "NO", font=F18, fill=(255, 130, 140))
    for first, second in zip(points, points[1:]):
        if second[0] == first[0] + 1:
            draw.line((first[1], first[2], second[1], second[2]), fill=(91, 211, 255), width=4)
    draw.text((690, 672), "coupled body/leg wave number", font=F20, fill=(180, 191, 202))
    draw.text((10, 325), "displacement\n(mm)", font=F18, fill=(180, 191, 202), spacing=2)
    image.save(output, quality=95)


def main() -> None:
    rows = read_jsonl(DATA / "broad_sobol_pilot_certified_trials.jsonl")
    for path in sorted(DATA.glob("adaptive_refinement_trials_s[123]_*.jsonl")):
        rows.extend(read_jsonl(path))
    selection = json.loads((DATA / "final_certified_selection.json").read_text(encoding="utf-8"))
    waves = [float(value) for value in selection["waves"]]
    FIGURES.mkdir(parents=True, exist_ok=True)
    coverage = FIGURES / "final_candidate_bank_domain_and_gates.png"
    summary = FIGURES / "final_certified_displacement_by_wave.png"
    domain_figure(rows, selection, waves, coverage)
    displacement_figure(selection, waves, summary)
    print(json.dumps({"coverage": str(coverage), "summary": str(summary)}, indent=2))


if __name__ == "__main__":
    main()
