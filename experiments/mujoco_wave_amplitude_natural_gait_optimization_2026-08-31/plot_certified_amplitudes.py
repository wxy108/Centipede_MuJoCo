"""Plot the certified natural-gait command amplitudes by coupled wave number."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs" / "data" / "final_certified_selection.csv"
OUTPUT = ROOT / "outputs" / "figures" / "certified_amplitudes_by_wave.png"


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F16 = font(16)
F18 = font(18)
F20 = font(20)
F22 = font(22)
F22B = font(22, True)
F26B = font(26, True)
F42B = font(42, True)


def read_rows():
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["wave"] = float(row["wave"])
        row["feasible"] = row["status"] == "certified_feasible"
        for field in ("body_yaw_deg", "leg_yaw_deg", "leg_pitch_deg"):
            row[field] = float(row[field]) if row[field] else None
    return rows


def main() -> None:
    rows = read_rows()
    width, height = 2100, 1280
    image = Image.new("RGB", (width, height), (13, 18, 24))
    draw = ImageDraw.Draw(image)

    ink = (236, 242, 247)
    muted = (164, 177, 190)
    grid = (48, 59, 70)
    axis = (102, 117, 130)
    infeasible = (244, 105, 112)
    series = [
        ("body_yaw_deg", "Body yaw", (255, 176, 80), "circle"),
        ("leg_yaw_deg", "Leg yaw", (79, 204, 255), "square"),
        ("leg_pitch_deg", "Leg pitch", (91, 224, 153), "diamond"),
    ]

    draw.text((70, 35), "认证自然步态：各命令自由度的优化幅值", font=F42B, fill=ink)
    draw.text(
        (72, 94),
        "Centipede Model · flat ground · coupled body/leg wave number · amplitude in degrees",
        font=F22,
        fill=(170, 204, 224),
    )
    draw.text(
        (72, 130),
        "仅显示通过全部 stability + naturality gates 并完成 cold replay 的有限候选集最优值；N/A 不是 0°。",
        font=F20,
        fill=muted,
    )

    left, top, right, bottom = 115, 235, 2025, 845
    draw.rounded_rectangle((left, top, right, bottom), radius=18, fill=(18, 25, 32), outline=(72, 85, 98), width=2)

    waves = [row["wave"] for row in rows]
    x_min, x_max = min(waves), max(waves)
    y_min, y_max = 0.0, 55.0

    def xp(value: float) -> float:
        return left + 36 + (value - x_min) / (x_max - x_min) * (right - left - 72)

    def yp(value: float) -> float:
        return bottom - 48 - (value - y_min) / (y_max - y_min) * (bottom - top - 90)

    for tick in range(0, 56, 5):
        y = yp(float(tick))
        draw.line((left + 36, y, right - 36, y), fill=grid, width=1)
        label = f"{tick}°"
        box = draw.textbbox((0, 0), label, font=F18)
        draw.text((left + 20 - (box[2] - box[0]), y - 11), label, font=F18, fill=muted)

    for row in rows:
        x = xp(row["wave"])
        if not row["feasible"]:
            draw.rounded_rectangle((x - 25, top + 22, x + 25, bottom - 48), radius=8, fill=(62, 31, 37))
            draw.line((x - 8, top + 49, x + 8, top + 65), fill=infeasible, width=4)
            draw.line((x + 8, top + 49, x - 8, top + 65), fill=infeasible, width=4)
        label = f"{row['wave']:g}"
        box = draw.textbbox((0, 0), label, font=F18)
        draw.text((x - (box[2] - box[0]) / 2, bottom - 36), label, font=F18, fill=ink)

    draw.text((900, 862), "Coupled wave number", font=F22B, fill=muted)
    draw.text((24, 468), "Amplitude (deg)", font=F20, fill=muted)

    for field, name, color, marker in series:
        points = [(xp(row["wave"]), yp(row[field])) for row in rows if row["feasible"]]
        for first, second in zip(points, points[1:]):
            draw.line((*first, *second), fill=color, width=5)
        for row in rows:
            if not row["feasible"]:
                continue
            x, y = xp(row["wave"]), yp(row[field])
            if marker == "circle":
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline=(12, 16, 20), width=2)
            elif marker == "square":
                draw.rounded_rectangle((x - 8, y - 8, x + 8, y + 8), radius=2, fill=color, outline=(12, 16, 20), width=2)
            else:
                draw.polygon(((x, y - 10), (x + 10, y), (x, y + 10), (x - 10, y)), fill=color, outline=(12, 16, 20))
            value = f"{row[field]:.1f}"
            if field == "leg_yaw_deg":
                offset_y = -32
            elif field == "leg_pitch_deg":
                offset_y = 12
            else:
                offset_y = -28 if row["wave"] in (2.0, 5.0, 6.0) else 11
            box = draw.textbbox((0, 0), value, font=F16)
            draw.text((x - (box[2] - box[0]) / 2, y + offset_y), value, font=F16, fill=color)

    legend_x = 1320
    for index, (_, name, color, marker) in enumerate(series):
        x = legend_x + index * 215
        y = 190
        draw.line((x, y, x + 34, y), fill=color, width=5)
        if marker == "circle":
            draw.ellipse((x + 10, y - 7, x + 24, y + 7), fill=color)
        elif marker == "square":
            draw.rectangle((x + 10, y - 7, x + 24, y + 7), fill=color)
        else:
            draw.polygon(((x + 17, y - 9), (x + 26, y), (x + 17, y + 9), (x + 8, y)), fill=color)
        draw.text((x + 44, y - 13), name, font=F20, fill=ink)

    # Exact-value table keeps the plot auditable without treating missing waves as zero.
    table_left, table_top, table_right, table_bottom = 70, 935, 2030, 1172
    draw.rounded_rectangle((table_left, table_top, table_right, table_bottom), radius=16, fill=(18, 25, 32), outline=(72, 85, 98), width=2)
    feasible_rows = [row for row in rows if row["feasible"]]
    columns = ["DOF"] + [f"w={row['wave']:g}" for row in feasible_rows]
    column_w = (table_right - table_left - 36) / len(columns)
    row_h = 49
    start_x, start_y = table_left + 18, table_top + 18
    for col, value in enumerate(columns):
        x = start_x + col * column_w
        draw.text((x + 8, start_y), value, font=F18 if col else F18, fill=(191, 206, 218))
    table_series = [
        ("Body yaw", "body_yaw_deg", series[0][2]),
        ("Leg yaw", "leg_yaw_deg", series[1][2]),
        ("Leg pitch", "leg_pitch_deg", series[2][2]),
    ]
    for row_index, (name, field, color) in enumerate(table_series, start=1):
        y = start_y + row_index * row_h
        draw.line((start_x, y - 8, table_right - 18, y - 8), fill=(46, 56, 66), width=1)
        draw.text((start_x + 8, y), name, font=F18, fill=color)
        for col, row in enumerate(feasible_rows, start=1):
            draw.text((start_x + col * column_w + 8, y), f"{row[field]:.2f}°", font=F18, fill=ink)

    draw.text(
        (72, 1200),
        "N/A waves: 0.5, 1, 7, 8, 9. Source: outputs/data/final_certified_selection.csv · 4,174 unique candidates.",
        font=F18,
        fill=muted,
    )
    draw.text(
        (72, 1232),
        "Claim boundary: finite-bank gated optimum, not a continuous mathematical global optimum. Robot Step 18/19 untouched.",
        font=F18,
        fill=(138, 153, 166),
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
