"""Build a read-only visual comparison from existing historical demo videos."""

from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\wxy22\Documents\Centipede_MUJOCO-main")
OUT = ROOT / "experiments/mujoco_low_wave_roll_onset_audit_2026-08-30/outputs/figures/historical_demo_comparison.png"
VIDEOS = [
    (
        "Aug 26: old plant + optimized flat controller",
        ROOT / "experiments/mujoco_zero_armature_controller_optimization_2026-08-26/outputs/videos/optimized_flat_leg_lift_oblique.mp4",
    ),
    (
        "Aug 26: q2 k20 fidelity stage",
        ROOT / "experiments/mujoco_template_q2_compliance_2026-08-26/outputs/videos/q2compliance_bounded_k20_c18_flat_oblique.mp4",
    ),
    (
        "Aug 27: locked-head current lineage",
        ROOT / "experiments/mujoco_template_locked_head_2026-08-27/outputs/videos/locked_head_m29p0_flat_oblique.mp4",
    ),
]
FRACTIONS = (0.30, 0.45, 0.60, 0.75)


def font(size: int):
    path = Path(r"C:\Windows\Fonts\arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def frame_at(cap: cv2.VideoCapture, fraction: float) -> Image.Image:
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(count - 1, round(fraction * (count - 1)))))
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame at {fraction:.0%}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).resize((480, 270), Image.Resampling.LANCZOS)


def main():
    margin, label_w, gap = 24, 370, 12
    cell_w, cell_h = 480, 270
    title_h, header_h, row_gap = 74, 42, 16
    width = margin * 2 + label_w + gap + len(FRACTIONS) * cell_w + (len(FRACTIONS) - 1) * gap
    height = margin * 2 + title_h + header_h + len(VIDEOS) * cell_h + (len(VIDEOS) - 1) * row_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), "Historical low-wave flat demos — same normalized gait phases", fill="#111827", font=font(34))
    y0 = margin + title_h
    for col, fraction in enumerate(FRACTIONS):
        x = margin + label_w + gap + col * (cell_w + gap)
        draw.text((x + cell_w // 2, y0), f"{fraction:.0%} of video", anchor="ma", fill="#374151", font=font(23))
    y0 += header_h

    for row, (label, path) in enumerate(VIDEOS):
        y = y0 + row * (cell_h + row_gap)
        draw.multiline_text((margin, y + 92), label, fill="#111827", font=font(25), spacing=7)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {path}")
        for col, fraction in enumerate(FRACTIONS):
            x = margin + label_w + gap + col * (cell_w + gap)
            canvas.paste(frame_at(cap, fraction), (x, y))
        cap.release()
        draw.line((margin, y + cell_h + row_gap // 2, width - margin, y + cell_h + row_gap // 2), fill="#D1D5DB", width=2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
