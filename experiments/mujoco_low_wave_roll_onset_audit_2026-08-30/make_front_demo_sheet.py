"""Create a matched-camera frame sheet for the two diagnostic demo videos."""

from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/figures/matched_front_demo_comparison.png"
VIDEOS = (
    ("A  OLD PLANT + OLD FLAT CONTROLLER", ROOT / "outputs/videos/diagnostic_A_old_plant_old_controller_front.mp4"),
    ("E  q2 k20 PLANT + INHERITED CONTROLLER", ROOT / "outputs/videos/diagnostic_E_q2_k20_inherited_controller_front.mp4"),
)
FRACTIONS = (0.30, 0.45, 0.60, 0.75)


def get_font(size: int):
    path = Path(r"C:\Windows\Fonts\arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def main():
    cell_w, cell_h, gap, margin = 480, 270, 12, 24
    label_h, title_h, header_h = 42, 64, 38
    width = margin * 2 + len(FRACTIONS) * cell_w + (len(FRACTIONS) - 1) * gap
    height = margin * 2 + title_h + header_h + len(VIDEOS) * (label_h + cell_h + gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), "Matched front camera: the old low-wave run already rolls", fill="#111827", font=get_font(32))
    y = margin + title_h
    for col, fraction in enumerate(FRACTIONS):
        x = margin + col * (cell_w + gap)
        draw.text((x + cell_w // 2, y), f"{fraction:.0%} of run", anchor="ma", fill="#374151", font=get_font(22))
    y += header_h

    for label, path in VIDEOS:
        draw.text((margin, y), label, fill="#111827", font=get_font(24))
        y += label_h
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for col, fraction in enumerate(FRACTIONS):
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(fraction * (count - 1)))
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read {path} at {fraction}")
            frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize(
                (cell_w, cell_h), Image.Resampling.LANCZOS
            )
            canvas.paste(frame, (margin + col * (cell_w + gap), y))
        cap.release()
        y += cell_h + gap

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
