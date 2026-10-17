from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / ".python_packages"))

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "outputs" / "data" / "audit_demo_manifest.json").read_text(encoding="utf-8"))
try:
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
except OSError:
    font = ImageFont.load_default()

tiles = []
for item in manifest:
    reader = imageio.get_reader(item["video"])
    frames = []
    for frame_id in (45, 70, 95, 120):
        frame = Image.fromarray(reader.get_data(frame_id))
        frame.thumbnail((640, 330), Image.Resampling.LANCZOS)
        frames.append(frame.copy())
    reader.close()
    tiles.append((Path(item["video"]).stem, frames))

sheet = Image.new("RGB", (4 * 640, len(tiles) * 380), (245, 247, 249))
draw = ImageDraw.Draw(sheet)
for row, (name, frames) in enumerate(tiles):
    draw.text((12, row * 380 + 8), name, font=font, fill=(20, 25, 30))
    for col, frame in enumerate(frames):
        x = col * 640 + (640 - frame.width) // 2
        y = row * 380 + 46
        sheet.paste(frame, (x, y))
path = ROOT / "outputs" / "figures" / "audit_demo_contact_sheet.png"
path.parent.mkdir(parents=True, exist_ok=True)
sheet.save(path)
print(path)

