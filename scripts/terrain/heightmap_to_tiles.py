#!/usr/bin/env python3
"""
heightmap_to_tiles.py — Split a MuJoCo terrain PNG into a 4×4 grid of
                        watertight STL tiles by direct pixel slicing.

NO resampling. NO smoothing. Each tile is built from the RAW pixels of the
source heightmap: tile 1 = first 256×256 pixels, tile 2 = next 256×256, etc.
A 1-pixel replicate-pad is added at the far edge so that every pair of
adjacent tiles shares a boundary vertex → gapless join when printed.

The source PNG represents a 1 m × 1 m (1000 mm × 1000 mm) world, so each
tile is exactly 250 mm × 250 mm at 1:1 simulation scale.

Numbering (row-major, top-left = 01):
    01 02 03 04
    05 06 07 08
    09 10 11 12
    13 14 15 16

Outputs land directly in terrain/stl/   (no intermediate folders).

Usage:
  python scripts/terrain/heightmap_to_tiles.py \
      --png outputs/wavelength_sweep/18-500/terrains/wl18.0mm_s46/1.png \
      --z-max 0.016
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from heightmap_to_stl import build_watertight_mesh, write_binary_stl  # type: ignore


WORLD_SIZE_MM = 1000.0   # MuJoCo world: 1 m × 1 m
GRID          = 4        # 4 × 4 = 16 tiles
TILE_MM       = WORLD_SIZE_MM / GRID   # 250 mm per tile


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--png", required=True, help="Input heightmap PNG")
    p.add_argument("--z-max", type=float, required=True,
                   help="MuJoCo z_max in metres (pixel 255 → this height). "
                        "For amplitude=0.008 m use 0.016.")
    p.add_argument("--out", default=None,
                   help="Output folder (default: terrain/stl relative to "
                        "the project root)")
    p.add_argument("--base", type=float, default=2.0,
                   help="Base plate thickness in mm (default: 2)")
    args = p.parse_args()

    # ── Resolve output path ───────────────────────────────────────────────
    if args.out:
        out_dir = args.out
    else:
        project_root = os.path.normpath(
            os.path.join(SCRIPT_DIR, "..", ".."))
        out_dir = os.path.join(project_root, "terrain", "stl")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load PNG (no resampling, no smoothing) ────────────────────────────
    print(f"[1/4] Loading {args.png}")
    img = Image.open(args.png).convert("L")
    h_full = np.array(img, dtype=np.float32) / 255.0
    H, W = h_full.shape
    print(f"      Source resolution: {H} × {W}  "
          f"(each pixel = {WORLD_SIZE_MM / W:.4f} mm in world)")

    if W % GRID != 0 or H % GRID != 0:
        print(f"  WARNING: source not divisible by {GRID}; cropping to "
              f"{(H // GRID) * GRID} × {(W // GRID) * GRID}")
        h_full = h_full[: (H // GRID) * GRID, : (W // GRID) * GRID]
        H, W = h_full.shape

    cells = H // GRID   # pixels per tile side, before the shared-edge pad
    # E.g. 1024 / 4 = 256 → each tile 256 px, 250 mm at exact scale.

    # ── Add a 1-pixel replicate pad at the far (south & east) edges ──────
    # This makes adjacent tiles share their boundary pixel so they join
    # without any gap. The pad is copied from the edge row/col, so no
    # artificial height is introduced.
    h_padded = np.pad(h_full, ((0, 1), (0, 1)), mode="edge")
    print(f"[2/4] Padded to {h_padded.shape[0]} × {h_padded.shape[1]} "
          f"(replicate-edge, for shared-vertex joining)")

    z_range_mm = args.z_max * 1000.0
    print(f"[3/4] Tile footprint: {TILE_MM:.1f} × {TILE_MM:.1f} mm   "
          f"Z relief: {z_range_mm:.2f} mm   Base: {args.base:.1f} mm")
    print(f"      Output folder:  {out_dir}")
    print(f"      Writing 16 tiles (direct pixel slice, no resampling)...")
    print()

    # ── Slice and write each tile ─────────────────────────────────────────
    idx = 1
    for row in range(GRID):
        for col in range(GRID):
            r0 = row * cells
            c0 = col * cells
            # Take cells+1 pixels per side so the last pixel of this tile
            # is the first pixel of the next → shared vertex → gapless.
            tile = h_padded[r0 : r0 + cells + 1, c0 : c0 + cells + 1]

            verts, faces = build_watertight_mesh(
                tile,
                xy_size_mm=TILE_MM,
                z_range_mm=z_range_mm,
                base_mm=args.base,
            )

            stl_path = os.path.join(out_dir, f"tile_{idx:02d}.stl")
            write_binary_stl(stl_path, verts, faces)
            size_mb = os.path.getsize(stl_path) / (1024 * 1024)
            print(f"  [{idx:2d}/16]  row={row} col={col}  "
                  f"px[{r0}:{r0+cells+1}, {c0}:{c0+cells+1}]  "
                  f"→  tile_{idx:02d}.stl   ({size_mb:.1f} MB)")
            idx += 1

    print()
    print(f"[4/4] Done. 16 tiles saved in {out_dir}")
    print()
    print(f"  Layout (place on table in this order — row-major, top-left = 01):")
    print(f"    01 02 03 04    ← north edge (image row 0)")
    print(f"    05 06 07 08")
    print(f"    09 10 11 12")
    print(f"    13 14 15 16    ← south edge")
    print()
    print(f"  Each tile: {TILE_MM:.0f} × {TILE_MM:.0f} × "
          f"{args.base + z_range_mm:.1f} mm  (base {args.base} + relief "
          f"{z_range_mm:.1f} mm)")
    print(f"  Adjacent tiles share boundary pixels — they join GAPLESS.")
    print(f"  Set 100% infill in your slicer for a fully solid print.")


if __name__ == "__main__":
    main()
