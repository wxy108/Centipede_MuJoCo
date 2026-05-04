#!/usr/bin/env python3
"""
heightmap_to_stl.py — Convert a MuJoCo terrain PNG into a watertight STL
                      that exactly matches what the simulation saw.

Pixel mapping (matches MuJoCo hfield convention used by wavelength_sweep.py):
  - The simulated world is 1 m × 1 m (world_half = 0.5 in configs/terrain.yaml)
  - Pixel value 0   → physical height 0
  - Pixel value 255 → physical height = z_max  (metres)
  - So physical_z(p) = (p / 255) * z_max

The script builds a watertight solid with:
  - top surface = the heightmap
  - 4 side walls dropped to a flat base
  - flat bottom
matching exactly the geometry MuJoCo loaded.

───────────────────────────────────────────────────────────────────────────────
USAGE
───────────────────────────────────────────────────────────────────────────────
  # True-scale (1:1 with simulation, 1000 mm × 1000 mm STL):
  python scripts/terrain/heightmap_to_stl.py \
      --png outputs/wavelength_sweep/<run>/terrains/<tag>/1.png \
      --z-max 0.016 --true-scale --out terrain.stl

  # True-scale but crop a centred 150 mm × 150 mm patch (recommended for
  # desktop 3D printers). Shape, heights and millimetre scale are all exact:
  python scripts/terrain/heightmap_to_stl.py \
      --png <png> --z-max 0.016 --true-scale --crop-mm 150 --out patch.stl

  # Scaled-down tile (keeps 1:1 XY:Z aspect ratio):
  python scripts/terrain/heightmap_to_stl.py \
      --png <png> --z-max 0.016 --print-size 100 --out small.stl

NOTES
  --z-max      metres. This is `2 * peak_m` from the sweep
               (e.g. amplitude 0.008 m → z_max ≈ 0.016 m).
               wavelength_sweep.py writes it into the MJCF hfield size.
  --true-scale Force 1000 mm XY (1 m) and z_max mm on Z.  Overrides --print-size.
  --crop-mm    If set, keep only a centred N × N mm patch of the terrain.
               Shape/heights/scale inside that patch are untouched.
  --print-size mm. XY side length when NOT in true-scale mode (default 100 mm).
               Z is scaled the same ratio to preserve aspect.
  --base       Base plate thickness in mm (default 2 mm).
  --downsample Max grid resolution before meshing (default 256). 1024×1024
               STLs are ~50 MB; 256×256 is plenty for printing.
"""

import argparse
import os
import struct
import numpy as np
from PIL import Image


# ═════════════════════════════════════════════════════════════════════════════
# Heightmap loading and cropping
# ═════════════════════════════════════════════════════════════════════════════

def load_heightmap(png_path: str):
    """Load PNG as float array in [0,1]. No resizing."""
    img = Image.open(png_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def crop_center(arr: np.ndarray, keep_fraction: float) -> np.ndarray:
    """Crop a centred square of keep_fraction × original side length."""
    H, W = arr.shape
    keep = max(2, int(round(min(H, W) * keep_fraction)))
    i0 = (H - keep) // 2
    j0 = (W - keep) // 2
    return arr[i0:i0 + keep, j0:j0 + keep]


def downsample_array(arr: np.ndarray, target: int) -> np.ndarray:
    """Lanczos-downsample a float heightmap via PIL."""
    if target <= 0 or (arr.shape[0] <= target and arr.shape[1] <= target):
        return arr
    img = Image.fromarray((arr * 65535).astype(np.uint16), mode="I;16")
    img = img.resize((target, target), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 65535.0


# ═════════════════════════════════════════════════════════════════════════════
# Mesh construction
# ═════════════════════════════════════════════════════════════════════════════

def build_watertight_mesh(height_norm: np.ndarray,
                          xy_size_mm: float,
                          z_range_mm: float,
                          base_mm: float):
    """Build (vertices, faces) for a closed solid.

    height_norm : (H, W) float in [0,1] — normalised pixel heights
    xy_size_mm  : side length of square in millimetres
    z_range_mm  : physical peak-to-base height in mm (pixel 255 → base+this)
    base_mm     : thickness of solid base plate under the lowest terrain point
    """
    H, W = height_norm.shape
    xs = np.linspace(0.0, xy_size_mm, W, dtype=np.float32)
    ys = np.linspace(0.0, xy_size_mm, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    Z_top = base_mm + height_norm * z_range_mm

    n_top = H * W
    top_verts = np.stack([X.ravel(), Y.ravel(), Z_top.ravel()], axis=1)
    bot_verts = np.stack([X.ravel(), Y.ravel(),
                          np.zeros(n_top, dtype=np.float32)], axis=1)
    vertices = np.vstack([top_verts, bot_verts])  # bottom offset = n_top

    faces = []

    # Top surface (CCW from above, +Z normal)
    for i in range(H - 1):
        for j in range(W - 1):
            a = i * W + j
            b = a + 1
            c = (i + 1) * W + j
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])

    # Bottom surface (reverse winding, −Z normal)
    bo = n_top
    for i in range(H - 1):
        for j in range(W - 1):
            a = bo + i * W + j
            b = a + 1
            c = bo + (i + 1) * W + j
            d = c + 1
            faces.append([a, d, b])
            faces.append([a, c, d])

    # Side walls — each quad becomes two triangles CCW from outside
    def wall_quad(t0, t1, b0, b1):
        faces.append([t0, t1, b1])
        faces.append([t0, b1, b0])

    # wall_quad invariant: b0 is directly below t0, b1 is directly below t1.
    # This makes the two triangles (t0,t1,b1) and (t0,b1,b0) tile the quad
    # along the t0↔b1 diagonal with NO gaps and NO overlaps.

    # y=0 edge (south, −Y normal)
    for j in range(W - 1):
        t0 = 0 * W + j
        t1 = 0 * W + (j + 1)
        b0 = bo + 0 * W + j
        b1 = bo + 0 * W + (j + 1)
        # Reverse winding so outward normal is −Y; swap b0/b1 to keep invariant.
        wall_quad(t1, t0, b1, b0)

    # y=max edge (north, +Y normal)
    for j in range(W - 1):
        t0 = (H - 1) * W + j
        t1 = (H - 1) * W + (j + 1)
        b0 = bo + (H - 1) * W + j
        b1 = bo + (H - 1) * W + (j + 1)
        wall_quad(t0, t1, b0, b1)

    # x=0 edge (west, −X normal)
    for i in range(H - 1):
        t0 = i * W + 0
        t1 = (i + 1) * W + 0
        b0 = bo + i * W + 0
        b1 = bo + (i + 1) * W + 0
        wall_quad(t0, t1, b0, b1)

    # x=max edge (east, +X normal)
    for i in range(H - 1):
        t0 = i * W + (W - 1)
        t1 = (i + 1) * W + (W - 1)
        b0 = bo + i * W + (W - 1)
        b1 = bo + (i + 1) * W + (W - 1)
        # Reverse winding so outward normal is +X; swap b0/b1 to keep invariant.
        wall_quad(t1, t0, b1, b0)

    return vertices, np.asarray(faces, dtype=np.int32)


def write_binary_stl(path: str, vertices: np.ndarray, faces: np.ndarray):
    n_tri = len(faces)
    v = vertices[faces]                              # (n_tri, 3, 3)
    u = v[:, 1] - v[:, 0]
    w = v[:, 2] - v[:, 0]
    nrm = np.cross(u, w)
    lens = np.linalg.norm(nrm, axis=1, keepdims=True)
    lens[lens == 0] = 1.0
    nrm = nrm / lens

    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", n_tri))
        for i in range(n_tri):
            f.write(struct.pack("<3f", *nrm[i]))
            f.write(struct.pack("<3f", *v[i, 0]))
            f.write(struct.pack("<3f", *v[i, 1]))
            f.write(struct.pack("<3f", *v[i, 2]))
            f.write(b"\x00\x00")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

WORLD_SIZE_MM = 1000.0  # simulation world is 1 m × 1 m (world_half=0.5)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--png", required=True, help="Input heightmap PNG")
    p.add_argument("--z-max", type=float, required=True,
                   help="MuJoCo z_max in metres (pixel 255 → this height). "
                        "This is 2*peak_m used by wavelength_sweep.py.")
    p.add_argument("--out", default=None, help="Output STL path (default: <png>.stl)")
    p.add_argument("--true-scale", action="store_true",
                   help="Use 1000 mm × 1000 mm × (z_max mm) — exact simulation size.")
    p.add_argument("--crop-mm", type=float, default=None,
                   help="Crop a centred N×N mm patch (in true-scale millimetres) "
                        "out of the 1000 mm tile before building the STL.")
    p.add_argument("--print-size", type=float, default=100.0,
                   help="XY side length in mm when NOT in true-scale mode "
                        "(default 100). Z is scaled the same ratio.")
    p.add_argument("--base", type=float, default=2.0,
                   help="Base plate thickness in mm (default 2)")
    p.add_argument("--downsample", type=int, default=256,
                   help="Max grid resolution before meshing (default 256)")
    args = p.parse_args()

    out = args.out or (os.path.splitext(args.png)[0] + ".stl")

    print(f"[1/5] Loading {args.png}")
    h_full = load_heightmap(args.png)
    H0, W0 = h_full.shape
    print(f"      source resolution: {H0}×{W0}")

    # ── Crop (in true simulation millimetres) ─────────────────────────────
    if args.crop_mm is not None:
        if args.crop_mm <= 0 or args.crop_mm > WORLD_SIZE_MM:
            raise SystemExit(f"--crop-mm must be in (0, {WORLD_SIZE_MM}]")
        keep_fraction = args.crop_mm / WORLD_SIZE_MM
        h_cropped = crop_center(h_full, keep_fraction)
        crop_xy_mm_full = args.crop_mm   # physical mm the crop represents
        print(f"[2/5] Cropped centred {args.crop_mm:.1f} mm × {args.crop_mm:.1f} mm "
              f"→ {h_cropped.shape[0]}×{h_cropped.shape[1]} px")
    else:
        h_cropped = h_full
        crop_xy_mm_full = WORLD_SIZE_MM
        print(f"[2/5] No crop (full {WORLD_SIZE_MM:.0f} mm tile)")

    # ── Downsample for mesh size ──────────────────────────────────────────
    h_mesh = downsample_array(h_cropped, args.downsample)
    print(f"[3/5] Mesh grid: {h_mesh.shape[0]}×{h_mesh.shape[1]}")

    # ── Determine physical dimensions ─────────────────────────────────────
    z_max_mm_true = args.z_max * 1000.0

    if args.true_scale:
        xy_mm = crop_xy_mm_full            # 1000 mm, or the crop size
        z_mm  = z_max_mm_true              # z_max in mm — unchanged
        mode  = f"TRUE SCALE (1:1 with simulation)"
    else:
        # Proportional shrink: xy_mm / 1000 = z_mm / z_max_mm_true
        xy_mm = args.print_size
        ratio = xy_mm / crop_xy_mm_full
        z_mm  = z_max_mm_true * ratio
        mode  = f"scaled {crop_xy_mm_full:.0f}mm → {xy_mm:.0f}mm ({ratio:.3f}×)"

    print(f"[4/5] {mode}")
    print(f"      XY tile: {xy_mm:.2f} mm   Z relief: {z_mm:.3f} mm   "
          f"Base: {args.base:.1f} mm")

    verts, faces = build_watertight_mesh(h_mesh, xy_mm, z_mm, args.base)
    print(f"      Mesh: {len(verts)} vertices, {len(faces)} triangles")

    print(f"[5/5] Writing {out}")
    write_binary_stl(out, verts, faces)
    file_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"      Done. {file_mb:.1f} MB")

    total_h = args.base + z_mm
    print()
    print(f"  Final print box: {xy_mm:.2f} × {xy_mm:.2f} × {total_h:.2f} mm")
    if args.true_scale:
        print(f"  ✓ Heights and XY layout are IDENTICAL to the simulated terrain.")
        if args.crop_mm is not None:
            print(f"  ✓ This is a centred {args.crop_mm:.0f} mm × {args.crop_mm:.0f} mm "
                  f"patch of the original 1000 mm tile.")
    else:
        print(f"  Shape preserved; XY and Z both shrunk by "
              f"{xy_mm / crop_xy_mm_full:.3f}× relative to simulation.")


if __name__ == "__main__":
    main()
