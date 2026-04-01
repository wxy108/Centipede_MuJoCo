#!/usr/bin/env python3
"""
Multi-frequency spectral terrain generator for centipede MuJoCo/FARMS simulation.

Design (Option A + per-level z_max)
------------------------------------
Each roughness level gets its OWN PNG with unique spatial patterns (different
band amplitudes → different bump shapes/locations).  Every PNG is normalised
to the full 0-255 range so MuJoCo gets maximum height resolution (no
staircases).  Physical roughness is then controlled by setting a per-level
z_max in the XML — mild terrain gets a small z_max, rough terrain gets a
large z_max.

  MuJoCo normalises the PNG internally:
    min pixel → height 0
    max pixel → height z_max
  So a full-range PNG + per-level z_max = full control.

Three non-overlapping spectral bands, each with independent amplitude:
  Low  (0.5–3 cyc/m)  → body-scale hills       → --low-amp
  Mid  (3–10 cyc/m)   → segment-scale mounds    → --mid-amp
  High (10–40 cyc/m)  → leg-scale texture        → --high-amp

Different amplitudes create different spatial patterns even though the PNG
is normalised — the RATIO between bands determines the terrain character.

Usage
-----
  python generate_terrain_multifreq.py --low-amp 0.004 --mid-amp 0.002 --high-amp 0.001 --seed 0
  python generate_terrain_multifreq.py --low-amp 0 --mid-amp 0 --high-amp 0 --seed 0   # flat
"""

import argparse
import os
import yaml
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "..", "configs", "terrain.yaml")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "..", "output")


# ── config ────────────────────────────────────────────────────────────────────
def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── core ──────────────────────────────────────────────────────────────────────
def _spectral_band(image_size, world_size, freq_min, freq_max,
                   n_components, amplitude, orientation_spread, rng):
    """Sum of n random sinusoids in [freq_min, freq_max] cycles/m, normalised to given amplitude."""
    x = np.linspace(-world_size / 2, world_size / 2, image_size)
    X, Y = np.meshgrid(x, x)
    h = np.zeros((image_size, image_size), dtype=np.float64)
    for _ in range(n_components):
        freq  = rng.uniform(freq_min, freq_max)
        angle = rng.uniform(0.0, np.pi * orientation_spread)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        kx = 2.0 * np.pi * freq * np.cos(angle)
        ky = 2.0 * np.pi * freq * np.sin(angle)
        h += np.sin(kx * X + ky * Y + phase)
    std = h.std()
    if std > 1e-12:
        h = h / std * amplitude
    return h


def _apply_centering(h, image_size, cfg_centering):
    """
    Blend heightmap to zero inside a circle centred at pixel (N//2, N//2),
    which corresponds to world origin (0, 0) for a centred MuJoCo hfield geom.

    blend = 0  inside  radius        → perfectly flat (h = 0)
    blend = 1  outside radius+blend  → full roughness
    smooth cosine transition in between.
    """
    if not cfg_centering.get("enabled", False):
        return h

    radius_px = cfg_centering["radius_fraction"] * image_size
    blend_px  = cfg_centering["blend_width"]     * image_size

    cx = cy = image_size // 2    # pixel (512,512) ↔ world (0, 0)
    row, col = np.ogrid[0:image_size, 0:image_size]
    dist = np.sqrt((col - cx) ** 2 + (row - cy) ** 2)

    # Smooth cosine blend: 0 at centre → 1 at radius+blend_px
    t = np.clip((dist - radius_px) / max(blend_px, 1.0), 0.0, 1.0)
    blend = 0.5 * (1.0 - np.cos(np.pi * t))   # smoother than linear
    return h * blend


def generate_terrain(cfg, low_amp, mid_amp, high_amp, seed):
    """
    Generate a float heightmap in metres (mean ≈ 0).

    Each of the three spectral bands has its own explicit amplitude:
      low_amp  → body-scale hills     (0.5–3 cyc/m)
      mid_amp  → segment-scale mounds (3–10 cyc/m)
      high_amp → leg-scale texture    (10–40 cyc/m)

    Frequency ranges come from terrain_config.yaml — they define the
    physical scale of each band and stay fixed across levels.

    Returns
    -------
    h_m           : np.ndarray float64, shape (N, N), metres
    roughness_idx : float  — σ_h / terrain_world_height
    rms_cm        : float  — σ_h in centimetres
    """
    rng        = np.random.default_rng(int(seed))
    cfg_world  = cfg["world"]
    image_size = int(cfg_world["image_size"])
    world_size = float(cfg_world["size"])
    sigma      = float(cfg_world["smooth_sigma"])
    T_height   = float(cfg_world["terrain_world_height"])
    bands      = cfg["spectral_bands"]

    A_low  = float(low_amp)
    A_mid  = float(mid_amp)
    A_high = float(high_amp)

    # ── pure-flat shortcut ────────────────────────────────────────────────
    if A_low == 0.0 and A_mid == 0.0 and A_high == 0.0:
        return np.zeros((image_size, image_size), dtype=np.float64), 0.0, 0.0

    # ── low band (body-scale hills) ──────────────────────────────────────
    h = _spectral_band(
        image_size, world_size,
        freq_min=float(bands["low"]["freq_min"]),
        freq_max=float(bands["low"]["freq_max"]),
        n_components=int(bands["low"]["n_components"]),
        amplitude=A_low,
        orientation_spread=float(bands["low"]["orientation_spread"]),
        rng=rng,
    )

    # ── mid band (segment-scale mounds) ──────────────────────────────────
    h += _spectral_band(
        image_size, world_size,
        freq_min=float(bands["mid"]["freq_min"]),
        freq_max=float(bands["mid"]["freq_max"]),
        n_components=int(bands["mid"]["n_components"]),
        amplitude=A_mid,
        orientation_spread=float(bands["mid"]["orientation_spread"]),
        rng=rng,
    )

    # ── high band (leg-scale texture) ────────────────────────────────────
    h += _spectral_band(
        image_size, world_size,
        freq_min=float(bands["high"]["freq_min"]),
        freq_max=float(bands["high"]["freq_max"]),
        n_components=int(bands["high"]["n_components"]),
        amplitude=A_high,
        orientation_spread=float(bands["high"]["orientation_spread"]),
        rng=rng,
    )

    # ── Gaussian smooth ───────────────────────────────────────────────────
    h = gaussian_filter(h, sigma=sigma)

    # ── Zero-centre ──────────────────────────────────────────────────────
    h -= h.mean()

    # ── centering (flat pad disabled — see terrain_config.yaml) ──────────
    h = _apply_centering(h, image_size, cfg["centering"])

    # ── metrics (before normalisation, in physical units) ────────────────
    rms_m  = float(np.std(h))
    peak_m = float(max(abs(h.min()), abs(h.max())))

    return h, rms_m, peak_m


def heightmap_to_png(h_m):
    """
    Map float heightmap → uint8 PNG, always using the FULL 0-255 range.

    MuJoCo normalises hfield data internally (min pixel → 0, max pixel →
    z_max), so we must use the full pixel range for maximum height
    resolution.  Physical roughness is controlled by z_max in the XML,
    not by the pixel spread.

    Mapping:  h_min → pixel 0
              h_max → pixel 255
    """
    h_min = h_m.min()
    h_max = h_m.max()
    span  = h_max - h_min
    if span < 1e-15:
        # Perfectly flat — return uniform mid-grey
        return np.full(h_m.shape, 128, dtype=np.uint8)
    img_f = (h_m - h_min) / span * 255.0
    return np.round(img_f).astype(np.uint8)


def save_terrain(h_m, rms_m, peak_m, cfg, low_amp, mid_amp, high_amp, seed):
    """Save PNG to terrain_output/ and print stats.

    Returns (png_path, rms_m, peak_m) so the caller can compute z_max.
    """
    arr     = heightmap_to_png(h_m)
    tag     = f"low{low_amp:.4f}_mid{mid_amp:.4f}_high{high_amp:.4f}_s{seed}"
    out_dir = os.path.join(OUTPUT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "1.png")
    Image.fromarray(arr, mode="L").save(png_path)
    print(f"Terrain saved to: {out_dir}")
    print(f"  low={low_amp:.4f}  mid={mid_amp:.4f}  high={high_amp:.4f}")
    print(f"  σ_h (RMS) : {rms_m*100:.3f} cm")
    print(f"  peak      : {peak_m*100:.3f} cm")
    print(f"  Pixel range: [{arr.min()}, {arr.max()}]  (should be 0–255)")
    # Recommended z_max = 2 * peak (so peak pixel maps to z_max/2,
    # matching the spawn-at-midpoint convention)
    z_max_rec = 2.0 * peak_m
    print(f"  Recommended z_max: {z_max_rec:.4f} m")
    return png_path, rms_m, peak_m


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--low-amp",  type=float, default=0.004,
                   help="Low-band amplitude (body-scale hills, 0.5–3 cyc/m)")
    p.add_argument("--mid-amp",  type=float, default=0.002,
                   help="Mid-band amplitude (segment-scale mounds, 3–10 cyc/m)")
    p.add_argument("--high-amp", type=float, default=0.001,
                   help="High-band amplitude (leg-scale bumps, 10–40 cyc/m)")
    p.add_argument("--seed",     type=int,   default=0,
                   help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    h_m, rms_m, peak_m = generate_terrain(cfg, args.low_amp, args.mid_amp,
                                           args.high_amp, args.seed)
    save_terrain(h_m, rms_m, peak_m, cfg, args.low_amp, args.mid_amp,
                 args.high_amp, args.seed)


if __name__ == "__main__":
    main()
