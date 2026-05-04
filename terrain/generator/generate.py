#!/usr/bin/env python3
"""
Morphology-driven spectral terrain generator.

Four bands, each at EXACTLY the robot's characteristic wavelength:
  World   → λ = L_w (arena half-extent)   → --world-amp
  Body    → λ = L_b (body length)          → --body-amp
  Segment → λ = L_s = L_b / n_wave         → --segment-amp
  Leg     → λ = L_ℓ (leg length)           → --leg-amp

To adapt to a different robot: change configs/terrain.yaml → morphology.

Usage
-----
  python generate.py --world-amp 0.006 --body-amp 0.003 --segment-amp 0.002 --leg-amp 0.001
  python generate.py --body-amp 0.005                   # body-only terrain
  python generate.py --world-amp 0 --body-amp 0 --segment-amp 0 --leg-amp 0  # flat
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


def resolve_morphology(cfg):
    """Compute all characteristic lengths from the morphology section."""
    morph = cfg["morphology"]
    L_b    = float(morph["L_b"])
    n_wave = float(morph["n_wave"])
    L_s    = L_b / n_wave
    L_ell  = float(morph["L_ell"])
    L_w    = float(cfg["world"]["size"])   # arena half-extent

    return {
        "L_w":   L_w,
        "L_b":   L_b,
        "L_s":   L_s,
        "L_ell": L_ell,
        "n_wave": n_wave,
    }


def resolve_bands(cfg, lengths):
    """
    Convert band definitions into concrete frequencies.
    Each band uses EXACTLY its anchor wavelength (narrow band).
    """
    bands_cfg = cfg["spectral_bands"]
    anchor_map = {
        "L_w":   lengths["L_w"],
        "L_b":   lengths["L_b"],
        "L_s":   lengths["L_s"],
        "L_ell": lengths["L_ell"],
    }

    resolved = []
    for band_name in ["world", "body", "segment", "leg"]:
        if band_name not in bands_cfg:
            continue
        bc = bands_cfg[band_name]
        anchor_key = bc["anchor"]
        wavelength = anchor_map[anchor_key]
        freq = 1.0 / wavelength   # exact frequency (cycles/m)

        # Narrow band: ±20% around the exact frequency
        # This gives slight variation so terrain isn't a perfect single sinusoid
        freq_min = freq * 0.8
        freq_max = freq * 1.2

        resolved.append({
            "name":               band_name,
            "anchor":             anchor_key,
            "wavelength_m":       wavelength,
            "freq_exact":         freq,
            "freq_min":           freq_min,
            "freq_max":           freq_max,
            "n_components":       int(bc["n_components"]),
            "amplitude":          float(bc["amplitude"]),
            "orientation_spread": float(bc.get("orientation_spread", 1.0)),
        })

    return resolved


# ── core ──────────────────────────────────────────────────────────────────────
def _spectral_band(image_size, world_size, freq_min, freq_max,
                   n_components, amplitude, orientation_spread, rng):
    """Sum of n random sinusoids in [freq_min, freq_max] cycles/m."""
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
    """Blend heightmap to zero inside a centred circle (flat spawn pad)."""
    if not cfg_centering.get("enabled", False):
        return h
    radius_px = cfg_centering["radius_fraction"] * image_size
    blend_px  = cfg_centering["blend_width"]     * image_size
    cx = cy = image_size // 2
    row, col = np.ogrid[0:image_size, 0:image_size]
    dist = np.sqrt((col - cx) ** 2 + (row - cy) ** 2)
    t = np.clip((dist - radius_px) / max(blend_px, 1.0), 0.0, 1.0)
    blend = 0.5 * (1.0 - np.cos(np.pi * t))
    return h * blend


def generate_terrain(cfg, amp_world, amp_body, amp_segment, amp_leg, seed):
    """
    Generate a float heightmap in metres (mean ≈ 0).

    Returns: (h_m, rms_m, peak_m, bands)
    """
    rng        = np.random.default_rng(int(seed))
    cfg_world  = cfg["world"]
    image_size = int(cfg_world["image_size"])
    world_size = float(cfg_world["size"]) * 2.0   # half-extent → full extent
    sigma      = float(cfg_world["smooth_sigma"])

    lengths = resolve_morphology(cfg)
    bands   = resolve_bands(cfg, lengths)

    # Map amplitude overrides to band names
    amp_map = {
        "world":   float(amp_world),
        "body":    float(amp_body),
        "segment": float(amp_segment),
        "leg":     float(amp_leg),
    }

    # Pure-flat shortcut
    if all(amp_map.get(b["name"], 0) == 0.0 for b in bands):
        return (np.zeros((image_size, image_size), dtype=np.float64),
                0.0, 0.0, bands)

    # Accumulate bands
    h = np.zeros((image_size, image_size), dtype=np.float64)

    for band in bands:
        amp = amp_map.get(band["name"], band["amplitude"])
        if amp <= 0:
            continue
        h += _spectral_band(
            image_size, world_size,
            freq_min=band["freq_min"],
            freq_max=band["freq_max"],
            n_components=band["n_components"],
            amplitude=amp,
            orientation_spread=band["orientation_spread"],
            rng=rng,
        )

    # Gaussian smooth
    h = gaussian_filter(h, sigma=sigma)

    # Zero-centre
    h -= h.mean()

    # Centering pad
    h = _apply_centering(h, image_size, cfg["centering"])

    # Metrics
    rms_m  = float(np.std(h))
    peak_m = float(max(abs(h.min()), abs(h.max())))

    return h, rms_m, peak_m, bands


def heightmap_to_png(h_m):
    """Map float heightmap → uint8 PNG, full 0-255 range."""
    h_min = h_m.min()
    h_max = h_m.max()
    span  = h_max - h_min
    if span < 1e-15:
        return np.full(h_m.shape, 128, dtype=np.uint8)
    img_f = (h_m - h_min) / span * 255.0
    return np.round(img_f).astype(np.uint8)


def save_terrain(h_m, rms_m, peak_m, bands, cfg,
                 amp_w, amp_b, amp_s, amp_l, seed):
    """Save PNG and print stats."""
    arr = heightmap_to_png(h_m)
    tag = f"W{amp_w:.4f}_B{amp_b:.4f}_S{amp_s:.4f}_L{amp_l:.4f}_s{seed}"
    out_dir = os.path.join(OUTPUT_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "1.png")
    Image.fromarray(arr, mode="L").save(png_path)

    print(f"\nTerrain saved to: {out_dir}")
    print(f"  world={amp_w:.4f}  body={amp_b:.4f}  segment={amp_s:.4f}  leg={amp_l:.4f}")
    print(f"  σ_h (RMS) : {rms_m*1000:.2f} mm")
    print(f"  peak      : {peak_m*1000:.2f} mm")
    print(f"  Pixel range: [{arr.min()}, {arr.max()}]")
    print(f"  Recommended z_max: {2.0 * peak_m:.4f} m")

    print(f"\n  Bands (exact wavelength from morphology):")
    for b in bands:
        print(f"    {b['name']:>7s}: λ = {b['wavelength_m']*1000:>7.1f} mm  "
              f"(f = {b['freq_exact']:>6.1f} cyc/m)  "
              f"n={b['n_components']}  "
              f"amp={amp_map_from_args(amp_w,amp_b,amp_s,amp_l).get(b['name'],0):.4f}")

    return png_path, rms_m, peak_m


def amp_map_from_args(amp_w, amp_b, amp_s, amp_l):
    return {"world": amp_w, "body": amp_b, "segment": amp_s, "leg": amp_l}


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--world-amp",   type=float, default=None,
                   help="World-band amplitude (λ = arena size)")
    p.add_argument("--body-amp",    type=float, default=None,
                   help="Body-band amplitude (λ = L_b)")
    p.add_argument("--segment-amp", type=float, default=None,
                   help="Segment-band amplitude (λ = L_b / n_wave)")
    p.add_argument("--leg-amp",     type=float, default=None,
                   help="Leg-band amplitude (λ = L_ℓ)")
    p.add_argument("--seed",        type=int,   default=0,
                   help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resolve morphology
    lengths = resolve_morphology(cfg)
    bands   = resolve_bands(cfg, lengths)
    band_defaults = {b["name"]: b["amplitude"] for b in bands}

    # Amplitudes: CLI overrides config defaults
    amp_w = args.world_amp   if args.world_amp   is not None else band_defaults.get("world",   0.006)
    amp_b = args.body_amp    if args.body_amp    is not None else band_defaults.get("body",    0.003)
    amp_s = args.segment_amp if args.segment_amp is not None else band_defaults.get("segment", 0.002)
    amp_l = args.leg_amp     if args.leg_amp     is not None else band_defaults.get("leg",     0.001)

    # Print morphology
    print("=" * 60)
    print("Morphology-Driven Terrain Generator")
    print("=" * 60)
    print(f"  L_w     = {lengths['L_w']*1000:.1f} mm  (world / arena half-extent)")
    print(f"  L_b     = {lengths['L_b']*1000:.1f} mm  (body length)")
    print(f"  n_wave  = {lengths['n_wave']:.0f}")
    print(f"  L_s     = {lengths['L_s']*1000:.1f} mm  (segment = L_b / n_wave)")
    print(f"  L_ell   = {lengths['L_ell']*1000:.1f} mm  (leg length)")

    h_m, rms_m, peak_m, bands = generate_terrain(
        cfg, amp_w, amp_b, amp_s, amp_l, args.seed)
    save_terrain(h_m, rms_m, peak_m, bands, cfg,
                 amp_w, amp_b, amp_s, amp_l, args.seed)


if __name__ == "__main__":
    main()
