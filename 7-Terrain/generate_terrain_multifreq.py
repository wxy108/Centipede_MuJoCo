#!/usr/bin/env python3
"""
generate_terrain_multifreq.py
==============================
Multi-frequency spectral terrain generator for MuJoCo heightmap terrains.

Philosophy
----------
Terrain is represented as a sum of sinusoidal components across three
spectral bands (low / mid / high frequency), each with random phases:

    h(x, y) = Σ_k  A_k · sin(2π·f_k·(x·cosθ_k + y·sinθ_k) + φ_k)

  Low  (0.3–1.5 /m)  : landscape undulation — hills and valleys
  Mid  (2–6 /m)      : secondary features — rocks, mounds
  High (8–24 /m)     : surface roughness / obstacles at leg-spacing scale

Why sinusoidal decomposition instead of Perlin noise?
  - Spectral content is explicit and controllable.
  - Band amplitudes directly map to physical terrain statistics.
  - A scalar roughness_index (RMS height) is analytically derivable,
    giving the ML model a clean continuous target.
  - Reproducible: seed → exact same terrain, every time.

Roughness index
---------------
    σ_h = sqrt(Σ_k  (A_k/√2)² )   [RMS height of the full spectrum]
    roughness_index = σ_h / world_height   (normalised 0–1)

This is the ground-truth label stored in metadata.json that the terrain
recognition model will try to predict from centipede joint sensor signals.

Usage
-----
    # Single terrain (uses config defaults)
    python generate_terrain_multifreq.py --config terrain_config.yaml

    # Override amplitudes on the fly
    python generate_terrain_multifreq.py --low-amp 0.02 --high-amp 0.005 --seed 3

    # Run the full parameter sweep (writes all terrains to output directory)
    python generate_terrain_multifreq.py --sweep
"""

import argparse
import json
import math
import os
import shutil

import numpy as np
import yaml
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _MPL = True
except ImportError:
    _MPL = False


# ── config loader ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── spectral terrain core ─────────────────────────────────────────────────────

def _sinusoidal_band(
    X: np.ndarray, Y: np.ndarray,
    n_components: int,
    freq_min: float, freq_max: float,
    amplitude: float,
    orientation_spread: float,
    rng: np.random.Generator,
    freq_centre_override: float = None,
) -> np.ndarray:
    """
    Sum n_components sinusoidal waves with random frequencies, orientations,
    and phases, all scaled to the given amplitude.

    Parameters
    ----------
    X, Y          : 2-D coordinate grids (metres)
    n_components  : number of sinusoidal components
    freq_min/max  : spatial frequency band (cycles / metre)
    amplitude     : peak amplitude (metres) — each component gets A/√n
                    so the RMS of the sum ≈ A/√2 regardless of n
    orientation_spread : 0 → all waves travel along X; 1 → fully random
    rng           : numpy random Generator (seeded externally)
    freq_centre_override : if set, draw frequencies around this centre
                    instead of uniformly in [freq_min, freq_max]
    """
    height = np.zeros_like(X)
    per_component_amp = amplitude / math.sqrt(n_components)

    for _ in range(n_components):
        # Spatial frequency
        if freq_centre_override is not None:
            half_bw = (freq_max - freq_min) / 4.0
            f = np.clip(
                rng.normal(freq_centre_override, half_bw),
                freq_min, freq_max
            )
        else:
            f = rng.uniform(freq_min, freq_max)

        # Wave orientation
        theta = rng.uniform(0, math.pi * orientation_spread)

        # Random phase
        phi = rng.uniform(0, 2.0 * math.pi)

        # Projection onto wave direction
        proj = X * math.cos(theta) + Y * math.sin(theta)
        height += per_component_amp * np.sin(2.0 * math.pi * f * proj + phi)

    return height


def generate_terrain(
    cfg: dict,
    low_amplitude: float = None,
    mid_amplitude: float = None,
    high_amplitude: float = None,
    high_freq_centre: float = None,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """
    Generate a terrain heightmap as a uint8 image and return metadata.

    Parameters
    ----------
    cfg              : parsed terrain_config.yaml dict
    low_amplitude    : override low-band amplitude (metres); None → use config
    mid_amplitude    : override mid-band amplitude
    high_amplitude   : override high-band amplitude
    high_freq_centre : override high-band frequency centre (cycles/m)
    seed             : random seed for reproducibility

    Returns
    -------
    heightmap : (SIZE, SIZE) uint8 array  →  0=lowest, 255=highest point
    metadata  : dict with roughness_index and all generation parameters
    """
    img_cfg = cfg["image"]
    SIZE        = int(img_cfg["size"])
    WORLD       = float(img_cfg["world_size"])
    WORLD_H     = float(img_cfg["world_height"])
    BASELINE    = int(img_cfg["baseline_grey"])

    bands = cfg["spectral_bands"]

    # ── allow per-call amplitude overrides ──
    A_low  = low_amplitude  if low_amplitude  is not None else float(bands["low"]["amplitude"])
    A_high = high_amplitude if high_amplitude is not None else float(bands["high"]["amplitude"])

    # Mid-band scales with max(A_low, A_high) so it's always proportional.
    # When both are zero → pure flat ground, no mid-band at all.
    if mid_amplitude is not None:
        A_mid = mid_amplitude
    else:
        A_mid = float(bands["mid"]["amplitude"]) * max(A_low, A_high) / max(
            float(bands["low"]["amplitude"]) + float(bands["high"]["amplitude"]), 1e-9
        )

    # ── coordinate grids (metres) ──
    lin = np.linspace(0, WORLD, SIZE, endpoint=False)
    X, Y = np.meshgrid(lin, lin)

    rng = np.random.default_rng(seed)

    # ── low band ──
    low_cfg = bands["low"]
    h_low = _sinusoidal_band(
        X, Y,
        n_components        = int(low_cfg["n_components"]),
        freq_min            = float(low_cfg["freq_min"]),
        freq_max            = float(low_cfg["freq_max"]),
        amplitude           = A_low,
        orientation_spread  = float(low_cfg["orientation_spread"]),
        rng                 = rng,
    )

    # ── mid band ──
    mid_cfg = bands["mid"]
    h_mid = _sinusoidal_band(
        X, Y,
        n_components        = int(mid_cfg["n_components"]),
        freq_min            = float(mid_cfg["freq_min"]),
        freq_max            = float(mid_cfg["freq_max"]),
        amplitude           = A_mid,
        orientation_spread  = float(mid_cfg["orientation_spread"]),
        rng                 = rng,
    )

    # ── high band ──
    high_cfg = bands["high"]
    h_high = _sinusoidal_band(
        X, Y,
        n_components        = int(high_cfg["n_components"]),
        freq_min            = float(high_cfg["freq_min"]),
        freq_max            = float(high_cfg["freq_max"]),
        amplitude           = A_high,
        orientation_spread  = float(high_cfg["orientation_spread"]),
        rng                 = rng,
        freq_centre_override= high_freq_centre,
    )

    h_total = h_low + h_mid + h_high   # metres, zero-mean by construction

    # ── smooth to remove angular edges ──────────────────────────────────
    # Each sinusoidal band is already band-limited, but the sum can have
    # sharp pixel-level transitions when high-freq content is strong.
    # A light Gaussian blur (sigma ~ 1 pixel) preserves all features
    # visible at the centipede scale while eliminating staircase artefacts.
    from scipy.ndimage import gaussian_filter
    smooth_sigma = max(0.5, float(img_cfg.get("smooth_sigma", 1.0)))
    h_total = gaussian_filter(h_total, sigma=smooth_sigma)

    # ── centering: flatten spawn zone at world origin (pixel grid centre) ──
    center_cfg = cfg.get("centering", {})
    if center_cfg.get("enabled", True):
        # Pixel grid runs 0→WORLD, so centre pixel = WORLD/2 = world origin (0,0)
        cx = cy = WORLD / 2.0
        r_flat  = float(center_cfg.get("radius_fraction", 0.025)) * WORLD
        r_blend = float(center_cfg.get("blend_width",     0.015)) * WORLD
        dist    = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        # alpha=0 inside flat zone, alpha=1 outside blend zone
        alpha   = np.clip((dist - r_flat) / r_blend, 0.0, 1.0)
        h_total = h_total * alpha

    # ── convert to uint8 (pixel = height in world_height range) ──
    # Map h_total (metres) → pixel: baseline_grey + h / world_height * 127.5
    scale = 127.5 / (WORLD_H / 2.0)
    pixel = BASELINE + h_total * scale
    pixel = np.clip(pixel, 0, 255).astype(np.uint8)

    # ── roughness index (RMS / world_height) ──
    # Analytical RMS of the sum of sinusoids: σ = sqrt(Σ (A_k/√2)²)
    n_low  = int(low_cfg["n_components"])
    n_mid  = int(mid_cfg["n_components"])
    n_high = int(high_cfg["n_components"])
    sigma_h = math.sqrt(
        (A_low  ** 2) / (2 * n_low)  * n_low  +
        (A_mid  ** 2) / (2 * n_mid)  * n_mid  +
        (A_high ** 2) / (2 * n_high) * n_high
    )
    # Simplifies to:
    sigma_h = math.sqrt(A_low**2 / 2 + A_mid**2 / 2 + A_high**2 / 2)
    roughness_index = sigma_h / WORLD_H

    metadata = {
        "seed":             seed,
        "low_amplitude":    A_low,
        "mid_amplitude":    A_mid,
        "high_amplitude":   A_high,
        "high_freq_centre": high_freq_centre,
        "roughness_index":  round(roughness_index, 6),   # ML target
        "sigma_h_metres":   round(sigma_h, 6),
        "world_size":       WORLD,
        "world_height":     WORLD_H,
        "image_size":       SIZE,
    }

    return pixel, metadata


# ── terrain folder helpers ────────────────────────────────────────────────────

def terrain_folder_name(
    low_amp: float,
    high_amp: float,
    high_freq: float,
    seed: int,
) -> str:
    """Human-readable folder name encoding the sweep point."""
    return (f"low{low_amp:.4f}_high{high_amp:.4f}"
            f"_hf{high_freq:.1f}_s{seed}")


def save_terrain(
    heightmap: np.ndarray,
    metadata: dict,
    output_dir: str,
    cfg: dict,
    preview: bool = True,
) -> str:
    """
    Save terrain PNG, metadata JSON, and optional preview figure.

    Returns the path to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1.png  — MuJoCo heightmap convention (matches terrain.xml)
    png_path = os.path.join(output_dir, "1.png")
    Image.fromarray(heightmap, mode="L").save(png_path)

    # metadata.json — ground-truth labels for ML
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Config copy for reproducibility
    if cfg.get("output", {}).get("save_config_copy", True):
        pass   # caller saves config if needed

    # Preview figure
    if preview and _MPL:
        _save_preview(heightmap, metadata, output_dir)

    return output_dir


def _save_preview(heightmap: np.ndarray, metadata: dict, output_dir: str):
    """Save a 2-panel preview: 2D colourmap + 3D surface."""
    SIZE = heightmap.shape[0]
    fig = plt.figure(figsize=(12, 5))

    # 2D colourmap
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(heightmap, cmap="terrain", vmin=0, vmax=255, origin="lower")
    plt.colorbar(im, ax=ax1, label="Pixel value (0=low, 255=high)")
    ax1.set_title(
        f"Terrain heightmap\n"
        f"low={metadata['low_amplitude']*100:.2f} cm  "
        f"high={metadata['high_amplitude']*100:.2f} cm  "
        f"hf={metadata['high_freq_centre']} /m\n"
        f"roughness_index={metadata['roughness_index']:.4f}  seed={metadata['seed']}",
        fontsize=9
    )
    ax1.axis("off")

    # 3D surface (downsampled)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    step = max(1, SIZE // 64)
    Z = heightmap[::step, ::step].astype(float)
    r = np.arange(Z.shape[0])
    Xp, Yp = np.meshgrid(r, r)
    ax2.plot_surface(Xp, Yp, Z, cmap="terrain", linewidth=0, antialiased=True)
    ax2.set_axis_off()
    ax2.set_title("3D view", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "preview.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── full sweep ────────────────────────────────────────────────────────────────

def run_sweep(cfg: dict, base_dir: str = None, preview: bool = True) -> list[dict]:
    """
    Generate all terrain combinations defined in cfg['sweep'].

    Returns a list of metadata dicts (one per terrain).
    """
    sweep   = cfg["sweep"]
    out_cfg = cfg.get("output", {})
    base    = base_dir or out_cfg.get("base_dir", "terrain_output")

    seeds           = sweep["seeds"]
    low_amps        = sweep["low_amplitudes"]
    high_amps       = sweep["high_amplitudes"]
    high_freqs      = sweep["high_freq_centres"]

    total = len(seeds) * len(low_amps) * len(high_amps) * len(high_freqs)

    print("=" * 60)
    print("Multi-frequency terrain sweep")
    print("=" * 60)
    print(f"  Seeds            : {seeds}")
    print(f"  Low amplitudes   : {low_amps} m")
    print(f"  High amplitudes  : {high_amps} m")
    print(f"  High freq centres: {high_freqs} /m")
    print(f"  Total terrains   : {total}")
    print(f"  Output dir       : {base}")
    print()

    all_meta = []
    count    = 0

    for seed in seeds:
        for A_low in low_amps:
            for A_high in high_amps:
                for hf in high_freqs:
                    count += 1
                    folder_name = terrain_folder_name(A_low, A_high, hf, seed)
                    out_dir = os.path.join(base, folder_name)

                    heightmap, meta = generate_terrain(
                        cfg,
                        low_amplitude    = A_low,
                        high_amplitude   = A_high,
                        high_freq_centre = hf,
                        seed             = seed,
                    )
                    meta["folder"] = folder_name
                    save_terrain(heightmap, meta, out_dir, cfg, preview=preview)
                    all_meta.append(meta)

                    if count % 20 == 0 or count == total:
                        print(f"  [{count:4d}/{total}] {folder_name}"
                              f"  roughness={meta['roughness_index']:.4f}")

    # Save master index
    index_path = os.path.join(base, "sweep_index.json")
    with open(index_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\nSweep index → {index_path}")
    print(f"Done: {total} terrains generated.")
    return all_meta


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Multi-frequency spectral terrain generator for MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate one terrain with defaults
  python generate_terrain_multifreq.py --config terrain_config.yaml

  # Override amplitudes
  python generate_terrain_multifreq.py --low-amp 0.02 --high-amp 0.005 --seed 3

  # Run full sweep (all combinations in config)
  python generate_terrain_multifreq.py --config terrain_config.yaml --sweep

  # Sweep without preview images (faster)
  python generate_terrain_multifreq.py --sweep --no-preview
        """
    )
    p.add_argument("--config",     default=None,
                   help="Path to terrain_config.yaml (default: same folder as this script)")
    p.add_argument("--sweep",      action="store_true",
                   help="Run all parameter combinations from config sweep section")
    p.add_argument("--no-preview", action="store_true",
                   help="Skip preview PNG generation (faster)")
    p.add_argument("--out-dir",    default=None,
                   help="Override output base directory")
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--low-amp",    type=float, default=None,
                   help="Low-freq amplitude (m)")
    p.add_argument("--mid-amp",    type=float, default=None,
                   help="Mid-freq amplitude (m)")
    p.add_argument("--high-amp",   type=float, default=None,
                   help="High-freq amplitude (m)")
    p.add_argument("--high-freq",  type=float, default=None,
                   help="High-freq band centre (cycles/m)")
    args = p.parse_args()

    # Resolve config relative to script directory (not working directory)
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(_SCRIPT_DIR, "terrain_config.yaml")
    if not os.path.exists(config_path):
        print(f"ERROR: config not found: {config_path}")
        print(f"  Place terrain_config.yaml in {_SCRIPT_DIR} or pass --config <path>")
        return

    cfg = load_config(config_path)
    preview = not args.no_preview

    # Default output dir relative to script
    _default_out = os.path.join(_SCRIPT_DIR, cfg.get("output", {}).get("base_dir", "terrain_output"))

    if args.sweep:
        base = args.out_dir or _default_out
        run_sweep(cfg, base_dir=base, preview=preview)
    else:
        # Single terrain
        heightmap, meta = generate_terrain(
            cfg,
            low_amplitude    = args.low_amp,
            mid_amplitude    = args.mid_amp,
            high_amplitude   = args.high_amp,
            high_freq_centre = args.high_freq,
            seed             = args.seed,
        )
        base = args.out_dir or _default_out
        name = terrain_folder_name(
            meta["low_amplitude"], meta["high_amplitude"],
            meta["high_freq_centre"] or 0.0, args.seed
        )
        out_dir = os.path.join(base, name)
        save_terrain(heightmap, meta, out_dir, cfg, preview=preview)
        print(f"Terrain saved to: {out_dir}")
        print(f"Roughness index : {meta['roughness_index']:.4f}")
        print(f"σ_h (RMS height): {meta['sigma_h_metres']*100:.3f} cm")


if __name__ == "__main__":
    main()
