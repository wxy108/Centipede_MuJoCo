"""
run_rough.py — Run the standard ImpedanceTravelingWaveController on
patched heightfield terrain.  Same CLI as run.py PLUS three terrain
flags so you can verify rough-terrain optimization results without
hand-patching the XML:

    --terrain-wavelength    λ in mm  (default 18 — matches CLAUDE.md)
    --terrain-amplitude     A in m   (default 0.01 = 10 mm peak)
    --terrain-seed          RNG seed (default 42; deterministic)
    --flat                  override: use flat plane

This wrapper does NOT change the controller — it's the original
ImpedanceTravelingWaveController exactly as in run.py.  The only
difference vs running run.py directly is that it generates and
patches in a heightfield before launching the simulation.

Usage
-----
    # default: 18 mm wavelength, 10 mm peak, viewer mode
    python controllers/farms/run_rough.py

    # headless 10 s with video, current optimizer's gains
    python controllers/farms/run_rough.py \\
        --duration 10 --headless --video test_rough.mp4

    # different terrain wavelength + amplitude
    python controllers/farms/run_rough.py \\
        --terrain-wavelength 30 --terrain-amplitude 0.005 \\
        --duration 10 --headless --video test_wl30.mp4

    # flat ground (skip terrain patching, behaves identically to run.py)
    python controllers/farms/run_rough.py --flat
"""

import argparse
import os
import sys

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "sweep"))

import run as _run                                              # noqa: E402


def _pop_terrain_args():
    """Pull our terrain-specific CLI flags out of sys.argv so run.py's
    own argparse (which doesn't know about them) doesn't complain."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--terrain-wavelength", type=float, default=18.0)
    parser.add_argument("--terrain-amplitude",  type=float, default=0.01)
    parser.add_argument("--terrain-seed",       type=int,   default=42)
    parser.add_argument("--flat",               action="store_true")
    known, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return (known.terrain_wavelength, known.terrain_amplitude,
            known.terrain_seed,       known.flat)


def _make_terrain_xml(wavelength_mm, amplitude_m, seed, base_xml):
    """Generate heightfield PNG + patched XML, return patched XML path."""
    from wavelength_sweep import (generate_single_wavelength_terrain,
                                  save_wavelength_terrain,
                                  patch_xml_terrain)

    wavelength_m = wavelength_mm * 1e-3
    h, rms_m, peak_m = generate_single_wavelength_terrain(
        wavelength_m=wavelength_m,
        amplitude_m=amplitude_m,
        seed=seed)

    scratch_dir = os.path.join(PROJECT_ROOT, "outputs",
                               "run_rough_terrain_cache")
    os.makedirs(scratch_dir, exist_ok=True)
    png_path = save_wavelength_terrain(h, wavelength_m, seed, scratch_dir)
    z_max = max(2.0 * amplitude_m, 1e-3)
    patched_xml = patch_xml_terrain(base_xml, png_path, z_max=z_max)

    print(f"[terrain] wavelength={wavelength_mm:.1f}mm  "
          f"amplitude={amplitude_m*1000:.1f}mm  seed={seed}")
    print(f"[terrain] rms={rms_m*1000:.2f}mm  peak={peak_m*1000:.2f}mm")
    print(f"[terrain] patched xml -> {patched_xml}")
    return patched_xml


def main():
    wavelength_mm, amplitude_m, seed, flat = _pop_terrain_args()

    if not flat:
        # Resolve --model override BEFORE terrain patching so the
        # caller can patch terrain into a different base model.
        base_xml = _run.DEFAULT_XML
        for i, a in enumerate(sys.argv[1:], start=1):
            if a == "--model" and i + 1 < len(sys.argv):
                base_xml = sys.argv[i + 1]
                break
            if a.startswith("--model="):
                base_xml = a.split("=", 1)[1]
                break

        patched_xml = _make_terrain_xml(
            wavelength_mm, amplitude_m, seed, base_xml)

        # Replace (or insert) --model in sys.argv with the patched XML
        argv = sys.argv[:]
        new_argv = [argv[0]]
        i = 1
        while i < len(argv):
            if argv[i] == "--model":
                i += 2
                continue
            if argv[i].startswith("--model="):
                i += 1
                continue
            new_argv.append(argv[i])
            i += 1
        new_argv.extend(["--model", patched_xml])
        sys.argv = new_argv

    return _run.main()


if __name__ == "__main__":
    sys.exit(main() or 0)
