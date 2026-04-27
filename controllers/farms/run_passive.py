"""
run_passive.py — Runner for the PassiveAxialController
=======================================================
Same CLI as run.py, with three extra flags for on-the-fly terrain
generation so you don't have to patch the XML by hand:

    --terrain-wavelength   λ in mm  (default 18 — matches CLAUDE.md)
    --terrain-amplitude    A in m   (default 0.01 = 10 mm peak)
    --terrain-seed         RNG seed (default 42; deterministic)
    --flat                 override: skip terrain, use flat plane

Leg controllers, pitch / roll impedance, CPG for legs, head-pitch
servo, and the settle+ramp envelope are ALL inherited unchanged from
the parent ImpedanceTravelingWaveController.  The only thing this
runner changes is the body-yaw torque computation.

Usage
-----
    # viewer, rough terrain wavelength 18 mm (default config)
    python controllers/farms/run_passive.py

    # headless 10-second run, record video
    python controllers/farms/run_passive.py \\
        --duration 10 --headless --video passive_rough.mp4

    # flat ground — same passive controller, no terrain
    python controllers/farms/run_passive.py --flat --duration 10 --headless

    # switch config mode on the fly — edit the yaml first, then:
    python controllers/farms/run_passive.py \\
        --config configs/farms_controller_passive.yaml
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "sweep"))

import run as _run                                              # noqa: E402
from passive_axial_controller import PassiveAxialController     # noqa: E402


# ── Monkey-patch: swap the controller class in run.py ───────────────────────
_run.ImpedanceTravelingWaveController = PassiveAxialController
_run.DEFAULT_CFG = os.path.join(PROJECT_ROOT, "configs",
                                "farms_controller_passive.yaml")


# ── Intercept terrain flags before run.main() sees them ─────────────────────
def _pop_terrain_args():
    """Pull our terrain-specific CLI flags out of sys.argv so run.py's
    own argparse (which doesn't know about them) doesn't complain.
    Returns (wavelength_mm, amplitude_m, seed, flat)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--terrain-wavelength", type=float, default=18.0)
    parser.add_argument("--terrain-amplitude",  type=float, default=0.01)
    parser.add_argument("--terrain-seed",       type=int,   default=42)
    parser.add_argument("--flat",               action="store_true")
    known, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return known.terrain_wavelength, known.terrain_amplitude, \
           known.terrain_seed,       known.flat


def _make_terrain_xml(wavelength_mm, amplitude_m, seed, base_xml):
    """Generate a heightfield PNG + patched XML and return the patched
    XML path.  The patched XML is written next to `base_xml` so the
    relative `<compiler meshdir="meshes">` still resolves."""
    from wavelength_sweep import (generate_single_wavelength_terrain,
                                  save_wavelength_terrain,
                                  patch_xml_terrain)

    wavelength_m = wavelength_mm * 1e-3
    h, rms_m, peak_m = generate_single_wavelength_terrain(
        wavelength_m=wavelength_m,
        amplitude_m=amplitude_m,
        seed=seed)

    # Drop heightmap into a scratch dir next to the xml
    scratch_dir = os.path.join(PROJECT_ROOT, "outputs",
                               "run_passive_terrain_cache")
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
        # User explicitly wants rough terrain (default).  Generate it
        # and override --model in sys.argv so run.main() loads it.
        base_xml = _run.DEFAULT_XML
        # Allow --model override BEFORE terrain patch so the user can
        # patch terrain into a different base model.
        for i, a in enumerate(sys.argv[1:], start=1):
            if a == "--model" and i + 1 < len(sys.argv):
                base_xml = sys.argv[i + 1]
                break
            if a.startswith("--model="):
                base_xml = a.split("=", 1)[1]
                break

        patched_xml = _make_terrain_xml(
            wavelength_mm, amplitude_m, seed, base_xml)

        # Replace (or insert) --model in sys.argv
        argv = sys.argv[:]
        new_argv = [argv[0]]
        i = 1
        while i < len(argv):
            if argv[i] == "--model":
                i += 2   # skip existing
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
