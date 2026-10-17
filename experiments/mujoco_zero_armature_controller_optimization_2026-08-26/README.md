# Zero-armature Centipede Model controller optimization

This directory is an isolated **Centipede Model** experiment. It uses the
geometry-derived, zero-armature MuJoCo plant from the 2026-08-25 rebuild. It
does not write to the physical Centipede Robot project or its Step 18/19 data.

The optimization follows the live RL contract:

- flat: wave 1.5 and straight-line viability;
- mild rough: L90/A8 and wave 2;
- rough: L60/A12 and wave 3;
- 1 Hz gait and frozen amplitudes;
- pitch/roll compliance retained and measured, rather than inferred from a
  low scalar Kp alone;
- armature exactly zero;
- no weighted scalar reward or flat-only winner.

The frozen controller is in `config/optimized_controller.json`. The complete
9.6 s validation is in `outputs/data/final_validation.json`. Four videos are
under `outputs/videos`.

Reproduce with the local MuJoCo environment:

```powershell
& 'C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe' controller_optimization.py prepare
& 'C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe' controller_optimization.py validate --duration 9.6
& 'C:\Users\wxy22\anaconda3\envs\isaaclab_50\python.exe' controller_optimization.py demos --duration 9.6
```

The terrain generator is a reduced-raster port of the clean RL homogeneous
terrain algorithm. Its exact seed, amplitudes, relief and generated model
hashes are recorded in `outputs/data/terrain_manifest.json`.
