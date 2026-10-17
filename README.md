# Centipede MuJoCo — historical model and alignment research

Local research sources synchronized on 2026-10-17. **This repository remains archived.** Its old optimizer trials are not the current Isaac research mainline and are not promoted as validated working defaults.

## Material worth retaining

- `experiments/`: September amplitude, model/contact and representation investigations, with original protocols and limits.
- [Amplitude history audit](experiments/amplitude_history_audit_2026-09-08/RESULTS.md).
- [Shared-contact sim-to-sim audit](sim2sim/AUDIT_AND_STATUS.zh-CN.md).
- [Shared-contact operating profile](sim2sim/COMMON_CONTACT_OPERATING_PROFILE.json) and reproduction manifest.
- [Backbone wave notes](docs/backbone-wave/backbone_wave_documentation.md), preserved research notes rather than newly verified derivations.

The shared compliant-contact model passed its recorded fixed-gait, short-duration, centered-layout mechanical comparisons. This explicitly changes the contact model. Native PGS/TGS rigid-contact equivalence was not established; the common-contact baseline is slow, has displaced-origin/long-travel limits, and does not migrate RL observations, rewards, resets or training.

Earlier amplitude optimizations use different models, contact laws, PD settings, objectives and windows. They must not be ranked against the current Isaac mainline as matched engine comparisons. Historical scripts and plans are retained for provenance, not as current run authorization.

The newer biological/terrain research entry is [Centipede_isaac_rl](https://github.com/wxy108/Centipede_isaac_rl); the separate CAD robot reliability entry is [Centipede_Robot_Sim](https://github.com/wxy108/Centipede_Robot_Sim). The complete Lab variant remains on `legacy/mujoco-lab/main`.

Source Python was syntax-checked and the shared-contact reproduction pins were compared without starting physics. Local-path dependencies and original environment requirements remain; this is not a new portability or physics acceptance. Original files, failed attempts and raw trajectories remain on disk, and prior GitHub versions remain in history. `LOCAL_SOURCE_MANIFEST.json` records the copied sources.
