# Centipede repository map

Repository consolidation: 2026-10-17.

## Maintained projects

| Project | Purpose |
| --- | --- |
| [Centipede_MuJoCo](https://github.com/wxy108/Centipede_MuJoCo) | This repository: biological centipede locomotion, controllers and MuJoCo experiments |
| [Centipede_Robot_Sim](https://github.com/wxy108/Centipede_Robot_Sim) | CAD-derived physical robot models and segment/terrain benchmarks |
| [Centipede_isaac_rl](https://github.com/wxy108/Centipede_isaac_rl) | Isaac Lab reinforcement learning, plus the consolidated rendering reference |
| [Terrain_model](https://github.com/wxy108/Terrain_model) | Standalone terrain-to-STL generation for physical experiments |

The CAD-derived robot and biological simulation have different model structures and remain separate. FARMS modules remain separate packages so their installation paths and upstream relationships are retained.

## Consolidated sources

| Original repository | Preserved source head | Location here |
| --- | --- | --- |
| [Centipede_MUJOCO_Lab](https://github.com/wxy108/Centipede_MUJOCO_Lab) | `605b0a9b5b62ddcd61ae54660b37a6192ae7955d` | [`legacy/mujoco-lab/main`](https://github.com/wxy108/Centipede_MuJoCo/tree/legacy/mujoco-lab/main), exact historical workspace and full history |
| [Repo](https://github.com/wxy108/Repo) | `3539fa540007bbda70e85f4abca4fb58b7519896` | [Backbone wave notes](backbone-wave/backbone_wave_documentation.md), with full original history on [`legacy/backbone-notes/main`](https://github.com/wxy108/Centipede_MuJoCo/tree/legacy/backbone-notes/main) |

Both source histories are also parents of the consolidation history. Existing main-branch controllers, configuration and model parameters are retained. The historical Lab workspace is preserved independently rather than replacing the newer main-branch files. The original repositories remain available as archives.

The empty one-byte `Backbone wave documentation` placeholder is retained in source history, while the substantive document is the entry point on main. Tracked Visual Studio `.vs/` caches have been removed from the current tree and ignored; they remain in existing Git history.

## Working with a historical workspace

From a clone of this repository:

```bash
git fetch origin
git worktree add ../centipede-mujoco-lab origin/legacy/mujoco-lab/main
```

This keeps the historical model, controllers and configuration together. No historical simulation results or mathematical claims were revalidated as part of repository organization.
