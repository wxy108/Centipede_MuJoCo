# Centipede Model flat wave/amplitude optimization protocol

## Scope and separation

- Model only: `centipede_flat_locked_head_m29p0.xml`.
- SHA-256: `3815AB6CC8F6BC60834A9A85C246C19DC6B9FB94BD18DF893E072A0345132520`.
- Flat ground only; zero non-root armature; current locked-head geometry and
  passive compliant tips are frozen.
- The body and legs always use the same wave number. No split-wave controller,
  projected-gravity term, contact feedback, or state machine is introduced.
- Centipede Robot Step 18/19 files are sentinels only and are never outputs or
  inputs to this experiment.

## Search variables

For each integer wave number 1 through 9, only three absolute target
amplitudes change:

| Variable | Values (degrees) |
|---|---|
| body yaw | 8, 12, 16, 20, 24, 28, 32 |
| leg yaw | 24, 30, 36, 42, 48, 54, 60 |
| leg pitch | 12, 18, 24, 30, 36, 42, 48 |

This is 343 deterministic trials per wave and 3,087 trials total. The ranges
bracket the earlier IsaacLab kinematic anchors (body yaw 0.4 rad, leg yaw
0.6 rad, leg pitch 0.2 rad) and the prior high-speed leg-yaw proposal of about
0.9 rad. The 60-degree leg-yaw ceiling is already above that proposal; it is
an admissible-domain limit, not an invitation to increase excursion until
the model produces a visually large displacement.

## Objective and hard gates

The objective is forward displacement from 1.5 to 4.0 seconds. It is compared
only after all gates pass. There is no weighted reward.

| Gate | Threshold |
|---|---:|
| finite and MuJoCo warning-free | required |
| forward displacement | > 0 |
| path efficiency | >= 0.80 |
| minimum root upright cosine | >= cos(50 deg) |
| maximum spine tortuosity | <= 1.20 |
| coherent body roll P95 | <= 10 deg |
| front-seven roll P95 | <= 12 deg |
| chassis centroid z peak-to-peak | <= 4 mm |
| chassis vertical speed absolute P99 | <= 0.10 m/s |
| active-joint speed absolute maximum | <= 100 rad/s |
| passive-tip speed absolute maximum | <= 1,000 rad/s |
| estimated active torque absolute maximum | <= 2e-4 N m |
| mean contacting feet | 6 to 28 |
| longest all-feet ballistic gap | <= 0.08 s |
| chassis-ground contact fraction | <= 0.05 |
| foot penetration P95 | <= 0.10 mm |
| passive-tip roll P95 | <= 33 deg |
| passive-tip stop occupancy | <= 0.03 |

The reported optimum is the exact global optimum on this declared finite
admissible grid. It is not a continuous-domain global-optimum claim, which is
not supportable for nonsmooth contact dynamics from a finite simulation set.

## Random-contact validation

Random contact erasure is deliberately excluded from selection. Every nominal
feasible winner is first checked for 6 seconds; where the 4-second winner
fails, higher-ranked candidates are tested in order. Each retained selection
is then re-run on flat ground with `q=0.3` and seeds 0 through 4. Waves with no
feasible nominal point remain explicit diagnostics. The protocol is inherited from the earlier
zero-armature Model study: within phase-aligned 0.5-second support windows,
burst duration is log-uniform from 0.02 to 1.0 window and burst start is
uniform conditional on duration. An event disables both complete collision
subtrees of one segment pair. This stage measures robustness; it cannot change
the selected amplitudes.
