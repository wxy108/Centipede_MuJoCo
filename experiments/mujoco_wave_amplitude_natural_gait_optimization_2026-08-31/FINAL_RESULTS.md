# Final natural-gait amplitude search results

Date: 2026-09-01  
Scope: abstract/biological Centipede Model only.  Centipede Robot Step 18/19
was not modified or used as a controller definition.

## Frozen experiment

- Model: `centipede_flat_locked_head_m29p0.xml`
- SHA-256: `3815AB6CC8F6BC60834A9A85C246C19DC6B9FB94BD18DF893E072A0345132520`
- Total mass: 2.536031 g; armature: 0
- Flat ground, 1 Hz, 6 s replay, 1.5 s warm-up
- Body and leg wave numbers are identical
- The only optimized variables are body-yaw, leg-yaw, and leg-pitch amplitude
- Controller remains the clean open-loop sinusoidal CPG

## Correction of the rejected search

The earlier grid result is rejected.  It omitted wave 1.5 and the half-wave
map, excluded zero/near-zero and the historical 18.75 deg Model leg-yaw
anchor, and ranked displacement without measuring realized oversweep.  Its
renderer simultaneously showed coincident collision and visual geometry,
causing the gray head flicker, and recolored invisible collision geometry
rather than the visible erased leg.

Protocol V2 searched body yaw 0–36 deg, leg yaw 0–90 deg, and leg pitch
0–60 deg.  It enumerated 4,174 unique candidates: 2,039 broad Sobol/anchor/
boundary points plus 1,092, 679, and 364 three-parameter coupled refinement
points.  Forward displacement was ranked only after all stability and
naturality gates passed.

## Certified nominal flat-ground selection

Displacement is measured over the common 1.5–6.0 s evaluation window.  These
are maxima within the explicitly completed finite candidate bank, not a claim
of continuous mathematical global optimality.

| Wave | Body yaw | Leg yaw | Leg pitch | Displacement | Natural candidates |
|---:|---:|---:|---:|---:|---:|
| 0.5 | — | — | — | no feasible point | 0 |
| 1 | — | — | — | no feasible point | 0 |
| 1.5 | 24.02° | 29.24° | 22.37° | 141.36 mm | 35 |
| 2 | 8.98° | 50.32° | 23.83° | 128.23 mm | 4 |
| 2.5 | 16.92° | 49.38° | 22.92° | **151.38 mm** | 27 |
| 3 | 14.52° | 48.24° | 23.62° | 129.63 mm | 42 |
| 3.5 | 19.24° | 41.03° | 28.61° | 122.67 mm | 46 |
| 4 | 18.82° | 38.55° | 24.82° | 110.74 mm | 31 |
| 4.5 | 13.03° | 37.94° | 24.49° | 97.08 mm | 23 |
| 5 | 7.54° | 26.19° | 28.29° | 65.86 mm | 25 |
| 6 | 7.09° | 21.58° | 32.07° | 52.98 mm | 30 |
| 7 | — | — | — | no feasible point | 0 |
| 8 | — | — | — | no feasible point | 0 |
| 9 | — | — | — | no feasible point | 0 |

Every selected nominal row was replayed in a new stride-1 MuJoCo instance and
reproduced its displacement exactly while passing every declared base and
naturality gate.

## Random-contact result

Random contact is a separate diagnostic, not part of nominal amplitude
selection.  Integer-wave winners were replayed at q=0.35 for seeds 0–9.

| Wave | Mean displacement retention | P05 | Minimum | Strict base-gate survival |
|---:|---:|---:|---:|---:|
| 2 | 0.89 | 0.83 | 0.82 | 0/10 |
| 3 | 0.88 | 0.80 | 0.77 | 0/10 |
| 4 | 0.90 | 0.85 | 0.84 | 0/10 |
| 5 | 0.92 | 0.88 | 0.88 | 0/10 |
| 6 | 0.93 | 0.85 | 0.83 | 0/10 |

The 0/10 result is material: q=0.35 commonly violates the active-joint-speed
and accepted-clean chassis-contact envelope, with additional tip, path,
upright, or tortuosity failures depending on wave.  Therefore the red-erasure
videos are labelled diagnostic failures and must not be presented as stable
random-contact gaits.

## Visualization contract

- Physical collision meshes remain active but are hidden from the camera.
- Only the ground and visual animal geometry are rendered; this removes the
  coincident-surface head flicker.
- During erasure the complete visible left/right leg pair is bright red.
- A persistent red legend shows active erased pair IDs.
- Every nominal winner has matched front, side, and oblique views.
- The wave-2.5 displacement winner also has a side/oblique local leg/foot zoom.

The final video list and exact parameter provenance are recorded in
`outputs/data/final_demo_manifest.json`.
