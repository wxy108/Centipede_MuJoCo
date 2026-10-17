# Flat amplitude-only optimization

Primary plant: latest local-Z compliant tip. Rigid remains a matched ablation; it did not replace the primary model. No Robot data or code changed.

Only body yaw, leg yaw and leg pitch amplitudes vary. All other physical and controller settings remain frozen per wave, including the historically saved PD for that wave. PD differs between some wave numbers, so this is conditional per-wave optimization, not an isolated causal wave-number comparison. Body and leg wave numbers remain equal; frequency is 1 Hz; erasure OFF.

The initial domain is body [0,36], leg yaw [0,90], leg pitch [0,60] degrees. The registered broad bank is followed by two distinct-seed local refinements and one bounded coupled interpolation pass from feasible anchors toward faster near-feasible endpoints (at most two endpoints and five fractions per wave). Gates are unchanged. Only a finite tested optimum is reported; a mathematical continuous global optimum is NOT proved.

Objective: world-forward geometric chassis-center displacement from t=1.5 s to 5.999 s, maximized only after all absolute, posture and registered contact/slip/work non-regression gates pass. The inherited 90%-of-reference displacement floor is diagnostic only; see OBJECTIVE_CLARIFICATION.md. Nominal candidates are independently tested at 12 s and at half timestep (0.125 ms), in descending-displacement batches of three until a candidate qualifies or the nominal feasible list is exhausted; displacement must agree within 10% across timesteps. All nominally faster candidates than the selected gait are thus checked. Failed waves remain unresolved, not substituted with a visually attractive or merely fast failure.

| Wave | Baseline D mm | Body / leg yaw / leg pitch amplitude deg | Qualified D mm | Gain | Status |
|---:|---:|---|---:|---:|---|
| 0.5 | 53.744 | — | — | — | unresolved |
| 1 | 123.950 | — | — | — | unresolved |
| 1.5 | 158.007 | 19.078 / 43.977 / 16.918 | 156.999 | -0.64% | qualified |
| 2 | 183.669 | 16.918 / 49.377 / 20.518 | 185.973 | +1.25% | qualified |
| 2.5 | 156.929 | 19.954 / 40.865 / 24.150 | 142.824 | -8.99% | qualified |
| 3 | 130.941 | 16.918 / 49.377 / 22.918 | 130.941 | +0.00% | qualified |
| 3.5 | 125.018 | 22.655 / 36.382 / 24.184 | 113.803 | -8.97% | qualified |
| 4 | 117.958 | 18.033 / 36.505 / 18.662 | 113.429 | -3.84% | qualified |
| 4.5 | 116.037 | 30.280 / 40.066 / 21.058 | 105.711 | -8.90% | qualified |
| 5 | 110.873 | 16.918 / 42.177 / 21.718 | 97.391 | -12.16% | qualified |
| 6 | 94.516 | 7.739 / 26.489 / 17.459 | 59.413 | -37.14% | qualified |
| 7 | 93.422 | 7.056 / 19.999 / 25.310 | 55.144 | -40.97% | qualified |
| 8 | 61.260 | — | — | — | unresolved |
| 9 | 45.321 | — | — | — | unresolved |

Tested distinct amplitude conditions: 3298. Waves qualified after all checks: 10 / 14.

Baseline displacement alone does not indicate baseline gate feasibility. Diagnostic records are retained for unresolved waves. A failed amplitude search is not proof that no feasible amplitude exists; it is evidence that the declared finite search did not find and validate one with the frozen PD.

## Coverage and unresolved waves

- wave=0.5: 234 candidates, 0 nominal all-gate passes; unresolved.
  Best diagnostic (NOT a selected gait) failed: feet_count, body_wave, duty, fixed_amp_work_per_distance.
- wave=1: 232 candidates, 0 nominal all-gate passes; unresolved.
  Best diagnostic (NOT a selected gait) failed: body_wave.
- wave=1.5: 239 candidates, 35 nominal all-gate passes; qualified.
- wave=2: 244 candidates, 42 nominal all-gate passes; qualified.
- wave=2.5: 216 candidates, 24 nominal all-gate passes; qualified.
- wave=3: 216 candidates, 32 nominal all-gate passes; qualified.
- wave=3.5: 244 candidates, 50 nominal all-gate passes; qualified.
- wave=4: 239 candidates, 38 nominal all-gate passes; qualified.
- wave=4.5: 244 candidates, 40 nominal all-gate passes; qualified.
- wave=5: 244 candidates, 38 nominal all-gate passes; qualified.
- wave=6: 234 candidates, 24 nominal all-gate passes; qualified.
- wave=7: 244 candidates, 31 nominal all-gate passes; qualified.
- wave=8: 234 candidates, 0 nominal all-gate passes; unresolved.
  Best diagnostic (NOT a selected gait) failed: chassis_contact, pitch_tracking.
- wave=9: 234 candidates, 0 nominal all-gate passes; unresolved.
  Best diagnostic (NOT a selected gait) failed: path_efficiency, chassis_contact.

## Artifacts

- Paired rigid/flexible experiment: PAIRED_RESULTS.md and outputs/data/paired_ablation.json.
- Exact parameter banks and provenance: outputs/data/*_manifest.json.
- All candidate outcomes, including strict early-failure proofs: outputs/trials/.
- Validation details: outputs/data/validated_selection.json.
- Posture/contact summary: STABILITY_RESULTS.md and outputs/figures/stability_by_wave.png.
- Achieved joint motion, distinct from commanded amplitude: outputs/figures/joint_motion_w*.png.
- Final provenance/selection/video audit: outputs/data/final_delivery_audit.json.
- Videos: outputs/videos/. Every delivery demo contains front, side and oblique views. Unresolved wave demos are explicitly labelled DIAGNOSTIC ONLY / NOT QUALIFIED.
- Prior models and selected profile files were not overwritten. No new candidate was silently promoted to the canonical baseline.
