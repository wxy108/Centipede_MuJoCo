# Sim-to-sim decision ledger

| Question or tempting inference | Evidence | Decision |
|---|---|---|
| Almost equal total mass means the same plant | Tail mass differs 22.915%; one complete leg differs 14.06%; topology and head DOFs differ | Export one explicit per-body physical definition to both engines |
| Same-named tiny USD carrier equals a whole MuJoCo leg | A leg is spread across carrier and welded child bodies | Compare complete rigid physical composition |
| Matching Kp/Kd field values matches the update | PhysX implicit drives include end-position stiffness | Derive D+hK; preserve physical D |
| A passing isolated drive proves contact locomotion matches | Full gait bank still diverged | Also share effective inertia between contact and velocity solves; independently test loaded contact |
| A passing calibration gait validates the chosen step | Initial full bank changed 6%–17% on refinement | Retain failed candidate; check every predeclared gait |
| TGS probe drift came from friction acting in free fall | Full velocity trace shows reversal at impact | Correct that interpretation; retain the trace |
| Reducing contact offset or velocity iterations fixes the probe | Those isolated ablations retained the reversal | Do not adopt them as a fix |
| A generic oblique-friction error explains the whole gait gap | Independent oblique stopping positions differ by only a few micrometres | No such large causal claim |
| Tiny numeric masses alone explain the CPU large-gait error | Consistent kg→mg unit conversion changes D only about 0.07% | Do not adopt unit scaling as a performance adjustment |
| A 9.3% raw generalized-matrix mismatch is a physical inertia error | Root velocity reference points differ; transform removes the discrepancy | Compare root-COM and root-origin coordinates correctly |
| Small errors in matrix entries imply, or rule out, inverse-inertia errors | Seven measured force-to-velocity probes agree within 0.0026% in kinetic norm | Use dynamic impulse evidence, not a mixed-unit condition number alone |
| CPU calibration agreement means GPU can be substituted | Full large-yaw gaits show backend differences | Record CPU/GPU separately and validate the actual GPU backend |
| Increasing CPU iterations must explain CPU/GPU disagreement | Fourfold CPU iterations change all eight D values by less than 0.9% | Insufficient CPU iteration count is not supported as that explanation |
| A faster target path is automatically equivalent | A mixed change in transport, batch size and diagnostics did not reproduce the full-bank trace | Do not adopt; do not identify one factor as the cause from this confounded test |
| Numerical armature hD+h²K is new physical rotor inertia | It follows from rearranging the same backward-Euler PD equation | Label it as a numerical coefficient; physical rotor inertia remains zero |
| A favourable eight-gait nominal comparison is enough | Both engines also need independent resolution checks and geometry review | Keep the profile pending until the full prescribed checks finish |
| A benchmark pass means all RL training and old optimizers have migrated | Only the common-physics open-loop entries are implemented and measured | State the verified scope and preserve historical entry points |

The old amplitude-optimization and contact-erasure queues were not resumed. No fitted wave-dependent physical PD or amplitude changes were used to meet these comparison criteria. Unfinished or superseded runs retain their receipts and are not counted as completed physical outcomes.

## Contact diagnostics and rejected candidates, 2026-09-11

| Question | New evidence | Decision |
|---|---|---|
| Does the nominal PGS GPU maximum gap of 8.919% establish alignment? | Complete GPU half-step changes sensitive cases by 8.495% and 18.366% | Reject the operating profile; archive the report and real pose-replay videos with this status |
| Does equal scalar friction mean the same force boundary? | Installed CPU/GPU PGS hold a 45° force inside a square but outside a circle; installed TGS and Mu elliptic move | Use explicit force-limit probes; PGS/TGS must not be treated as the same friction law |
| Does an analytically matched PGS square cone fix the full robot? | Fixed/velocity tangent-basis Mu adapters worsen 2.5/B to 136.248/146.845 mm | Preserve, do not adopt or fit friction |
| Is the original TGS impact anomaly unavoidable? | Valid external-forces-each-iteration scene flag improves drop x from 114.098 to 127.498/127.500 µm CPU/GPU | Carry the supported solver correction into a separate TGS candidate; retain the small GPU velocity residual |
| Are CFM, contact offset or correlation the dominant remaining PGS causes? | Independent two-sensitive-gait ablations make small or zero changes | No such attribution or production change |
| Does the whole coupled drive already diverge without ground? | Same-step PGS CPU/Mu full-chain max q error 0.000037°, though body-origin error reaches 41.85 µm | Contact behavior is the next focus; do not assert exact equality of all poses |
| Do close TGS CPU main results establish success? | 250→125 µs full-bank CPU changes 12.241% and 5.372%; GPU main 2.5/B gap 11.977% | Reject candidate; independently diagnose time grid, iterations and command clock under unchanged gates |

References and exact run names are retained in NUMERICAL_AMENDMENT_10.md through NUMERICAL_AMENDMENT_14.md and the machine-readable audit files. The public PhysX main branch is corroborating context, not proof of the exact installed binary; the installed-backend probes establish the reported behavior.

## Subsequent evidence and the separate compliant-contact branch

| Question | Evidence | Decision |
|---|---|---|
| Can the TGS CPU thread-count change explain later diagnostics? | Controlled 8→2-worker full nine-gait comparison has bitwise identical q, v and body origins at all 601 frames | Remove that confound for these measured CPU cases; do not generalize to every backend |
| Does TGS one-position-iteration remove the resolution problem? | CPU125→62.5, velocity32: sensitive holdouts rise 95.840→120.116 and 135.829→169.358 mm | Reject as an aligned profile; retain geometry audits (no inverted spine samples) |
| Does more MuJoCo NoSlip fix the sensitive gait? | NoSlip20 changes 2.5/B by about 0.145% | Not adopted as a fix |
| Is sustained sliding-normal discrepancy just a prescribed-state artifact? | Independent continuous horizontal physical PD gives Mu mean height 6.041 µm / vertical RMS 8.121 mm/s, TGS approximately zero | Record an independent contact-law discrepancy; do not claim it uniquely explains all gait error |
| Does an isolated successful De Saxce correction validate a robot implementation? | Full-robot safeguarded solves fail at 1.47625 and 1.367 s; saved-state root retries retain excessive residuals | Reject; stop on unresolved contact, preserve partial state, produce no final displacement |
| Does a root solver success flag establish physical convergence? | Some root methods report success while velocity residual remains approximately 2.678e-5 m/s | Apply the predeclared physical residual gate, not the optimizer flag |
| Can shared external contact forces be called unchanged native rigid contact? | COMMON_CONTACT_V2.json explicitly introduces fixed normal/tangential compliance and a common support-point discretization | Treat as a separate optional physical model, not validation of the prior rigid model |
| Are external forces applied at equivalent reference points? | Seven COM force/torque impulses through the actual native API match Mu inverse dynamics within 0.002478% CPU / 0.002552% GPU in kinetic norm | Accept the force/frame application check; still require full locomotion and refinement |
| Is vectorized support-point calculation a different contact law? | 1000 sequential randomized states across nine environments yield bitwise identical force, torque, tangential memory and support indices; CPU/GPU short initialization trajectories also identical | Use equivalent vectorized implementation where recorded; preserve reference and source hashes |
| Does a very small first-two-second CPU/Mu difference complete the task? | Eight startup/ramp displacement differences are at most 0.01272 mm, endpoint joint error at most 0.00902 degrees | Encouraging partial evidence only; wait for every complete 2–6 s holdout and independent half-step |

The first shared-contact short CPU initialization inherited a TGS-only external-force iteration flag while selecting PGS. The backend explicitly reported it unsupported. The corrected full-bank runner sets it false and saves its readback. A first GPU body-wrench launch used a one-environment bank where the probe required seven; it exited at its assertion and was rerun with the correct seven cases. Neither failed launch is counted as passed evidence.

## Complete main-grid shared-contact results and coordinate diagnostics

All eight fixed 2–6 s main-grid cases completed. Maximum symmetric forward gaps are 0.026617% CPU/Mu, 0.020956% GPU/Mu and 0.015637% CPU/GPU. Mu refinement is at most 0.017754%, CPU refinement 0.013406%; GPU refinement remains pending at this entry. The fixed bank has been reused across candidate development, so it is not described as a never-seen blind test.

Forward agreement does not imply full trajectory equality. CPU fine-grid case 4.5/A at global Y=1.75 m has maximum XY path difference 3.550056 mm. The complete same-step dt31 causal origin comparison reproduces the high-Y control bitwise. Moving only Y to zero reduces maximum XY difference from 0.750733 to 0.003139 mm. The fine-grid origin diagnostic was predeclared next and does not replace the frozen bank.

Independent zero-gravity, zero-contact constant-velocity tests keep root velocity 3 mm/s, with essentially zero joint motion. At Y=0 the 0.5 s translations are 1.499807 / 1.500286 mm at dt31 / dt15; at Y=1.75 m they are 0.000596 / 0.000000 mm, versus the analytical 1.5 mm. A simple whole-step float32-addition prediction disagrees with the dt31 high-Y result, so it is not asserted to be the exact internal PhysX update. These data establish origin dependence, consistent with the positional precision scale, not the unique cause of the historical native-contact gap.

The first constant-velocity initialization failed before physics because integer zero gravity reached an installed USD binding as NumPy integers. Floating-point zeros fixed initialization without changing the intended physical test or editing the framework. The failed record remains preserved.

## Centered independent environments

Per-step free translation records show that a root-only float32 update does not explain the dt31 high-Y trajectory. Adding a sequential float32 whole-articulation COM correction reproduces all 64 high-Y root-COM steps exactly; the root-only update matches one. The origin-zero arithmetic residual remains tiny but not exactly zero. This isolates the diagnosed constant-velocity mechanism, not the exact complete installed engine or all historical contact errors.

Amendment 28 changes only static environment translations and initial Y offsets to zero. Native collisions were already disabled, and per-environment force histories remain independent. No repeated root resets, velocity edits, gain fitting or contact-parameter changes. CPU short trajectories reproduce both the existing zero-origin calibration and the independent zero-origin 4.5/A control bitwise. Full CPU/GPU main and fine banks are required, with additional maximum XY-path difference 0.5 mm and chord-heading difference 0.25 degrees across every backend pair and each refinement.

Amendment 29 intentionally closes the superseded displaced-origin GPU fine run after its last saved 2.5 s partial trace. No final score is manufactured. Exact owned-process cancellation records are retained. GPU time is allocated to complete validation of the centered candidate; the old forward agreement alone cannot promote the superseded layout.

## Final centered benchmark acceptance — 2026-09-11

All eight fixed gaits completed in MuJoCo, PhysX CPU and PhysX GPU at both31.25 and15.625 microseconds:48 formal outcomes. All72 additional XY/heading comparisons passed. Main-grid forward gaps are at most0.023491% CPU/Mu and0.020025% GPU/Mu; the maximum across every same-grid backend pair at both grids is0.048325% (CPU/GPU, fine grid,4.5/A). Each backend's forward half-step change is below0.023%. Maximum XY difference is0.195881mm and chord-heading difference0.235118degrees, within the unchanged0.5mm/0.25degree gates. GPU heading differences increased at the finer step; no monotonic convergence or longer-horizon claim is made.

Complete quality review found finite states and no inverted spine samples. All four full unchanged-origin controls were independently rechecked bitwise. Original41 source hashes remain unchanged. The finalizer promoted `COMMON_CONTACT_OPERATING_PROFILE.json`, wrote its validation receipt and froze47 required inputs. All three original-project entrypoint dry runs passed the accepted-profile checks. The30-file presentation bundle and both fully decoded pose-replay videos passed checksum verification. Current centered workers, dispatchers and collector have exited; no further physics, optimization, erasure or training is queued by this task.

This accepts the explicitly new shared compliant-contact mechanical benchmark. The native rigid-contact candidates remain rejected. Historical defaults and Clean are not migrated; the original RL observation/reward/contact-sensor/reset pipeline remains outside this adoption.
