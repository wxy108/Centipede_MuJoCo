# Amplitude optimization recovery: completed and audited

Completed on 2026-09-08. The delivered results are in [fine_step/RESULTS.md](fine_step/RESULTS.md), the reusable full-precision parameters in [selected_gaits.json](fine_step/outputs/data/selected_gaits.json), and the sealed evidence in [FINAL_RECEIPT.json](fine_step/FINAL_RECEIPT.json).

The final audit passed for 2,096 distinct assessment records, 1,908 nominal amplitude conditions, 42 completed banks and 53 frozen inputs. The receipt seals 7,126 files. Its SHA-256 is `DEC797624D36D81E08CDBE1496DD26901485DDB13433E2CB3897276212B84A57`.

Thirteen of fourteen waves have selected gaits passing nominal, 12-second and half-timestep checks. Wave 8 remains unresolved under the unchanged gates. All thirteen selected waves completed the declared directional resolution check; this does not prove a continuous global optimum. Ten waves with previous selected amplitudes improved by 6.06%–45.66% when both sets were measured at the same 0.125 ms timestep. Nine selected-gait videos and one unresolved-case video are delivered, with representative frames visually checked. All owned computation processes have ended.

The coarse-step preparation exposed timestep bias in low-wave scores. Its remaining local-search plan was superseded by the finer-step protocol; the coarse preparation is not a delivered optimum.

The completed experiment follows [fine_step/PROTOCOL.md](fine_step/PROTOCOL.md) and its recorded qualified local-check supplement. Every wave uses nominal 0.125 ms, long-horizon validation at the same timestep, and an independent 0.0625 ms check. Physical model and established per-wave controller are unchanged.

Evidence for the change is in `outputs/data/resolution_diagnostic.json` and `outputs/data/fine_step_amendment.json`. Coarse records remain available as proposal evidence; only exact current-session finer conditions were imported by hash. The largest residual nominal/half-step displacement difference among selected gaits is 5.18% at wave 0.5.

The earlier proposal rule also overemphasized ensemble spread when its predicted mean was below the incumbent. The finer protocol separates performance proposals, expected-improvement proposals, uncertainty and coverage. No surrogate score can select a final winner without a full simulation.

No random-contact sweep was run in this amplitude recovery. The earlier crossover result used different amplitudes and timestep and must not be merged with these zero-loss scores. Future loss work should freeze these parameters, use the same finer timestep, and address the previously documented contact-restoration/body-contact issues before interpreting a crossing.
