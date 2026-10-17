# Completed — 2026-09-11

The centered shared-contact benchmark is finalized and delivered. **Do not restart any simulation, collector, finalizer or paused optimization from the historical notes below.** All native queues and collector exited0; no current centered physics worker remains. All48 formal gait outcomes and72 motion comparisons passed. Main CPU/Mu maxgap0.023491%, main GPU/Mu0.020025%; maximum across both same-step grids and every pair0.048325%. Maxforwardrefinement0.022933%, maxXY0.195881mm, maxheading0.235118degrees.

Accepted files: `COMMON_CONTACT_OPERATING_PROFILE.json`, `COMMON_CONTACT_VALIDATION_RECEIPT.json`, `COMMON_CONTACT_REPRODUCTION_MANIFEST.json`. All47 frozen input hashes verified. All3 original-project entrypoint dry runs passed. Final report and current23/24/25/26 plus diagnostic22 were visually reviewed; both videos fully decoded and QA'd. `presentation_ready_common_contact` exists with30 hashed artifacts plus README/manifest; package and video checksums pass. Original41 source hashes unchanged. No Clean, RL training, old optimization or contact erasure migration.

Do not regenerate frozen summaries or full-control JSON after promotion. Current figures24/25 include all6 forward criteria; figure26 near-limit labels are inside the bars to avoid the threshold line. The reporter distinguishes completed quality review from the frozen numeric evaluator's original review-required status. Video reconstruction from recorded joints/body positions is explicitly described. All-steps maximum forward gap is now also recorded in the accepted profile.

## Historical continuation notes (superseded)

Current candidate is **centered shared compliant contact**, amendment 28, not the displaced-origin bank. Do not resume old PGS/TGS, amplitude optimization or contact erasure. Do not edit Clean or frozen simulation inputs. No profile is accepted yet.

Running bounded jobs:

- CPU queue `run_common_contact_centered_cpu.py`, exec session28178 and owned workerPID238480 have completed exit0. CPUfine full6s wall5225.64s. Collector collected it. All CPUrefinement criteria pass, maximum0.0144623%; all additional CPUfine motion checks pass. Four full unchanged-origin controls now all bitwise identical and audited. Do not rerun CPU physics.
- GPU queue `run_common_contact_centered_gpu.py`, exec session 57428, owned worker PID236032: main dt31 complete; now `native_common_contact_centered_validation_gpu_dt15_000` (started12:46:21 UTC), saved1.5s. One GPU physics worker only. Expected all GPU work around15:00 UTC, subject to observed speed.
- Collector `collect_common_contact_centered_results.py`, session 85314: fixed four native runs; readback, quality, mesh-compression, forward and 72 motion comparisons. No physics launch or promotion. **Does not refresh plots/reports/replays**.
- Both centered demos complete. GPU render session97526 also exited0. Both fully decoded:176frames,H264,1280×800,25fps,7.04s. Startup,4s andterminal poses visually checked; no clipping. See `audit/centered_replay_media.json` and contact sheet. Do not rerender unnecessarily.

Old displaced GPU fine worker and its two old dispatch/collection processes were intentionally stopped. Last saved old GPU fine state is 2.5 s, no final score. Exact PIDs and provenance: `audit/common_contact_displaced_cancellation.json` and `...displaced_closeout.json`. Old sessions 93003, 59000 and 44740 are obsolete. Their incomplete data are never counted as passed.

Completed centered CPU main: max forward gap 0.023491%, max XY path gap 0.040666 mm, max chord-heading gap 0.005907°. No failed criteria. Main calibration and 4.5/A origin-zero controls are bitwise identical over all 601 frames. CPU and GPU short startup controls both passed bitwise. Native backend readbacks passed. The same 16 complete Mu main/fine runs are reused unchanged.

Completed centered GPU main: max forward gap0.020025%, XY0.077744mm,heading0.101794°. CPU/GPU main forward gap0.013945%,XY0.0797mm. Both main runs collected: forward16missing, motion32/72complete,40missing,0failures. Figures23/25/26 and report refreshed for both main runs and visually checked; figure24 also generated. Figure22 final paired data/footer visually checked. GPU short backend and currently running fine backend readback pass.

Update13:41: CPUfine collected, forward8missing(GPUfine only), motion48/72complete,24missing,0failures. Figures23/24/25/26, limits and report refreshed with CPUfine. All41 original source hashes preserved. GPUfine saved2.0s with finite states, partial maximum heading gap0.137653° (limit0.25). This is mostly an initial offset: its largest case0 main-grid gap at2s−0.06849°, at6s−0.07008°. No failed criterion or new physics branch. GPUfine remains required, expected~15:00UTC.

New reproducibility detail: `audit/hardware_identity.json` records actual driver/Torch RTX5070 Laptop GPU,driver610.88. Graphics startup table labels GTX1050Ti; this discrepancy is explicitly recorded, cause undiagnosed. All four current logs report same driver; native CUDA device and GPU scene flags verified. `verify_alignment_runtime.py --check-gpu-driver` passes, including31 external runtime code files. GPU reproducer and finalizer now additionally check device/driver. Snapshot script `capture_alignment_hardware.py` refuses overwrite; do not rerun it. Runtime code snapshot was captured12:46 during validation, not a retroactive initial hash.

Finalizer now also requires all four full unchanged-origin controls and independently rechecks their six-second q/v/body equality. Run `audit_common_contact_centered_controls.py` after CPUfine is collected. Its JSON is now included in the final frozen manifest, along with hardware snapshot. Do not refresh this control JSON after promotion. The finalizer's default remains read-only/pending; no adoption files exist yet.

`read_centered_validation_progress.py` only reads saved partial traces and reports finite state/partial XY/heading differences. It launches no work and writes no acceptance file. Use occasionally for pending status, not as a replacement for full results.

Origin diagnostics complete: 4.5/A high-Y controls match the old banks bitwise. Moving Y=1.75 to zero reduces max XY error 0.750733→0.003139 mm at dt31, 3.550056→0.024404 mm at dt15. Zero-force translation stays at 3 mm/s but loses far-origin position updates. Per-step COM-correction arithmetic reproduces all 64 dt31 high-Y root-COM updates exactly. See amendments 24–29 and coordinate audit; do not claim this uniquely explains all historical native-contact differences.

Required gates: original all-eight forward gap ≤10%, each backend half-step change ≤5%, each displacement ≥50 mm, full finite six seconds, same physical readback; plus **all 72** same-step pair and within-engine half-step XY-path differences ≤0.5 mm, heading differences ≤0.25°. Main31.25 µs, fine15.625 µs, command250 µs. No physical/PD/contact fitting.

Remaining work:

1. Wait for the centered complete banks; inspect all criteria and quality, preserving failures. Keep meaningful user updates. No new physics branches unless an actual failure needs diagnosis.
2. After each collected result as useful: refresh `plot_common_contact_centered_results.py`, `plot_common_contact_centered_difference_detail.py`, `plot_common_contact_centered_motion.py`, `audit_common_joint_limits.py`, `audit_common_contact_centered_controls.py`, then `build_current_alignment_report.py`.
3. Current figures are **23/24/25/26**, plus diagnostic22. Older19–21 are displaced-origin history. Both CPU/GPU centered replays have been rendered and QA'd. All rendering is CPU saved-pose projection, not native Isaac rendering.
4. On all four centered completions, read collector `all_collected`, check motion32 native/Mu rows and72 gate rows, no missing/failures; inspect warnings, readbacks and physical quality. Refresh scientific figures and visually QA; verify videos/metadata.
5. Run source preservation (41 original hashes) and finalizer default. `finalize_common_contact.py` now delegates to `finalize_centered_common_contact.py`; only `--finalize` after actual full passage/review. It checks all20 run packages, additional motion gates, limits, exact centered layout, GPU backend, source hashes, and old mapping/mesh provenance. It writes current common-contact operating profile, receipt and manifest. Do not run old PGS freeze or displaced finalizer.
6. Reproducer and both original-project optional wrappers already point to the centered candidate when pending. Verify accepted dry runs for Mu/CPU/GPU after promotion. **Do not regenerate hashed summaries after promotion.** Report generation can still run; it does not change accepted inputs.
7. Generate final report, centered demos, and `package_common_contact_results.py` (guarded until accepted) into `presentation_ready_common_contact`. Confirm package links/data hashes. Final Chinese response must clearly state the new shared contact law, common physical PD and measured scope; original RL pipeline and Clean remain unmigrated. Include report, final comparison image, and saved-pose replay links.

Current `COMMON_CONTACT_V2.json` keeps its frozen creation status; current adopted state will be in the separate operating profile. Original `OPERATING_PROFILE.json` / `REPRODUCTION_MANIFEST.json` remain old rejected PGS provenance and must not be overwritten. The finalized manifest deliberately anchors old mapping/meshes against that existing manifest.
