# Run / resume notes

Started 2026-09-03 12:07 EDT. Pipeline PID at launch: 52608. This is a launch
record, NOT a claim that the process is still running when this file is read.
Check the current process and the newest logs before resuming; do not start a
second copy while one is active.

The setup audit checked all 5,036 prior conditions. Cold wave 2 and wave 3.5
replays matched every saved metric exactly. The pipeline then entered the
220-condition transfer/axial-neighborhood challenge. No new optimum is claimed
by these verification checks.

Live evidence:

- `outputs/pipeline_stdout.log`, `outputs/pipeline_stderr.log`
- `outputs/data/pipeline_progress.json` (stage transitions and pipeline PID)
- `outputs/data/live_status.json` (latest completed-work progress)
- `outputs/data/amplitude_manifest.json` (frozen dependencies and policy)
- `outputs/data/setup_audit.json`, `outputs/data/verification.json`

The sequential queue is transfer -> interim validation -> interim reports and
three-view/local demos -> two independent global banks -> exact-body profiles
and 2-D refinement -> multistart 3-D refinement -> final validation -> delivery.
Full-search results must not be inferred from an interim report.

The runner is the bundled Python at:
`C:/Users/wxy22/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`.
Run `pipeline.py --workers 20 --resume-from <stage>` from this experiment
directory ONLY after checking that the old process is no longer running and
that frozen hashes still pass. Existing batches/results are retained and reused.
Dependency reads may require the same approved elevated tool execution used at
launch; ordinary restricted execution cannot read the project's MuJoCo package.

Preserve all old experiments. Never edit frozen files to make the audit pass.
If a real bug requires an implementation change, stop this owned queue, record
the amendment and its affected results explicitly, and re-verify before reuse.
Do not change model/controller/PD/thresholds, force a monotone curve, or promote
parameters to defaults. No Robot Step 18/19 or IsaacLab/Clean changes are in scope.

This remains a bounded search. Poll-budget stops are unresolved, and even a
stationary poll is not a proof of continuous global optimality.
