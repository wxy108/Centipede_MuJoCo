# Compliant-tip amplitude optimization with relaxed engineering gates

This is a separate Centipede Model experiment authorized after the strict study.
No Robot Step 18/19 files, prior results, physical model, PD gains, controller,
frequency, phase, passive tip stiffness/damping/range, mass, inertia or armature
are modified. The latest local-Z compliant tip remains the only plant.

Flat ground; contact erasure OFF; body and leg wave numbers remain coupled.
The 14 previous wave numbers and their individual frozen PD profiles are used.
PD differs across waves, so this remains conditional per-wave optimization.

Only body-yaw / leg-yaw / leg-pitch amplitudes vary within [0,36] / [0,90] /
[0,60] degrees. Body yaw's 36-degree maximum is the physical joint limit;
commands above the limit are not used to manufacture visible differences.

## Gate changes frozen before reclassification

The machine-readable source of truth is gates.py SPEC and its SHA-256.

| Constraint | Previous | Relaxed |
|---|---:|---:|
| Body / leg yaw / leg pitch waveform residual P95 | .40 / .18 / .40 | .60 / .25 / .55 |
| Leg yaw / pitch speed P99 (rad/s) | 8 / 7 | 10 / 9 |
| Leg yaw / pitch reversals per cycle P95 | 3 / 6.25 | 4 / 7.5 |
| Swing height P95 (mm) | 5.5 | 7 |
| Cycle displacement coefficient of variation | .15 | .20 |
| Mean contacting feet | 6–28 | 4–32 |
| Contact duty / left-right duty difference | .15–.65 / .10 | .10–.80 / .15 |
| Work per distance vs matching current-plant reference | 1.25x | 1.50x |
| Slip and left-right force imbalance vs reference | 1.10x | 1.35x |
| Normal/chassis 10 ms impulse vs reference | 1.10x | 1.25x |

Old-axis reference comparisons and extra reference-dependent posture caps
become diagnostics. The absolute body posture caps are NOT relaxed.

All other absolute gates remain unchanged, including warning-free finite
simulation; positive forward displacement/path efficiency; upright, spine,
root roll 20 degrees, coherent/front roll P95 10/12 degrees; height range 2 mm;
body speed and 10 ms jump caps; force/torque limits; penetration; chassis
contact fraction 3%; no long ballistic interval; tip angle/speed/stops;
and active tracking error caps. See prior raw records for all thresholds.

## Search and validation

The objective is maximum fixed-axis forward displacement of the unweighted
21-body geometric center over t=1.5–5.999 s, after every new gate passes.
There is no weighted score and no reward for changing amplitudes.

Reassess all 3,298 old conditions under the declared gates. A full old rollout
can be reused after provenance checks. An old early rejection can only be
reused if at least one irreversible proof still violates an unchanged gate;
otherwise the condition must be simulated again. Never rank a truncated run
by its partial displacement.

Add a separately frozen scrambled Sobol bank (64 new samples per wave), then
two bounded multistart refinement rounds with 8% and 2% box-span steps.
Retain spatially distinct seeds; candidate lists are written before execution.
Any optional final boundary pass must also be bounded and frozen first.

Rank feasible candidates by nominal six-second displacement; validate in
descending batches until a candidate passes cold 12 s and cold half-step 6 s
tests, or all nominal feasible candidates are exhausted. Half-step displacement
must agree within 10%. The new gates apply consistently in every duration.
The output is finite-tested-set best, not a mathematical global-optimality proof.

"Noticeable amplitude difference" is only a reporting diagnostic: at least
one amplitude changes by 5 degrees or by 25% (using max(old amplitude,5 degrees)
as denominator). It never affects selection. Compare both the previous strict
optimum and the common original fixed amplitudes; unresolved old waves are
not assigned a fictitious strict optimum.

Every wave receives unsmoothed front/side/oblique demos, with unresolved cases
clearly marked. Record actual joint motion and individual-body roll envelopes,
not only commanded amplitude or a whole-body averaged roll.
