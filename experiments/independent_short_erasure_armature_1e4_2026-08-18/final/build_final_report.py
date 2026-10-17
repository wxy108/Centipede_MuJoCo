"""Build the evidence-first report for the independent short-erasure rerun."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "data"
FIGURES = ROOT / "outputs" / "figures"
VIDEOS = ROOT / "outputs" / "videos"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float:
    return float(value)


def main() -> None:
    raw = read_csv(DATA / "short_erasure_trials.csv")
    summary = read_csv(DATA / "short_erasure_summary.csv")
    recovery = read_csv(DATA / "recovery_transition_summary.csv")
    manifest = json.loads((DATA / "experiment_manifest.json").read_text(encoding="utf-8"))

    unstable = [row for row in raw if row["stable"].lower() != "true"]
    nonfinite = [row for row in raw if row["finite"].lower() != "true"]
    self_contact = [
        row for row in raw if int(row["maximum_dynamic_self_contact_count"]) > 0
    ]
    max_roll = max(number(row["max_abs_root_roll_deg"]) for row in raw)
    max_pitch = max(number(row["max_abs_root_pitch_deg"]) for row in raw)
    max_segment_tilt = max(number(row["max_segment_tilt_deg"]) for row in raw)
    min_floor = min(number(row["minimum_dynamic_floor_distance_m"]) for row in raw)

    by_key = {(int(row["wave"]), number(row["q"])): row for row in summary}
    table_rows = []
    for wave in (1, 2, 3, 4):
        base = by_key[(wave, 0.0)]
        high = by_key[(wave, 0.4)]
        d0 = number(base["mean_displacement_m"])
        d4 = number(high["mean_displacement_m"])
        loss = 100.0 * (d0 - d4) / d0
        table_rows.append(
            "| {wave} | {G} | {mmin}–{mmax} | {d0:.4f} | {d4:.4f} ± {sd:.4f} | "
            "{loss:.2f}% | {r2:.3f} | {r5:.3f} | {r10:.3f} |".format(
                wave=wave,
                G=int(high["phase_group_count"]),
                mmin=int(high["m_min"]),
                mmax=int(high["m_max"]),
                d0=d0,
                d4=d4,
                sd=number(high["sd_displacement_m"]),
                loss=loss,
                r2=number(high["reliability_02pct"]),
                r5=number(high["reliability_05pct"]),
                r10=number(high["reliability_10pct"]),
            )
        )

    theory_pairs = [
        (
            number(row["predicted_any_group_blackout_fraction"]),
            number(row["mean_any_group_blackout_fraction"]),
        )
        for row in summary
        if number(row["q"]) > 0.0
    ]
    theory_mae = sum(abs(predicted - measured) for predicted, measured in theory_pairs) / len(
        theory_pairs
    )
    theory_rmse = math.sqrt(
        sum((predicted - measured) ** 2 for predicted, measured in theory_pairs)
        / len(theory_pairs)
    )

    def recovery_value(transition: str, metric: str, field: str) -> float:
        row = next(
            item
            for item in recovery
            if item["transition"] == transition and item["metric"] == metric
        )
        return number(row[field])

    ordinary_linear_p95 = recovery_value("ordinary", "root_linear_delta_mps", "p95")
    restore_linear_p95 = recovery_value("restore", "root_linear_delta_mps", "p95")
    ordinary_linear_max = recovery_value("ordinary", "root_linear_delta_mps", "max")
    restore_linear_max = recovery_value("restore", "root_linear_delta_mps", "max")
    overlap_fraction = recovery_value(
        "restore", "fraction_restored_while_overlapping_floor", "median"
    )

    video_names = (
        "baseline_w3_q000.mp4",
        "single_erasure_w3_segment09_100ms.mp4",
        "short_erasure_w1_q030.mp4",
        "short_erasure_w2_q030.mp4",
        "short_erasure_w3_q030.mp4",
        "short_erasure_w4_q030.mp4",
    )
    video_lines = "\n".join(f"- `outputs/videos/{name}`" for name in video_names)

    report = fr"""# Independent short-contact-erasure rerun

## Status

This is the completed flat-ground rerun on the detailed FARMS MuJoCo centipede. It uses the validated local FARMS impedance/CPG controller, preserves the original `1e-4 kg m^2` joint armature, and varies only the integer spatial wave number and the independent short-erasure channel. Correlated erasure and rough terrain are outside this experiment.

The 164-trial sweep passed the preregistered dynamics gate: {len(unstable)} unstable trials, {len(nonfinite)} non-finite trials, and {len(self_contact)} robot self-contact trials. Across all runs, the largest absolute root roll was {max_roll:.2f} deg, the largest absolute root pitch was {max_pitch:.2f} deg, the largest segment tilt was {max_segment_tilt:.2f} deg, and the deepest dynamic floor contact was {-1000.0 * min_floor:.2f} mm.

## Controlled model delta

- Source: `{manifest['model_refinement']['source']}`
- Final model SHA-256: `{manifest['model_sha256']}`
- Time step and integrator: 0.5 ms, RK4
- Armature: unchanged at `1e-4 kg m^2`
- Active-joint passive stiffness/damping: removed for 211 controller-driven joints
- Passive leg-roll/foot springs and dampers: preserved on 76 hinges
- Robot-floor contact: enabled
- Robot self-collision: disabled for the 211 robot collision geoms
- Mass, inertia, geometry, friction, actuator definitions, and topology: unchanged
- Floor texture, lighting, shadow quality, camera, and red erasure highlighting are visualization-only changes

`audit_model_delta.py` is the executable contract for these statements.

## Channel definition

Let segment-pair index \(i\) range from 0 to 18. Pair \(i\) contains the complete left and right leg subtrees attached to that segment. Each support opportunity \(e\) draws an independent trigger

\[
Z_{{i,e}}\sim\operatorname{{Bernoulli}}(q).
\]

If \(Z_{{i,e}}=1\), all collision geoms of both legs in pair \(i\) are disabled for

\[
\tau_{{i,e}}=A_{{i,e}}\tau_s,\qquad \tau_s=0.5\ \mathrm{{s}},
\]

where \(A\) is log-uniform on \([0.02,1]\). Thus bursts range from 10 to 500 ms and

\[
\mathbb E[A]=\frac{{1-0.02}}{{\log(1/0.02)}}={manifest['channel']['expected_duration_fraction']:.6f},
\qquad \nu=q\mathbb E[A].
\]

Here \(q\) is the event-trigger probability per support opportunity; it is not the fraction of wall-clock time erased. The realized erased-time fraction is recorded separately in every trial.

For phase group \(\mathcal G_g\) with \(m_g=|\mathcal G_g|\), the independent random-snapshot approximation is

\[
P(B_g=1)=\nu^{{m_g}},\qquad
P\!\left(\bigcup_g B_g\right)
=1-\prod_g\left(1-\nu^{{m_g}}\right).
\]

The mean absolute error between this prediction and measured any-group blackout-time fraction over the 20 nonzero-\(q\) cells is {theory_mae:.4f}; RMSE is {theory_rmse:.4f}. This validates the event-level channel calculation, not a claim that displacement must equal blackout probability.

## Protocol

- Spatial wave number: \(w\in\{{1,2,3,4\}}\)
- Trigger probability: \(q\in\{{0,0.05,0.1,0.2,0.3,0.4\}}\)
- Trial duration: 8 s; measurement/noise interval: 2–8 s
- Replicates: one deterministic matched baseline per wave and eight seeds per nonzero \(q\)
- Total: 164 trials
- Terrain: flat and fixed
- Controller: fixed local FARMS configuration; only `body_wave.wave_number` changes
- Forward direction: native FARMS world \(-x\), reported as positive forward displacement

Wave number is not synonymous with redundancy. In this 19-pair/18-phase lattice, \(w=2\) and \(w=4\) both yield \(G=9\) and group sizes 2–3, whereas \(w=3\) yields \(G=6\) and sizes 3–4. The pair \(w=2,4\) is therefore an internal check that gait kinematics can change performance even when \(m_g\) is the same.

## Main results

Reliability at tolerance \(\epsilon\) is

\[
\widehat{{\mathcal R}}_{{\epsilon,w}}
=P\!\left(\text{{stable}}\ \land\ 
|\widehat D-D_{{0,w}}|\le \epsilon |D_{{0,w}}|\right),
\]

where \(D_{{0,w}}\) is the matched no-erasure baseline for wave number \(w\).

| w | G | group size \(m_g\) | baseline \(D_0\) (m) | \(D\) at \(q=0.4\) (m) | mean loss | \(\mathcal R_{{2\%}}\) | \(\mathcal R_{{5\%}}\) | \(\mathcal R_{{10\%}}\) |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

The primary absolute-displacement result is that 10–500 ms independent local erasures are mild for this controller over the tested range. The largest mean loss at \(q=0.4\) is only a few percent. A ±10% task tolerance therefore saturates at reliability 1.0. The separate ±2%, ±5%, and ±10% figures expose this threshold dependence without selecting a tolerance after seeing the data. The non-monotonic empirical points for \(w=1\) at ±2% reflect only eight seeds per cell and should not be interpreted as noise-enhanced locomotion.

## Contact-restoration audit

In the two-seed \(q=0.3\) transition audit, {100.0 * overlap_fraction:.1f}% of contact restorations occurred while at least one restored geom had negative signed floor distance. The root linear-velocity increment at restoration had 95th percentile {restore_linear_p95:.6f} m/s versus {ordinary_linear_p95:.6f} m/s on ordinary steps. Its maximum was {restore_linear_max:.6f} m/s versus {ordinary_linear_max:.6f} m/s on ordinary steps. Restoration therefore produces a small measurable distribution shift, but not a larger worst-case impulse than the unperturbed contact dynamics in this audit. The raw transition samples are retained so a later soft-contact or delayed-reactivation variant can be compared directly.

## Figures

- `outputs/figures/absolute_displacement_vs_q.png` — primary performance plot
- `outputs/figures/displacement_vs_realized_erased_load.png` — performance versus realized time erasure
- `outputs/figures/reliability_02pct_vs_q.png` — strict task reliability
- `outputs/figures/reliability_05pct_vs_q.png`
- `outputs/figures/reliability_vs_q.png` — proposal's current ±10% criterion
- `outputs/figures/stability_vs_q.png`
- `outputs/figures/group_blackout_vs_q.png`
- `outputs/figures/group_blackout_theory_vs_measured.png` — information-theory bridge
- `outputs/figures/wave_number_vs_redundancy.png`

## Videos

{video_lines}

Red legs and a red status dot indicate active contact erasure. The single-event video forces one 100 ms erasure on segment-pair 9 at 3 s; the other erasure videos use the exact seeded stochastic protocol.

## Evidence files

- `outputs/data/short_erasure_trials.csv` — all 164 trial-level records
- `outputs/data/short_erasure_summary.csv` — cell summaries
- `outputs/data/experiment_manifest.json` — exact controller snapshot, hashes, groups, and protocol
- `outputs/data/recovery_transition_samples.csv`
- `outputs/data/recovery_transition_summary.csv`
- `audit_model_delta.py`, `stability_gate.py`, `diagnose_recovery.py`, and `run_pipeline.py`

## Interpretation and next experiment

The event-level result agrees with the independent-channel prediction: larger homologous group size strongly suppresses complete phase-group blackout. The locomotion result is subtler: the detailed robot is so tolerant of short local erasures that mean displacement changes little, and \(w=2\) versus \(w=4\) shows that gait dynamics remain a confound even at identical group-size structure. The next defensible step is not correlated noise yet. It is a calibrated severity extension under the same independent channel—longer burst distributions or increasing numbers of simultaneously erased segment-pairs—until the ±10% reliability curve leaves saturation, while retaining the same stability gate and matched baselines.
"""
    (ROOT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(ROOT / "FINAL_REPORT.md")


if __name__ == "__main__":
    main()
