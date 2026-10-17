"""Strong independent partial-contact-erasure severity sweep.

The x-axis is nu: the expected fraction of time for which one segment-pair's
contact channel is erased. Bursts remain shorter than or equal to one 0.5 s
support window. For nu <= 0.75, A is uniform on [0.5, 1] and q=nu/0.75.
For nu > 0.75, every support opportunity triggers and A is uniform on
[2*nu-1, 1], preserving E[A]=nu continuously up to complete erasure.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_pipeline as base


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "strong_independent_erasure"
PILOT = OUT / "pilot"
FULL = OUT / "full"
HIGH = OUT / "high_extension"
EXTENDED = OUT / "extended_full"
WAVE_5_TO_9_PILOT = OUT / "wave_5_to_9_pilot"
WAVE_5_TO_9 = OUT / "wave_5_to_9"
ALL_WAVES_OUTPUT = OUT / "all_waves_1_to_9"
VIDEO = OUT / "videos"

BURST_MIN = 0.5
BURST_MAX = 1.0
MEAN_BURST_FRACTION = 0.5 * (BURST_MIN + BURST_MAX)
PILOT_NU = (0.0, 0.30, 0.45, 0.60)
FULL_NU = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
HIGH_NU = (0.0, 0.70, 0.80, 0.90, 1.0)
WAVES = (1, 2, 3, 4)
MORE_WAVES = (5, 6, 7, 8, 9)
ALL_WAVES = tuple(range(1, 10))
ALL_NU = tuple(float(value) / 10.0 for value in range(11))


@dataclass(frozen=True)
class SeverityCase:
    wave: int
    nu: float
    seed: int
    duration_s: float
    eval_start_s: float = 2.0


def channel_parameters(nu: float) -> tuple[float, float, float, float]:
    """Return q, minimum A, maximum A, and E[A] for a target erased load."""
    nu = float(nu)
    if not 0.0 <= nu <= 1.0:
        raise ValueError(f"nu must lie in [0, 1], received {nu}")
    if nu <= MEAN_BURST_FRACTION:
        return nu / MEAN_BURST_FRACTION, BURST_MIN, BURST_MAX, MEAN_BURST_FRACTION
    minimum = 2.0 * nu - 1.0
    return 1.0, minimum, 1.0, nu


def trigger_probability(nu: float) -> float:
    return channel_parameters(nu)[0]


def build_strong_events(case: SeverityCase) -> tuple[list[base.ErasureEvent], int]:
    q, burst_min, burst_max, _ = channel_parameters(case.nu)
    rng = np.random.default_rng(case.seed + 100_003 * case.wave + 7_000_001)
    events: list[base.ErasureEvent] = []
    opportunities = 0
    omega = 2.0 * math.pi * base.FREQUENCY_HZ
    for segment in range(base.N_LEG_PAIRS):
        phase = 2.0 * math.pi * case.wave * segment / (base.N_BODY_YAW - 1)
        phase_offset = (
            0.0
            if math.isclose(case.nu, 1.0, abs_tol=1e-12)
            else (phase / omega) % base.SUPPORT_WINDOW_S
        )
        window_start = case.eval_start_s + phase_offset
        channel_use = 0
        while window_start < case.duration_s - 1e-12:
            opportunities += 1
            u_trigger, u_duration, u_offset = rng.random(3)
            fraction = burst_min + (burst_max - burst_min) * u_duration
            duration = fraction * base.SUPPORT_WINDOW_S
            start = window_start + u_offset * (base.SUPPORT_WINDOW_S - duration)
            if u_trigger < q:
                events.append(
                    base.ErasureEvent(
                        segment=segment,
                        channel_use=channel_use,
                        window_start_s=window_start,
                        start_s=start,
                        end_s=start + duration,
                        duration_s=duration,
                        duration_fraction=fraction,
                        trigger_uniform=u_trigger,
                    )
                )
            channel_use += 1
            window_start += base.SUPPORT_WINDOW_S
    events.sort(key=lambda event: event.start_s)
    return events, opportunities


def run_case(case: SeverityCase, render_path: Path | None = None) -> dict:
    q, burst_min, burst_max, expected_burst_fraction = channel_parameters(case.nu)
    events, opportunities = build_strong_events(case)
    config = base.TrialConfig(
        wave=case.wave,
        q=q,
        seed=case.seed,
        duration_s=case.duration_s,
        eval_start_s=case.eval_start_s,
        minimum_burst_fraction=burst_min,
    )
    result = base.run_trial(
        config,
        render_path=render_path,
        events_override=events,
        render_noise_label=f"nu={case.nu:.2f}",
    )
    result.update(
        {
            "noise_level_nu": case.nu,
            "event_trigger_probability_q": q,
            "burst_distribution": f"uniform_fraction_[{burst_min:.6g},{burst_max:.6g}]",
            "minimum_burst_fraction": burst_min,
            "maximum_burst_fraction": burst_max,
            "expected_burst_fraction": expected_burst_fraction,
            "expected_erased_load": case.nu,
            "channel_opportunities": opportunities,
            "triggered_events": len(events),
            "realized_trigger_rate": len(events) / max(opportunities, 1),
        }
    )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def predicted_any_group_blackout(wave: int, nu: float) -> float:
    survival = 1.0
    for group in base.phase_groups(wave):
        survival *= 1.0 - float(nu) ** len(group)
    return 1.0 - survival


def predicted_mean_group_blackout(wave: int, nu: float) -> float:
    groups = base.phase_groups(wave)
    return float(np.mean([float(nu) ** len(group) for group in groups]))


def summarize(raw: list[dict]) -> list[dict]:
    baselines = {
        int(row["wave"]): float(row["absolute_forward_displacement_m"])
        for row in raw
        if float(row["noise_level_nu"]) == 0.0
    }
    rows_out: list[dict] = []
    waves = sorted({int(row["wave"]) for row in raw})
    for wave in waves:
        levels = sorted(
            {float(row["noise_level_nu"]) for row in raw if int(row["wave"]) == wave}
        )
        for nu in levels:
            cell = [
                row
                for row in raw
                if int(row["wave"]) == wave
                and float(row["noise_level_nu"]) == nu
            ]
            displacement = np.asarray(
                [float(row["absolute_forward_displacement_m"]) for row in cell]
            )
            stable = np.asarray(
                [str(row["stable"]).lower() == "true" for row in cell], dtype=bool
            )
            baseline = baselines[wave]
            stable_displacement = displacement[stable]
            baseline_valid = abs(baseline) > 1e-9
            template = cell[0]
            result = {
                "wave": wave,
                "phase_group_count": int(template["phase_group_count"]),
                "m_min": int(template["m_min"]),
                "m_max": int(template["m_max"]),
                "noise_level_nu": nu,
                "event_trigger_probability_q": float(
                    template["event_trigger_probability_q"]
                ),
                "n": len(cell),
                "baseline_displacement_m": baseline,
                # Primary performance statistics exclude dynamically invalid trials.
                # The all-trial value and stable_fraction preserve the failure evidence.
                "mean_displacement_m": float(np.mean(stable_displacement))
                if len(stable_displacement)
                else float("nan"),
                "mean_displacement_all_trials_m": float(np.mean(displacement)),
                "sd_displacement_m": float(np.std(stable_displacement, ddof=1))
                if len(stable_displacement) > 1
                else 0.0,
                "mean_retained_performance": float(
                    np.mean(stable_displacement / baseline)
                )
                if len(stable_displacement) and baseline_valid
                else float("nan"),
                "mean_performance_loss": float(
                    np.mean((baseline - stable_displacement) / baseline)
                )
                if len(stable_displacement) and baseline_valid
                else float("nan"),
                "sd_fractional_performance_loss": float(
                    np.std((baseline - stable_displacement) / baseline, ddof=1)
                )
                if len(stable_displacement) > 1 and baseline_valid
                else 0.0,
                "stable_fraction": float(np.mean(stable)),
                "mean_realized_erased_load": float(
                    np.mean([float(row["realized_erased_load"]) for row in cell])
                ),
                "mean_any_group_blackout_fraction": float(
                    np.mean([float(row["any_group_blackout_fraction"]) for row in cell])
                ),
                "mean_group_blackout_fraction": float(
                    np.mean([float(row["mean_group_blackout_fraction"]) for row in cell])
                ),
                "predicted_any_group_blackout_fraction": predicted_any_group_blackout(
                    wave, nu
                ),
                "predicted_mean_group_blackout_fraction": predicted_mean_group_blackout(
                    wave, nu
                ),
            }
            for percentage in (2, 5, 10):
                tolerance = percentage / 100.0
                success = stable & (
                    np.abs(displacement - baseline) <= tolerance * abs(baseline)
                )
                result[f"reliability_{percentage:02d}pct"] = float(np.mean(success))
            rows_out.append(result)
    return rows_out


def label_for(row: dict) -> str:
    wave = int(row["wave"])
    minimum = int(row["m_min"])
    maximum = int(row["m_max"])
    return rf"$N_{{\mathrm{{wave}}}}={wave},\ m_g={minimum}{{-}}{maximum}$"


def style(axis, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2, linewidth=0.7)


def make_figures(
    summary: list[dict], directory: Path, waves: tuple[int, ...] | None = None
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if waves is None:
        waves = tuple(sorted({int(row["wave"]) for row in summary}))
    colors = dict(
        zip(waves, plt.get_cmap("viridis")(np.linspace(0.08, 0.94, len(waves))))
    )
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
    marker_for = {wave: markers[index % len(markers)] for index, wave in enumerate(waves)}

    def curves(key: str, ylabel: str, filename: str, error_key: str | None = None):
        width = 7.2 if len(waves) > 5 else 6.3
        fig, axis = plt.subplots(figsize=(width, 4.4), constrained_layout=True)
        for wave in waves:
            cell = sorted(
                [row for row in summary if int(row["wave"]) == wave],
                key=lambda row: float(row["noise_level_nu"]),
            )
            x = np.asarray([float(row["noise_level_nu"]) for row in cell])
            y = np.asarray([float(row[key]) for row in cell])
            if error_key:
                error = np.asarray([float(row[error_key]) for row in cell])
                axis.errorbar(
                    x,
                    y,
                    yerr=error,
                    fmt=marker_for[wave],
                    linestyle="-",
                    color=colors[wave],
                    capsize=2.5,
                    label=label_for(cell[0]),
                )
            else:
                axis.plot(
                    x,
                    y,
                    marker=marker_for[wave],
                    linestyle="-",
                    color=colors[wave],
                    label=label_for(cell[0]),
                )
        style(axis, r"Noise level $\nu$ (erased-time fraction)", ylabel)
        axis.legend(frameon=False, fontsize=7.5, ncol=2 if len(waves) > 5 else 1)
        fig.savefig(directory / filename, dpi=220)
        plt.close(fig)

    curves(
        "mean_displacement_m",
        "Forward displacement (m)",
        "absolute_displacement_vs_nu.png",
        "sd_displacement_m",
    )
    curves(
        "mean_displacement_m",
        "Forward displacement (m)",
        "forward_displacement_vs_nu.png",
        "sd_displacement_m",
    )
    curves(
        "mean_retained_performance",
        r"Retained displacement $D/D_{0,k}$",
        "retained_displacement_vs_nu.png",
    )
    curves(
        "mean_performance_loss",
        "Mean fractional performance loss",
        "performance_loss_vs_nu.png",
        "sd_fractional_performance_loss",
    )
    curves(
        "mean_performance_loss",
        "Mean fractional performance loss",
        "mean_fractional_performance_loss_vs_nu.png",
        "sd_fractional_performance_loss",
    )
    curves("reliability_02pct", "Reliability (2% tolerance)", "reliability_02pct_vs_nu.png")
    curves("reliability_05pct", "Reliability (5% tolerance)", "reliability_05pct_vs_nu.png")
    curves("reliability_10pct", "Reliability (10% tolerance)", "reliability_10pct_vs_nu.png")
    curves("stable_fraction", "Dynamically stable trial fraction", "stability_vs_nu.png")
    curves(
        "mean_any_group_blackout_fraction",
        "Time with any phase group erased",
        "group_blackout_vs_nu.png",
    )
    curves(
        "mean_group_blackout_fraction",
        "Mean phase-group blackout fraction",
        "mean_group_blackout_vs_nu.png",
    )

    fig, axis = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    for wave in waves:
        cell = [
            row
            for row in summary
            if int(row["wave"]) == wave and float(row["noise_level_nu"]) > 0.0
        ]
        axis.scatter(
            [float(row["predicted_any_group_blackout_fraction"]) for row in cell],
            [float(row["mean_any_group_blackout_fraction"]) for row in cell],
            color=colors[wave],
            s=35,
            label=label_for(cell[0]),
        )
    axis.plot([0, 1], [0, 1], "--", color="0.4", linewidth=1.2)
    style(axis, "Independent-channel prediction", "Measured blackout-time fraction")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(directory / "group_blackout_theory_vs_measured.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    for wave in waves:
        cell = [
            row
            for row in summary
            if int(row["wave"]) == wave and float(row["noise_level_nu"]) > 0.0
        ]
        axis.scatter(
            [float(row["predicted_mean_group_blackout_fraction"]) for row in cell],
            [float(row["mean_group_blackout_fraction"]) for row in cell],
            color=colors[wave],
            s=35,
            label=label_for(cell[0]),
        )
    axis.plot([0, 1], [0, 1], "--", color="0.4", linewidth=1.2)
    style(axis, "Independent-channel prediction", "Measured mean group-blackout fraction")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(directory / "mean_group_blackout_theory_vs_measured.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    for wave in waves:
        cell = sorted(
            [
                row
                for row in summary
                if int(row["wave"]) == wave and float(row["noise_level_nu"]) > 0.0
            ],
            key=lambda row: float(row["mean_group_blackout_fraction"]),
        )
        axis.plot(
            [float(row["mean_group_blackout_fraction"]) for row in cell],
            [float(row["mean_performance_loss"]) for row in cell],
            marker=marker_for[wave],
            linestyle="-",
            color=colors[wave],
            label=label_for(cell[0]),
        )
    style(axis, "Measured mean group-blackout fraction", "Mean fractional performance loss")
    axis.legend(frameon=False, fontsize=8)
    fig.savefig(directory / "performance_loss_vs_group_blackout.png", dpi=220)
    plt.close(fig)


def run_sweep(
    destination: Path,
    levels: tuple[float, ...],
    seeds: int,
    duration_s: float,
    workers: int,
    waves: tuple[int, ...] = WAVES,
) -> tuple[list[dict], list[dict]]:
    cases: list[SeverityCase] = []
    for wave in waves:
        for nu in levels:
            cell_seeds = (0,) if nu in (0.0, 1.0) else tuple(range(seeds))
            for seed in cell_seeds:
                cases.append(SeverityCase(wave, nu, seed, duration_s))
    started = time.time()
    raw: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_case, case): case for case in cases}
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            case = futures[future]
            result = future.result()
            raw.append(result)
            print(
                f"[{completed:03d}/{len(cases):03d}] N_wave={case.wave} "
                f"nu={case.nu:.2f} seed={case.seed} "
                f"D={result['absolute_forward_displacement_m']:.4f} "
                f"stable={result['stable']} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    raw.sort(key=lambda row: (int(row["wave"]), float(row["noise_level_nu"]), int(row["seed"])))
    summary = summarize(raw)
    write_csv(destination / "trials.csv", raw)
    write_csv(destination / "summary.csv", summary)
    make_figures(summary, destination / "figures", waves)
    manifest = {
        "noise_level": "nu = expected erased-time fraction of one segment-pair",
        "channel_schedule": {
            "nu_le_0.75": "q=nu/0.75, A~uniform[0.5,1]",
            "nu_gt_0.75": "q=1, A~uniform[2nu-1,1]",
            "nu_eq_1": "all segment-pair leg contacts continuously erased",
        },
        "waves": list(waves),
        "levels": list(levels),
        "seeds_per_nonzero_cell": seeds,
        "duration_s": duration_s,
        "eval_start_s": 2.0,
        "model": str(base.MODEL_PATH),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return raw, summary


def combine_all_waves(new_raw: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge the archived N_wave=1--4 and N_wave=5--9 scans."""
    original_raw = read_csv(EXTENDED / "trials.csv")
    combined = original_raw + new_raw
    combined.sort(
        key=lambda row: (
            int(row["wave"]),
            float(row["noise_level_nu"]),
            int(row["seed"]),
        )
    )
    summary = summarize(combined)
    write_csv(ALL_WAVES_OUTPUT / "trials.csv", combined)
    write_csv(ALL_WAVES_OUTPUT / "summary.csv", summary)
    make_figures(summary, ALL_WAVES_OUTPUT / "figures_all", ALL_WAVES)

    # These panels answer different questions and should not be visually conflated.
    # N_wave=1,2,3,6,9 is the increasing-multiplicity sequence.
    make_figures(
        summary,
        ALL_WAVES_OUTPUT / "figures_redundancy_scan",
        (1, 2, 3, 6, 9),
    )
    # Same-multiplicity families isolate gait ordering from nominal redundancy.
    make_figures(
        summary,
        ALL_WAVES_OUTPUT / "figures_same_m_low",
        (1, 5, 7),
    )
    make_figures(
        summary,
        ALL_WAVES_OUTPUT / "figures_same_m_mid",
        (2, 4, 8),
    )
    multiplicities = {
        str(wave): {
            "phase_group_count": len(base.phase_groups(wave)),
            "group_sizes": [len(group) for group in base.phase_groups(wave)],
        }
        for wave in ALL_WAVES
    }
    manifest = {
        "source_waves_1_to_4": str(EXTENDED / "trials.csv"),
        "source_waves_5_to_9": str(WAVE_5_TO_9 / "trials.csv"),
        "noise_levels": list(ALL_NU),
        "total_trials": len(combined),
        "multiplicity_by_wave": multiplicities,
        "ordered_redundancy_scan": [1, 2, 3, 6, 9],
        "same_multiplicity_controls": [[1, 5, 7], [2, 4, 8]],
        "performance_gate": (
            "mean_displacement_m and retention use dynamically stable trials only; "
            "stable_fraction and all-trial displacement preserve invalid-run evidence"
        ),
    }
    (ALL_WAVES_OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return combined, summary


def parse_levels(text: str) -> tuple[float, ...]:
    return tuple(float(value.strip()) for value in text.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--levels", default=",".join(map(str, PILOT_NU)))
    pilot.add_argument("--seeds", type=int, default=3)
    pilot.add_argument("--duration", type=float, default=6.0)
    pilot.add_argument("--workers", type=int, default=4)
    full = sub.add_parser("full")
    full.add_argument("--levels", default=",".join(map(str, FULL_NU)))
    full.add_argument("--seeds", type=int, default=8)
    full.add_argument("--duration", type=float, default=8.0)
    full.add_argument("--workers", type=int, default=4)
    extend = sub.add_parser("extend")
    extend.add_argument("--levels", default=",".join(map(str, HIGH_NU)))
    extend.add_argument("--seeds", type=int, default=8)
    extend.add_argument("--duration", type=float, default=8.0)
    extend.add_argument("--workers", type=int, default=4)
    more_pilot = sub.add_parser("waves5to9-pilot")
    more_pilot.add_argument("--levels", default="0,0.3,0.6,1.0")
    more_pilot.add_argument("--seeds", type=int, default=1)
    more_pilot.add_argument("--duration", type=float, default=6.0)
    more_pilot.add_argument("--workers", type=int, default=4)
    more = sub.add_parser("waves5to9")
    more.add_argument("--levels", default=",".join(map(str, ALL_NU)))
    more.add_argument("--seeds", type=int, default=8)
    more.add_argument("--duration", type=float, default=8.0)
    more.add_argument("--workers", type=int, default=4)
    more_demo = sub.add_parser("waves5to9-demo")
    more_demo.add_argument("--nu", type=float, default=0.6)
    more_demo.add_argument("--seed", type=int, default=0)
    more_demo.add_argument("--duration", type=float, default=8.0)
    demo = sub.add_parser("demo")
    demo.add_argument("--wave", type=int, default=3)
    demo.add_argument("--nu", type=float, default=0.6)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()

    if args.command == "pilot":
        run_sweep(PILOT, parse_levels(args.levels), args.seeds, args.duration, args.workers)
    elif args.command == "full":
        run_sweep(FULL, parse_levels(args.levels), args.seeds, args.duration, args.workers)
    elif args.command == "extend":
        high_raw, _ = run_sweep(
            HIGH, parse_levels(args.levels), args.seeds, args.duration, args.workers
        )
        original_raw = read_csv(FULL / "trials.csv")
        combined = original_raw + [
            row for row in high_raw if float(row["noise_level_nu"]) > 0.60
        ]
        combined.sort(
            key=lambda row: (
                int(row["wave"]),
                float(row["noise_level_nu"]),
                int(row["seed"]),
            )
        )
        summary = summarize(combined)
        write_csv(EXTENDED / "trials.csv", combined)
        write_csv(EXTENDED / "summary.csv", summary)
        make_figures(summary, EXTENDED / "figures")
        (EXTENDED / "manifest.json").write_text(
            json.dumps(
                {
                    "source_low_noise": str(FULL / "trials.csv"),
                    "source_high_noise": str(HIGH / "trials.csv"),
                    "noise_levels": sorted(
                        {float(row["noise_level_nu"]) for row in combined}
                    ),
                    "total_trials": len(combined),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    elif args.command == "waves5to9-pilot":
        run_sweep(
            WAVE_5_TO_9_PILOT,
            parse_levels(args.levels),
            args.seeds,
            args.duration,
            args.workers,
            MORE_WAVES,
        )
    elif args.command == "waves5to9":
        new_raw, _ = run_sweep(
            WAVE_5_TO_9,
            parse_levels(args.levels),
            args.seeds,
            args.duration,
            args.workers,
            MORE_WAVES,
        )
        combine_all_waves(new_raw)
    elif args.command == "waves5to9-demo":
        VIDEO.mkdir(parents=True, exist_ok=True)
        for wave in MORE_WAVES:
            path = VIDEO / (
                f"strong_erasure_k{wave}_nu{args.nu:.2f}_seed{args.seed}.mp4"
            )
            result = run_case(
                SeverityCase(wave, args.nu, args.seed, args.duration),
                render_path=path,
            )
            print(
                f"N_wave={wave} video={path.name} "
                f"D={result['absolute_forward_displacement_m']:.4f} "
                f"stable={result['stable']}",
                flush=True,
            )
    elif args.command == "demo":
        VIDEO.mkdir(parents=True, exist_ok=True)
        path = VIDEO / f"strong_erasure_k{args.wave}_nu{args.nu:.2f}_seed{args.seed}.mp4"
        result = run_case(
            SeverityCase(args.wave, args.nu, args.seed, args.duration),
            render_path=path,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
