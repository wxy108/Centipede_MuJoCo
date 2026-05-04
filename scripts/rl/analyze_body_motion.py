"""
analyze_body_motion.py — Pure-analysis script that compares body-joint
trajectories between baseline (CPG only) and RL policy from PRE-RECORDED
NPZ files.

This script does NOT run any simulation.  It expects the NPZ files
written by `compare_baseline_rl.py` (`baseline_trajectory.npz` /
`rl_trajectory.npz`) or by `sweep_compare.py`
(`<output>/trajectories/wl<W>_baseline.npz` / `wl<W>_rl.npz`).

Each NPZ must contain:
    t        (T,)           time stamps (s)
    q_yaw    (T, N_BODY)    body yaw joint angles (rad)
    q_pitch  (T, N_PITCH)   body pitch joint angles (rad)  [optional]
    qd_yaw   (T, N_BODY)    body yaw joint velocities (rad/s)
    action   (T, A)         action issued to env (RL only meaningful)
    com      (T, 3)         COM xyz (m)
    dt       (1,)           RL step dt (s)

Usage
-----
  # Single (baseline, rl) pair, e.g. from compare_baseline_rl.py
  python scripts/rl/analyze_body_motion.py \\
      --baseline-npz outputs/rl/<run>/comparison_wl18_v25/baseline_trajectory.npz \\
      --rl-npz       outputs/rl/<run>/comparison_wl18_v25/rl_trajectory.npz \\
      --output-dir   outputs/rl/<run>/comparison_wl18_v25/analysis/

  # Auto-discover all wavelengths from sweep_compare's trajectories/ dir
  python scripts/rl/analyze_body_motion.py \\
      --sweep-trajectories-dir outputs/rl/<run>/sweep_compare/trajectories/

  # Pair-mode shortcut: pass a directory holding {baseline,rl}_trajectory.npz
  python scripts/rl/analyze_body_motion.py \\
      --input-dir outputs/rl/<run>/comparison_wl18_v25/
"""

import argparse
import csv
import os
import re
import sys

import numpy as np

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "controllers", "farms"))

# Only used for action-statistics interpretation; no env / mujoco imports.
from modulation_controller import (ACTION_DIM, AMP_SCALE_LO, AMP_SCALE_HI,
                                   PHASE_NUDGE_MAX_RAD)


# ─────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────
def load_trajectory(path):
    """Load an NPZ trajectory written by run_episode."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory not found: {path}")
    z = np.load(path, allow_pickle=False)
    qpitch = z["q_pitch"] if "q_pitch" in z.files and z["q_pitch"].size > 0 else None
    return {
        "t":       z["t"],
        "q_yaw":   z["q_yaw"],
        "q_pitch": qpitch,
        "qd_yaw":  z["qd_yaw"] if "qd_yaw" in z.files else None,
        "action":  z["action"],
        "com":     z["com"],
        "dt":      float(z["dt"][0]) if "dt" in z.files else float(np.mean(np.diff(z["t"]))),
    }


# ─────────────────────────────────────────────────────────────────────────
# Per-segment trajectory metrics
# ─────────────────────────────────────────────────────────────────────────
def per_segment_metrics(q_seg, dt):
    """Given a body-yaw trajectory of shape (T, N_seg), compute per-segment
    metrics: RMS amplitude, peak-to-peak, dominant frequency, phase lag
    relative to segment 0 (head)."""
    T, N = q_seg.shape
    rms = np.sqrt(np.mean(np.square(q_seg - q_seg.mean(axis=0)), axis=0))
    p2p = q_seg.max(axis=0) - q_seg.min(axis=0)

    freqs = np.fft.rfftfreq(T, d=dt)
    fft_seg = np.fft.rfft(q_seg - q_seg.mean(axis=0), axis=0)
    mag = np.abs(fft_seg)
    if mag.shape[0] > 1:
        mag[0] = 0   # discard DC
    f_dom_idx = mag.argmax(axis=0)
    f_dom = freqs[f_dom_idx]
    phase_at_dom = np.angle(fft_seg[f_dom_idx, np.arange(N)])
    phase_rel = np.angle(np.exp(1j * (phase_at_dom - phase_at_dom[0])))

    return {
        "rms":        rms,
        "p2p":        p2p,
        "f_dom_hz":   f_dom,
        "phase_lag":  phase_rel,
    }


def action_stats(actions):
    """Decode action vector (T, ACTION_DIM=36) into physical units.

    First half = phase nudges (raw [-1,1] → ±PHASE_NUDGE_MAX_RAD)
    Second half = amp scales  (raw [-1,1] → [AMP_SCALE_LO, AMP_SCALE_HI])
    """
    if actions is None or actions.size == 0:
        return None
    T, A = actions.shape
    half = A // 2
    a_phi = actions[:, :half]
    a_amp = actions[:, half:]
    phi_rad   = a_phi * PHASE_NUDGE_MAX_RAD
    amp_phys  = (0.5 * (AMP_SCALE_LO + AMP_SCALE_HI)
                 + 0.5 * (AMP_SCALE_HI - AMP_SCALE_LO) * a_amp)
    return {
        "phi_mean_rad": phi_rad.mean(axis=0),
        "phi_std_rad":  phi_rad.std (axis=0),
        "amp_mean":     amp_phys.mean(axis=0),
        "amp_std":      amp_phys.std (axis=0),
    }


# ─────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────
def plot_comparison(rec_b, rec_r, m_b, m_r, act_r, out_png, title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 11))
    gs  = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.35)

    qy_b = rec_b["q_yaw"]
    qy_r = rec_r["q_yaw"]
    t    = rec_b["t"]
    N    = qy_b.shape[1]

    # 1. Heatmaps of q_yaw[seg, t] for both controllers + diff
    extent = [t[0], t[-1], 0, N]
    vmax = max(np.max(np.abs(qy_b)), np.max(np.abs(qy_r)))
    ax = fig.add_subplot(gs[0, 0])
    im1 = ax.imshow(qy_b.T, aspect="auto", origin="lower",
                    extent=extent, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax.set_title("Baseline body yaw q[seg,t] (rad)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("segment")
    plt.colorbar(im1, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    im2 = ax.imshow(qy_r.T, aspect="auto", origin="lower",
                    extent=extent, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax.set_title("RL body yaw q[seg,t] (rad)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("segment")
    plt.colorbar(im2, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    diff = qy_r - qy_b
    dmax = max(np.max(np.abs(diff)), 1e-6)
    im3 = ax.imshow(diff.T, aspect="auto", origin="lower",
                    extent=extent, vmin=-dmax, vmax=dmax, cmap="PuOr_r")
    ax.set_title(f"Δq (RL − baseline)   max|Δ|={dmax:.3f} rad")
    ax.set_xlabel("time (s)"); ax.set_ylabel("segment")
    plt.colorbar(im3, ax=ax, fraction=0.046)

    # 2. Per-segment RMS amplitude
    ax = fig.add_subplot(gs[1, 0])
    seg_idx = np.arange(N)
    w = 0.4
    ax.bar(seg_idx - w/2, m_b["rms"], width=w, label="baseline", color="#888")
    ax.bar(seg_idx + w/2, m_r["rms"], width=w, label="RL",       color="#c0392b")
    ax.set_xlabel("segment"); ax.set_ylabel("RMS amplitude (rad)")
    ax.set_title("Per-segment RMS amplitude")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 3. Per-segment dominant frequency
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(seg_idx, m_b["f_dom_hz"], "o-", label="baseline", color="#888")
    ax.plot(seg_idx, m_r["f_dom_hz"], "s-", label="RL",       color="#c0392b")
    ax.set_xlabel("segment"); ax.set_ylabel("dominant freq (Hz)")
    ax.set_title("Per-segment dominant frequency (peak FFT)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 4. Phase-lag profile — traveling wave check
    ax = fig.add_subplot(gs[1, 2])
    pl_b = np.unwrap(m_b["phase_lag"])
    pl_r = np.unwrap(m_r["phase_lag"])
    ax.plot(seg_idx, np.degrees(pl_b), "o-", label="baseline", color="#888")
    ax.plot(seg_idx, np.degrees(pl_r), "s-", label="RL",       color="#c0392b")
    ax.set_xlabel("segment")
    ax.set_ylabel("phase lag rel. head (deg, unwrapped)")
    ax.set_title("Traveling-wave phase profile")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 5. RL action statistics
    if act_r is not None:
        n_mod = len(act_r["phi_mean_rad"])
        seg_mod = np.arange(1, 1 + n_mod)

        ax = fig.add_subplot(gs[2, 0])
        ax.errorbar(seg_mod, np.degrees(act_r["phi_mean_rad"]),
                    yerr=np.degrees(act_r["phi_std_rad"]),
                    fmt="o-", color="#c0392b", capsize=2)
        ax.axhline(0, color="black", lw=0.6)
        ax.fill_between(seg_mod,
                        -np.degrees(PHASE_NUDGE_MAX_RAD),
                        +np.degrees(PHASE_NUDGE_MAX_RAD),
                        color="#c0392b", alpha=0.07,
                        label=f"clip ±{np.degrees(PHASE_NUDGE_MAX_RAD):.1f}°")
        ax.set_xlabel("modulated segment")
        ax.set_ylabel("phase nudge (deg, mean ± std)")
        ax.set_title("RL phase-nudge action — what the policy chose")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[2, 1])
        ax.errorbar(seg_mod, act_r["amp_mean"], yerr=act_r["amp_std"],
                    fmt="s-", color="#c0392b", capsize=2)
        ax.axhline(1.0, color="black", lw=0.6, ls="--", label="full CPG amp")
        ax.axhline(AMP_SCALE_LO, color="grey", lw=0.6, ls=":")
        ax.axhline(AMP_SCALE_HI, color="grey", lw=0.6, ls=":")
        ax.set_ylim(AMP_SCALE_LO - 0.05, AMP_SCALE_HI + 0.05)
        ax.set_xlabel("modulated segment")
        ax.set_ylabel("amp scale (mean ± std)")
        ax.set_title("RL amp-scale action — attenuation per segment")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[2, 2])
        d_rms = m_r["rms"] - m_b["rms"]
        colors = ["#1f77b4" if d < 0 else "#d62728" for d in d_rms]
        ax.bar(seg_idx, d_rms, color=colors)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("segment"); ax.set_ylabel("Δ RMS amplitude (rad)")
        ax.set_title("Per-segment Δ amplitude (RL − baseline)")
        ax.grid(alpha=0.3)

    # 6. Sample time series — head, mid, tail
    sample_segs = [0, N // 2, N - 1]
    for col, s in enumerate(sample_segs):
        ax = fig.add_subplot(gs[3, col])
        ax.plot(t, qy_b[:, s], color="#888", lw=1.0, label="baseline")
        ax.plot(t, qy_r[:, s], color="#c0392b", lw=1.0, label="RL")
        ax.set_title(f"segment {s} time-series")
        ax.set_xlabel("time (s)"); ax.set_ylabel("q_yaw (rad)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"Body-motion comparison: BASELINE vs RL{title_suffix}",
                 fontsize=12)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] PNG  → {out_png}")


def write_csv(path, ctrl_name, m, act_stats=None):
    N = len(m["rms"])
    fieldnames = ["controller", "segment",
                  "rms_rad", "p2p_rad", "f_dom_hz", "phase_lag_rad"]
    if act_stats is not None:
        fieldnames += ["phi_mean_rad", "phi_std_rad", "amp_mean", "amp_std"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for i in range(N):
            row = {
                "controller":     ctrl_name,
                "segment":        i,
                "rms_rad":        float(m["rms"][i]),
                "p2p_rad":        float(m["p2p"][i]),
                "f_dom_hz":       float(m["f_dom_hz"][i]),
                "phase_lag_rad":  float(m["phase_lag"][i]),
            }
            if act_stats is not None:
                if i < len(act_stats["phi_mean_rad"]):
                    row["phi_mean_rad"] = float(act_stats["phi_mean_rad"][i])
                    row["phi_std_rad"]  = float(act_stats["phi_std_rad"][i])
                    row["amp_mean"]     = float(act_stats["amp_mean"][i])
                    row["amp_std"]      = float(act_stats["amp_std"][i])
                else:
                    row["phi_mean_rad"] = float("nan")
                    row["phi_std_rad"]  = float("nan")
                    row["amp_mean"]     = float("nan")
                    row["amp_std"]      = float("nan")
            w.writerow(row)


def print_summary(m_b, m_r, act_r, header=""):
    if header:
        print("\n" + header)
    print("=" * 90)
    print("Per-segment body-yaw amplitude/frequency comparison")
    print("=" * 90)
    print("{:>3} | {:>10} {:>10} {:>9} | {:>10} {:>10} {:>9} | {:>10}".format(
        "seg", "RMS_B(rad)", "RMS_R(rad)", "Δ",
        "fdom_B", "fdom_R", "Δ", "ΔRMS%"))
    print("-" * 90)
    for i in range(len(m_b["rms"])):
        rb = m_b["rms"][i]; rr = m_r["rms"][i]
        fb = m_b["f_dom_hz"][i]; fr = m_r["f_dom_hz"][i]
        d_pct = 100.0 * (rr - rb) / rb if rb > 1e-9 else 0.0
        print(f"{i:>3} | {rb:10.4f} {rr:10.4f} {rr-rb:+9.4f} | "
              f"{fb:10.3f} {fr:10.3f} {fr-fb:+9.3f} | {d_pct:+9.1f}%")
    print("=" * 90)


# ─────────────────────────────────────────────────────────────────────────
# CLI dispatch
# ─────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--baseline-npz",
                   help="Path to baseline_trajectory.npz (use with --rl-npz)")
    g.add_argument("--input-dir",
                   help="Directory containing baseline_trajectory.npz + "
                        "rl_trajectory.npz (e.g., a comparison_wlX_vY/ folder)")
    g.add_argument("--sweep-trajectories-dir",
                   help="Directory containing wl<W>_baseline.npz + wl<W>_rl.npz "
                        "pairs for multiple wavelengths (e.g., sweep_compare/trajectories/)")
    p.add_argument("--rl-npz",
                   help="Path to rl_trajectory.npz (used only with --baseline-npz)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write CSV+PNG. Defaults to --input-dir or to "
                        "the parent dir of --baseline-npz.")
    return p.parse_args()


def analyze_pair(baseline_npz, rl_npz, out_dir, label=""):
    """Load a (baseline, rl) NPZ pair and emit CSV+PNG into out_dir."""
    rec_b = load_trajectory(baseline_npz)
    rec_r = load_trajectory(rl_npz)

    if rec_b["q_yaw"].shape != rec_r["q_yaw"].shape:
        print(f"WARNING: q_yaw shape mismatch: "
              f"baseline {rec_b['q_yaw'].shape} vs rl {rec_r['q_yaw'].shape}.  "
              f"Truncating to common length.")
        T = min(rec_b["q_yaw"].shape[0], rec_r["q_yaw"].shape[0])
        for d in (rec_b, rec_r):
            for k in ("t", "q_yaw", "qd_yaw", "action", "com"):
                if d.get(k) is not None and d[k].shape[0] > T:
                    d[k] = d[k][:T]
            if d.get("q_pitch") is not None and d["q_pitch"].shape[0] > T:
                d["q_pitch"] = d["q_pitch"][:T]

    m_b = per_segment_metrics(rec_b["q_yaw"], rec_b["dt"])
    m_r = per_segment_metrics(rec_r["q_yaw"], rec_r["dt"])
    act_r = action_stats(rec_r["action"])

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "body_motion.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    write_csv(csv_path, "baseline", m_b, act_stats=None)
    write_csv(csv_path, "rl",       m_r, act_stats=act_r)
    print(f"[saved] CSV  → {csv_path}")

    png_path = os.path.join(out_dir, "body_motion.png")
    plot_comparison(rec_b, rec_r, m_b, m_r, act_r,
                    out_png=png_path,
                    title_suffix=f"  ({label})" if label else "")

    print_summary(m_b, m_r, act_r,
                  header=f"--- {label} ---" if label else "")


def main():
    args = parse_args()

    # ── Mode 1: explicit pair ────────────────────────────────────────
    if args.baseline_npz:
        if not args.rl_npz:
            print("ERROR: --baseline-npz requires --rl-npz")
            sys.exit(1)
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.baseline_npz))
        analyze_pair(args.baseline_npz, args.rl_npz, out_dir)
        return

    # ── Mode 2: directory containing baseline_trajectory.npz + rl_trajectory.npz
    if args.input_dir:
        b = os.path.join(args.input_dir, "baseline_trajectory.npz")
        r = os.path.join(args.input_dir, "rl_trajectory.npz")
        if not (os.path.exists(b) and os.path.exists(r)):
            print(f"ERROR: expected baseline_trajectory.npz and rl_trajectory.npz "
                  f"inside {args.input_dir}")
            sys.exit(1)
        out_dir = args.output_dir or args.input_dir
        analyze_pair(b, r, out_dir)
        return

    # ── Mode 3: sweep_compare trajectories directory ─────────────────
    if args.sweep_trajectories_dir:
        d = args.sweep_trajectories_dir
        files = sorted(os.listdir(d))
        # Match BOTH single-trial (wl18_baseline.npz) and multi-trial
        # (wl18_t0_yaw047_baseline.npz) layouts.
        pat = re.compile(r"^(wl(\d+)(?:_t(\d+)_yaw(\d+))?)_baseline\.npz$")
        pairs = []   # list of (wl_mm, trial_idx_or_None, base_path, rl_path, label, sub_name)
        for f in files:
            m = pat.match(f)
            if not m:
                continue
            stem, wl_str, t_str, yaw_str = m.groups()
            wl = int(wl_str)
            base_path = os.path.join(d, f)
            rl_path   = os.path.join(d, f"{stem}_rl.npz")
            if not os.path.exists(rl_path):
                print(f"[skip] missing rl pair for {stem}")
                continue
            if t_str is not None:
                tidx = int(t_str)
                label = f"wl = {wl} mm, trial {tidx} (yaw {yaw_str}°)"
                sub_name = f"wl{wl}/trial{tidx}"
            else:
                tidx = None
                label = f"wl = {wl} mm"
                sub_name = f"wl{wl}"
            pairs.append((wl, tidx, base_path, rl_path, label, sub_name))

        if not pairs:
            print(f"ERROR: no wl<W>(_t<T>_yaw<D>)?_baseline.npz files in {d}")
            sys.exit(1)

        out_root = args.output_dir or os.path.join(
            os.path.dirname(d.rstrip("/")), "body_motion")
        # Sort by (wl, trial) for stable ordering
        pairs.sort(key=lambda x: (x[0], x[1] if x[1] is not None else -1))
        for wl, tidx, b, r, label, sub_name in pairs:
            sub = os.path.join(out_root, sub_name)
            print(f"\n=== Analyzing {label} ===")
            analyze_pair(b, r, sub, label=label)
        return


if __name__ == "__main__":
    main()
