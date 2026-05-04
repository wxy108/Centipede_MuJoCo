#!/usr/bin/env python3
"""
analyze_sensor_data.py — Analyse a rich per-frame trial dump (.npz)
====================================================================
Companion to ``scripts/sweep/sensor_recorder.py``.  Given one or more
``trial_*.npz`` files produced by ``--sensor-data`` on ``wave_number_sweep.py``,
produces a folder of diagnostic plots + a concise text/JSON summary.

Examples
--------
# Analyse a single trial
python analysis/analyze_sensor_data.py \
    outputs/wave_number_sweep/sweep_<TS>/sensors_rich/k3.0/wl_18mm/trial_00.npz

# Analyse every trial in a run directory (produces one subfolder per .npz)
python analysis/analyze_sensor_data.py \
    outputs/wave_number_sweep/sweep_<TS>/sensors_rich --recursive

# Pipe the outputs elsewhere (default is alongside each npz)
python analysis/analyze_sensor_data.py trial_00.npz --out analysis_out/

What gets plotted
-----------------
  tracking_body.png        Body-yaw: q vs target (all 19 joints)
  tracking_legs.png        Leg hip yaw/pitch: q vs target (L/R, all 19 legs)
  tracking_pitch_roll.png  Pitch/roll joint angles (all segments)
  torque_saturation.png    |τ_cmd| / |τ_actual| envelopes per joint family
  contact_forces.png       Per-leg foot force magnitudes + grid heatmap
  contact_imbalance.png    L/R and front/back contact imbalance timeseries
  body_rpy.png             Root body roll / pitch / yaw
  gait_diagram.png         Contact on/off pattern for L + R legs over time
  cot_and_power.png        Mechanical power + cumulative mechanical work
  summary.json             Scalar metrics (tracking RMSE, saturation, CoT, ...)
  summary.txt              Same, human-readable

The analysis skips the first ``--settle`` seconds (default 3.0 s: matches the
settle + ramp window in the sweep).  Use ``--settle 0`` to include the whole
trajectory.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

# Matplotlib is used only at plot-time so the script can be imported for
# scripted use without the backend dependency.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


# ══════════════════════════════════════════════════════════════════════════════
# Quaternion → Euler (z-y-x intrinsic) helper.  Duplicated here so this script
# is stand-alone and doesn't need to import sensor_recorder.
# ══════════════════════════════════════════════════════════════════════════════

def quat_to_euler(q):
    q = np.asarray(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.stack([roll, pitch, yaw], axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Trial loader
# ══════════════════════════════════════════════════════════════════════════════

class Trial:
    """Lazy-ish wrapper around a sensor_recorder .npz dump."""

    def __init__(self, path):
        self.path = path
        self._d = dict(np.load(path, allow_pickle=False))
        # Scalars come back as arrays of shape (1,)
        self.dt              = float(self._d["dt"][0])
        self.dt_record       = float(self._d["dt_record"][0])
        self.contact_threshold = float(self._d["contact_threshold"][0])
        self.settle_time     = float(self._d.get("settle_time",
                                                 np.array([0.0]))[0])
        self.total_mass      = float(self._d["total_mass_kg"][0])
        self.gravity_z       = float(self._d["gravity_z"][0])
        self.n_body_joints   = int(self._d["n_body_joints"][0])
        self.n_body_segments = int(self._d["n_body_segments"][0])
        self.n_legs          = int(self._d["n_legs"][0])
        self.n_leg_dof       = int(self._d["n_leg_dof"][0])

    def __getitem__(self, key):
        return self._d[key]

    @property
    def T(self):
        return len(self._d["time"])

    def mask_after(self, settle):
        """Boolean index into the time axis selecting t >= settle (seconds)."""
        t = self._d["time"]
        return t >= float(settle)


# ══════════════════════════════════════════════════════════════════════════════
# Individual plots
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_tracking_body(tr, out_dir, mask):
    t = tr["time"][mask]
    q = np.degrees(tr["body_yaw_q"][mask])
    g = np.degrees(tr["body_yaw_target"][mask])
    n = q.shape[1]

    fig, ax = plt.subplots(n, 1, figsize=(10, 1.1 * n), sharex=True)
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].plot(t, g[:, i], color="C0", lw=0.9, label="target")
        ax[i].plot(t, q[:, i], color="C1", lw=0.8, label="actual")
        err_rms = float(np.sqrt(np.mean((q[:, i] - g[:, i]) ** 2)))
        ax[i].set_ylabel(f"body {i+1}\nRMSE {err_rms:.1f}°", fontsize=7)
        ax[i].tick_params(labelsize=7)
    ax[0].legend(loc="upper right", fontsize=7)
    ax[-1].set_xlabel("time (s)")
    fig.suptitle(os.path.basename(tr.path) + "  — body yaw tracking",
                 fontsize=9)
    return _savefig(fig, out_dir, "tracking_body.png")


def plot_tracking_legs(tr, out_dir, mask):
    t = tr["time"][mask]
    # Focus on the active DOFs (hip yaw = 0, hip pitch = 1)
    q = np.degrees(tr["leg_q"][mask])       # (T, 19, 2, 4)
    g = np.degrees(tr["leg_target"][mask])  # (T, 19, 2, 4)

    dofs_to_plot = [(0, "hip yaw"), (1, "hip pitch")]
    fig, axes = plt.subplots(len(dofs_to_plot), 2, figsize=(14, 6),
                             sharex=True)
    if len(dofs_to_plot) == 1:
        axes = axes[None, :]

    sides = ["L", "R"]
    for row, (dof, label) in enumerate(dofs_to_plot):
        for col in range(2):
            a = axes[row, col]
            a.set_prop_cycle(None)
            for leg in range(q.shape[1]):
                a.plot(t, q[:, leg, col, dof], lw=0.4, alpha=0.7)
                a.plot(t, g[:, leg, col, dof], ":", lw=0.4, alpha=0.7)
            rmse = float(np.sqrt(np.mean(
                (q[:, :, col, dof] - g[:, :, col, dof]) ** 2
            )))
            a.set_title(f"{label} — side {sides[col]}   "
                        f"RMSE {rmse:.1f}° (all legs)", fontsize=9)
            a.set_ylabel("angle (deg)")
            a.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(os.path.basename(tr.path) + "  — leg joint tracking "
                 "(solid=actual, dotted=target)", fontsize=10)
    return _savefig(fig, out_dir, "tracking_legs.png")


def plot_pitch_roll_joints(tr, out_dir, mask):
    t = tr["time"][mask]
    pitch = np.degrees(tr["pitch_q"][mask])  # (T, n_pitch)
    roll  = np.degrees(tr["roll_q"][mask])   # (T, n_roll)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    if pitch.size:
        for i in range(pitch.shape[1]):
            a1.plot(t, pitch[:, i], lw=0.6, alpha=0.7,
                    label=f"j{i}" if i < 5 else None)
        a1.axhline(0, color="k", lw=0.5, alpha=0.3)
        a1.set_ylabel("pitch joint (deg)")
        a1.set_title("Body pitch joint angles (all segments)")
    else:
        a1.set_title("Body pitch joint angles — no pitch joints found")
    a1.grid(alpha=0.3)

    if roll.size:
        for i in range(roll.shape[1]):
            a2.plot(t, roll[:, i], lw=0.6, alpha=0.7)
        a2.axhline(0, color="k", lw=0.5, alpha=0.3)
    a2.set_ylabel("roll joint (deg)")
    a2.set_xlabel("time (s)")
    a2.set_title("Body roll joint angles (all segments)")
    a2.grid(alpha=0.3)

    fig.suptitle(os.path.basename(tr.path) + "  — pitch/roll joint angles",
                 fontsize=10)
    return _savefig(fig, out_dir, "tracking_pitch_roll.png")


def plot_torque_saturation(tr, out_dir, mask):
    """Commanded vs actuator_force for each joint family.  In MuJoCo
    `<general>` actuators without a `forcerange`, they're equal — but if
    `forcerange`/`forcelimited` is set the cmd can exceed the applied force
    and this plot makes the saturation visible."""
    t = tr["time"][mask]

    groups = [
        ("body yaw", tr["body_yaw_cmd_torque"][mask],
                     tr["body_yaw_act_torque"][mask]),
        ("pitch",    tr["pitch_cmd_torque"][mask],
                     tr["pitch_act_torque"][mask]),
        ("roll",     tr["roll_cmd_torque"][mask],
                     tr["roll_act_torque"][mask]),
        ("leg", tr["leg_cmd_torque"][mask].reshape(len(t), -1),
                tr["leg_act_torque"][mask].reshape(len(t), -1)),
    ]

    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 10), sharex=True)
    for ax, (label, cmd, act) in zip(axes, groups):
        if cmd.size == 0:
            ax.set_title(f"{label} — empty")
            continue
        cmd_mag = np.abs(cmd).max(axis=1)
        act_mag = np.abs(act).max(axis=1)
        ax.plot(t, cmd_mag, color="C3", lw=0.8, label="|cmd| max")
        ax.plot(t, act_mag, color="C0", lw=0.8, label="|actual| max")
        ax.set_ylabel(f"{label}\ntorque (Nm)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        cmd_p99 = float(np.percentile(np.abs(cmd), 99))
        act_p99 = float(np.percentile(np.abs(act), 99))
        ax.set_title(f"{label}  p99 |cmd|={cmd_p99*1000:.2f} mNm, "
                     f"p99 |actual|={act_p99*1000:.2f} mNm", fontsize=9)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(os.path.basename(tr.path) + "  — torque envelopes",
                 fontsize=10)
    return _savefig(fig, out_dir, "torque_saturation.png")


def plot_contact_forces(tr, out_dir, mask):
    t    = tr["time"][mask]
    mag  = tr["foot_contact_mag"][mask]     # (T, 19, 2)
    T, n_legs, _ = mag.shape

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 1.5])

    # Per-leg (left side)
    a_l = fig.add_subplot(gs[0, 0])
    for leg in range(n_legs):
        a_l.plot(t, mag[:, leg, 0], lw=0.5, alpha=0.8)
    a_l.set_title("|F_contact| — left feet")
    a_l.set_ylabel("force (N)")
    a_l.grid(alpha=0.3)

    a_r = fig.add_subplot(gs[0, 1], sharey=a_l)
    for leg in range(n_legs):
        a_r.plot(t, mag[:, leg, 1], lw=0.5, alpha=0.8)
    a_r.set_title("|F_contact| — right feet")
    a_r.grid(alpha=0.3)

    # Heatmap: rows = leg index, cols = time (separately for L and R)
    a_hl = fig.add_subplot(gs[1, 0])
    im_l = a_hl.imshow(mag[:, :, 0].T, aspect="auto", origin="lower",
                       extent=[t[0], t[-1], 0, n_legs],
                       cmap="viridis")
    a_hl.set_title("|F_contact| heatmap — left (darker = no contact)")
    a_hl.set_ylabel("leg #")
    plt.colorbar(im_l, ax=a_hl, fraction=0.04)

    a_hr = fig.add_subplot(gs[1, 1], sharey=a_hl)
    im_r = a_hr.imshow(mag[:, :, 1].T, aspect="auto", origin="lower",
                       extent=[t[0], t[-1], 0, n_legs],
                       cmap="viridis")
    a_hr.set_title("|F_contact| heatmap — right")
    plt.colorbar(im_r, ax=a_hr, fraction=0.04)

    # Distribution of peak force per foot (bar chart)
    a_b = fig.add_subplot(gs[2, :])
    peak_L = mag[:, :, 0].max(axis=0)
    peak_R = mag[:, :, 1].max(axis=0)
    x = np.arange(n_legs)
    a_b.bar(x - 0.2, peak_L, 0.4, label="peak L")
    a_b.bar(x + 0.2, peak_R, 0.4, label="peak R")
    body_weight = tr.total_mass * tr.gravity_z
    a_b.axhline(body_weight, color="k", ls="--", lw=0.8,
                label=f"body weight = {body_weight*1000:.1f} mN")
    a_b.set_xlabel("leg index")
    a_b.set_ylabel("peak |F| (N)")
    a_b.set_title("Per-foot peak contact force magnitude")
    a_b.legend(fontsize=8)
    a_b.grid(alpha=0.3)
    a_b.set_xticks(x)

    fig.suptitle(os.path.basename(tr.path) + "  — contact forces",
                 fontsize=10)
    return _savefig(fig, out_dir, "contact_forces.png")


def plot_contact_imbalance(tr, out_dir, mask):
    t   = tr["time"][mask]
    mag = tr["foot_contact_mag"][mask]     # (T, 19, 2)
    in_ct = tr["foot_in_contact"][mask]    # (T, 19, 2)
    n_legs = mag.shape[1]

    F_L = mag[:, :, 0].sum(axis=1)
    F_R = mag[:, :, 1].sum(axis=1)
    front = n_legs // 2
    F_front = mag[:, :front, :].sum(axis=(1, 2))
    F_back  = mag[:, front:, :].sum(axis=(1, 2))

    n_ct_L = in_ct[:, :, 0].sum(axis=1)
    n_ct_R = in_ct[:, :, 1].sum(axis=1)
    n_ct_front = in_ct[:, :front, :].sum(axis=(1, 2))
    n_ct_back  = in_ct[:, front:, :].sum(axis=(1, 2))

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, F_L, label="ΣF left", color="C0")
    axes[0].plot(t, F_R, label="ΣF right", color="C1")
    axes[0].set_ylabel("N")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_title("Total contact force — L vs R")

    denom = (F_L + F_R + 1e-12)
    axes[1].plot(t, (F_L - F_R) / denom, color="C3")
    axes[1].axhline(0, color="k", lw=0.5, alpha=0.3)
    axes[1].set_ylabel("(L−R)/(L+R)")
    axes[1].set_title("Normalized L/R force imbalance  (0 = balanced)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, F_front, label="front half", color="C0")
    axes[2].plot(t, F_back,  label="back half",  color="C1")
    axes[2].set_ylabel("N")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    axes[2].set_title("Total contact force — front vs back half")

    axes[3].plot(t, n_ct_L, label="#contacts L", color="C0")
    axes[3].plot(t, n_ct_R, label="#contacts R", color="C1")
    axes[3].plot(t, n_ct_front, label="#contacts front", color="C2", lw=0.8)
    axes[3].plot(t, n_ct_back,  label="#contacts back",  color="C3", lw=0.8)
    axes[3].set_ylabel("count")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)
    axes[3].set_title(f"Number of feet in contact "
                      f"(threshold {tr.contact_threshold*1000:.2f} mN)")

    fig.suptitle(os.path.basename(tr.path) + "  — contact imbalance",
                 fontsize=10)
    return _savefig(fig, out_dir, "contact_imbalance.png")


def plot_body_rpy(tr, out_dir, mask):
    t    = tr["time"][mask]
    rpy  = np.degrees(quat_to_euler(tr["root_quat"][mask]))   # (T, 3)
    com  = tr["com_pos"][mask]
    head = np.degrees(tr["heading_rad"][mask])

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, rpy[:, 0], color="C0"); axes[0].set_ylabel("roll (°)")
    axes[0].set_title("Root body roll / pitch / yaw  (world-frame)")
    axes[1].plot(t, rpy[:, 1], color="C1"); axes[1].set_ylabel("pitch (°)")
    axes[2].plot(t, rpy[:, 2], color="C2"); axes[2].set_ylabel("yaw (°)")
    for a in axes[:3]:
        a.axhline(0, color="k", lw=0.5, alpha=0.3)
        a.grid(alpha=0.3)

    axes[3].plot(t, com[:, 2] * 1000, color="C3")
    axes[3].set_ylabel("COM z (mm)")
    axes[3].set_xlabel("time (s)")
    axes[3].set_title("COM height above world origin")
    axes[3].grid(alpha=0.3)

    fig.suptitle(os.path.basename(tr.path) + "  — body RPY + COM z",
                 fontsize=10)
    return _savefig(fig, out_dir, "body_rpy.png")


def plot_gait_diagram(tr, out_dir, mask):
    t = tr["time"][mask]
    in_ct = tr["foot_in_contact"][mask]      # (T, 19, 2)
    n_legs = in_ct.shape[1]

    # Pack into a 2-row diagram: rows 0..n-1 = L, n..2n-1 = R
    img = np.zeros((2 * n_legs, len(t)))
    for leg in range(n_legs):
        img[leg, :]        = in_ct[:, leg, 0].astype(float)
        img[n_legs + leg, :] = in_ct[:, leg, 1].astype(float)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img, aspect="auto", origin="lower",
              extent=[t[0], t[-1], 0, 2 * n_legs],
              cmap="Greys", vmin=0, vmax=1)
    ax.axhline(n_legs, color="r", lw=1.0)
    ax.set_yticks([n_legs / 2, n_legs + n_legs / 2])
    ax.set_yticklabels(["Left legs", "Right legs"])
    ax.set_xlabel("time (s)")
    ax.set_title(os.path.basename(tr.path) + "  — gait diagram "
                 "(black = foot in contact)")
    return _savefig(fig, out_dir, "gait_diagram.png")


def plot_power_work(tr, out_dir, mask):
    """Mechanical power = sum_j τ_j · q̇_j.  Cumulative work is the integral."""
    t  = tr["time"][mask]
    dt = np.diff(t, prepend=t[0])

    # Body yaw
    P_body = (tr["body_yaw_act_torque"][mask]
              * tr["body_yaw_qdot"][mask]).sum(axis=1)

    # Pitch
    P_pitch = (tr["pitch_act_torque"][mask]
               * tr["pitch_qdot"][mask]).sum(axis=1) \
              if tr["pitch_act_torque"][mask].size else np.zeros_like(t)

    # Roll
    P_roll = (tr["roll_act_torque"][mask]
              * tr["roll_qdot"][mask]).sum(axis=1) \
             if tr["roll_act_torque"][mask].size else np.zeros_like(t)

    # Legs
    P_leg = (tr["leg_act_torque"][mask]
             * tr["leg_qdot"][mask]).sum(axis=(1, 2, 3))

    P_total = P_body + P_pitch + P_roll + P_leg
    E_cum   = np.cumsum(np.abs(P_total) * dt)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    a1.plot(t, P_body,  label="body yaw", lw=0.7)
    a1.plot(t, P_pitch, label="pitch",    lw=0.7)
    a1.plot(t, P_roll,  label="roll",     lw=0.7)
    a1.plot(t, P_leg,   label="legs",     lw=0.7)
    a1.plot(t, P_total, label="total",    color="k", lw=0.8)
    a1.set_ylabel("P (W)")
    a1.set_title("Mechanical power (signed)")
    a1.legend(fontsize=8, ncol=5); a1.grid(alpha=0.3)

    a2.plot(t, E_cum * 1000, color="C3")
    a2.set_ylabel("|E|_cum  (mJ)")
    a2.set_xlabel("time (s)")
    a2.set_title("Cumulative absolute mechanical work")
    a2.grid(alpha=0.3)

    fig.suptitle(os.path.basename(tr.path) + "  — power / work",
                 fontsize=10)
    return _savefig(fig, out_dir, "cot_and_power.png")


# ══════════════════════════════════════════════════════════════════════════════
# Scalar summary metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_summary(tr, settle):
    mask = tr.mask_after(settle)
    t = tr["time"][mask]
    if len(t) < 2:
        return {"error": "not enough post-settle samples"}

    dt = np.diff(t, prepend=t[0])

    # ── Tracking RMSE (deg) ─────────────────────────────────────
    def rmse_deg(q, g):
        return float(np.degrees(np.sqrt(np.mean((q - g) ** 2))))

    rmse_body  = rmse_deg(tr["body_yaw_q"][mask], tr["body_yaw_target"][mask])
    rmse_leg   = rmse_deg(tr["leg_q"][mask],      tr["leg_target"][mask])
    rmse_pitch = (rmse_deg(tr["pitch_q"][mask], tr["pitch_target"][mask])
                  if tr["pitch_q"][mask].size else 0.0)

    # ── Peak joint excursion (deg) ──────────────────────────────
    max_body  = float(np.degrees(np.abs(tr["body_yaw_q"][mask]).max()))
    max_pitch = float(np.degrees(np.abs(tr["pitch_q"][mask]).max())
                      if tr["pitch_q"][mask].size else 0.0)
    max_roll  = float(np.degrees(np.abs(tr["roll_q"][mask]).max())
                      if tr["roll_q"][mask].size else 0.0)

    # ── Torque envelopes (Nm → mNm for readability) ─────────────
    def p_stats(x):
        if x.size == 0:
            return {"rms": 0.0, "p99": 0.0, "max": 0.0}
        a = np.abs(x)
        return {"rms": float(np.sqrt(np.mean(x ** 2)) * 1000),
                "p99": float(np.percentile(a, 99) * 1000),
                "max": float(a.max() * 1000)}

    tau_body = p_stats(tr["body_yaw_act_torque"][mask])
    tau_leg  = p_stats(tr["leg_act_torque"][mask])
    tau_pitch = p_stats(tr["pitch_act_torque"][mask])
    tau_roll  = p_stats(tr["roll_act_torque"][mask])

    # ── Contact forces ──────────────────────────────────────────
    mag = tr["foot_contact_mag"][mask]           # (T, 19, 2)
    in_ct = tr["foot_in_contact"][mask]
    body_weight = tr.total_mass * tr.gravity_z

    peak_force = float(mag.max())
    peak_over_weight = peak_force / max(body_weight, 1e-12)
    mean_feet_in_contact = float(in_ct.sum(axis=(1, 2)).mean())
    duty_factor = float(in_ct.mean())            # per-foot avg contact time

    # L/R and front/back imbalance (RMS of normalized difference)
    F_L = mag[:, :, 0].sum(axis=1); F_R = mag[:, :, 1].sum(axis=1)
    lr_norm = (F_L - F_R) / (F_L + F_R + 1e-12)
    n_legs = mag.shape[1]; split = n_legs // 2
    F_front = mag[:, :split, :].sum(axis=(1, 2))
    F_back  = mag[:, split:, :].sum(axis=(1, 2))
    fb_norm = (F_front - F_back) / (F_front + F_back + 1e-12)

    # ── CoT ──────────────────────────────────────────────────────
    P_body = (tr["body_yaw_act_torque"][mask]
              * tr["body_yaw_qdot"][mask]).sum(axis=1)
    P_pitch = ((tr["pitch_act_torque"][mask]
                * tr["pitch_qdot"][mask]).sum(axis=1)
               if tr["pitch_act_torque"][mask].size else np.zeros_like(t))
    P_roll = ((tr["roll_act_torque"][mask]
               * tr["roll_qdot"][mask]).sum(axis=1)
              if tr["roll_act_torque"][mask].size else np.zeros_like(t))
    P_leg = (tr["leg_act_torque"][mask]
             * tr["leg_qdot"][mask]).sum(axis=(1, 2, 3))
    E_total = float(np.sum(np.abs(P_body + P_pitch + P_roll + P_leg) * dt))

    com = tr["com_pos"][mask]
    dist = float(np.sqrt((com[-1, 0] - com[0, 0]) ** 2
                         + (com[-1, 1] - com[0, 1]) ** 2))
    duration = float(t[-1] - t[0])
    speed = dist / max(duration, 1e-12)
    cot = (E_total / (body_weight * max(dist, 1e-9)))

    # ── Body attitude ──────────────────────────────────────────
    rpy_deg = np.degrees(quat_to_euler(tr["root_quat"][mask]))
    mean_pitch = float(np.mean(np.abs(rpy_deg[:, 1])))
    max_pitch_root = float(np.max(np.abs(rpy_deg[:, 1])))
    mean_roll_root = float(np.mean(np.abs(rpy_deg[:, 0])))
    max_roll_root  = float(np.max(np.abs(rpy_deg[:, 0])))

    return {
        "file":               os.path.basename(tr.path),
        "n_frames":           int(tr.T),
        "duration_s":         duration,
        "settle_s":           float(settle),
        "distance_m":         dist,
        "forward_speed_mps":  speed,
        "total_mass_kg":      tr.total_mass,
        "body_weight_N":      float(body_weight),
        "cot":                float(cot),
        "energy_abs_J":       E_total,
        "rmse_body_yaw_deg":  rmse_body,
        "rmse_leg_deg":       rmse_leg,
        "rmse_pitch_deg":     rmse_pitch,
        "max_body_yaw_deg":   max_body,
        "max_pitch_joint_deg": max_pitch,
        "max_roll_joint_deg":  max_roll,
        "tau_body_mNm":       tau_body,
        "tau_leg_mNm":        tau_leg,
        "tau_pitch_mNm":      tau_pitch,
        "tau_roll_mNm":       tau_roll,
        "peak_foot_force_N":  peak_force,
        "peak_force_over_body_weight": float(peak_over_weight),
        "mean_feet_in_contact": mean_feet_in_contact,
        "duty_factor_mean":   duty_factor,
        "lr_imbalance_rms":   float(np.sqrt(np.mean(lr_norm ** 2))),
        "fb_imbalance_rms":   float(np.sqrt(np.mean(fb_norm ** 2))),
        "root_mean_pitch_deg": mean_pitch,
        "root_max_pitch_deg":  max_pitch_root,
        "root_mean_roll_deg":  mean_roll_root,
        "root_max_roll_deg":   max_roll_root,
    }


def _format_summary(s):
    def fmt(k, v):
        if isinstance(v, dict):
            body = ", ".join(f"{kk}={vv:.3f}" for kk, vv in v.items())
            return f"  {k:28s}: {{ {body} }}"
        if isinstance(v, float):
            return f"  {k:28s}: {v:.4f}"
        return f"  {k:28s}: {v}"
    lines = [f"=== Summary: {s.get('file','<?>')} ==="]
    for k, v in s.items():
        if k == "file":
            continue
        lines.append(fmt(k, v))
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_inputs(paths, recursive):
    npz = []
    for p in paths:
        if os.path.isdir(p):
            if recursive:
                for root, _, files in os.walk(p):
                    for f in files:
                        if f.endswith(".npz"):
                            npz.append(os.path.join(root, f))
            else:
                for f in os.listdir(p):
                    if f.endswith(".npz"):
                        npz.append(os.path.join(p, f))
        elif os.path.isfile(p) and p.endswith(".npz"):
            npz.append(p)
        else:
            print(f"[WARN] skip (not .npz or dir): {p}")
    return sorted(npz)


def _analyse_one(npz_path, out_dir, settle, plot=True):
    tr = Trial(npz_path)
    os.makedirs(out_dir, exist_ok=True)

    summary = compute_summary(tr, settle)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(_format_summary(summary) + "\n")

    if plot and _HAVE_MPL:
        mask = tr.mask_after(settle)
        if mask.sum() < 2:
            print(f"  [skip plots] {npz_path} has <2 samples after settle")
        else:
            plot_tracking_body(tr, out_dir, mask)
            plot_tracking_legs(tr, out_dir, mask)
            plot_pitch_roll_joints(tr, out_dir, mask)
            plot_torque_saturation(tr, out_dir, mask)
            plot_contact_forces(tr, out_dir, mask)
            plot_contact_imbalance(tr, out_dir, mask)
            plot_body_rpy(tr, out_dir, mask)
            plot_gait_diagram(tr, out_dir, mask)
            plot_power_work(tr, out_dir, mask)
    elif plot and not _HAVE_MPL:
        print("  [WARN] matplotlib not available — skipping plots")

    return summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help=".npz file(s) or directory(ies) containing trial_*.npz")
    ap.add_argument("--out", default=None,
                    help="Output root directory. If omitted, each trial's "
                         "analysis goes next to the .npz in <file>_analysis/.")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subdirectories when input is a dir")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="Skip the first N seconds (default 3.0s — matches "
                         "the 1s settle + 2s ramp in wave_number_sweep)")
    ap.add_argument("--no-plots", action="store_true",
                    help="Only write summary.json / summary.txt, no figures")
    args = ap.parse_args()

    npz_list = _resolve_inputs(args.paths, args.recursive)
    if not npz_list:
        print("No .npz files found.")
        return 1

    print(f"Analysing {len(npz_list)} trial(s). settle={args.settle:.1f}s")
    all_summaries = []
    for p in npz_list:
        if args.out:
            rel = os.path.splitext(os.path.basename(p))[0]
            out_dir = os.path.join(args.out, rel)
        else:
            out_dir = os.path.splitext(p)[0] + "_analysis"
        print(f"  [{os.path.relpath(p)}] → {os.path.relpath(out_dir)}")
        s = _analyse_one(p, out_dir, args.settle, plot=not args.no_plots)
        s["_npz_path"] = p
        s["_analysis_dir"] = out_dir
        all_summaries.append(s)

    # If we processed more than one, write a top-level aggregate CSV
    if len(all_summaries) > 1 and args.out:
        csv_path = os.path.join(args.out, "all_trials_summary.csv")
        os.makedirs(args.out, exist_ok=True)
        # Flatten tau_* dicts
        flat = []
        keys = set()
        for s in all_summaries:
            row = {}
            for k, v in s.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        key = f"{k}.{kk}"
                        row[key] = vv
                        keys.add(key)
                else:
                    row[k] = v
                    keys.add(k)
            flat.append(row)
        keys = sorted(keys)
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in flat:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(f"Aggregate → {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
