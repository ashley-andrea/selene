#!/usr/bin/env python3
"""
evaluate_simulator.py
=====================
Tests the trained Simulator Model on the held-out test split and produces:

  1. Per-symptom performance metrics
  2. Satisfaction score regression metrics
  3. Treatment trajectory visualisations for a curated set of showcase patients

Performance metrics
-------------------
For each binary target:
  • AUROC          — discrimination ability (threshold-independent)
  • Average Precision (AP) — area under PR curve, better for imbalanced classes
  • Brier Score    — calibration of predicted probabilities (lower = better)

For satisfaction_score (regression):
  • RMSE, MAE, R²

Note on serious events (evt_dvt / evt_pe / evt_stroke):
  These have near-zero prevalence in the test set. Their AUROC will reflect
  near-random discrimination — this is expected and correct. The primary
  safety gate for serious events is the WHO MEC profile blocking layer.

Visualisations
--------------
For each showcase patient we generate one figure comparing N_PILLS_SHOWN pills.
Each figure has 5 row panels (symptom groups) x N_PILLS_SHOWN columns:
  • Tolerability  : nausea, headache, breast tenderness, spotting
  • Mood / MH     : mood worsened, depression episode, anxiety, libido decreased
  • Positive FX   : acne improved, cramps relieved, PCOS improvement
  • Cosmetic      : weight gain, acne worsened, hair loss
  • Summary       : still_taking probability + satisfaction score

Predicted probability = solid line.
Actual binary realisation (from diaries) = jittered scatter.

Duration parameter
------------------
  --n-months N   : number of months to show in trajectory plots (default 12)
  Any value up to 12 is valid (training data goes to month 12).

Usage
-----
    python evaluate_simulator.py
    python evaluate_simulator.py --n-months 6 --n-showcase 6

Outputs
-------
    models/simulation/artifacts/eval_metrics.json
    models/simulation/artifacts/plots/01_metrics_heatmap.png
    models/simulation/artifacts/plots/02_satisfaction_metrics.png
    models/simulation/artifacts/plots/03_patient_XX_trajectory.png   (per patient)
"""

import os
import json
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    mean_squared_error, mean_absolute_error, r2_score,
)

# Import helpers from train_simulator
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_simulator import (
    load_models, build_pill_features_lookup, predict_trajectory,
    ALL_FEATURES, PATIENT_FEATURES, PILL_FEATURES, BINARY_TARGETS,
    PATIENTS_CSV, DIARIES_CSV, PILLS_CSV, ARTIFACTS_DIR,
    encode_pills,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
TEST_IDS_TXT  = os.path.join(PROJECT_ROOT, "data/splits/test_ids.txt")
PLOTS_DIR     = os.path.join(ARTIFACTS_DIR, "plots")

# ──────────────────────────────────────────────────────────────────────────────
#  Symptom groups for visualisation
# ──────────────────────────────────────────────────────────────────────────────
SYMPTOM_GROUPS = {
    "Tolerability":    ["sym_nausea", "sym_headache", "sym_breast_tenderness", "sym_spotting"],
    "Mood / MH":       ["sym_mood_worsened", "sym_depression_episode", "sym_anxiety",
                        "sym_libido_decreased"],
    "Positive FX":     ["sym_acne_improved", "sym_cramps_relieved", "sym_pcos_improvement"],
    "Cosmetic":        ["sym_weight_gain", "sym_acne_worsened", "sym_hair_loss"],
}

# Readable short labels
LABEL_MAP = {
    "sym_nausea":             "Nausea",
    "sym_headache":           "Headache",
    "sym_breast_tenderness":  "Breast tend.",
    "sym_spotting":           "Spotting",
    "sym_mood_worsened":      "Mood ↓",
    "sym_depression_episode": "Depression ep.",
    "sym_anxiety":            "Anxiety",
    "sym_libido_decreased":   "Libido ↓",
    "sym_weight_gain":        "Weight gain",
    "sym_acne_improved":      "Acne improved",
    "sym_acne_worsened":      "Acne worsened",
    "sym_hair_loss":          "Hair loss",
    "sym_cramps_relieved":    "Cramps ↓",
    "sym_pcos_improvement":   "PCOS improv.",
    "evt_dvt":                "DVT",
    "evt_pe":                 "PE",
    "evt_stroke":             "Stroke",
    "still_taking":           "Still taking",
    "satisfaction_score":     "Satisfaction",
}

PILL_COLORS = {
    "EE20_LNG90":       "#4e79a7",
    "EE30_LNG150":      "#f28e2b",
    "EE35_NET500_1000": "#e15759",
    "EE20_NET1000":     "#76b7b2",
    "EE25_35_NGM":      "#59a14f",
    "EE30_DSG150":      "#edc948",
    "EE30_DRSP3":       "#b07aa1",
    "EE20_DRSP3":       "#ff9da7",
    "NET_PO_350":       "#9c755f",
}


# ──────────────────────────────────────────────────────────────────────────────
#  Data loader for test split
# ──────────────────────────────────────────────────────────────────────────────
def load_test_data(meta: dict) -> tuple:
    print(f"  Loading test data …")
    patients = pd.read_csv(PATIENTS_CSV)
    test_ids = set(open(TEST_IDS_TXT).read().split())
    patients = patients[patients["patient_id"].astype(str).isin(test_ids)].copy()
    patients["patient_id"] = patients["patient_id"].astype(str)
    print(f"  Test patients : {len(patients):,}")

    t0 = time.time()
    diaries = pd.read_csv(DIARIES_CSV, dtype={"patient_id": str})
    diaries = diaries[diaries["patient_id"].isin(test_ids)].copy()
    print(f"  Test diary rows : {len(diaries):,}  ({time.time()-t0:.1f}s)")

    pills_df   = pd.read_csv(PILLS_CSV)
    pill_feats = encode_pills(pills_df)

    feat_names = meta["feature_names"]
    patient_cols = ["patient_id"] + [f for f in PATIENT_FEATURES if f in patients.columns]
    merged = diaries.merge(patients[patient_cols], on="patient_id", how="left")
    for feat in PILL_FEATURES:
        merged[feat] = merged["combo_id"].map(pill_feats[feat])

    avail = [f for f in feat_names if f in merged.columns]
    X_test = merged[avail].astype(float)

    binary_cols = [t for t in meta["binary_targets"] if t in merged.columns]
    Y_bin  = merged[binary_cols].astype(float)
    Y_cont = merged[["satisfaction_score"]].astype(float) \
             if "satisfaction_score" in merged.columns else None

    return X_test, Y_bin, Y_cont, merged, patients, pill_feats


# ──────────────────────────────────────────────────────────────────────────────
#  Main evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(n_months: int = 12, n_showcase: int = 6, seed: int = 42):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"\n{'─'*62}")
    print("  LOAD models")
    print(f"{'─'*62}")
    symptoms_model, satisfaction_model, meta = load_models()
    pills_lookup = build_pill_features_lookup()

    print(f"\n{'─'*62}")
    print("  LOAD test data")
    print(f"{'─'*62}")
    X_test, Y_bin, Y_cont, merged_df, patients_df, pill_feats = load_test_data(meta)

    # ══════════════════════════════════════════════════════════════════════
    #  Predictions on test set
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("  Computing predictions on test set …")
    print(f"{'─'*62}")
    t0 = time.time()
    proba_list = symptoms_model.predict_proba(X_test.values)
    Y_pred_bin = np.column_stack([p[:, 1] for p in proba_list])
    Y_pred_sat = satisfaction_model.predict(X_test.values)
    print(f"  Done in {time.time()-t0:.1f}s")

    binary_cols = list(Y_bin.columns)

    # ══════════════════════════════════════════════════════════════════════
    #  A) Per-symptom binary metrics
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("  A) Per-symptom metrics")
    print(f"{'─'*62}")
    print(f"  {'Target':<30}  {'AUROC':>6}  {'AP':>6}  {'Brier':>6}  {'n_pos':>7}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}")

    metrics = {}
    for i, col in enumerate(binary_cols):
        y_true = Y_bin[col].values
        y_pred = Y_pred_bin[:, i]
        n_pos  = int(y_true.sum())

        if n_pos == 0:
            auc = ap = brier = float("nan")
        elif n_pos == len(y_true):
            auc = ap = brier = float("nan")
        else:
            try:
                auc   = roc_auc_score(y_true, y_pred)
                ap    = average_precision_score(y_true, y_pred)
                brier = brier_score_loss(y_true, y_pred)
            except Exception:
                auc = ap = brier = float("nan")

        metrics[col] = {"auroc": round(float(auc), 4),
                        "ap": round(float(ap), 4),
                        "brier": round(float(brier), 4),
                        "n_pos": n_pos}

        flag = ("✓" if not np.isnan(auc) and auc >= 0.70
                else ("~" if not np.isnan(auc) and auc >= 0.55
                      else "?"))
        auc_s   = f"{auc:.3f}"  if not np.isnan(auc)   else "  n/a"
        ap_s    = f"{ap:.3f}"   if not np.isnan(ap)     else "  n/a"
        brier_s = f"{brier:.4f}" if not np.isnan(brier) else "   n/a"
        print(f"  {flag} {col:<30}  {auc_s:>6}  {ap_s:>6}  {brier_s:>6}  {n_pos:>7,}")

    # ══════════════════════════════════════════════════════════════════════
    #  B) Satisfaction regression metrics
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("  B) Satisfaction score regression metrics")
    print(f"{'─'*62}")
    sat_mask = Y_cont["satisfaction_score"].notna()
    y_sat_true = Y_cont["satisfaction_score"].values[sat_mask]
    y_sat_pred = Y_pred_sat[sat_mask]

    rmse = float(np.sqrt(mean_squared_error(y_sat_true, y_sat_pred)))
    mae  = float(mean_absolute_error(y_sat_true, y_sat_pred))
    r2   = float(r2_score(y_sat_true, y_sat_pred))

    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")
    metrics["satisfaction_score"] = {"rmse": round(rmse,4), "mae": round(mae,4), "r2": round(r2,4)}

    # ══════════════════════════════════════════════════════════════════════
    #  Persist metrics
    # ══════════════════════════════════════════════════════════════════════
    eval_out = {"n_months": n_months, "binary_metrics": metrics}
    with open(os.path.join(ARTIFACTS_DIR, "eval_metrics.json"), "w") as f:
        json.dump(eval_out, f, indent=2)
    print(f"\n  eval_metrics.json saved")

    # ══════════════════════════════════════════════════════════════════════
    #  PLOTS
    # ══════════════════════════════════════════════════════════════════════
    _plot_metrics_heatmap(metrics, binary_cols)
    _plot_satisfaction(y_sat_true, y_sat_pred, rmse, r2)
    _plot_showcase_patients(
        patients_df, merged_df, symptoms_model, satisfaction_model,
        meta, pills_lookup, pill_feats, n_months, n_showcase, rng
    )

    print(f"\n{'═'*62}")
    print(f"  EVALUATION COMPLETE")
    valid_aucs = [v["auroc"] for v in metrics.values()
                  if "auroc" in v and not np.isnan(v["auroc"])]
    if valid_aucs:
        print(f"  Mean AUROC (excluding n/a): {np.mean(valid_aucs):.3f}")
    print(f"  Satisfaction R²: {r2:.3f}   RMSE: {rmse:.3f}")
    print(f"  All plots → {PLOTS_DIR}")
    print(f"{'═'*62}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Plot helpers
# ──────────────────────────────────────────────────────────────────────────────
def _plot_metrics_heatmap(metrics: dict, binary_cols: list):
    """Heatmap of AUROC and AP per symptom (rows) × metric (columns)."""
    show_cols = [c for c in binary_cols if c in metrics and not np.isnan(metrics[c]["auroc"])]
    if not show_cols:
        print("  [WARN] No valid AUROC — skipping heatmap")
        return

    data = np.array([[metrics[c]["auroc"], metrics[c]["ap"], 1 - metrics[c]["brier"]]
                     for c in show_cols])
    row_labels = [LABEL_MAP.get(c, c) for c in show_cols]
    col_labels = ["AUROC", "Avg Precision", "1−Brier\n(calibration)"]

    fig, ax = plt.subplots(figsize=(7, max(4, len(show_cols) * 0.45 + 1)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    plt.colorbar(im, ax=ax, label="Score  (green = better)")
    for i in range(len(show_cols)):
        for j in range(3):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if 0.6 < v < 0.95 else "white",
                    fontweight="bold")
    ax.set_title("Simulator — Per-Symptom Performance (test set)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "01_metrics_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 01_metrics_heatmap.png")


def _plot_satisfaction(y_true, y_pred, rmse, r2):
    """Scatter + residuals for satisfaction score."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Scatter
    axes[0].scatter(y_true, y_pred, alpha=0.08, s=4, color="#4e79a7")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect")
    axes[0].set_xlabel("Actual satisfaction score", fontsize=10)
    axes[0].set_ylabel("Predicted", fontsize=10)
    axes[0].set_title(f"Satisfaction: Predicted vs Actual\nR²={r2:.3f}  RMSE={rmse:.3f}",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)

    # Residuals
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=50, color="#f28e2b", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Residual (pred − actual)", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title(f"Residual Distribution\nmean={residuals.mean():.3f}  "
                      f"std={residuals.std():.3f}", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "02_satisfaction_metrics.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 02_satisfaction_metrics.png")


def _select_showcase_patients(patients_df, rng, n_showcase):
    """
    Pick n_showcase patients covering diverse risk profiles:
    - 2 baseline (no high-risk conditions)
    - 1-2 PCOS (acne improvement signal)
    - 1-2 depression (mood signal)
    - 1 thrombophilia / migraine+aura (risk differentiation signal)
    """
    def sample(mask_series, n, avoid):
        cands = patients_df[mask_series & ~patients_df["patient_id"].isin(avoid)]
        n_take = min(n, len(cands))
        if n_take == 0:
            return []
        return cands.sample(n_take, random_state=int(rng.integers(1e6)))["patient_id"].tolist()

    baseline_mask = (
        patients_df.get("cond_pcos",           pd.Series(0, index=patients_df.index)).fillna(0) == 0
    ) & (
        patients_df.get("cond_depression",     pd.Series(0, index=patients_df.index)).fillna(0) == 0
    ) & (
        patients_df.get("cond_thrombophilia",  pd.Series(0, index=patients_df.index)).fillna(0) == 0
    ) & (
        patients_df.get("cond_migraine_with_aura", pd.Series(0, index=patients_df.index)).fillna(0) == 0
    )

    selected = []
    selected += sample(baseline_mask, 2, set())
    selected += sample(patients_df.get("cond_pcos", pd.Series(0, index=patients_df.index)).fillna(0) == 1, 1, set(selected))
    selected += sample(patients_df.get("cond_depression", pd.Series(0, index=patients_df.index)).fillna(0) == 1, 1, set(selected))
    extra_cond = (
        patients_df.get("cond_thrombophilia",  pd.Series(0, index=patients_df.index)).fillna(0) +
        patients_df.get("cond_migraine_with_aura", pd.Series(0, index=patients_df.index)).fillna(0)
    ) >= 1
    selected += sample(extra_cond, 1, set(selected))
    # fill remainder
    remaining = patients_df[~patients_df["patient_id"].isin(selected)]
    need = n_showcase - len(selected)
    if need > 0 and len(remaining) > 0:
        selected += remaining.sample(min(need, len(remaining)),
                                     random_state=int(rng.integers(1e6)))["patient_id"].tolist()
    return selected[:n_showcase]


def _choose_showcase_pills(patient_row: dict) -> list:
    """
    Pick 3 representative pills to compare per patient:
    - DRSP (4th gen, anti-androgenic, high VTE)
    - LNG  (2nd gen, moderate androgenic, moderate VTE)
    - NET_PO_350 (progestin-only, always available)
    """
    return ["EE30_DRSP3", "EE30_LNG150", "NET_PO_350"]


def _plot_showcase_patients(
    patients_df, merged_df, symptoms_model, satisfaction_model,
    meta, pills_lookup, pill_feats, n_months, n_showcase, rng
):
    print(f"\n  Selecting {n_showcase} showcase patients …")
    showcase_ids = _select_showcase_patients(patients_df, rng, n_showcase)
    patient_lookup = patients_df.set_index("patient_id")

    for idx, pid in enumerate(showcase_ids):
        if pid not in patient_lookup.index:
            continue
        pat_row = patient_lookup.loc[pid].to_dict()
        chosen_pills = _choose_showcase_pills(pat_row)

        # Get actual realised diaries for this patient
        actual = merged_df[merged_df["patient_id"].astype(str) == str(pid)].copy()

        # Build predicted trajectories
        trajectories = {}
        for combo in chosen_pills:
            trajectories[combo] = predict_trajectory(
                pat_row, combo, pills_lookup,
                symptoms_model, satisfaction_model, meta, n_months
            )

        _draw_patient_figure(pid, idx + 1, pat_row, trajectories, actual, n_months, chosen_pills)

    print(f"  All showcase plots saved.")


def _draw_patient_figure(pid, idx, pat_row, trajectories, actual, n_months, chosen_pills):
    """
    Draw one figure per showcase patient:
    5 rows (symptom groups + summary), N_pills columns.
    """
    n_pills   = len(chosen_pills)
    group_names = list(SYMPTOM_GROUPS.keys()) + ["Summary"]
    n_rows    = len(group_names)

    fig = plt.figure(figsize=(5.5 * n_pills, 3.8 * n_rows))
    outer = gridspec.GridSpec(n_rows, n_pills, figure=fig, hspace=0.55, wspace=0.35)

    months = list(range(1, n_months + 1))

    for col_idx, combo in enumerate(chosen_pills):
        traj   = trajectories[combo]
        sym_p  = traj["symptom_probs"]
        sat_p  = np.array(traj["satisfaction"])
        color  = PILL_COLORS.get(combo, "#888888")
        act    = actual[actual["combo_id"] == combo]

        for row_idx, group_name in enumerate(group_names):
            ax = fig.add_subplot(outer[row_idx, col_idx])

            if group_name == "Summary":
                # still_taking + satisfaction
                still_p = np.array(sym_p.get("still_taking", [0] * n_months))
                ax2     = ax.twinx()

                ax.plot(months, still_p * 100, color=color, linewidth=2,
                        label="P(still taking) %")
                ax2.plot(months, sat_p, color=color, linewidth=2,
                         linestyle="--", alpha=0.7, label="Satisfaction")

                # Actual scatter
                if len(act) > 0:
                    act_m = act[act["month"].isin(months)]
                    ax.scatter(act_m["month"],
                               act_m["still_taking"].astype(float) * 100,
                               color=color, s=15, alpha=0.3, zorder=5)
                    if "satisfaction_score" in act_m.columns:
                        ax2.scatter(act_m["month"], act_m["satisfaction_score"],
                                    color=color, marker="^", s=15, alpha=0.3, zorder=5)

                ax.set_ylim(-5, 105)
                ax2.set_ylim(0, 10)
                ax.set_ylabel("P(still taking) %", fontsize=8, color=color)
                ax2.set_ylabel("Satisfaction (1–10)", fontsize=8, color=color, alpha=0.7)
                ax.set_xlabel("Month", fontsize=8)
                ax.set_title("Summary", fontsize=9, fontweight="bold")

            else:
                group_syms = SYMPTOM_GROUPS[group_name]
                palette    = plt.cm.tab10(np.linspace(0, 0.9, len(group_syms)))

                for s_idx, sym in enumerate(group_syms):
                    if sym not in sym_p:
                        continue
                    p_line = np.array(sym_p[sym]) * 100
                    lbl    = LABEL_MAP.get(sym, sym)
                    ax.plot(months, p_line, color=palette[s_idx],
                            linewidth=2, label=lbl)

                    # Actual scatter (jittered y slightly for visibility)
                    if len(act) > 0 and sym in act.columns:
                        act_m = act[act["month"].isin(months)]
                        jitter = rng_global.uniform(-1, 1, len(act_m))
                        y_actual = act_m[sym].astype(float).values * 100 + jitter
                        ax.scatter(act_m["month"].values, y_actual,
                                   color=palette[s_idx], s=8, alpha=0.25, zorder=5)

                ax.set_ylim(-2, None)
                ax.set_ylabel("Probability (%)", fontsize=8)
                ax.set_xlabel("Month", fontsize=8)
                ax.legend(fontsize=7, loc="upper right", framealpha=0.6)
                ax.set_title(group_name, fontsize=9, fontweight="bold")

            ax.set_xticks(months)
            ax.tick_params(labelsize=7)

            # Column header (pill name) on top row only
            if row_idx == 0:
                ax.set_title(f"{combo}\n{group_name}", fontsize=9, fontweight="bold", color=color)

    # Build patient card text for the suptitle
    age   = pat_row.get("age", "?")
    bmi   = pat_row.get("obs_bmi", "?")
    smoke = "smoker" if pat_row.get("obs_smoker", 0) == 1 else "non-smoker"
    conds = [c.replace("cond_", "").replace("_", " ")
             for c in PATIENT_FEATURES if "cond_" in c and pat_row.get(c, 0) == 1]
    cond_str = ", ".join(conds[:5]) if conds else "none"
    try: age_s = f"{float(age):.0f}y"
    except: age_s = "?"
    try: bmi_s = f"BMI {float(bmi):.1f}"
    except: bmi_s = "?"

    fig.suptitle(
        f"Patient {idx} | {age_s}, {bmi_s}, {smoke} | Conditions: {cond_str}\n"
        f"ID: ...{str(pid)[-10:]}  |  Trajectory horizon: {n_months} months  |  "
        f"Lines = predicted P(%), dots = actual realised outcomes",
        fontsize=10, fontweight="bold", y=1.01
    )

    fname = f"03_patient_{idx:02d}_trajectory.png"
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ──────────────────────────────────────────────────────────────────────────────
#  Module-level RNG for jitter (shared across helper calls)
# ──────────────────────────────────────────────────────────────────────────────
rng_global = np.random.default_rng(42)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Simulator Model on the held-out test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-months",   type=int, default=12,
                        help="Simulation horizon for trajectory plots (1-12)")
    parser.add_argument("--n-showcase", type=int, default=6,
                        help="Number of patients to visualise")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    evaluate(n_months=args.n_months, n_showcase=args.n_showcase, seed=args.seed)


if __name__ == "__main__":
    main()
