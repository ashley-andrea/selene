#!/usr/bin/env python3
"""
evaluate_profiles.py
====================
Tests the trained GMM profile model on the held-out test split and:

  1. Assigns each test patient soft profile memberships  P(profile_k | patient)
  2. Derives hard blocking rules per patient via threshold θ (default 0.40)
  3. Computes performance metrics explained below
  4. Generates visualisations for a curated sample of 8 test patients

─── Performance metrics ────────────────────────────────────────────────────────
Because this is an unsupervised model there is no single numeric label to predict.
We evaluate three complementary dimensions:

A) SAFETY RECALL  (most important)
   Ground truth: `has_absolute_contraindication_combined_ocp` flag (Cat-4 only)
   Prediction  : at least one combined OCP blocked for this patient
   → False Negative = patient who SHOULD be blocked but is NOT → safety failure
   → We want Safety Recall = TP / (TP + FN) to be as close to 1.0 as possible.
   → Precision < 1 is acceptable (over-blocking is conservative, not dangerous).

B) PER-CONDITION BLOCK RATE
   For every Cat-3/4 condition individually: of the patients who carry that
   condition, what fraction had at least one relevant pill blocked?
   Tells us which conditions the clustering "sees" well vs. misses.

C) CLUSTER QUALITY  (test-set silhouette score)
   Measures how well separated the 12 profiles are on unseen data.
   A silhouette score > 0.30 on a mixed binary/continuous matrix is good;
   > 0.20 is acceptable for clinical feature spaces.

D) CONFIDENCE DISTRIBUTION
   Per-patient entropy of the probability vector P(k | patient).
   Low entropy = patient is cleanly assigned to one profile (good).
   High entropy = patient is ambiguous across profiles.
   We also report how many patients trigger ≥2 profiles above threshold.

Usage
-----
    python evaluate_profiles.py
    python evaluate_profiles.py --threshold 0.40 --n-showcase 8 --seed 42

Outputs (all in models/clustering/artifacts/)
----------------------------------------------
    eval_results.json          full metrics dict
    test_assignments.csv       one row per test patient with profile probs + blocks
    plots/                     patient showcase + aggregate visualisations
"""

import os
import json
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.metrics import silhouette_score

# ──────────────────────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
PATIENTS_CSV  = os.path.join(PROJECT_ROOT, "data/output/patients_flat.csv")
TEST_IDS_TXT  = os.path.join(PROJECT_ROOT, "data/splits/test_ids.txt")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
PLOTS_DIR     = os.path.join(ARTIFACTS_DIR, "plots")

# ──────────────────────────────────────────────────────────────────────────────
#  Conditions to audit for per-condition block-rate check
# ──────────────────────────────────────────────────────────────────────────────
AUDIT_CONDITIONS = {
    # cat4
    "cond_migraine_with_aura":     ("Cat 4", "All combined blocked"),
    "cond_stroke":                 ("Cat 4", "All combined blocked"),
    "cond_mi":                     ("Cat 4", "All combined blocked"),
    "cond_dvt":                    ("Cat 4", "All combined blocked"),
    "cond_breast_cancer":          ("Cat 4", "ALL pills blocked"),
    "cond_lupus":                  ("Cat 4", "All combined blocked"),
    "cond_thrombophilia":          ("Cat 4", "All combined blocked"),
    "cond_atrial_fibrillation":    ("Cat 4", "All combined blocked"),
    "cond_liver_disease":          ("Cat 4", "ALL pills blocked"),
    # cat3 (hard-blocked per policy)
    "cond_hypertension":           ("Cat 3", "All combined blocked"),
    "cond_gallstones":             ("Cat 3", "All combined blocked"),
    "cond_diabetes":               ("Cat 3", "All combined blocked"),
    "cond_chronic_kidney_disease": ("Cat 3", "All combined blocked"),
    "cond_epilepsy":               ("Cat 3", "All combined blocked"),
}

ALL_COMBINED = [
    "EE20_LNG90", "EE30_LNG150", "EE35_NET500_1000", "EE20_NET1000",
    "EE25_35_NGM", "EE30_DSG150", "EE30_DRSP3", "EE20_DRSP3",
]
ALL_PILLS = ALL_COMBINED + ["NET_PO_350"]


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_artifacts():
    def _pkl(name):
        with open(os.path.join(ARTIFACTS_DIR, name), "rb") as f:
            return pickle.load(f)
    gmm      = _pkl("gmm_model.pkl")
    scaler   = _pkl("scaler.pkl")
    imp_dict = _pkl("imputer.pkl")
    rules    = json.load(open(os.path.join(ARTIFACTS_DIR, "profile_rules.json")))
    return gmm, scaler, imp_dict, rules


def prepare_X(df, imp_dict, scaler, feature_order):
    """Impute + scale exactly as done at training time."""
    avail_cont = imp_dict["avail_cont"]
    avail_bin  = imp_dict["avail_bin"]
    imp_cont   = imp_dict["continuous"]
    imp_bin    = imp_dict["binary"]

    X_raw = df[feature_order].copy()

    X_cont = pd.DataFrame(
        imp_cont.transform(X_raw[avail_cont]), columns=avail_cont, index=df.index
    )
    X_bin = pd.DataFrame(
        imp_bin.transform(X_raw[avail_bin]), columns=avail_bin, index=df.index
    )
    X = pd.concat([X_cont, X_bin], axis=1)[feature_order]
    X[avail_cont] = scaler.transform(X[avail_cont])
    return X.values, X_cont, X_bin


def blocking_for_patient(probs_row, rules, threshold):
    """Union of blocked combos from all profiles where P > threshold."""
    k = rules["k"]
    blocked = set()
    reasons = {}
    triggered = []
    for kid in range(k):
        p = probs_row[kid]
        if p >= threshold:
            profile = rules["profiles"][str(kid)]
            triggered.append((kid, p))
            for combo, reason_list in profile["blocked_combos"].items():
                blocked.add(combo)
                reasons.setdefault(combo, []).extend(reason_list)
    return blocked, reasons, triggered


# ──────────────────────────────────────────────────────────────────────────────
#  Main evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(threshold=0.40, n_showcase=8, seed=42):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)

    print(f"\n{'─'*62}")
    print("  LOAD artifacts + test data")
    print(f"{'─'*62}")
    gmm, scaler, imp_dict, rules = load_artifacts()
    feature_order = rules["feature_order"]

    df_all = pd.read_csv(PATIENTS_CSV)
    test_ids = set(open(TEST_IDS_TXT).read().split())
    test_df = df_all[df_all["patient_id"].astype(str).isin(test_ids)].reset_index(drop=True)
    print(f"  Test patients: {len(test_df):,}")

    # ── Prepare feature matrix ────────────────────────────────────────────
    X_matrix, X_cont_df, X_bin_df = prepare_X(
        test_df, imp_dict, scaler, feature_order
    )

    # ── Soft assignments ──────────────────────────────────────────────────
    probs = gmm.predict_proba(X_matrix)            # (n_test, k)
    hard_labels = gmm.predict(X_matrix)

    # ── Per-patient blocking ──────────────────────────────────────────────
    records = []
    for i, pid in enumerate(test_df["patient_id"]):
        blocked, reasons, triggered = blocking_for_patient(
            probs[i], rules, threshold
        )
        record = {"patient_id": pid}
        # profile probabilities
        for kid in range(rules["k"]):
            record[f"p_profile_{kid}"] = round(float(probs[i, kid]), 4)
        record["hard_profile"]        = int(hard_labels[i])
        record["n_profiles_triggered"] = len(triggered)
        record["blocked_combos"]      = "|".join(sorted(blocked))
        record["n_blocked"]           = len(blocked)
        record["blocks_any_combined"] = int(any(c in ALL_COMBINED for c in blocked))
        record["blocks_all_combined"] = int(
            all(c in blocked for c in ALL_COMBINED)
        )
        # ground truth flags (for metric computation)
        gt = test_df.iloc[i]
        record["gt_abs_contra"] = int(
            gt.get("has_absolute_contraindication_combined_ocp", 0)
        )
        records.append(record)

    results_df = pd.DataFrame(records)
    results_df.to_csv(os.path.join(ARTIFACTS_DIR, "test_assignments.csv"), index=False)
    print(f"  test_assignments.csv saved")

    # ══════════════════════════════════════════════════════════════════════
    #  A) SAFETY RECALL
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("  A) SAFETY RECALL  (combined OCP blocking vs. ground truth)")
    print(f"{'─'*62}")

    y_true = results_df["gt_abs_contra"].values
    y_pred = results_df["blocks_any_combined"].values

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    safety_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    safety_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    safety_f1        = (2 * safety_precision * safety_recall /
                        (safety_precision + safety_recall + 1e-9))
    over_block_rate  = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"  Safety Recall    : {safety_recall:.3f}  ← must be ~1.0")
    print(f"  Safety Precision : {safety_precision:.3f}")
    print(f"  Safety F1        : {safety_f1:.3f}")
    print(f"  Over-block rate  : {over_block_rate:.3f}  "
          f"(healthy patients conservatively blocked)")

    if fn > 0:
        missed = results_df[(y_true == 1) & (y_pred == 0)]["patient_id"].tolist()
        print(f"  ⚠️  MISSED patients (FN): {missed}")

    # ══════════════════════════════════════════════════════════════════════
    #  B) PER-CONDITION BLOCK RATE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("  B) PER-CONDITION BLOCK RATE")
    print(f"{'─'*62}")

    cond_stats = []
    for cond, (mec_cat, policy) in AUDIT_CONDITIONS.items():
        if cond not in test_df.columns:
            continue
        mask = test_df[cond].fillna(0).astype(int) == 1
        n_with_cond = mask.sum()
        if n_with_cond == 0:
            continue
        sub = results_df[mask.values]

        # what counts as "correctly blocked" depends on the condition
        if cond == "cond_breast_cancer" or cond == "cond_liver_disease":
            # should block ALL pills
            blocked_correctly = (sub["n_blocked"] == len(ALL_PILLS)).sum()
        else:
            # should block all COMBINED
            blocked_correctly = sub["blocks_all_combined"].sum()

        block_rate = blocked_correctly / n_with_cond
        cond_stats.append({
            "condition":   cond,
            "mec_cat":     mec_cat,
            "n_patients":  int(n_with_cond),
            "n_blocked":   int(blocked_correctly),
            "block_rate":  round(float(block_rate), 4),
        })
        flag = "✓" if block_rate >= 0.80 else ("⚠" if block_rate >= 0.50 else "✗")
        print(f"  {flag} {cond:<35} {mec_cat}  "
              f"n={n_with_cond:3d}  blocked={block_rate:.0%}")

    # ══════════════════════════════════════════════════════════════════════
    #  C) CLUSTER QUALITY — silhouette score
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("  C) CLUSTER QUALITY  (test-set silhouette score)")
    print(f"{'─'*62}")

    # Only compute if we have variation in hard labels
    unique_labels = np.unique(hard_labels)
    if len(unique_labels) > 1:
        sil = silhouette_score(X_matrix, hard_labels, sample_size=min(1000, len(X_matrix)),
                               random_state=seed)
        print(f"  Silhouette score: {sil:.4f}  "
              f"({'good' if sil > 0.30 else 'acceptable' if sil > 0.20 else 'weak'})")
    else:
        sil = float("nan")
        print("  Only one label present in sample — silhouette not computable")

    # ══════════════════════════════════════════════════════════════════════
    #  D) CONFIDENCE / ENTROPY DISTRIBUTION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print("  D) CONFIDENCE DISTRIBUTION")
    print(f"{'─'*62}")

    # Entropy: H = -sum(p * log2(p))
    eps = 1e-12
    entropy = -(probs * np.log2(probs + eps)).sum(axis=1)
    max_entropy = np.log2(rules["k"])

    n_multitrigger = (results_df["n_profiles_triggered"] >= 2).sum()
    n_no_trigger   = (results_df["n_profiles_triggered"] == 0).sum()

    print(f"  Max possible entropy  : {max_entropy:.2f} bits")
    print(f"  Mean patient entropy  : {entropy.mean():.3f} bits")
    print(f"  Median patient entropy: {np.median(entropy):.3f} bits")
    print(f"  Mean max-P confidence : {probs.max(axis=1).mean():.3f}")
    print(f"  Patients with 0 profiles triggered  : {n_no_trigger}")
    print(f"  Patients with ≥2 profiles triggered : {n_multitrigger}")

    # ══════════════════════════════════════════════════════════════════════
    #  Persist metrics
    # ══════════════════════════════════════════════════════════════════════
    eval_results = {
        "threshold": threshold,
        "n_test":    len(test_df),
        "safety": {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall":     round(safety_recall, 4),
            "precision":  round(safety_precision, 4),
            "f1":         round(safety_f1, 4),
            "over_block_rate": round(over_block_rate, 4),
        },
        "per_condition_block_rates": cond_stats,
        "cluster_quality": {
            "silhouette_score": round(float(sil), 4) if not np.isnan(sil) else None,
        },
        "confidence": {
            "mean_entropy":   round(float(entropy.mean()), 4),
            "median_entropy": round(float(np.median(entropy)), 4),
            "mean_max_p":     round(float(probs.max(axis=1).mean()), 4),
            "n_multi_trigger": int(n_multitrigger),
            "n_no_trigger":    int(n_no_trigger),
        },
    }
    with open(os.path.join(ARTIFACTS_DIR, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n  eval_results.json saved")

    # ══════════════════════════════════════════════════════════════════════
    #  VISUALISATIONS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*62}")
    print(f"  PLOTS")
    print(f"{'─'*62}")

    _plot_confusion_matrix(y_true, y_pred)
    _plot_condition_block_rates(cond_stats)
    _plot_entropy_distribution(entropy, max_entropy)
    _plot_profile_size_distribution(rules)
    _plot_patient_showcase(test_df, results_df, probs, rules, threshold, n_showcase, rng)

    print(f"\n  All plots saved → {PLOTS_DIR}")
    print(f"\n{'═'*62}")
    print(f"  EVALUATION COMPLETE")
    print(f"  Safety Recall {safety_recall:.1%}  ·  "
          f"Silhouette {sil:.3f}  ·  "
          f"Mean confidence {probs.max(axis=1).mean():.3f}")
    print(f"{'═'*62}\n")
    return eval_results


# ──────────────────────────────────────────────────────────────────────────────
#  Plot helpers
# ──────────────────────────────────────────────────────────────────────────────
PROFILE_COLORS = [
    "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
    "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac",
    "#d37295","#fabfd2",
]

def _profile_label_short(rules, kid):
    return rules["profiles"][str(kid)]["label"].replace("Profile ", "P").split(":")[0] \
           + ":" + rules["profiles"][str(kid)]["label"].split(":")[1][:22]


# ── Plot 1: Confusion matrix ─────────────────────────────────────────────────
def _plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Not blocked\n(no abs. contra)", "Blocked\n(abs. contra triggered)"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No abs.\ncontra", "Abs.\ncontra"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Safety Gate — Blocking vs. Abs. Contraindication\n(test set)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted: combined OCP blocked", fontsize=9)
    ax.set_ylabel("Actual: has abs. contraindication", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "01_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 01_confusion_matrix.png")


# ── Plot 2: Per-condition block rates ────────────────────────────────────────
def _plot_condition_block_rates(cond_stats):
    if not cond_stats:
        return
    df = pd.DataFrame(cond_stats).sort_values("block_rate")
    colors = ["#e15759" if r < 0.80 else "#59a14f" for r in df["block_rate"]]
    short_names = [c.replace("cond_", "").replace("_", " ") for c in df["condition"]]
    labels = [f"{sn}\n(n={n}, {cat})"
              for sn, n, cat in zip(short_names, df["n_patients"], df["mec_cat"])]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(labels, df["block_rate"] * 100, color=colors, edgecolor="white",
                   height=0.65)
    ax.axvline(80, color="#333", linestyle="--", linewidth=1.2, label="80% target")
    ax.set_xlabel("% of patients with condition → correctly blocked (%)", fontsize=10)
    ax.set_title("Per-Condition Block Rate\n(test set, θ = 0.40)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 105)
    for bar, val in zip(bars, df["block_rate"] * 100):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=8.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "02_per_condition_block_rates.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 02_per_condition_block_rates.png")


# ── Plot 3: Entropy distribution ─────────────────────────────────────────────
def _plot_entropy_distribution(entropy, max_entropy):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(entropy, bins=40, color="#4e79a7", edgecolor="white", alpha=0.85)
    ax.axvline(entropy.mean(), color="#e15759", linewidth=2,
               label=f"Mean = {entropy.mean():.2f} bits")
    ax.axvline(max_entropy, color="#59a14f", linewidth=1.5, linestyle="--",
               label=f"Max (uniform) = {max_entropy:.2f} bits")
    ax.set_xlabel("Profile assignment entropy (bits)", fontsize=10)
    ax.set_ylabel("Number of patients", fontsize=10)
    ax.set_title("Profile Assignment Confidence (Test Set)\n"
                 "Low entropy → patient fits cleanly into one profile",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "03_entropy_distribution.png"), dpi=150)
    plt.close(fig)
    print("  ✓ 03_entropy_distribution.png")


# ── Plot 4: Profile size distribution ───────────────────────────────────────
def _plot_profile_size_distribution(rules):
    k = rules["k"]
    labels = [rules["profiles"][str(i)]["label"].split(": ", 1)[1][:28]
              for i in range(k)]
    sizes  = [rules["profiles"][str(i)]["n_patients_train"] for i in range(k)]
    blocked_counts = [len(rules["profiles"][str(i)]["all_blocked_ids"]) for i in range(k)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: bar chart sizes
    colors = [PROFILE_COLORS[i % len(PROFILE_COLORS)] for i in range(k)]
    axes[0].barh(labels[::-1], sizes[::-1], color=colors[::-1], edgecolor="white")
    axes[0].set_xlabel("Patients in training set", fontsize=10)
    axes[0].set_title("Profile Population Size (Training)", fontsize=11, fontweight="bold")
    for i, (v, lbl) in enumerate(zip(sizes[::-1], labels[::-1])):
        axes[0].text(v + 5, i, str(v), va="center", fontsize=8)

    # right: blocked combo count per profile
    block_colors = ["#e15759" if b > 0 else "#59a14f" for b in blocked_counts[::-1]]
    axes[1].barh(labels[::-1], blocked_counts[::-1], color=block_colors, edgecolor="white")
    axes[1].axvline(len(ALL_COMBINED), color="#333", linestyle="--",
                    linewidth=1.2, label="All combined pills")
    axes[1].set_xlabel("Number of pills blocked", fontsize=10)
    axes[1].set_title("Blocked Pills per Profile", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=8)
    for i, v in enumerate(blocked_counts[::-1]):
        axes[1].text(v + 0.1, i, str(v), va="center", fontsize=8.5)

    plt.suptitle("GMM Patient Risk Profiles — Overview", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "04_profile_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 04_profile_overview.png")


# ── Plot 5: Patient showcase ─────────────────────────────────────────────────
def _plot_patient_showcase(test_df, results_df, probs, rules, threshold, n_showcase, rng):
    """
    Select n_showcase patients that cover diverse profile types:
    - 2 baseline (no blocks)
    - 2 hard-blocked (abs. contraindication)
    - 2 partially blocked (Cat 3)
    - 2 multi-trigger (≥2 profiles above threshold)
    then draw one figure per patient.
    """
    k = rules["k"]
    profile_labels = [rules["profiles"][str(i)]["label"].split(": ", 1)[1]
                      for i in range(k)]

    def pick(mask_series, n, avoid=set()):
        cands = results_df[mask_series & ~results_df["patient_id"].isin(avoid)]["patient_id"]
        if len(cands) == 0:
            return []
        chosen = cands.sample(min(n, len(cands)), random_state=int(rng.integers(1e6)))
        return chosen.tolist()

    selected_ids = []
    # 2 baseline
    selected_ids += pick(results_df["n_blocked"] == 0, 2)
    # 2 fully blocked (all combined)
    selected_ids += pick(results_df["blocks_all_combined"] == 1, 2, set(selected_ids))
    # 2 partially blocked (some but not all combined)
    part_mask = (results_df["n_blocked"] > 0) & (results_df["blocks_all_combined"] == 0)
    selected_ids += pick(part_mask, 2, set(selected_ids))
    # 2 multi-trigger
    selected_ids += pick(results_df["n_profiles_triggered"] >= 2, 2, set(selected_ids))

    # deduplicate preserving order
    seen = set()
    unique_ids = [x for x in selected_ids if not (x in seen or seen.add(x))]
    unique_ids = unique_ids[:n_showcase]

    # Key binary features to show in radar (reduce to 10 for readability)
    RADAR_FEATS = [
        ("obs_smoker",              "Smoker"),
        ("cond_migraine_with_aura", "Migraine+Aura"),
        ("cond_thrombophilia",      "Thrombophilia"),
        ("cond_dvt",                "DVT History"),
        ("cond_hypertension",       "Hypertension"),
        ("cond_diabetes",           "Diabetes"),
        ("cond_pcos",               "PCOS"),
        ("cond_endometriosis",      "Endometriosis"),
        ("cond_depression",         "Depression"),
        ("cond_epilepsy",           "Epilepsy"),
    ]
    radar_cols   = [f for f, _ in RADAR_FEATS if f in test_df.columns]
    radar_labels = [lbl for f, lbl in RADAR_FEATS if f in test_df.columns]

    # Build combined patient lookup
    patient_lookup = test_df.set_index("patient_id")
    results_lookup = results_df.set_index("patient_id")

    for idx, pid in enumerate(unique_ids):
        pat   = patient_lookup.loc[pid]
        res   = results_lookup.loc[pid]
        p_vec = probs[test_df[test_df["patient_id"] == pid].index[0]]

        blocked_list = [c for c in res["blocked_combos"].split("|") if c] \
                       if isinstance(res["blocked_combos"], str) else []
        available    = [c for c in ALL_PILLS if c not in blocked_list]

        # ── Layout: left=profile bar, middle=feature presence, right=text card
        fig = plt.figure(figsize=(16, 5.5))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
        ax_bar  = fig.add_subplot(gs[0])
        ax_feat = fig.add_subplot(gs[1])
        ax_card = fig.add_subplot(gs[2])

        # ── Left: profile probability bars ───────────────────────────────
        sorted_idx = np.argsort(p_vec)[::-1]
        top_k      = min(8, k)
        top_ids    = sorted_idx[:top_k]
        bar_colors = [
            ("#e15759" if p_vec[i] >= threshold else
             "#f28e2b" if p_vec[i] >= 0.15 else "#bab0ac")
            for i in top_ids
        ]
        bar_labels = [f"P{i}: {profile_labels[i][:22]}" for i in top_ids]
        ax_bar.barh(bar_labels[::-1], [p_vec[i] * 100 for i in top_ids[::-1]],
                    color=bar_colors[::-1], edgecolor="white", height=0.6)
        ax_bar.axvline(threshold * 100, color="#333", linestyle="--",
                       linewidth=1.3, label=f"θ = {threshold:.0%}")
        ax_bar.set_xlim(0, 103)
        ax_bar.set_xlabel("P(profile | patient) %", fontsize=9)
        ax_bar.set_title("Profile Probabilities", fontsize=10, fontweight="bold")
        ax_bar.legend(fontsize=8)
        for bar, val in zip(ax_bar.patches, [p_vec[i] * 100 for i in top_ids[::-1]]):
            ax_bar.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=7.5)

        # ── Middle: feature presence bars ────────────────────────────────
        feat_vals = []
        for col in radar_cols:
            v = pat.get(col, 0)
            try:
                feat_vals.append(float(v) if not pd.isna(v) else 0.0)
            except (ValueError, TypeError):
                feat_vals.append(0.0)
        feat_colors = ["#e15759" if v > 0 else "#d0d0d0" for v in feat_vals]
        ax_feat.barh(radar_labels[::-1], [v * 100 for v in feat_vals[::-1]],
                     color=feat_colors[::-1], edgecolor="white", height=0.6)
        ax_feat.set_xlim(-5, 120)
        ax_feat.set_xlabel("Condition present (%)", fontsize=9)
        ax_feat.set_title("Key Risk Features", fontsize=10, fontweight="bold")
        ax_feat.set_xticks([0, 100])
        ax_feat.set_xticklabels(["Absent", "Present"], fontsize=8)

        # ── Right: text card ──────────────────────────────────────────────
        ax_card.axis("off")
        age  = pat.get("age", "?")
        bmi  = pat.get("obs_bmi", "?")
        sbp  = pat.get("obs_systolic_bp", "?")
        phq  = pat.get("obs_phq9_score", "?")

        try: age = f"{float(age):.0f} y"
        except: age = "?"
        try: bmi = f"{float(bmi):.1f}"
        except: bmi = "?"
        try: sbp = f"{float(sbp):.0f} mmHg"
        except: sbp = "?"
        try: phq = f"{float(phq):.0f}/27"
        except: phq = "?"

        dom_profile_id = int(np.argmax(p_vec))
        dom_p = p_vec[dom_profile_id] * 100

        card_lines = [
            f"Patient: ...{str(pid)[-8:]}",
            "",
            f"Age     : {age}",
            f"BMI     : {bmi}",
            f"Syst.BP : {sbp}",
            f"PHQ-9   : {phq}",
            "",
            f"Top profile:",
            f"  P{dom_profile_id}: {profile_labels[dom_profile_id][:26]}",
            f"  ({dom_p:.1f}%)",
            "",
        ]
        if blocked_list:
            card_lines += [f"[BLOCKED] ({len(blocked_list)}):"]
            for c in blocked_list[:5]:
                card_lines.append(f"  {c}")
            if len(blocked_list) > 5:
                card_lines.append(f"  (+{len(blocked_list)-5} more)")
        else:
            card_lines += ["[OK] No pills blocked"]

        card_lines += ["", f"Available: {len(available)}/{len(ALL_PILLS)}"]

        card_text = "\n".join(card_lines)
        bg_color = "#fff3cd" if blocked_list else "#d4edda"
        ax_card.add_patch(FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02",
            facecolor=bg_color, edgecolor="#aaa", linewidth=1.2,
            transform=ax_card.transAxes
        ))
        ax_card.text(0.07, 0.96, card_text, va="top", ha="left",
                     fontsize=8.5, fontfamily="monospace",
                     transform=ax_card.transAxes, linespacing=1.55)
        ax_card.set_title("Patient Card", fontsize=10, fontweight="bold")

        gt_str = "[!] Has abs. contraindication" if res.get("gt_abs_contra", 0) == 1 \
                 else "[OK] No abs. contraindication"
        fig.suptitle(
            f"Patient Profile Assignment — Patient {idx + 1} / {len(unique_ids)}  "
            f"({gt_str})",
            fontsize=11, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        fname = f"05_patient_{idx+1:02d}_showcase.png"
        fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {fname}  —  {profile_labels[dom_profile_id][:40]}")


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GMM patient profiles on the held-out test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--threshold",   type=float, default=0.40,
                        help="P threshold for triggering profile blocking rules")
    parser.add_argument("--n-showcase",  type=int,   default=8,
                        help="Number of patient showcase plots to generate")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    evaluate(
        threshold=args.threshold,
        n_showcase=args.n_showcase,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
