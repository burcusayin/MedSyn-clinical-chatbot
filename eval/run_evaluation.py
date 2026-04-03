#!/usr/bin/env python3
"""
MedSyn Unified Evaluation Script
=================================
Produces all tables and figures for the MedSyn paper (Parts A-C).

Part A: Automated diagnosis evaluation (fuzzy matching metrics)
Part B: Manual evaluation alignment & manual score analysis
Part C: Inter-user concordance (ground-truth independent)

Usage:
    python run_evaluation.py \
        --session_dir eval/session_outputs \
        --manual_dir  eval/manual_eval/inputs \
        --out_dir     eval/results \
        --threshold   80 \
        --bootstrap_n 20000 \
        --seed        42

Requirements: pandas, numpy, rapidfuzz, matplotlib, scipy
"""

import argparse, os, json, re, unicodedata, warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants ──────────────────────────────────────────────────────────────────
SENIORS = ["phy1", "phy2", "phy3"]
RESIDENTS = ["res1", "res3", "res5", "res6"]
ALL_PARTICIPANTS = SENIORS + RESIDENTS
SESSIONS = {1: "baseline", 2: "interactive", 3: "baseline", 4: "interactive"}
DIFF_WEIGHTS = {"Easy": 3 / 13, "Medium": 6 / 13, "Hard": 4 / 13}
MANUAL_SCORES = {"WRONG": 0.0, "PARTIALLY CORRECT": 0.5, "COMPLETELY CORRECT": 1.0}
FIG_DPI = 400  # npj Digital Medicine: ≥300 dpi
PALETTE = {"Senior": "#2166ac", "Resident": "#b2182b"}
COND_PALETTE = {"Baseline": "#7fbf7b", "Interactive": "#af8dc3"}

# ── Text normalisation ─────────────────────────────────────────────────────────
def normalize_text(s: str) -> str:
    if pd.isna(s) or not isinstance(s, str):
        return ""
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("&", "and")
    s = re.sub(r"[\(\)\[\]\{\}]", "", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def split_diagnoses(text: str, is_gold: bool = False) -> list[str]:
    """Split a diagnosis string into individual items."""
    if pd.isna(text) or not str(text).strip():
        return []
    text = str(text).strip()
    # Handle primary/secondary markers
    text = re.split(r"(?i)\bsecondary\s*(?:diagnos[ei]s)?\s*:", text)[0]
    text = re.sub(r"(?i)\bprimary\s*(?:diagnos[ei]s)?\s*:", "", text)
    # Split by semicolons and newlines
    items = re.split(r"[;\n\r]+", text)
    result = []
    for item in items:
        n = normalize_text(item)
        if n and len(n) > 2:
            result.append(n)
    return result


# ── Fuzzy matching ─────────────────────────────────────────────────────────────
def fuzzy_match_greedy(pred_list: list[str], gold_list: list[str],
                       threshold: int = 80) -> tuple[int, list[tuple]]:
    """Greedy one-to-one fuzzy matching using RapidFuzz token_set_ratio."""
    if not pred_list or not gold_list:
        return 0, []
    pairs = []
    for pi, p in enumerate(pred_list):
        for gi, g in enumerate(gold_list):
            score = fuzz.token_set_ratio(p, g)
            if score >= threshold:
                pairs.append((pi, gi, score))
    pairs.sort(key=lambda x: x[2], reverse=True)
    used_pred, used_gold, matches = set(), set(), []
    for pi, gi, score in pairs:
        if pi not in used_pred and gi not in used_gold:
            used_pred.add(pi)
            used_gold.add(gi)
            matches.append((pi, gi, score))
    return len(matches), matches


def compute_case_metrics(pred_text: str, gold_text: str,
                         threshold: int = 80) -> dict:
    """Compute all per-case metrics for one (participant, case)."""
    pred = split_diagnoses(pred_text, is_gold=False)
    gold = split_diagnoses(gold_text, is_gold=True)
    m, _ = fuzzy_match_greedy(pred, gold, threshold)
    p_count, g_count = len(pred), len(gold)
    prec = m / p_count if p_count else 0.0
    rec = m / g_count if g_count else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    any_match = 1.0 if m > 0 else 0.0
    exact = 1.0 if (m == p_count == g_count and m > 0) else 0.0
    return dict(n_pred=p_count, n_gold=g_count, n_match=m,
                precision=prec, recall=rec, f1=f1,
                any_match=any_match, exact_match=exact)


# ── Difficulty standardisation ─────────────────────────────────────────────────
def standardize(df: pd.DataFrame, metric: str) -> float:
    """Weighted mean over difficulty levels for one participant-session."""
    result = 0.0
    for diff, w in DIFF_WEIGHTS.items():
        subset = df[df["difficulty"] == diff]
        result += w * (subset[metric].mean() if len(subset) else 0.0)
    return result


# ── Bootstrap ──────────────────────────────────────────────────────────────────
def paired_bootstrap(baseline_vals: np.ndarray, interactive_vals: np.ndarray,
                     n_boot: int = 20000, seed: int = 42) -> dict:
    """Paired bootstrap over participants: Interactive minus Baseline."""
    rng = np.random.default_rng(seed)
    n = len(baseline_vals)
    deltas = interactive_vals - baseline_vals
    observed_mean = deltas.mean()
    observed_median = np.median(deltas)
    boot_means = np.array([
        rng.choice(deltas, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    p_val = 2 * min((boot_means <= 0).mean(), (boot_means >= 0).mean())
    p_val = max(p_val, 1 / n_boot)  # floor
    return dict(
        mean_delta=observed_mean, median_delta=observed_median,
        ci_lo=ci_lo, ci_hi=ci_hi, p_value=p_val,
        sig=p_val < 0.05,
        baseline_median=np.median(baseline_vals),
        baseline_q1=np.percentile(baseline_vals, 25),
        baseline_q3=np.percentile(baseline_vals, 75),
        interactive_median=np.median(interactive_vals),
        interactive_q1=np.percentile(interactive_vals, 25),
        interactive_q3=np.percentile(interactive_vals, 75),
    )


def cohens_d(baseline: np.ndarray, interactive: np.ndarray) -> float:
    """Paired Cohen's d (Hedges' correction for small n)."""
    diff = interactive - baseline
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
    n = len(diff)
    correction = 1 - 3 / (4 * n - 5)  # Hedges' g
    return d * correction


# ── Load data ──────────────────────────────────────────────────────────────────
def load_session_data(session_dir: str) -> pd.DataFrame:
    """Load 4 session CSVs into long-format DataFrame."""
    rows = []
    for sess, cond in SESSIONS.items():
        path = Path(session_dir) / f"auto_eval_session{sess}_{cond}.csv"
        df = pd.read_csv(path)
        for _, case in df.iterrows():
            for p in ALL_PARTICIPANTS:
                ans_col = f"{p}_answer"
                time_col = f"{p}_time"
                rows.append(dict(
                    session=sess, condition=cond,
                    note_id=case["note_id"],
                    difficulty=case["Difficulty"],
                    gold=case["discharge diagnosis"],
                    participant=p,
                    expertise="Senior" if p in SENIORS else "Resident",
                    answer=case.get(ans_col, ""),
                    time_s=case.get(time_col, np.nan),
                ))
    return pd.DataFrame(rows)


def load_manual_data(manual_dir: str) -> pd.DataFrame:
    """Load 4 manual evaluation CSVs into long-format DataFrame."""
    rows = []
    for sess, cond in SESSIONS.items():
        path = Path(manual_dir) / f"manual_eval_session{sess}_{cond}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, case in df.iterrows():
            for p in ALL_PARTICIPANTS:
                corr_col = f"{p}_correctness"
                if corr_col not in df.columns:
                    continue
                label = str(case.get(corr_col, "")).strip().upper()
                rows.append(dict(
                    session=sess, condition=cond,
                    note_id=case["note_id"],
                    difficulty=case["Difficulty"],
                    participant=p,
                    expertise="Senior" if p in SENIORS else "Resident",
                    manual_label=label,
                    manual_score=MANUAL_SCORES.get(label, np.nan),
                    manual_binary=0.0 if label == "WRONG" else 1.0,
                    manual_complete=1.0 if label == "COMPLETELY CORRECT" else 0.0,
                ))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PART A: Automated Diagnosis Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_part_a(long_df: pd.DataFrame, out_dir: Path, threshold: int,
               n_boot: int, seed: int):
    """Compute automated metrics, bootstrap tests, and generate outputs."""
    print("═" * 60)
    print("PART A: Automated Diagnosis Evaluation")
    print("═" * 60)

    # ── A1: Per-case metrics ───────────────────────────────────────────────
    metrics_list = []
    for _, row in long_df.iterrows():
        m = compute_case_metrics(row["answer"], row["gold"], threshold)
        m["time_min"] = row["time_s"] / 60 if pd.notna(row["time_s"]) else np.nan
        m.update({k: row[k] for k in
                  ["session", "condition", "note_id", "difficulty",
                   "participant", "expertise"]})
        metrics_list.append(m)
    mdf = pd.DataFrame(metrics_list)
    mdf.to_csv(out_dir / "a_case_level_metrics.csv", index=False)
    print(f"  Case-level metrics: {len(mdf)} rows")

    # ── A2: Participant-level standardised scores ──────────────────────────
    endpoints = ["any_match", "exact_match", "f1", "precision", "recall", "time_min"]
    pstd_rows = []
    for (sess, p), g in mdf.groupby(["session", "participant"]):
        row = dict(session=sess, participant=p,
                   condition=SESSIONS[sess],
                   expertise="Senior" if p in SENIORS else "Resident")
        for ep in endpoints:
            row[f"{ep}_std"] = standardize(g, ep)
        pstd_rows.append(row)
    pstd = pd.DataFrame(pstd_rows)

    # Aggregate: Baseline = mean(S1,S3), Interactive = mean(S2,S4)
    agg_rows = []
    for p in ALL_PARTICIPANTS:
        for cond in ["baseline", "interactive"]:
            sub = pstd[(pstd["participant"] == p) & (pstd["condition"] == cond)]
            row = dict(participant=p, condition=cond,
                       expertise="Senior" if p in SENIORS else "Resident")
            for ep in endpoints:
                row[f"{ep}_std"] = sub[f"{ep}_std"].mean()
            agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_dir / "a_participant_aggregated.csv", index=False)

    # ── A3: Bootstrap tests ────────────────────────────────────────────────
    test_rows = []
    for ep in endpoints:
        col = f"{ep}_std"
        for group_name, participants in [
            ("All", ALL_PARTICIPANTS),
            ("Senior", SENIORS),
            ("Resident", RESIDENTS),
        ]:
            bl = agg[(agg["condition"] == "baseline") &
                      (agg["participant"].isin(participants))][col].values
            it = agg[(agg["condition"] == "interactive") &
                      (agg["participant"].isin(participants))][col].values
            bt = paired_bootstrap(bl, it, n_boot, seed)
            bt["endpoint"] = ep
            bt["group"] = group_name
            bt["n"] = len(bl)
            bt["cohens_d"] = cohens_d(bl, it)
            test_rows.append(bt)
    tests = pd.DataFrame(test_rows)
    tests.to_csv(out_dir / "a_bootstrap_tests.csv", index=False)
    print(f"  Bootstrap tests: {len(tests)} comparisons")

    # ── A4: Threshold sensitivity ──────────────────────────────────────────
    sweep_rows = []
    for thr in [60, 65, 70, 75, 80, 85, 90]:
        for _, row in long_df.iterrows():
            m = compute_case_metrics(row["answer"], row["gold"], thr)
            m.update({k: row[k] for k in
                      ["session", "condition", "participant", "difficulty", "expertise"]})
            m["threshold"] = thr
            sweep_rows.append(m)
    sweep_df = pd.DataFrame(sweep_rows)
    # Standardise and aggregate
    sweep_summary = []
    for thr in [60, 65, 70, 75, 80, 85, 90]:
        sub = sweep_df[sweep_df["threshold"] == thr]
        for ep in ["any_match", "f1", "exact_match"]:
            for cond in ["baseline", "interactive"]:
                vals = []
                for p in ALL_PARTICIPANTS:
                    ps = sub[(sub["participant"] == p) &
                             (sub["condition"] == cond)]
                    # average across two sessions of same condition
                    session_vals = []
                    for sess in [s for s, c in SESSIONS.items() if c == cond]:
                        sess_sub = ps[ps["session"] == sess]
                        session_vals.append(standardize(sess_sub, ep))
                    vals.append(np.mean(session_vals))
                sweep_summary.append(dict(
                    threshold=thr, endpoint=ep, condition=cond,
                    mean=np.mean(vals), std=np.std(vals),
                ))
    pd.DataFrame(sweep_summary).to_csv(
        out_dir / "a_threshold_sensitivity.csv", index=False)

    # ── A5: Per-case improvement (which cases benefit most?) ───────────────
    case_imp_rows = []
    for note_id in mdf["note_id"].unique():
        for p in ALL_PARTICIPANTS:
            bl_vals = mdf[(mdf["note_id"] == note_id) &
                          (mdf["participant"] == p) &
                          (mdf["condition"] == "baseline")]
            it_vals = mdf[(mdf["note_id"] == note_id) &
                          (mdf["participant"] == p) &
                          (mdf["condition"] == "interactive")]
            if len(bl_vals) and len(it_vals):
                case_imp_rows.append(dict(
                    note_id=note_id, participant=p,
                    difficulty=bl_vals.iloc[0]["difficulty"],
                    expertise="Senior" if p in SENIORS else "Resident",
                    bl_any=bl_vals.iloc[0]["any_match"],
                    it_any=it_vals.iloc[0]["any_match"],
                    delta_any=it_vals.iloc[0]["any_match"] - bl_vals.iloc[0]["any_match"],
                    bl_f1=bl_vals.iloc[0]["f1"],
                    it_f1=it_vals.iloc[0]["f1"],
                    delta_f1=it_vals.iloc[0]["f1"] - bl_vals.iloc[0]["f1"],
                ))
    # Note: cases appear in different sessions, so each participant sees
    # each case only once (either baseline or interactive, not both).
    # Per-case improvement is computed across participants.
    case_agg = mdf.groupby(["note_id", "condition", "difficulty"]).agg(
        mean_any=("any_match", "mean"),
        mean_f1=("f1", "mean"),
        n=("any_match", "count"),
    ).reset_index()
    case_agg.to_csv(out_dir / "a_case_level_by_condition.csv", index=False)

    # ── A6: Difficulty-stratified means ────────────────────────────────────
    diff_strat = mdf.groupby(["difficulty", "condition", "expertise"]).agg(
        mean_any=("any_match", "mean"),
        mean_f1=("f1", "mean"),
        mean_exact=("exact_match", "mean"),
        mean_time=("time_min", "mean"),
        n=("any_match", "count"),
    ).reset_index()
    diff_strat.to_csv(out_dir / "a_difficulty_stratified.csv", index=False)

    return mdf, agg, tests


# ══════════════════════════════════════════════════════════════════════════════
# PART B: Manual Evaluation & Alignment
# ══════════════════════════════════════════════════════════════════════════════

def run_part_b(mdf: pd.DataFrame, manual_df: pd.DataFrame, out_dir: Path,
               n_boot: int, seed: int):
    """Manual score analysis and automated-manual alignment."""
    print("\n" + "═" * 60)
    print("PART B: Manual Evaluation & Alignment")
    print("═" * 60)

    if manual_df.empty:
        print("  No manual evaluation data found. Skipping Part B.")
        return

    # Merge automated + manual
    merged = mdf.merge(
        manual_df[["session", "note_id", "participant",
                    "manual_label", "manual_score", "manual_binary", "manual_complete"]],
        on=["session", "note_id", "participant"],
        how="inner"
    )
    merged.to_csv(out_dir / "b_merged_auto_manual.csv", index=False)
    print(f"  Merged rows: {len(merged)}")

    # ── B1: Manual score bootstrap by condition × expertise ────────────────
    endpoints_manual = ["manual_score", "manual_binary", "manual_complete"]
    manual_tests = []
    for ep in endpoints_manual:
        pstd_rows = []
        for (sess, p), g in merged.groupby(["session", "participant"]):
            pstd_rows.append(dict(
                session=sess, participant=p,
                condition=SESSIONS[sess],
                expertise="Senior" if p in SENIORS else "Resident",
                value=standardize(g, ep),
            ))
        pstd = pd.DataFrame(pstd_rows)
        # Aggregate across sessions of same condition
        agg_rows = []
        for p in ALL_PARTICIPANTS:
            for cond in ["baseline", "interactive"]:
                sub = pstd[(pstd["participant"] == p) & (pstd["condition"] == cond)]
                agg_rows.append(dict(
                    participant=p, condition=cond,
                    expertise="Senior" if p in SENIORS else "Resident",
                    value=sub["value"].mean(),
                ))
        agg = pd.DataFrame(agg_rows)
        for group_name, participants in [
            ("All", ALL_PARTICIPANTS), ("Senior", SENIORS), ("Resident", RESIDENTS),
        ]:
            bl = agg[(agg["condition"] == "baseline") &
                      (agg["participant"].isin(participants))]["value"].values
            it = agg[(agg["condition"] == "interactive") &
                      (agg["participant"].isin(participants))]["value"].values
            bt = paired_bootstrap(bl, it, n_boot, seed)
            bt["endpoint"] = ep
            bt["group"] = group_name
            bt["cohens_d"] = cohens_d(bl, it)
            manual_tests.append(bt)
    pd.DataFrame(manual_tests).to_csv(
        out_dir / "b_manual_bootstrap_tests.csv", index=False)

    # ── B2: Manual scores by difficulty × expertise × condition ────────────
    #    (Comment 35: the "selling point" analysis)
    diff_exp_rows = []
    for (diff, exp, cond), g in merged.groupby(
            ["difficulty", "expertise", "condition"]):
        vals = g["manual_score"].values
        diff_exp_rows.append(dict(
            difficulty=diff, expertise=exp, condition=cond,
            mean=vals.mean(), std=vals.std(),
            median=np.median(vals),
            q1=np.percentile(vals, 25), q3=np.percentile(vals, 75),
            n=len(vals),
        ))
    diff_exp = pd.DataFrame(diff_exp_rows)
    diff_exp.to_csv(out_dir / "b_manual_by_difficulty_expertise.csv", index=False)

    # Detailed summary table with median, IQR, and deltas per difficulty × expertise
    # Uses participant-level means (averaged across sessions of same condition)
    # to match the boxplot figure methodology
    detail_rows = []
    for diff in ["Easy", "Medium", "Hard"]:
        for exp in ["Senior", "Resident"]:
            for cond, prefix in [("baseline", "bl"), ("interactive", "it")]:
                sub = merged[(merged["difficulty"] == diff) &
                             (merged["condition"] == cond) &
                             (merged["expertise"] == exp)]
                pmeans = sub.groupby("participant")["manual_score"].mean()
                if prefix == "bl":
                    bl_pmeans = pmeans
                else:
                    it_pmeans = pmeans
            if len(bl_pmeans) and len(it_pmeans):
                detail_rows.append(dict(
                    difficulty=diff, expertise=exp,
                    bl_median=bl_pmeans.median(),
                    bl_iqr=f"{bl_pmeans.quantile(0.25):.3f}-{bl_pmeans.quantile(0.75):.3f}",
                    bl_mean=bl_pmeans.mean(), bl_n=int(len(bl_pmeans)),
                    it_median=it_pmeans.median(),
                    it_iqr=f"{it_pmeans.quantile(0.25):.3f}-{it_pmeans.quantile(0.75):.3f}",
                    it_mean=it_pmeans.mean(), it_n=int(len(it_pmeans)),
                    mean_delta=it_pmeans.mean() - bl_pmeans.mean(),
                    median_delta=it_pmeans.median() - bl_pmeans.median(),
                ))
    pd.DataFrame(detail_rows).to_csv(
        out_dir / "b_manual_detail_table.csv", index=False)
    print("  b_manual_detail_table.csv")

    # Bootstrap for difficulty × expertise interaction
    diff_interaction_tests = []
    for diff in ["Easy", "Medium", "Hard"]:
        for group_name, participants in [
            ("All", ALL_PARTICIPANTS), ("Senior", SENIORS), ("Resident", RESIDENTS),
        ]:
            sub = merged[(merged["difficulty"] == diff) &
                         (merged["participant"].isin(participants))]
            bl_vals = sub[sub["condition"] == "baseline"]["manual_score"].values
            it_vals = sub[sub["condition"] == "interactive"]["manual_score"].values
            if len(bl_vals) > 1 and len(it_vals) > 1:
                # Non-paired bootstrap (different cases in different conditions)
                rng = np.random.default_rng(seed)
                boot_deltas = []
                for _ in range(n_boot):
                    bl_s = rng.choice(bl_vals, size=len(bl_vals), replace=True)
                    it_s = rng.choice(it_vals, size=len(it_vals), replace=True)
                    boot_deltas.append(it_s.mean() - bl_s.mean())
                boot_deltas = np.array(boot_deltas)
                ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
                p_val = 2 * min((boot_deltas <= 0).mean(),
                                (boot_deltas >= 0).mean())
                p_val = max(p_val, 1 / n_boot)
                diff_interaction_tests.append(dict(
                    difficulty=diff, group=group_name,
                    bl_mean=bl_vals.mean(), it_mean=it_vals.mean(),
                    mean_delta=it_vals.mean() - bl_vals.mean(),
                    ci_lo=ci_lo, ci_hi=ci_hi, p_value=p_val,
                    sig=p_val < 0.05,
                    n_bl=len(bl_vals), n_it=len(it_vals),
                ))
    pd.DataFrame(diff_interaction_tests).to_csv(
        out_dir / "b_manual_difficulty_interaction_tests.csv", index=False)
    print("  Manual difficulty × expertise interaction tests: done")

    # ── B3: Automated vs manual alignment ──────────────────────────────────
    # Binary alignment
    auto_bin = merged["any_match"].values.astype(int)
    man_bin = merged["manual_binary"].values.astype(int)
    agree_bin = (auto_bin == man_bin).mean()

    # Confusion matrix (binary)
    tp = ((auto_bin == 1) & (man_bin == 1)).sum()
    tn = ((auto_bin == 0) & (man_bin == 0)).sum()
    fp = ((auto_bin == 1) & (man_bin == 0)).sum()
    fn = ((auto_bin == 0) & (man_bin == 1)).sum()

    # Cohen's kappa
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / len(auto_bin) ** 2
    kappa_bin = (agree_bin - pe) / (1 - pe) if (1 - pe) > 0 else 0

    # 3-class alignment
    auto_3class = np.where(
        merged["exact_match"] == 1, "COMPLETELY CORRECT",
        np.where(merged["any_match"] == 1, "PARTIALLY CORRECT", "WRONG")
    )
    man_3class = merged["manual_label"].values
    agree_3 = (auto_3class == man_3class).mean()

    alignment = dict(
        binary_agreement=agree_bin, binary_kappa=kappa_bin,
        binary_tp=int(tp), binary_tn=int(tn),
        binary_fp=int(fp), binary_fn=int(fn),
        triclass_agreement=agree_3,
        n=len(merged),
    )
    pd.DataFrame([alignment]).to_csv(
        out_dir / "b_alignment_coefficients.csv", index=False)

    # Confusion matrix for 3-class
    labels_3 = ["WRONG", "PARTIALLY CORRECT", "COMPLETELY CORRECT"]
    conf_3 = pd.crosstab(
        pd.Categorical(auto_3class, categories=labels_3),
        pd.Categorical(man_3class, categories=labels_3),
        rownames=["Automated"], colnames=["Manual"]
    )
    conf_3.to_csv(out_dir / "b_confusion_matrix_3class.csv")

    # ── B4: Manual label distribution ──────────────────────────────────────
    dist = merged.groupby(["expertise", "condition", "manual_label"]).size()
    dist = dist.unstack(fill_value=0)
    dist.to_csv(out_dir / "b_manual_label_distribution.csv")

    # Per-session manual summary with CIs
    session_manual = []
    for sess in [1, 2, 3, 4]:
        sub = merged[merged["session"] == sess]
        for ep in endpoints_manual:
            vals = sub[ep].values
            rng = np.random.default_rng(seed)
            boots = [rng.choice(vals, size=len(vals), replace=True).mean()
                     for _ in range(4000)]
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            session_manual.append(dict(
                session=sess, condition=SESSIONS[sess],
                endpoint=ep, mean=vals.mean(),
                ci_lo=ci_lo, ci_hi=ci_hi,
            ))
    pd.DataFrame(session_manual).to_csv(
        out_dir / "b_manual_per_session.csv", index=False)

    # ── B5: Difficulty-standardised strict binary manual bootstrap ─────────
    # Mirrors Table 1 (Part A) but uses manual completely-correct rate
    # (COMPLETELY CORRECT = 1, PARTIALLY CORRECT + WRONG = 0).
    merged["manual_complete"] = (merged["manual_label"] == "COMPLETELY CORRECT").astype(float)

    strict_records = []
    for participant in ALL_PARTICIPANTS:
        for cond in ["baseline", "interactive"]:
            sub = merged[(merged["participant"] == participant) &
                         (merged["condition"] == cond)]
            weighted = sum(
                DIFF_WEIGHTS[d] * sub[sub["difficulty"] == d]["manual_complete"].mean()
                for d in DIFF_WEIGHTS
                if len(sub[sub["difficulty"] == d]) > 0
            )
            strict_records.append(dict(
                participant=participant, condition=cond,
                expertise="Senior" if participant in SENIORS else "Resident",
                std_complete_rate=weighted,
            ))
    strict_df = pd.DataFrame(strict_records)
    strict_df.to_csv(out_dir / "b_manual_strict_participant.csv", index=False)

    strict_tests = []
    for group_name, participants in [("All", ALL_PARTICIPANTS),
                                     ("Senior", SENIORS),
                                     ("Resident", RESIDENTS)]:
        bl = strict_df[(strict_df["condition"] == "baseline") &
                       (strict_df["participant"].isin(participants))
                       ].sort_values("participant")["std_complete_rate"].values
        it = strict_df[(strict_df["condition"] == "interactive") &
                       (strict_df["participant"].isin(participants))
                       ].sort_values("participant")["std_complete_rate"].values
        result = paired_bootstrap(bl, it, n_boot, seed)
        result["cohens_d"] = cohens_d(bl, it)
        result["endpoint"] = "complete_rate"
        result["group"] = group_name
        result["n"] = len(bl)
        strict_tests.append(result)

    pd.DataFrame(strict_tests).to_csv(
        out_dir / "b_manual_strict_bootstrap.csv", index=False)
    print("  Manual strict binary (completely-correct rate) bootstrap: done")

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# PART C: Inter-User Concordance
# ══════════════════════════════════════════════════════════════════════════════

def pairwise_f1(ans_a: str, ans_b: str, threshold: int = 80) -> float:
    """F1 between two participants' diagnosis sets."""
    a_list = split_diagnoses(ans_a, is_gold=False)
    b_list = split_diagnoses(ans_b, is_gold=False)
    if not a_list and not b_list:
        return 1.0
    if not a_list or not b_list:
        return 0.0
    m, _ = fuzzy_match_greedy(a_list, b_list, threshold)
    prec = m / len(a_list) if a_list else 0
    rec = m / len(b_list) if b_list else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def run_part_c(long_df: pd.DataFrame, out_dir: Path, threshold: int,
               n_boot: int, seed: int):
    """Inter-user concordance without ground truth."""
    print("\n" + "═" * 60)
    print("PART C: Inter-User Concordance")
    print("═" * 60)

    conc_rows = []
    for (sess, note_id), case_group in long_df.groupby(["session", "note_id"]):
        answers = {row["participant"]: row["answer"]
                   for _, row in case_group.iterrows()}
        diff = case_group.iloc[0]["difficulty"]
        cond = SESSIONS[sess]

        for p in ALL_PARTICIPANTS:
            if p not in answers:
                continue
            p_exp = "Senior" if p in SENIORS else "Resident"
            # Within-expertise
            same_exp = [q for q in (SENIORS if p in SENIORS else RESIDENTS)
                        if q != p and q in answers]
            if same_exp:
                within = np.mean([pairwise_f1(answers[p], answers[q], threshold)
                                  for q in same_exp])
            else:
                within = np.nan
            # Cross-expertise
            other_exp = [q for q in (RESIDENTS if p in SENIORS else SENIORS)
                         if q in answers]
            if other_exp:
                cross = np.mean([pairwise_f1(answers[p], answers[q], threshold)
                                 for q in other_exp])
            else:
                cross = np.nan
            conc_rows.append(dict(
                session=sess, condition=cond, note_id=note_id,
                difficulty=diff, participant=p, expertise=p_exp,
                within_expertise_f1=within, cross_expertise_f1=cross,
            ))

    cdf = pd.DataFrame(conc_rows)
    cdf.to_csv(out_dir / "c_concordance_case_level.csv", index=False)

    # Standardise and aggregate
    conc_tests = []
    for metric in ["within_expertise_f1", "cross_expertise_f1"]:
        pstd_rows = []
        for (sess, p), g in cdf.groupby(["session", "participant"]):
            pstd_rows.append(dict(
                session=sess, participant=p, condition=SESSIONS[sess],
                expertise="Senior" if p in SENIORS else "Resident",
                value=standardize(g, metric),
            ))
        pstd = pd.DataFrame(pstd_rows)
        agg_rows = []
        for p in ALL_PARTICIPANTS:
            for cond in ["baseline", "interactive"]:
                sub = pstd[(pstd["participant"] == p) &
                           (pstd["condition"] == cond)]
                agg_rows.append(dict(
                    participant=p, condition=cond,
                    expertise="Senior" if p in SENIORS else "Resident",
                    value=sub["value"].mean(),
                ))
        agg = pd.DataFrame(agg_rows)
        for group_name, participants in [
            ("All", ALL_PARTICIPANTS), ("Senior", SENIORS), ("Resident", RESIDENTS),
        ]:
            bl = agg[(agg["condition"] == "baseline") &
                      (agg["participant"].isin(participants))]["value"].values
            it = agg[(agg["condition"] == "interactive") &
                      (agg["participant"].isin(participants))]["value"].values
            bt = paired_bootstrap(bl, it, n_boot, seed)
            bt["endpoint"] = metric
            bt["group"] = group_name
            bt["cohens_d"] = cohens_d(bl, it)
            conc_tests.append(bt)
    pd.DataFrame(conc_tests).to_csv(
        out_dir / "c_concordance_bootstrap_tests.csv", index=False)
    print(f"  Concordance tests: {len(conc_tests)} comparisons")

    # Concordance by difficulty
    conc_by_diff = cdf.groupby(["difficulty", "condition", "expertise"]).agg(
        mean_within=("within_expertise_f1", "mean"),
        mean_cross=("cross_expertise_f1", "mean"),
        n=("within_expertise_f1", "count"),
    ).reset_index()
    conc_by_diff.to_csv(out_dir / "c_concordance_by_difficulty.csv", index=False)

    return cdf


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def generate_figures(mdf, agg, tests, merged, cdf, out_dir, fig_dir):
    """Generate all publication-quality figures."""
    print("\n" + "═" * 60)
    print("GENERATING FIGURES")
    print("═" * 60)

    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Paired trajectories ──────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Offset for each expertise group so lines align with their box plots
    exp_offset = {"Senior": -0.15, "Resident": 0.15}
    for idx, (ep, title) in enumerate([
        ("any_match_std", "Any-match accuracy"),
        ("exact_match_std", "Exact-match accuracy"),
        ("f1_std", "Diagnosis-set F1"),
        ("time_min_std", "Time per case (minutes)"),
    ]):
        ax = axes[idx // 2][idx % 2]
        # Draw box plots first (behind lines)
        for ci, cond in enumerate(["Baseline", "Interactive"]):
            cond_key = cond.lower()
            for ei, (exp, parts) in enumerate([("Senior", SENIORS), ("Resident", RESIDENTS)]):
                vals = agg[(agg["condition"] == cond_key) &
                           (agg["participant"].isin(parts))][ep].values
                pos = ci + exp_offset[exp]
                bp = ax.boxplot([vals], positions=[pos], widths=0.2,
                                patch_artist=True, showfliers=False,
                                zorder=1)
                bp["boxes"][0].set_facecolor(PALETTE[exp])
                bp["boxes"][0].set_alpha(0.3)
                for element in ['whiskers', 'caps', 'medians']:
                    for line in bp[element]:
                        line.set_alpha(0.5)
        # Draw individual participant lines on top
        for p in ALL_PARTICIPANTS:
            exp = "Senior" if p in SENIORS else "Resident"
            color = PALETTE[exp]
            off = exp_offset[exp]
            bl = agg[(agg["participant"] == p) &
                      (agg["condition"] == "baseline")][ep].values
            it = agg[(agg["participant"] == p) &
                      (agg["condition"] == "interactive")][ep].values
            if len(bl) and len(it):
                ax.plot([0 + off, 1 + off], [bl[0], it[0]],
                        "o-", color=color, alpha=0.7, markersize=6,
                        zorder=2,
                        label=exp if idx == 0 and p in [SENIORS[0], RESIDENTS[0]] else "")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "Interactive"])
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_paired_trajectories.png", dpi=FIG_DPI,
                bbox_inches="tight")
    plt.close()
    print("  fig_paired_trajectories.png")

    # ── Figure 2: Metrics by difficulty × condition × expertise ────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (ep, title) in enumerate([
        ("any_match", "Any-match accuracy"),
        ("exact_match", "Exact-match accuracy"),
        ("f1", "F1"),
        ("time_min", "Time (min)"),
    ]):
        ax = axes[idx // 2][idx % 2]
        diff_order = ["Easy", "Medium", "Hard"]
        x = np.arange(len(diff_order))
        width = 0.2
        for gi, (exp, cond, ls) in enumerate([
            ("Senior", "baseline", "--"), ("Senior", "interactive", "-"),
            ("Resident", "baseline", "--"), ("Resident", "interactive", "-"),
        ]):
            vals = []
            errs = []
            for diff in diff_order:
                sub = mdf[(mdf["difficulty"] == diff) &
                          (mdf["condition"] == cond) &
                          (mdf["expertise"] == exp)]
                vals.append(sub[ep].mean())
                errs.append(sub[ep].sem())
            offset = (gi - 1.5) * width
            color = PALETTE[exp]
            alpha = 1.0 if cond == "interactive" else 0.5
            label = f"{exp} - {cond.capitalize()}"
            ax.bar(x + offset, vals, width, yerr=errs, color=color,
                   alpha=alpha, label=label, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(diff_order)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Difficulty")
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_metrics_by_difficulty.png", dpi=FIG_DPI,
                bbox_inches="tight")
    plt.close()
    print("  fig_metrics_by_difficulty.png")

    # ── Figure 3: Manual 3-class distribution ──────────────────────────────
    if merged is not None and not merged.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = [("Senior", "baseline"), ("Senior", "interactive"),
                  ("Resident", "baseline"), ("Resident", "interactive")]
        labels = [f"{e}\n{c.capitalize()}" for e, c in groups]
        colors_3 = {"WRONG": "#d73027", "PARTIALLY CORRECT": "#fee08b",
                     "COMPLETELY CORRECT": "#1a9850"}
        bottoms = np.zeros(len(groups))
        for cat in ["WRONG", "PARTIALLY CORRECT", "COMPLETELY CORRECT"]:
            vals = []
            for exp, cond in groups:
                sub = merged[(merged["expertise"] == exp) &
                             (merged["condition"] == cond)]
                vals.append((sub["manual_label"] == cat).mean())
            ax.bar(labels, vals, bottom=bottoms, color=colors_3[cat],
                   label=cat.title(), edgecolor="white", linewidth=0.5)
            bottoms += vals
        ax.set_ylabel("Proportion of cases")
        ax.set_title("Manual correctness distribution by expertise and condition",
                     fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_manual_distribution.png", dpi=FIG_DPI,
                    bbox_inches="tight")
        plt.close()
        print("  fig_manual_distribution.png")

        # ── Figure 4: Manual score by difficulty × expertise × condition ───
        # Boxplot version matching original style (clean, minimal color)
        # Load interaction test p-values for annotation
        _int_test_path = out_dir / "b_manual_difficulty_interaction_tests.csv"
        _int_tests = {}
        if _int_test_path.exists():
            _idf = pd.read_csv(_int_test_path)
            for _, row in _idf.iterrows():
                _int_tests[(row["difficulty"], row["group"])] = row["p_value"]

        diff_order = ["Easy", "Medium", "Hard"]
        groups = [
            ("Senior", "baseline", "S-B"),
            ("Senior", "interactive", "S-I"),
            ("Resident", "baseline", "R-B"),
            ("Resident", "interactive", "R-I"),
        ]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        bp_data = []
        bp_colors = []
        positions = []
        tick_labels = []
        pos = 0
        for di, diff in enumerate(diff_order):
            if di > 0:
                pos += 1.2  # gap between difficulty groups
            for gi, (exp, cond, label) in enumerate(groups):
                sub = merged[(merged["difficulty"] == diff) &
                             (merged["condition"] == cond) &
                             (merged["expertise"] == exp)]
                # Per-participant means (averaged across sessions of same
                # condition), matching the v3 report methodology
                participant_means = sub.groupby(
                    "participant")["manual_score"].mean().values
                bp_data.append(participant_means)
                positions.append(pos)
                tick_labels.append(f"{diff}\n{label}")
                color = PALETTE[exp]
                alpha = 0.35 if cond == "baseline" else 0.7
                bp_colors.append((color, alpha))
                pos += 1

        from matplotlib.colors import to_rgba
        bp = ax.boxplot(bp_data, positions=positions, widths=0.65,
                        patch_artist=True, showmeans=False,
                        medianprops=dict(color="darkorange", linewidth=1.5),
                        whiskerprops=dict(color="black", linewidth=0.8),
                        capprops=dict(color="black", linewidth=0.8),
                        flierprops=dict(marker="o", markerfacecolor="gray",
                                        markersize=4, alpha=0.5,
                                        markeredgecolor="gray"))
        for patch, (color, alpha) in zip(bp["boxes"], bp_colors):
            patch.set_facecolor(to_rgba(color, alpha))
            patch.set_edgecolor(color)
            patch.set_linewidth(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_ylabel("Ordinal Correctness Score", fontsize=10)
        ax.set_title("Blinded Manual Assessment by Difficulty, Expertise, and Condition",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add p-value brackets for Resident baseline→interactive per difficulty
        for di, diff in enumerate(diff_order):
            p = _int_tests.get((diff, "Resident"))
            if p is not None:
                rb_pos = positions[di * 4 + 2]
                ri_pos = positions[di * 4 + 3]
                # Find max whisker top for these two boxes
                rb_idx = di * 4 + 2
                ri_idx = di * 4 + 3
                top = max(bp["whiskers"][rb_idx * 2 + 1].get_ydata().max(),
                          bp["whiskers"][ri_idx * 2 + 1].get_ydata().max())
                y = top + 0.03
                ax.plot([rb_pos, rb_pos, ri_pos, ri_pos],
                        [y, y + 0.02, y + 0.02, y],
                        color="black", linewidth=0.8)
                p_str = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
                ax.text((rb_pos + ri_pos) / 2, y + 0.025, p_str,
                        ha="center", va="bottom", fontsize=7)

        # Legend
        from matplotlib.patches import Patch
        from matplotlib.colors import to_rgba as _to_rgba
        legend_elements = [
            Patch(facecolor=_to_rgba(PALETTE["Senior"], 0.35),
                  edgecolor=PALETTE["Senior"], label="Senior - Baseline"),
            Patch(facecolor=_to_rgba(PALETTE["Senior"], 0.7),
                  edgecolor=PALETTE["Senior"], label="Senior - Interactive"),
            Patch(facecolor=_to_rgba(PALETTE["Resident"], 0.35),
                  edgecolor=PALETTE["Resident"], label="Resident - Baseline"),
            Patch(facecolor=_to_rgba(PALETTE["Resident"], 0.7),
                  edgecolor=PALETTE["Resident"], label="Resident - Interactive"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="lower left")

        plt.tight_layout()
        plt.savefig(fig_dir / "fig_manual_by_difficulty.png", dpi=FIG_DPI,
                    bbox_inches="tight")
        plt.close()
        print("  fig_manual_by_difficulty.png")

        # ── Figure 5: Confusion matrices ───────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Binary
        ax = axes[0]
        auto_bin = merged["any_match"].values.astype(int)
        man_bin = merged["manual_binary"].values.astype(int)
        conf_bin = np.array([
            [(auto_bin == r).sum() - ((auto_bin == r) & (man_bin == 1)).sum()
             if c == 0 else ((auto_bin == r) & (man_bin == 1)).sum()
             for c in [0, 1]]
            for r in [0, 1]
        ])
        # Recompute properly
        conf_bin = np.zeros((2, 2), dtype=int)
        for a, m in zip(auto_bin, man_bin):
            conf_bin[a][m] += 1
        im = ax.imshow(conf_bin, cmap="YlOrBr", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(conf_bin[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Wrong", "Correct"])
        ax.set_yticklabels(["No match", "Any match"])
        ax.set_xlabel("Manual label")
        ax.set_ylabel("Automated label")
        ax.set_title("Binary: auto any-match vs manual", fontweight="bold")

        # 3-class
        ax = axes[1]
        labels_3 = ["Wrong", "Partially\ncorrect", "Completely\ncorrect"]
        auto_3 = np.where(
            merged["exact_match"] == 1, 2,
            np.where(merged["any_match"] == 1, 1, 0)
        )
        man_3 = np.where(
            merged["manual_label"] == "COMPLETELY CORRECT", 2,
            np.where(merged["manual_label"] == "PARTIALLY CORRECT", 1, 0)
        )
        conf_3 = np.zeros((3, 3), dtype=int)
        for a, m in zip(auto_3, man_3):
            conf_3[a][m] += 1
        im = ax.imshow(conf_3, cmap="YlOrBr", aspect="auto")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(conf_3[i, j]), ha="center", va="center",
                        fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(labels_3, fontsize=9)
        ax.set_yticklabels(labels_3, fontsize=9)
        ax.set_xlabel("Manual label")
        ax.set_ylabel("Automated label")
        ax.set_title("3-class: automated vs manual", fontweight="bold")

        plt.tight_layout()
        plt.savefig(fig_dir / "fig_confusion_matrices.png", dpi=FIG_DPI,
                    bbox_inches="tight")
        plt.close()
        print("  fig_confusion_matrices.png")

    # ── Figure 6: Concordance ──────────────────────────────────────────────
    if cdf is not None and not cdf.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax_idx, (metric, title) in enumerate([
            ("within_expertise_f1", "Within-expertise concordance"),
            ("cross_expertise_f1", "Cross-expertise concordance"),
        ]):
            ax = axes[ax_idx]
            diff_order = ["Easy", "Medium", "Hard"]
            x = np.arange(len(diff_order))
            width = 0.2
            for gi, (exp, cond) in enumerate([
                ("Senior", "baseline"), ("Senior", "interactive"),
                ("Resident", "baseline"), ("Resident", "interactive"),
            ]):
                vals = []
                errs = []
                for diff in diff_order:
                    sub = cdf[(cdf["difficulty"] == diff) &
                              (cdf["condition"] == cond) &
                              (cdf["expertise"] == exp)]
                    vals.append(sub[metric].mean())
                    errs.append(sub[metric].sem())
                offset = (gi - 1.5) * width
                color = PALETTE[exp]
                alpha = 1.0 if cond == "interactive" else 0.5
                label = f"{exp} - {cond.capitalize()}"
                ax.bar(x + offset, vals, width, yerr=errs, color=color,
                       alpha=alpha, label=label, capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(diff_order)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Difficulty")
            ax.grid(axis="y", alpha=0.3)
            if ax_idx == 0:
                ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_concordance_by_difficulty.png", dpi=FIG_DPI,
                    bbox_inches="tight")
        plt.close()
        print("  fig_concordance_by_difficulty.png")

    # ── Figure 7: Threshold sensitivity ────────────────────────────────────
    sweep = pd.read_csv(out_dir / "a_threshold_sensitivity.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for idx, ep in enumerate(["any_match", "f1", "exact_match"]):
        ax = axes[idx]
        for cond, ls in [("baseline", "--"), ("interactive", "-")]:
            sub = sweep[(sweep["endpoint"] == ep) & (sweep["condition"] == cond)]
            ax.plot(sub["threshold"], sub["mean"], f"{ls}o",
                    color=COND_PALETTE[cond.capitalize()],
                    label=cond.capitalize(), markersize=5)
            ax.fill_between(sub["threshold"],
                            sub["mean"] - sub["std"],
                            sub["mean"] + sub["std"],
                            color=COND_PALETTE[cond.capitalize()], alpha=0.15)
        ax.set_xlabel("Matching threshold")
        ax.set_title(ep.replace("_", " ").title(), fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_threshold_sensitivity.png", dpi=FIG_DPI,
                bbox_inches="tight")
    plt.close()
    print("  fig_threshold_sensitivity.png")

    # ── Figure 8: Across-sessions trajectory (S1→S2→S3→S4) ─────────────
    # Compute per-session standardised means by expertise group
    sess_agg = []
    for (sess, p), g in mdf.groupby(["session", "participant"]):
        row = dict(session=sess, participant=p,
                   expertise="Senior" if p in SENIORS else "Resident")
        for ep in ["any_match", "exact_match", "f1", "time_min"]:
            row[ep] = standardize(g, ep)
        sess_agg.append(row)
    sess_df = pd.DataFrame(sess_agg)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (ep, ylabel) in enumerate([
        ("any_match", "Any-match accuracy"),
        ("exact_match", "Exact-match accuracy"),
        ("f1", "Diagnosis-set F1"),
        ("time_min", "Time per case (min)"),
    ]):
        ax = axes[idx // 2][idx % 2]
        for exp, parts in [("Senior", SENIORS), ("Resident", RESIDENTS)]:
            means = []
            sems = []
            for s in [1, 2, 3, 4]:
                vals = sess_df[(sess_df["session"] == s) &
                               (sess_df["expertise"] == exp)][ep].values
                means.append(vals.mean())
                sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            ax.errorbar([1, 2, 3, 4], means, yerr=sems, marker="o",
                        color=PALETTE[exp], label=exp, linewidth=2,
                        capsize=4, markersize=7)
        # Shade interactive sessions
        ax.axvspan(1.5, 2.5, alpha=0.08, color="#af8dc3")
        ax.axvspan(3.5, 4.5, alpha=0.08, color="#af8dc3")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["S1\n(Baseline)", "S2\n(Interactive)",
                             "S3\n(Baseline)", "S4\n(Interactive)"])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_across_sessions.png", dpi=FIG_DPI,
                bbox_inches="tight")
    plt.close()
    print("  fig_across_sessions.png")

    # ── Figure 9: Expertise gap narrowing by difficulty ───────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (ep, title) in enumerate([
        ("any_match", "Any-match accuracy gap"),
        ("exact_match", "Exact-match accuracy gap"),
        ("f1", "F1 gap"),
        ("time_min", "Time gap (min)"),
    ]):
        ax = axes[idx // 2][idx % 2]
        diff_order = ["Easy", "Medium", "Hard"]
        for cond, ls, marker in [("baseline", "--", "s"), ("interactive", "-", "o")]:
            gaps = []
            for diff in diff_order:
                senior_vals = mdf[(mdf["difficulty"] == diff) &
                                   (mdf["condition"] == cond) &
                                   (mdf["expertise"] == "Senior")][ep].mean()
                resident_vals = mdf[(mdf["difficulty"] == diff) &
                                     (mdf["condition"] == cond) &
                                     (mdf["expertise"] == "Resident")][ep].mean()
                gaps.append(senior_vals - resident_vals)
            ax.plot(diff_order, gaps, f"{ls}{marker}",
                    color=COND_PALETTE[cond.capitalize()],
                    label=cond.capitalize(), linewidth=2, markersize=8)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Difficulty")
        ax.set_ylabel("Gap (Senior − Resident)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_expertise_gap.png", dpi=FIG_DPI,
                bbox_inches="tight")
    plt.close()
    print("  fig_expertise_gap.png")

    # ── Figure 10: Ablation model comparison ─────────────────────────────
    ablation_dir = Path("eval/ablation_eval/results")
    bl_path = ablation_dir / "baseline_summary_primary.csv"
    it_path = ablation_dir / "interactive_summary_primary.csv"
    if bl_path.exists() and it_path.exists():
        bl_abl = pd.read_csv(bl_path)
        it_abl = pd.read_csv(it_path)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        metrics_abl = ["micro_precision", "micro_recall", "micro_f1"]
        metric_labels = ["Precision", "Recall", "F1"]
        model_colors = {
            "gpt-5.2-chat": "#2166ac",
            "gpt-oss-120b": "#4393c3",
            "gemini-3-pro-preview": "#92c5de",
            "llama-4-scout": "#f4a582",
            "gpt-5.1": "#d6604d",
            "gpt-5.1-chat": "#d6604d",
        }

        for ax_idx, (abl_df, scenario) in enumerate([
            (bl_abl, "Baseline scenario"),
            (it_abl, "Interactive scenario"),
        ]):
            ax = axes[ax_idx]
            x = np.arange(len(metrics_abl))
            n_models = len(abl_df)
            width = 0.8 / n_models
            for mi, (_, row) in enumerate(abl_df.iterrows()):
                model = row["model"]
                vals = [row[m] for m in metrics_abl]
                offset = (mi - n_models / 2 + 0.5) * width
                color = model_colors.get(model, "#999999")
                short_name = model.replace("-chat", "").replace("-preview", "")
                ax.bar(x + offset, vals, width, label=short_name,
                       color=color, edgecolor="white", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1.0)
            ax.set_title(scenario, fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=7, loc="upper left")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_ablation_comparison.png", dpi=FIG_DPI,
                    bbox_inches="tight")
        plt.close()
        print("  fig_ablation_comparison.png")
    else:
        print("  [SKIP] Ablation CSVs not found; skipping ablation figure.")

    # ── Figure 11: Forest plot of improvement effects (bootstrap) ────────
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_rows = []
    for _, t in tests.iterrows():
        plot_rows.append(t)
    # Order: group within endpoint
    ep_order = ["any_match", "exact_match", "f1", "precision", "recall", "time_min"]
    ep_labels = {"any_match": "Any-match", "exact_match": "Exact-match",
                 "f1": "Diagnosis-set F1", "precision": "Precision",
                 "recall": "Recall", "time_min": "Time (min)"}
    grp_order = ["All", "Senior", "Resident"]
    grp_colors = {"All": "#333333", "Senior": "#2166ac", "Resident": "#b2182b"}
    grp_markers = {"All": "D", "Senior": "s", "Resident": "o"}
    y_pos = 0
    y_labels = []
    y_positions = []
    for ep in ep_order:
        for grp in grp_order:
            t = tests[(tests["endpoint"] == ep) & (tests["group"] == grp)]
            if len(t) == 0:
                continue
            t = t.iloc[0]
            color = grp_colors[grp]
            marker = grp_markers[grp]
            facecolor = color if t["sig"] else "white"
            ax.errorbar(t["mean_delta"], y_pos,
                        xerr=[[t["mean_delta"] - t["ci_lo"]],
                              [t["ci_hi"] - t["mean_delta"]]],
                        fmt=marker, color=color, markerfacecolor=facecolor,
                        markersize=8, capsize=4, linewidth=1.5,
                        markeredgewidth=1.5)
            label = f"  {ep_labels[ep]} ({grp})"
            sig_str = ""
            if t["p_value"] < 0.001:
                sig_str = " ***"
            elif t["p_value"] < 0.01:
                sig_str = " **"
            elif t["sig"]:
                sig_str = " *"
            y_labels.append(f"{label}{sig_str}")
            y_positions.append(y_pos)
            y_pos += 1
        y_pos += 0.5  # gap between endpoints
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Mean improvement (Interactive − Baseline)", fontsize=11)
    ax.set_title("Effect sizes with 95% bootstrap CIs", fontsize=12,
                 fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_forest_plot.png", dpi=FIG_DPI,
                bbox_inches="tight")
    plt.close()
    print("  fig_forest_plot.png")

    # ── Figure 12: Overall baseline vs interactive grouped bar ────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for idx, (ep, title) in enumerate([
        ("any_match_std", "Any-match accuracy"),
        ("exact_match_std", "Exact-match accuracy"),
        ("f1_std", "Diagnosis-set F1"),
        ("time_min_std", "Time per case (min)"),
    ]):
        ax = axes[idx]
        x = np.arange(2)  # Senior, Resident
        width = 0.35
        for ci, (cond, alpha_val) in enumerate([("baseline", 0.5), ("interactive", 1.0)]):
            vals = []
            errs = []
            for exp, parts in [("Senior", SENIORS), ("Resident", RESIDENTS)]:
                sub = agg[(agg["condition"] == cond) &
                          (agg["participant"].isin(parts))][ep]
                vals.append(sub.mean())
                errs.append(sub.sem())
            offset = (ci - 0.5) * width
            ax.bar(x + offset, vals, width, yerr=errs, capsize=4,
                   color=[PALETTE["Senior"], PALETTE["Resident"]],
                   alpha=alpha_val,
                   label=cond.capitalize() if idx == 0 else "",
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Senior", "Resident"])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_overall_comparison.png", dpi=FIG_DPI,
                bbox_inches="tight")
    plt.close()
    print("  fig_overall_comparison.png")

    # ── Figure 13: Per-session performance (with error bars) ──────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (ep, title) in enumerate([
        ("any_match", "Any-match accuracy"),
        ("exact_match", "Exact-match accuracy"),
        ("f1", "Diagnosis-set F1"),
        ("time_min", "Time per case (min)"),
    ]):
        ax = axes[idx // 2][idx % 2]
        sessions = [1, 2, 3, 4]
        x = np.arange(len(sessions))
        width = 0.35
        for ei, (exp, parts) in enumerate([("Senior", SENIORS), ("Resident", RESIDENTS)]):
            means = []
            sems = []
            for s in sessions:
                # Compute standardized per-participant for this session
                sess_vals = []
                for p in parts:
                    pdata = mdf[(mdf["session"] == s) & (mdf["participant"] == p)]
                    if len(pdata):
                        sess_vals.append(standardize(pdata, ep))
                means.append(np.mean(sess_vals) if sess_vals else 0)
                sems.append(np.std(sess_vals, ddof=1) / np.sqrt(len(sess_vals))
                            if len(sess_vals) > 1 else 0)
            offset = (ei - 0.5) * width
            ax.bar(x + offset, means, width, yerr=sems, capsize=4,
                   color=PALETTE[exp], label=exp if idx == 0 else "",
                   edgecolor="white", linewidth=0.5)
        # Shade interactive sessions
        ax.axvspan(0.5, 1.5, alpha=0.06, color="#af8dc3")
        ax.axvspan(2.5, 3.5, alpha=0.06, color="#af8dc3")
        ax.set_xticks(x)
        ax.set_xticklabels(["S1\n(Baseline)", "S2\n(Interactive)",
                             "S3\n(Baseline)", "S4\n(Interactive)"])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_per_session_bars.png", dpi=FIG_DPI,
                bbox_inches="tight")
    plt.close()
    print("  fig_per_session_bars.png")

    # ── Figure 14: Manual evaluation per-session summary ──────────────────
    if merged is not None and not merged.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sessions_m = [1, 2, 3, 4]
        for ax_idx, (ep, title) in enumerate([
            ("manual_score", "Ordinal score (0 / 0.5 / 1)"),
            ("manual_binary", "Binary correctness"),
            ("manual_complete", "Completely correct rate"),
        ]):
            ax = axes[ax_idx]
            x = np.arange(len(sessions_m))
            width = 0.35
            for ei, (exp, parts) in enumerate([("Senior", SENIORS),
                                                ("Resident", RESIDENTS)]):
                means = []
                sems = []
                for s in sessions_m:
                    sub = merged[(merged["session"] == s) &
                                 (merged["expertise"] == exp)]
                    means.append(sub[ep].mean() if len(sub) else 0)
                    sems.append(sub[ep].sem() if len(sub) > 1 else 0)
                offset = (ei - 0.5) * width
                ax.bar(x + offset, means, width, yerr=sems, capsize=4,
                       color=PALETTE[exp], label=exp if ax_idx == 0 else "",
                       edgecolor="white", linewidth=0.5)
            ax.axvspan(0.5, 1.5, alpha=0.06, color="#af8dc3")
            ax.axvspan(2.5, 3.5, alpha=0.06, color="#af8dc3")
            ax.set_xticks(x)
            ax.set_xticklabels(["S1\n(Baseline)", "S2\n(Interactive)",
                                 "S3\n(Baseline)", "S4\n(Interactive)"])
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
        axes[0].legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_manual_per_session.png", dpi=FIG_DPI,
                    bbox_inches="tight")
        plt.close()
        print("  fig_manual_per_session.png")

    print("  All figures generated.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MedSyn Unified Evaluation")
    parser.add_argument("--session_dir", default="eval/session_outputs")
    parser.add_argument("--manual_dir", default="eval/manual_eval/inputs")
    parser.add_argument("--out_dir", default="eval/results")
    parser.add_argument("--threshold", type=int, default=80)
    parser.add_argument("--bootstrap_n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration:")
    print(f"  Threshold: {args.threshold}")
    print(f"  Bootstrap replicates: {args.bootstrap_n}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {out_dir}")
    print()

    # Save config
    config = vars(args)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Load data
    long_df = load_session_data(args.session_dir)
    manual_df = load_manual_data(args.manual_dir)
    print(f"Loaded: {len(long_df)} observations, "
          f"{len(manual_df)} manual labels\n")

    # Run parts
    mdf, agg, tests = run_part_a(long_df, out_dir, args.threshold,
                                  args.bootstrap_n, args.seed)
    merged = run_part_b(mdf, manual_df, out_dir, args.bootstrap_n, args.seed)
    cdf = run_part_c(long_df, out_dir, args.threshold,
                      args.bootstrap_n, args.seed)

    # Generate figures
    generate_figures(mdf, agg, tests, merged, cdf, out_dir, fig_dir)

    # Print key results summary
    print("\n" + "═" * 60)
    print("KEY RESULTS SUMMARY")
    print("═" * 60)
    sig_tests = tests[tests["sig"]]
    for _, t in tests.iterrows():
        star = "***" if t["p_value"] < 0.001 else ("**" if t["p_value"] < 0.01
               else ("*" if t["sig"] else ""))
        print(f"  {t['endpoint']:15s} | {t['group']:8s} | "
              f"Δ={t['mean_delta']:+.3f} | "
              f"CI=[{t['ci_lo']:.3f}, {t['ci_hi']:.3f}] | "
              f"p={t['p_value']:.4f} {star} | "
              f"d={t['cohens_d']:.2f}")

    print(f"\nAll outputs saved to {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
