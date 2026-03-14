#!/usr/bin/env python3
"""
MedSyn ablation evaluator (baseline + interactive).

This script implements:
- trustworthy parsing of model outputs that store literal "\\n" separators
- parsing of ground truth "discharge diagnosis" with primary/secondary sections
- soft matching (RapidFuzz token_set_ratio) with configurable threshold
- micro precision/recall/F1, top-1 primary-first accuracy, any-primary recall, avg #preds
- difficulty-stratified metrics
- threshold sweep and bootstrap confidence intervals

Example:
python evaluate_ablation.py \
  --baseline baseline_scenario_outputs.csv \
  --interactive interactive_scenario_outputs.csv \
  --outdir ablation_eval_out \
  --threshold 62 \
  --sweep \
  --bootstrap 2000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


ABBREV_MAP: Dict[str, str] = {
    "pna": "pneumonia",
    "cap": "community acquired pneumonia",
    "copd": "chronic obstructive pulmonary disease",
    "aki": "acute kidney injury",
    "uti": "urinary tract infection",
    "afib": "atrial fibrillation",
    "a": "a",  # placeholder: keep map conservative; do not expand common tokens.
    "rvr": "rapid ventricular response",
    "tia": "transient ischemic attack",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "dka": "diabetic ketoacidosis",
    "ckd": "chronic kidney disease",
    "esrd": "end stage renal disease",
    "hfpef": "heart failure with preserved ejection fraction",
    "hfreF".lower(): "heart failure with reduced ejection fraction",
    "nstemi": "non st elevation myocardial infarction",
    "stemi": "st elevation myocardial infarction",
    "mi": "myocardial infarction",
    "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis",
    "cva": "stroke",
    "ich": "intracerebral hemorrhage",
    "sah": "subarachnoid hemorrhage",
    "rll": "right lower lobe",
    "lll": "left lower lobe",
    "rul": "right upper lobe",
    "lul": "left upper lobe",
}

STOPWORDS = {
    "the", "and", "of", "with", "without", "due", "to", "in", "on", "for", "a", "an"
}

def normalize_dx(s: str) -> str:
    """Normalize a diagnosis string for similarity matching."""
    if s is None:
        return ""
    s = str(s).strip()
    # strip surrounding quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    s = s.lower().replace("&", " and ")
    s = re.sub(r"^\s*(primary|secondary)\s*:\s*", "", s)
    # keep only alnum and spaces
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    toks: List[str] = []
    for t in s.split():
        if t in ABBREV_MAP:
            toks.extend(ABBREV_MAP[t].split())
        else:
            toks.append(t)
    toks = [t for t in toks if t not in STOPWORDS]
    return " ".join(toks).strip()

def similarity(a: str, b: str) -> float:
    na, nb = normalize_dx(a), normalize_dx(b)
    if not na or not nb:
        return 0.0
    return float(fuzz.token_set_ratio(na, nb))

def extract_pred_text(cell: str) -> str:
    """Extract raw diagnosis text from a model cell."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return ""
    s = str(cell).strip()
    m = re.search(r"diagnosis\s*=\s*(.*)$", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()
    if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
        s = s[1:-1]
    # convert literal "\n" to real newline
    s = s.replace("\\n", "\n")
    return s.strip()

def split_pred_list(pred_text: str) -> List[str]:
    """Split raw prediction text into a list of diagnoses, preserving order."""
    s = (pred_text or "").strip()
    if not s:
        return []
    # normalize bullets
    s = s.replace("•", "\n").replace("- ", "\n")
    if "\n" in s:
        parts = [p.strip() for p in s.split("\n")]
    elif ";" in s:
        parts = [p.strip() for p in s.split(";")]
    elif "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [s]

    parts = [re.sub(r"^\s*[\-\*\d\.\)\]]+\s*", "", p).strip() for p in parts]
    parts = [p for p in parts if p and p.lower() not in ("none", "n/a", "na", "unknown")]
    # de-duplicate by normalized form
    seen = set()
    out: List[str] = []
    for p in parts:
        k = normalize_dx(p)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out

def parse_gt(gt: str) -> Tuple[List[str], List[str]]:
    """Parse ground truth into primary and secondary lists."""
    s = str(gt or "").replace("\r\n", "\n")
    low = s.lower()
    prim_idx = low.find("primary:")
    sec_idx = low.find("secondary:")
    if prim_idx != -1:
        if sec_idx != -1 and sec_idx > prim_idx:
            prim_text = s[prim_idx + len("primary:"):sec_idx]
            sec_text = s[sec_idx + len("secondary:"):]
        else:
            prim_text, sec_text = s[prim_idx + len("primary:"):], ""
    else:
        prim_text, sec_text = s, ""

    def split_lines(t: str) -> List[str]:
        lines = [ln.strip() for ln in t.split("\n")]
        out: List[str] = []
        for ln in lines:
            if not ln:
                continue
            ln = re.sub(r"\.+$", "", ln).strip()
            if ln in ("...", "…"):
                continue
            out.append(ln)
        return out

    return split_lines(prim_text), split_lines(sec_text)

def match_tp(preds: List[str], gts: List[str], threshold: float) -> int:
    """Greedy one-to-one matching between preds and gts above threshold."""
    if not preds or not gts:
        return 0
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            sc = similarity(p, g)
            if sc >= threshold:
                pairs.append((sc, i, j))
    pairs.sort(reverse=True)
    used_p, used_g = set(), set()
    tp = 0
    for sc, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        tp += 1
    return tp

@dataclass
class SummaryRow:
    model: str
    micro_precision: float
    micro_recall: float
    micro_f1: float
    top1_primary_first_acc: float
    any_primary_recall: float
    avg_n_pred: float
    tp: int
    fp: int
    fn: int
    n_cases: int
    target: str

def evaluate_df(df: pd.DataFrame, output_cols: List[str], threshold: float, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    totals = {col: {"tp":0, "fp":0, "fn":0, "n_pred":0, "top1":0, "anyprim":0, "n_cases":0} for col in output_cols}
    per_case_records = []

    for _, row in df.iterrows():
        prim, sec = parse_gt(row.get("discharge diagnosis", ""))
        gt_all = prim + sec
        gts = gt_all if target == "all" else prim

        for col in output_cols:
            preds = split_pred_list(extract_pred_text(row.get(col, "")))

            tp = match_tp(preds, gts, threshold)
            fp = max(len(preds) - tp, 0)
            fn = max(len(gts) - tp, 0)

            totals[col]["tp"] += tp
            totals[col]["fp"] += fp
            totals[col]["fn"] += fn
            totals[col]["n_pred"] += len(preds)
            totals[col]["n_cases"] += 1

            # top1 primary-first
            top1_match = False
            if preds:
                top1_match = match_tp([preds[0]], prim, threshold) > 0
            totals[col]["top1"] += int(top1_match)

            any_primary = match_tp(preds, prim, threshold) > 0 if preds else False
            totals[col]["anyprim"] += int(any_primary)

            per_case_records.append({
                "note_id": row.get("note_id", ""),
                "Difficulty": row.get("Difficulty", ""),
                "model_col": col,
                "target": target,
                "n_gt": len(gts),
                "n_pred": len(preds),
                "tp": tp, "fp": fp, "fn": fn,
                "top1_primary_first": int(top1_match),
                "any_primary": int(any_primary),
            })

    rows: List[SummaryRow] = []
    for col, v in totals.items():
        tp, fp, fn = v["tp"], v["fp"], v["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        n_cases = v["n_cases"]
        rows.append(SummaryRow(
            model=col.replace("output_", ""),
            micro_precision=prec,
            micro_recall=rec,
            micro_f1=f1,
            top1_primary_first_acc=v["top1"] / n_cases if n_cases else 0.0,
            any_primary_recall=v["anyprim"] / n_cases if n_cases else 0.0,
            avg_n_pred=v["n_pred"] / n_cases if n_cases else 0.0,
            tp=tp, fp=fp, fn=fn,
            n_cases=n_cases,
            target=target,
        ))

    summary_df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("micro_f1", ascending=False)
    per_case_df = pd.DataFrame(per_case_records)
    return summary_df, per_case_df

def newline_audit(df: pd.DataFrame, output_cols: List[str]) -> pd.DataFrame:
    audit = []
    for col in output_cols:
        vals = df[col].astype(str)
        audit.append({
            "column": col,
            "pct_literal_\\n": float(vals.str.contains(r"\\n", regex=True).mean()),
            "pct_real_newline": float(vals.str.contains("\n").mean())
        })
    return pd.DataFrame(audit)

def summarize_by_difficulty(per_case_df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (diff, model_col), g in per_case_df.groupby(["Difficulty", "model_col"]):
        tp, fp, fn = int(g["tp"].sum()), int(g["fp"].sum()), int(g["fn"].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        recs.append({
            "Difficulty": diff,
            "model": model_col.replace("output_", ""),
            "micro_precision": prec,
            "micro_recall": rec,
            "micro_f1": f1,
            "top1_primary_first_acc": float(g["top1_primary_first"].mean()),
            "any_primary_recall": float(g["any_primary"].mean()),
            "avg_n_pred": float(g["n_pred"].mean()),
            "n_cases": int(len(g)),
        })
    return pd.DataFrame(recs).sort_values(["Difficulty", "micro_f1"], ascending=[True, False])

def evaluate_thresholds(df: pd.DataFrame, output_cols: List[str], thresholds: List[int], target: str) -> pd.DataFrame:
    rows = []
    for th in thresholds:
        summary_df, _ = evaluate_df(df, output_cols, threshold=float(th), target=target)
        for _, r in summary_df.iterrows():
            rows.append({
                "threshold": th,
                "model": r["model"],
                "micro_f1": r["micro_f1"],
                "micro_precision": r["micro_precision"],
                "micro_recall": r["micro_recall"],
                "top1_primary_first_acc": r["top1_primary_first_acc"],
                "avg_n_pred": r["avg_n_pred"],
                "target": target,
            })
    return pd.DataFrame(rows)

def bootstrap_ci(per_case_df: pd.DataFrame, n_boot: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    case_ids = per_case_df["note_id"].unique()
    models = per_case_df["model_col"].unique()
    n = len(case_ids)

    per_model = {}
    for m in models:
        g = per_case_df[per_case_df["model_col"] == m].set_index("note_id")
        per_model[m] = {
            "tp": g.loc[case_ids, "tp"].to_numpy(),
            "fp": g.loc[case_ids, "fp"].to_numpy(),
            "fn": g.loc[case_ids, "fn"].to_numpy(),
            "top1": g.loc[case_ids, "top1_primary_first"].to_numpy(),
            "anyprim": g.loc[case_ids, "any_primary"].to_numpy(),
        }

    boot_rows = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for m in models:
            tp = int(per_model[m]["tp"][idx].sum())
            fp = int(per_model[m]["fp"][idx].sum())
            fn = int(per_model[m]["fn"][idx].sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            top1 = float(per_model[m]["top1"][idx].mean())
            anyprim = float(per_model[m]["anyprim"][idx].mean())
            boot_rows.append((m, f1, prec, rec, top1, anyprim))

    boot_df = pd.DataFrame(boot_rows, columns=["model_col", "micro_f1", "micro_precision", "micro_recall", "top1_primary_first_acc", "any_primary_recall"])

    ci = []
    for m in models:
        g = boot_df[boot_df["model_col"] == m]
        for met in ["micro_f1", "micro_precision", "micro_recall", "top1_primary_first_acc", "any_primary_recall"]:
            ci.append({
                "model": m.replace("output_", ""),
                "metric": met,
                "mean": float(g[met].mean()),
                "ci_low": float(g[met].quantile(0.025)),
                "ci_high": float(g[met].quantile(0.975)),
            })
    return pd.DataFrame(ci)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Path to baseline CSV")
    ap.add_argument("--interactive", required=True, help="Path to interactive CSV")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--threshold", type=float, default=62.0, help="Similarity threshold (0-100)")
    ap.add_argument("--sweep", action="store_true", help="Run threshold sweep (55,60,62,65,70,75)")
    ap.add_argument("--bootstrap", type=int, default=0, help="Bootstrap iterations for CI (e.g., 2000). 0 disables.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    baseline_df = pd.read_csv(args.baseline)
    interactive_df = pd.read_csv(args.interactive)

    output_cols_baseline = [c for c in baseline_df.columns if c.startswith("output_")]
    output_cols_interactive = [c for c in interactive_df.columns if c.startswith("output_")]

    # audits
    newline_audit(baseline_df, output_cols_baseline).to_csv(os.path.join(args.outdir, "baseline_output_newline_audit.csv"), index=False)
    newline_audit(interactive_df, output_cols_interactive).to_csv(os.path.join(args.outdir, "interactive_output_newline_audit.csv"), index=False)

    # eval (primary diagnoses only)
    for scenario_name, df, outcols in [
        ("baseline", baseline_df, output_cols_baseline),
        ("interactive", interactive_df, output_cols_interactive),
    ]:
        target = "primary"
        summary_df, per_case_df = evaluate_df(df, outcols, threshold=args.threshold, target=target)
        summary_df.to_csv(os.path.join(args.outdir, f"{scenario_name}_summary_{target}.csv"), index=False)
        per_case_df.to_csv(os.path.join(args.outdir, f"{scenario_name}_per_case_{target}.csv"), index=False)
        summarize_by_difficulty(per_case_df).to_csv(os.path.join(args.outdir, f"{scenario_name}_by_difficulty_{target}.csv"), index=False)

        if args.bootstrap and args.bootstrap > 0:
            bootstrap_ci(per_case_df, n_boot=args.bootstrap, seed=42).to_csv(
                os.path.join(args.outdir, f"{scenario_name}_bootstrap_ci_{target}.csv"), index=False
            )

        if args.sweep:
            thresholds = [55, 60, 62, 65, 70, 75]
            evaluate_thresholds(df, outcols, thresholds, target=target).to_csv(
                os.path.join(args.outdir, f"{scenario_name}_threshold_sweep_{target}.csv"), index=False
            )

    # environment info
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "threshold": args.threshold,
        "baseline_csv": os.path.basename(args.baseline),
        "interactive_csv": os.path.basename(args.interactive),
    }
    with open(os.path.join(args.outdir, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)

    print(f"Wrote outputs to: {args.outdir}")

if __name__ == "__main__":
    main()
