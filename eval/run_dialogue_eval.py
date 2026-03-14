#!/usr/bin/env python3
"""
MedSyn Dialogue Evaluation (Part D)

Evaluates interactive dialogue quality across Session 2 and Session 4.
- Reuses existing turn-level and case-level CSVs from dial-eval/
- Builds clinical context from session CSVs for LLM-as-judge
- Runs faithfulness & answer-relevancy scoring via OpenRouter (gemini-2.5-flash)
- Generates publication-quality figures (400 DPI)
- Statistical comparisons: clinician vs resident, session 2 vs session 4

Usage:
    python run_dialogue_eval.py \
        --turn_csv eval/dial-eval/turn_level.csv \
        --case_csv eval/dial-eval/case_level.csv \
        --session2_notes eval/session_outputs/auto_eval_session2_interactive.csv \
        --session4_notes eval/session_outputs/auto_eval_session4_interactive.csv \
        --out_dir eval/results/dialogue \
        --run_llm_judge \
        --judge_model google/gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── OpenAI client (for OpenRouter) ──────────────────────────

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ── Helpers ─────────────────────────────────────────────────

def build_context(row: pd.Series) -> str:
    parts = []
    for col in ["chief_complaint", "history", "physical_exam", "results"]:
        val = row.get(col, "")
        if isinstance(val, str) and val.strip():
            header = col.replace("_", " ").title()
            parts.append(f"{header}:\n{val.strip()}")
    return "\n\n".join(parts)


def safe_json(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def cache_key(metric: str, q: str, a: str, ctx: str) -> str:
    return hashlib.sha256(f"{metric}\n{q}\n{a}\n{ctx}".encode()).hexdigest()


# ── LLM-as-Judge ───────────────────────────────────────────

def run_llm_judge(
    turns: pd.DataFrame,
    cache_path: Path,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> pd.DataFrame:
    """Score each turn for faithfulness and answer relevancy."""
    if OpenAI is None:
        raise RuntimeError("openai package required. pip install openai>=1.0")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={"X-Title": "MedSyn-Eval"},
    )

    # Load cache
    cache: Dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                cache[obj["key"]] = obj

    def score_one(metric: str, question: str, answer: str, context: str) -> Tuple[Optional[float], Optional[str]]:
        key = cache_key(metric, question, answer, context)
        if key in cache:
            return cache[key].get("score"), cache[key].get("rationale")

        if metric == "faithfulness":
            system = "You are a meticulous clinical documentation auditor."
            prompt = (
                "Score how faithful the ANSWER is to the CONTEXT on a scale of 0.0 to 1.0.\n"
                "Faithfulness = every factual claim in ANSWER must be supported by CONTEXT.\n"
                "If the answer correctly states information is not in the note, that is faithful.\n"
                "Return JSON only: {\"score\": <float>, \"rationale\": \"<1 sentence>\"}\n\n"
                f"CONTEXT:\n{context[:3000]}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer[:1500]}"
            )
        else:  # answer_relevancy
            system = "You are a meticulous clinical Q&A evaluator."
            prompt = (
                "Score how relevant the ANSWER is to the QUESTION on a scale of 0.0 to 1.0.\n"
                "High if it directly addresses the question. Low if mostly irrelevant content.\n"
                "Return JSON only: {\"score\": <float>, \"rationale\": \"<1 sentence>\"}\n\n"
                f"CONTEXT:\n{context[:3000]}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer[:1500]}"
            )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            js = safe_json(content)
            score = float(js["score"]) if js and "score" in js else None
            rationale = str(js.get("rationale", ""))[:500] if js else None
        except Exception as e:
            print(f"  Judge error: {e}")
            score, rationale = None, str(e)[:200]
            time.sleep(2)

        obj = {"key": key, "metric": metric, "score": score, "rationale": rationale, "model": model}
        cache[key] = obj
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return score, rationale

    n = len(turns)
    faith_s, faith_r, rel_s, rel_r = [], [], [], []
    for i, (_, row) in enumerate(turns.iterrows()):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Judging turn {i+1}/{n} ...")
        q, a, ctx = str(row["question"]), str(row["answer"]), str(row.get("context", ""))
        s1, r1 = score_one("faithfulness", q, a, ctx)
        s2, r2 = score_one("answer_relevancy", q, a, ctx)
        faith_s.append(s1)
        faith_r.append(r1)
        rel_s.append(s2)
        rel_r.append(r2)

    turns = turns.copy()
    turns["judge_faithfulness"] = faith_s
    turns["judge_faithfulness_rationale"] = faith_r
    turns["judge_answer_relevancy"] = rel_s
    turns["judge_answer_relevancy_rationale"] = rel_r
    return turns


# ── Statistical helpers ─────────────────────────────────────

def mann_whitney(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    # rank-biserial r as effect size
    n1, n2 = len(a), len(b)
    r = 1 - (2 * stat) / (n1 * n2)
    return p, r


def bootstrap_ci(a, b, n_boot=10000, seed=42):
    """Paired or unpaired bootstrap for mean difference."""
    rng = np.random.RandomState(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    obs_diff = np.nanmean(a) - np.nanmean(b)
    diffs = []
    for _ in range(n_boot):
        idx_a = rng.choice(len(a), len(a), replace=True)
        idx_b = rng.choice(len(b), len(b), replace=True)
        diffs.append(np.nanmean(a[idx_a]) - np.nanmean(b[idx_b]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs_diff, lo, hi


# ── Figure generation ───────────────────────────────────────

DPI = 400
COLORS = {"clinician": "#2196F3", "resident": "#FF9800"}
SESSION_COLORS = {2: "#4CAF50", 4: "#9C27B0"}


def fig_turns_per_case(case_df, fig_dir):
    """Bar chart: mean turns per case by group × session."""
    agg = case_df.groupby(["session", "group"])["turn_pairs"].agg(["mean", "std", "count"]).reset_index()
    agg["se"] = agg["std"] / np.sqrt(agg["count"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(2)
    w = 0.3
    for i, grp in enumerate(["clinician", "resident"]):
        sub = agg[agg["group"] == grp].sort_values("session")
        ax.bar(x + i * w, sub["mean"], w, yerr=sub["se"], color=COLORS[grp],
               label=grp.capitalize(), capsize=3, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(["Session 2", "Session 4"])
    ax.set_ylabel("Mean turns per case")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_dial_turns_per_case.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_question_categories(turn_df, fig_dir):
    """Stacked bar: question type proportions by group × session."""
    cats = ["detail_request", "info_request", "suggestion", "other"]
    cat_colors = {"detail_request": "#1976D2", "info_request": "#42A5F5",
                  "suggestion": "#FF7043", "other": "#BDBDBD"}

    grps = turn_df.groupby(["session", "group", "q_category"]).size().reset_index(name="n")
    totals = turn_df.groupby(["session", "group"]).size().reset_index(name="total")
    grps = grps.merge(totals, on=["session", "group"])
    grps["prop"] = grps["n"] / grps["total"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, sess in zip(axes, [2, 4]):
        sub = grps[grps["session"] == sess]
        bottom_c = np.zeros(1)
        bottom_r = np.zeros(1)
        for cat in cats:
            c_val = sub[(sub["group"] == "clinician") & (sub["q_category"] == cat)]["prop"].values
            r_val = sub[(sub["group"] == "resident") & (sub["q_category"] == cat)]["prop"].values
            c_val = c_val[0] if len(c_val) > 0 else 0
            r_val = r_val[0] if len(r_val) > 0 else 0
            ax.bar([0], [c_val], 0.5, bottom=bottom_c, color=cat_colors[cat], label=cat if sess == 2 else "")
            ax.bar([1], [r_val], 0.5, bottom=bottom_r, color=cat_colors[cat])
            bottom_c += c_val
            bottom_r += r_val
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Clinician", "Resident"])
        ax.set_title(f"Session {sess}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Proportion of questions")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_dial_question_categories.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_answer_quality(turn_df, fig_dir):
    """Box plots for answer specificity and context overlap by group × session."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for ax, metric, label in [
        (axes[0], "a_specificity_score", "Answer specificity"),
        (axes[1], "overlap_context", "Context grounding"),
    ]:
        data = []
        labels = []
        positions = []
        colors = []
        for j, sess in enumerate([2, 4]):
            for k, grp in enumerate(["clinician", "resident"]):
                vals = turn_df[(turn_df["session"] == sess) & (turn_df["group"] == grp)][metric].dropna()
                data.append(vals)
                labels.append(f"S{sess}\n{grp[:4].capitalize()}")
                positions.append(j * 3 + k)
                colors.append(COLORS[grp])

        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(fig_dir / "fig_dial_answer_quality.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_judge_scores(turn_df, fig_dir):
    """Box plots for LLM judge faithfulness and relevancy."""
    if "judge_faithfulness" not in turn_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, col, title in [
        (axes[0], "judge_faithfulness", "Faithfulness"),
        (axes[1], "judge_answer_relevancy", "Answer relevancy"),
    ]:
        data, labels, positions, colors = [], [], [], []
        for j, sess in enumerate([2, 4]):
            for k, grp in enumerate(["clinician", "resident"]):
                vals = turn_df[(turn_df["session"] == sess) & (turn_df["group"] == grp)][col].dropna()
                data.append(vals)
                labels.append(f"S{sess}\n{grp[:4].capitalize()}")
                positions.append(j * 3 + k)
                colors.append(COLORS[grp])

        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(f"Judge {title} (0-1)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_dial_judge_scores.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_judge_by_category(turn_df, fig_dir):
    """Mean judge scores by question category."""
    if "judge_faithfulness" not in turn_df.columns:
        return

    cats = ["detail_request", "info_request", "suggestion", "other"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, col, title in [
        (axes[0], "judge_faithfulness", "Faithfulness"),
        (axes[1], "judge_answer_relevancy", "Relevancy"),
    ]:
        means = []
        sems = []
        for cat in cats:
            vals = turn_df[turn_df["q_category"] == cat][col].dropna()
            means.append(vals.mean() if len(vals) > 0 else 0)
            sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        ax.bar(range(len(cats)), means, yerr=sems, capsize=3,
               color=["#1976D2", "#42A5F5", "#FF7043", "#BDBDBD"], edgecolor="white")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=7)
        ax.set_ylabel(f"Mean {title}")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_dial_judge_by_category.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_duration_vs_turns(case_df, fig_dir):
    """Scatter: case duration vs number of turns, colored by group."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for grp in ["clinician", "resident"]:
        sub = case_df[case_df["group"] == grp]
        ax.scatter(sub["turn_pairs"], sub["duration_sec"] / 60, alpha=0.5,
                   s=20, color=COLORS[grp], label=grp.capitalize())
    ax.set_xlabel("Number of turns")
    ax.set_ylabel("Duration (minutes)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_dial_duration_vs_turns.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="MedSyn Dialogue Evaluation (Part D)")
    ap.add_argument("--turn_csv", required=True, type=Path)
    ap.add_argument("--case_csv", required=True, type=Path)
    ap.add_argument("--session2_notes", required=True, type=Path)
    ap.add_argument("--session4_notes", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--run_llm_judge", action="store_true")
    ap.add_argument("--judge_model", default="google/gemini-2.5-flash")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load data
    turn_df = pd.read_csv(args.turn_csv)
    case_df = pd.read_csv(args.case_csv)
    print(f"Loaded {len(turn_df)} turns, {len(case_df)} cases")

    # Build context map from session notes
    s2 = pd.read_csv(args.session2_notes)
    s4 = pd.read_csv(args.session4_notes)
    ctx_map: Dict[Tuple[int, str], str] = {}
    for sess, notes_df in [(2, s2), (4, s4)]:
        for _, row in notes_df.iterrows():
            ctx_map[(sess, str(row["note_id"]))] = build_context(row)

    # Add context to turns for judge
    turn_df["context"] = turn_df.apply(
        lambda r: ctx_map.get((r["session"], str(r["note_id"])), ""), axis=1
    )

    # ── LLM Judge ────────────────────────────────────
    if args.run_llm_judge:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            # Try loading from .env.prod
            env_path = Path(__file__).parent.parent / ".env.prod"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("OPENROUTER_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found")

        print(f"\nRunning LLM-as-judge with {args.judge_model}")
        print(f"  {len(turn_df)} turns × 2 metrics = {len(turn_df)*2} API calls")
        cache_path = args.out_dir / "judge_cache.jsonl"
        turn_df = run_llm_judge(turn_df, cache_path, args.judge_model, api_key)

    # ── Descriptive statistics ───────────────────────
    print("\n" + "=" * 60)
    print("DIALOGUE DESCRIPTIVE STATISTICS")
    print("=" * 60)

    # Turns per case
    case_stats = case_df.groupby(["session", "group"]).agg(
        mean_turns=("turn_pairs", "mean"),
        std_turns=("turn_pairs", "std"),
        mean_dur=("duration_sec", lambda x: x.mean() / 60),
        std_dur=("duration_sec", lambda x: x.std() / 60),
        n_cases=("turn_pairs", "count"),
    ).round(2)
    print("\nCase-level summary:")
    print(case_stats.to_string())

    # Question categories
    q_dist = turn_df.groupby(["session", "group", "q_category"]).size().reset_index(name="n")
    totals = turn_df.groupby(["session", "group"]).size().reset_index(name="total")
    q_dist = q_dist.merge(totals, on=["session", "group"])
    q_dist["prop"] = (q_dist["n"] / q_dist["total"]).round(3)
    print("\nQuestion category distribution:")
    print(q_dist.to_string(index=False))

    # Answer quality heuristics
    aq = turn_df.groupby(["session", "group"]).agg(
        mean_specificity=("a_specificity_score", "mean"),
        mean_overlap=("overlap_context", "mean"),
        mean_words=("a_words", "mean"),
        prop_not_in_note=("a_not_in_note", "mean"),
    ).round(3)
    print("\nAnswer quality (heuristic):")
    print(aq.to_string())

    # ── Statistical comparisons ──────────────────────
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISONS")
    print("=" * 60)

    comparisons = []

    # Clinician vs Resident (pooled sessions)
    for metric_col in ["a_specificity_score", "overlap_context", "a_words"]:
        c_vals = turn_df[turn_df["group"] == "clinician"][metric_col].dropna()
        r_vals = turn_df[turn_df["group"] == "resident"][metric_col].dropna()
        p, r = mann_whitney(c_vals, r_vals)
        diff, lo, hi = bootstrap_ci(c_vals, r_vals, seed=args.seed)
        comparisons.append({
            "comparison": "clinician_vs_resident",
            "metric": metric_col,
            "mean_A": c_vals.mean(), "mean_B": r_vals.mean(),
            "diff": diff, "CI_lo": lo, "CI_hi": hi,
            "MW_p": p, "rank_biserial_r": r,
        })

    # Session 2 vs Session 4 (pooled groups)
    for metric_col in ["a_specificity_score", "overlap_context", "a_words"]:
        s2_vals = turn_df[turn_df["session"] == 2][metric_col].dropna()
        s4_vals = turn_df[turn_df["session"] == 4][metric_col].dropna()
        p, r = mann_whitney(s2_vals, s4_vals)
        diff, lo, hi = bootstrap_ci(s2_vals, s4_vals, seed=args.seed)
        comparisons.append({
            "comparison": "session2_vs_session4",
            "metric": metric_col,
            "mean_A": s2_vals.mean(), "mean_B": s4_vals.mean(),
            "diff": diff, "CI_lo": lo, "CI_hi": hi,
            "MW_p": p, "rank_biserial_r": r,
        })

    # Turns: clinician vs resident
    c_turns = case_df[case_df["group"] == "clinician"]["turn_pairs"]
    r_turns = case_df[case_df["group"] == "resident"]["turn_pairs"]
    p, r = mann_whitney(c_turns, r_turns)
    diff, lo, hi = bootstrap_ci(c_turns, r_turns, seed=args.seed)
    comparisons.append({
        "comparison": "clinician_vs_resident",
        "metric": "turn_pairs",
        "mean_A": c_turns.mean(), "mean_B": r_turns.mean(),
        "diff": diff, "CI_lo": lo, "CI_hi": hi,
        "MW_p": p, "rank_biserial_r": r,
    })

    # Judge scores comparisons
    if "judge_faithfulness" in turn_df.columns:
        for metric_col in ["judge_faithfulness", "judge_answer_relevancy"]:
            # Clinician vs Resident
            c_vals = turn_df[turn_df["group"] == "clinician"][metric_col].dropna()
            r_vals = turn_df[turn_df["group"] == "resident"][metric_col].dropna()
            p, r = mann_whitney(c_vals, r_vals)
            diff, lo, hi = bootstrap_ci(c_vals, r_vals, seed=args.seed)
            comparisons.append({
                "comparison": "clinician_vs_resident",
                "metric": metric_col,
                "mean_A": c_vals.mean(), "mean_B": r_vals.mean(),
                "diff": diff, "CI_lo": lo, "CI_hi": hi,
                "MW_p": p, "rank_biserial_r": r,
            })

            # Session 2 vs 4
            s2_vals = turn_df[turn_df["session"] == 2][metric_col].dropna()
            s4_vals = turn_df[turn_df["session"] == 4][metric_col].dropna()
            p, r = mann_whitney(s2_vals, s4_vals)
            diff, lo, hi = bootstrap_ci(s2_vals, s4_vals, seed=args.seed)
            comparisons.append({
                "comparison": "session2_vs_session4",
                "metric": metric_col,
                "mean_A": s2_vals.mean(), "mean_B": s4_vals.mean(),
                "diff": diff, "CI_lo": lo, "CI_hi": hi,
                "MW_p": p, "rank_biserial_r": r,
            })

            # By question category
            for cat in ["detail_request", "info_request", "suggestion"]:
                cat_vals = turn_df[turn_df["q_category"] == cat][metric_col].dropna()
                other_vals = turn_df[turn_df["q_category"] != cat][metric_col].dropna()
                if len(cat_vals) >= 5:
                    p, r = mann_whitney(cat_vals, other_vals)
                    comparisons.append({
                        "comparison": f"{cat}_vs_rest",
                        "metric": metric_col,
                        "mean_A": cat_vals.mean(), "mean_B": other_vals.mean(),
                        "diff": cat_vals.mean() - other_vals.mean(),
                        "CI_lo": np.nan, "CI_hi": np.nan,
                        "MW_p": p, "rank_biserial_r": r,
                    })

    comp_df = pd.DataFrame(comparisons).round(4)
    print("\nStatistical tests:")
    for _, row in comp_df.iterrows():
        sig = "***" if row["MW_p"] < 0.001 else "**" if row["MW_p"] < 0.01 else "*" if row["MW_p"] < 0.05 else ""
        print(f"  {row['comparison']:30s} | {row['metric']:25s} | Δ={row['diff']:+.3f} | p={row['MW_p']:.4f} {sig}")

    # ── Save outputs ─────────────────────────────────
    turn_df.to_csv(args.out_dir / "turn_level_scored.csv", index=False)
    case_stats.to_csv(args.out_dir / "case_summary.csv")
    q_dist.to_csv(args.out_dir / "question_categories.csv", index=False)
    aq.to_csv(args.out_dir / "answer_quality_heuristic.csv")
    comp_df.to_csv(args.out_dir / "statistical_comparisons.csv", index=False)

    # ── Judge summary ────────────────────────────────
    if "judge_faithfulness" in turn_df.columns:
        judge_summary = turn_df.groupby(["session", "group"]).agg(
            mean_faithfulness=("judge_faithfulness", "mean"),
            std_faithfulness=("judge_faithfulness", "std"),
            mean_relevancy=("judge_answer_relevancy", "mean"),
            std_relevancy=("judge_answer_relevancy", "std"),
            n=("judge_faithfulness", "count"),
        ).round(3)
        print("\n" + "=" * 60)
        print("LLM JUDGE SUMMARY")
        print("=" * 60)
        print(judge_summary.to_string())
        judge_summary.to_csv(args.out_dir / "judge_summary.csv")

        # Overall
        f_all = turn_df["judge_faithfulness"].dropna()
        r_all = turn_df["judge_answer_relevancy"].dropna()
        print(f"\n  Overall faithfulness:  {f_all.mean():.3f} ± {f_all.std():.3f} (n={len(f_all)})")
        print(f"  Overall relevancy:    {r_all.mean():.3f} ± {r_all.std():.3f} (n={len(r_all)})")

    # ── Figures ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    fig_turns_per_case(case_df, fig_dir)
    print("  fig_dial_turns_per_case.png")

    fig_question_categories(turn_df, fig_dir)
    print("  fig_dial_question_categories.png")

    fig_answer_quality(turn_df, fig_dir)
    print("  fig_dial_answer_quality.png")

    fig_judge_scores(turn_df, fig_dir)
    if "judge_faithfulness" in turn_df.columns:
        print("  fig_dial_judge_scores.png")

    fig_judge_by_category(turn_df, fig_dir)
    if "judge_faithfulness" in turn_df.columns:
        print("  fig_dial_judge_by_category.png")

    fig_duration_vs_turns(case_df, fig_dir)
    print("  fig_dial_duration_vs_turns.png")

    # Save config
    config = {
        "judge_model": args.judge_model if args.run_llm_judge else None,
        "n_turns": len(turn_df),
        "n_cases": len(case_df),
        "run_llm_judge": args.run_llm_judge,
        "seed": args.seed,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\nAll outputs saved to {args.out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
