#!/usr/bin/env python3
"""
MedSyn dialogue interaction analysis (Session 2 & Session 4)

Outputs:
- case_level.csv: one row per (user, case)
- turn_level.csv: one row per (user question -> MedSyn answer) turn
- question_category_distribution.csv
- figures/*.png

Optional:
- LLM-as-judge scoring (OpenRouter): per-turn faithfulness & answer relevancy in [0,1]
  Enable with --run_llm_judge and OPENROUTER_API_KEY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional dependency (installed in many envs). We keep it optional.
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None


# -------------------------
# Text utilities
# -------------------------

def norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\-\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


LAB_TERMS = set("""
wbc hgb hb hematocrit plt platelet sodium na potassium k chloride cl bun creatinine cr glucose lactate troponin
bilirubin alt ast alk phos lipase amylase inr ptt pt d-dimer ddimer crp esr
""".split())

IMAGING_TERMS = set("ct mri ultrasound us x-ray xray cxr ekg ecg angiography echo tte".split())
ANATOMY_TERMS = set("abdomen abdominal pelvic pelvis chest lung heart brain head neck arm leg right left".split())
SUGGEST_TERMS = set("diagnosis differential suggest impression likely think could possible cause etiology".split())
NEXTSTEP_TERMS = set("treatment manage management next step workup recommend advice should".split())


def infer_user_id_from_filename(fn: str) -> str:
    # e.g. dialogue_interactive__phy1__phy1_session2_merged.jsonl
    m = re.search(r"dialogue_interactive__([^_]+)__", fn)
    return m.group(1) if m else fn


def infer_group(user_id: str) -> str:
    uid = (user_id or "").lower()
    if uid.startswith("res"):
        return "resident"
    if uid.startswith("phy") or uid.startswith("clin") or uid.startswith("sen"):
        return "clinician"
    return "unknown"


def classify_user_msg(msg: str) -> Tuple[str, int]:
    """
    Heuristic 4-way taxonomy + simple detail score.

    Categories:
      - detail_request: asks for a specific lab/imaging/report
      - info_request: asks for general clinical info
      - suggestion: asks for interpretation/differential/next steps
      - other: acknowledgements/ambiguous/short prompts
    """
    t = norm_text(msg)
    tokens = set(t.split())

    is_suggestion = any(w in tokens for w in SUGGEST_TERMS) or "what is the diagnosis" in t
    is_nextstep = any(w in tokens for w in NEXTSTEP_TERMS) or "what should i do" in t
    is_info = any(w in tokens for w in (LAB_TERMS | IMAGING_TERMS | ANATOMY_TERMS)) or any(
        kw in t
        for kw in [
            "age",
            "sex",
            "history",
            "symptom",
            "vital",
            "bp",
            "heart rate",
            "resp",
            "temp",
            "exam",
            "findings",
            "report",
            "results",
        ]
    )

    detailed = any(w in tokens for w in (LAB_TERMS | IMAGING_TERMS)) or any(
        kw in t for kw in ["ct", "mri", "ultrasound", "x-ray", "troponin", "wbc", "creatinine"]
    )

    category = "suggestion" if (is_suggestion or is_nextstep) and not detailed else "detail_request" if detailed else "info_request" if is_info else "other"
    num_mentions = len(re.findall(r"\b\d+(\.\d+)?\b", msg))
    detail_score = int(detailed) + int(any(w in tokens for w in IMAGING_TERMS)) + int(any(w in tokens for w in LAB_TERMS)) + num_mentions
    return category, detail_score


def extract_primary(dx_text: str) -> str:
    t = (dx_text or "").lower()
    m = re.search(r"primary:\s*(.*)", t)
    if not m:
        return t.strip()
    after = m.group(1)
    after = re.split(r"\s+secondary:\s*", after)[0]
    return after.strip()


def build_context(note_row: pd.Series) -> str:
    parts: List[str] = []
    for col in ["chief_complaint", "history", "physical_exam", "results"]:
        val = note_row.get(col, "")
        if isinstance(val, str) and val.strip():
            header = col.replace("_", " ").title()
            parts.append(f"{header}:\n{val.strip()}")
    return "\n\n".join(parts)


def overlap_score(text: str, context: str) -> float:
    def content_tokens(s: str) -> List[str]:
        s = norm_text(s)
        toks = [
            w
            for w in s.split()
            if len(w) > 3 and w not in {"with", "from", "that", "this", "have", "were", "been", "also", "into", "only", "note", "clinical"}
        ]
        return toks

    a = set(content_tokens(text))
    c = set(content_tokens(context))
    if not a:
        return 0.0
    return len(a & c) / len(a)


def answer_specificity(answer: str) -> Tuple[float, Dict[str, int]]:
    """
    A lightweight, deterministic proxy for "specificity" that does not equate verbosity with detail.
    Higher when the answer contains:
      - numbers (labs/vitals)
      - medical terms (labs/imaging/anatomy or long technical tokens)
      - explicit note anchoring
    """
    t = norm_text(answer)
    tokens = t.split()
    token_set = set(tokens)
    n_words = len(tokens)
    n_nums = len(re.findall(r"\b\d+(\.\d+)?\b", answer))

    has_note = 1 if ("clinical note" in t or "per clinical note" in t or "from clinical note" in t or "as documented" in t) else 0
    has_lab = 1 if any(w in token_set for w in LAB_TERMS) else 0
    has_img = 1 if any(w in token_set for w in IMAGING_TERMS) else 0

    medish = set([w for w in tokens if w in LAB_TERMS or w in IMAGING_TERMS or w in ANATOMY_TERMS or len(w) > 10])
    n_medish = len(medish)

    score = (
        0.15 * min(n_words / 80, 1.0)
        + 0.35 * min(n_nums, 6) / 6
        + 0.35 * min(n_medish, 10) / 10
        + 0.15 * (has_note + has_lab + has_img) / 3
    )
    return float(score), {"n_words": n_words, "n_nums": n_nums, "n_medish": n_medish, "has_note": has_note, "has_lab": has_lab, "has_img": has_img}


def is_not_in_note(ans: str) -> bool:
    t = (ans or "").lower()
    return ("not described" in t) or ("not specified" in t) or ("no mention" in t) or ("not documented" in t)


# -------------------------
# Dialogue parsing
# -------------------------

def load_jsonl(path: Path, session: int) -> pd.DataFrame:
    user_id = infer_user_id_from_filename(path.name)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            obj["file"] = path.name
            obj["session"] = session
            obj["user_id"] = user_id
            obj["group"] = infer_group(user_id)
            rows.append(obj)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["case_index"] = df["case_index"].astype(int)
    return df


def extract_final_from_case_end(case_df: pd.DataFrame) -> Optional[str]:
    ends = case_df[(case_df["sender"] == "system") & (case_df["message"].astype(str).str.startswith("CASE_END"))]
    if len(ends) == 0:
        return None
    msg = ends.sort_values("timestamp").iloc[-1]["message"]
    m = re.search(r"final_answer=(.*)$", msg)
    return m.group(1).strip() if m else None


def extract_turn_pairs(case_df: pd.DataFrame) -> List[Tuple[pd.Series, pd.Series]]:
    case_df = case_df.sort_values("timestamp")
    pairs: List[Tuple[pd.Series, pd.Series]] = []
    pending_user = None
    for _, row in case_df.iterrows():
        if row["sender"] == "user":
            if str(row["message"]).lower().startswith("final answer"):
                pending_user = None
                continue
            pending_user = row
        elif row["sender"] == "llm":
            if pending_user is not None:
                pairs.append((pending_user, row))
                pending_user = None
    return pairs


# -------------------------
# OpenRouter judge (optional)
# -------------------------

@dataclass
class JudgeConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 500


def _hash_key(metric: str, question: str, answer: str, context: str) -> str:
    payload = (metric + "\n" + question + "\n" + answer + "\n" + context).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_json_load(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        # try to extract first json object
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def run_openrouter_judge(
    turns: pd.DataFrame,
    cache_path: Path,
    config: JudgeConfig,
    http_referer: Optional[str] = None,
    x_title: Optional[str] = None,
) -> pd.DataFrame:
    """
    Adds two columns:
      - judge_faithfulness
      - judge_answer_relevancy
    plus rationales.

    Requires OPENROUTER_API_KEY and openai>=1.x.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not available. Install openai>=1.x.")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    headers = {}
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers=headers or None,
    )

    cache: Dict[str, dict] = {}
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cache[obj["key"]] = obj

    def score_one(metric: str, question: str, answer: str, context: str) -> Tuple[Optional[float], Optional[str]]:
        key = _hash_key(metric, question, answer, context)
        if key in cache:
            return cache[key].get("score"), cache[key].get("rationale")

        if metric == "faithfulness":
            system = "You are a meticulous clinical documentation auditor."
            user = f"""Score how faithful the ANSWER is to the CONTEXT. 
Faithfulness means: every factual claim in ANSWER must be supported by CONTEXT. 
Return JSON only: {{"score": <float 0..1>, "rationale": "<short>"}}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}
"""
        elif metric == "answer_relevancy":
            system = "You are a meticulous evaluator of question-answer relevance."
            user = f"""Score how relevant the ANSWER is to the QUESTION given CONTEXT.
High if it directly addresses the question with minimal irrelevant content.
Return JSON only: {{"score": <float 0..1>, "rationale": "<short>"}}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}
"""
        else:
            raise ValueError(metric)

        resp = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        content = resp.choices[0].message.content or ""
        js = _safe_json_load(content)
        score = None
        rationale = None
        if js and "score" in js:
            try:
                score = float(js["score"])
            except Exception:
                score = None
            rationale = str(js.get("rationale", ""))[:1000]
        cache_obj = {"key": key, "metric": metric, "score": score, "rationale": rationale, "model": config.model}
        cache[key] = cache_obj
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cache_obj, ensure_ascii=False) + "\n")
        return score, rationale

    faith_scores: List[Optional[float]] = []
    faith_rat: List[Optional[str]] = []
    rel_scores: List[Optional[float]] = []
    rel_rat: List[Optional[str]] = []

    for _, r in turns.iterrows():
        q = str(r["question"])
        a = str(r["answer"])
        ctx = str(r["context"])
        s1, r1 = score_one("faithfulness", q, a, ctx)
        s2, r2 = score_one("answer_relevancy", q, a, ctx)
        faith_scores.append(s1)
        faith_rat.append(r1)
        rel_scores.append(s2)
        rel_rat.append(r2)

    turns = turns.copy()
    turns["judge_faithfulness"] = faith_scores
    turns["judge_faithfulness_rationale"] = faith_rat
    turns["judge_answer_relevancy"] = rel_scores
    turns["judge_answer_relevancy_rationale"] = rel_rat
    return turns


# -------------------------
# Main analysis
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session2_dir", required=True, type=Path, help="Directory containing merged Session 2 JSONL files")
    ap.add_argument("--session4_dir", required=True, type=Path, help="Directory containing merged Session 4 JSONL files")
    ap.add_argument("--session2_notes_csv", required=True, type=Path, help="session2_interactive.csv with clinical notes + ground truth")
    ap.add_argument("--session4_notes_csv", required=True, type=Path, help="session4_interactive.csv with clinical notes + ground truth")
    ap.add_argument("--out_dir", required=True, type=Path, help="Output directory")

    ap.add_argument("--run_llm_judge", action="store_true", help="Run OpenRouter LLM-as-judge scoring (faithfulness & answer relevancy)")
    ap.add_argument("--judge_model", default=os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.2-chat"), help="OpenRouter judge model (provider/model)")
    ap.add_argument("--judge_temp", type=float, default=0.0)
    ap.add_argument("--judge_max_tokens", type=int, default=500)
    ap.add_argument("--http_referer", default=os.environ.get("OPENROUTER_HTTP_REFERER"), help="Optional OpenRouter HTTP-Referer header")
    ap.add_argument("--x_title", default=os.environ.get("OPENROUTER_X_TITLE"), help="Optional OpenRouter X-Title header")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load notes
    s2 = pd.read_csv(args.session2_notes_csv)
    s4 = pd.read_csv(args.session4_notes_csv)

    ctx_map: Dict[Tuple[int, str], str] = {}
    gt_map: Dict[Tuple[int, str], Dict[str, str]] = {}
    for sess, df_notes in [(2, s2), (4, s4)]:
        for _, row in df_notes.iterrows():
            note_id = row["note_id"]
            ctx_map[(sess, note_id)] = build_context(row)
            gt_map[(sess, note_id)] = {
                "Difficulty": str(row.get("Difficulty", "")),
                "primary_dx": extract_primary(str(row.get("discharge diagnosis", ""))),
            }

    # Load dialogues
    all_dfs: List[pd.DataFrame] = []
    for p in sorted(args.session2_dir.glob("*.jsonl")):
        all_dfs.append(load_jsonl(p, 2))
    for p in sorted(args.session4_dir.glob("*.jsonl")):
        all_dfs.append(load_jsonl(p, 4))
    df = pd.concat(all_dfs, ignore_index=True)

    df["is_case_start"] = (df["sender"] == "system") & df["message"].astype(str).str.startswith("CASE_START")
    df["is_case_end"] = (df["sender"] == "system") & df["message"].astype(str).str.startswith("CASE_END")

    # Case-level metrics
    case_rows = []
    for (session, user_id, group, case_index, note_id), g in df.groupby(["session", "user_id", "group", "case_index", "note_id"]):
        g = g.sort_values("timestamp")
        start_ts = g.loc[g["is_case_start"], "timestamp"].min()
        end_ts = g.loc[g["is_case_end"], "timestamp"].max()
        dur = (end_ts - start_ts).total_seconds() if pd.notnull(start_ts) and pd.notnull(end_ts) else np.nan

        user_msgs = g[g["sender"] == "user"]
        llm_msgs = g[g["sender"] == "llm"]
        user_nonfinal = user_msgs[~user_msgs["message"].astype(str).str.lower().str.startswith("final answer")]

        final_ans = extract_final_from_case_end(g)
        gt = gt_map.get((session, note_id), {})
        primary_dx = gt.get("primary_dx", "")
        difficulty = gt.get("Difficulty", "")

        case_rows.append(
            {
                "session": session,
                "user_id": user_id,
                "group": group,
                "case_index": int(case_index),
                "note_id": note_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_sec": dur,
                "n_user_msgs": int(len(user_msgs)),
                "n_user_nonfinal": int(len(user_nonfinal)),
                "n_llm_msgs": int(len(llm_msgs)),
                "turn_pairs": int(min(len(user_nonfinal), len(llm_msgs))),
                "final_answer": final_ans,
                "primary_dx": primary_dx,
                "Difficulty": difficulty,
            }
        )
    case_df = pd.DataFrame(case_rows)

    # Turn-level metrics (with context)
    turn_rows = []
    for (session, user_id, case_index, note_id), g in df.groupby(["session", "user_id", "case_index", "note_id"]):
        ctx = ctx_map.get((session, note_id), "")
        pairs = extract_turn_pairs(g)
        for i, (u, l) in enumerate(pairs, start=1):
            q = str(u["message"])
            a = str(l["message"])
            q_cat, q_detail = classify_user_msg(q)
            spec, spec_feats = answer_specificity(a)
            turn_rows.append(
                {
                    "session": session,
                    "user_id": user_id,
                    "group": infer_group(user_id),
                    "case_index": int(case_index),
                    "note_id": note_id,
                    "turn_index": i,
                    "question": q,
                    "answer": a,
                    "context": ctx,
                    "q_category": q_cat,
                    "q_detail_score": q_detail,
                    "a_specificity": spec,
                    "a_words": spec_feats["n_words"],
                    "a_nums": spec_feats["n_nums"],
                    "a_medish": spec_feats["n_medish"],
                    "a_has_note": spec_feats["has_note"],
                    "a_has_lab": spec_feats["has_lab"],
                    "a_has_img": spec_feats["has_img"],
                    "a_not_in_note": is_not_in_note(a),
                    "overlap_context": overlap_score(a, ctx) if ctx else np.nan,
                    "overlap_question": overlap_score(a, q),
                }
            )
    turn_df = pd.DataFrame(turn_rows)

    # Optional judge
    if args.run_llm_judge:
        cache_path = args.out_dir / "judge_cache.jsonl"
        config = JudgeConfig(model=args.judge_model, temperature=args.judge_temp, max_tokens=args.judge_max_tokens)
        turn_df = run_openrouter_judge(turn_df, cache_path, config, http_referer=args.http_referer, x_title=args.x_title)

    # Question category distribution (based on user messages)
    user_df = df[(df["sender"] == "user") & ~df["message"].astype(str).str.lower().str.startswith("final answer")].copy()
    user_df[["q_category", "q_detail_score"]] = user_df["message"].astype(str).apply(lambda m: pd.Series(classify_user_msg(m)))

    q_cat = user_df.groupby(["session", "group", "q_category"]).size().reset_index(name="n")
    totals = user_df.groupby(["session", "group"]).size().reset_index(name="total")
    q_cat = q_cat.merge(totals, on=["session", "group"])
    q_cat["prop"] = q_cat["n"] / q_cat["total"]

    # Save CSV outputs
    case_df.to_csv(args.out_dir / "case_level.csv", index=False)
    turn_df.to_csv(args.out_dir / "turn_level.csv", index=False)
    q_cat.to_csv(args.out_dir / "question_category_distribution.csv", index=False)

    # Figures
    # 1) turns per case
    fig, ax = plt.subplots(figsize=(6, 4))
    case_df.groupby(["session", "group"])["turn_pairs"].mean().unstack().plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean turns per case")
    ax.set_title("Turns per case by group and session")
    ax.legend(title="Group")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_turns_by_group_session.png", dpi=200)
    plt.close(fig)

    # 2) question category distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    prop_table = q_cat.pivot_table(index=["session", "group"], columns="q_category", values="prop").fillna(0)
    prop_table.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Proportion of questions")
    ax.set_title("Physician question types by group and session")
    ax.legend(title="Question category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_question_categories.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) context overlap
    fig, ax = plt.subplots(figsize=(6, 4))
    turn_df.groupby(["session", "group"])["overlap_context"].mean().unstack().plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean context overlap (0–1)")
    ax.set_title("Context grounding proxy by group and session")
    plt.tight_layout()
    fig.savefig(fig_dir / "fig_context_overlap.png", dpi=200)
    plt.close(fig)

    print(f"Done. Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
