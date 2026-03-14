#!/usr/bin/env python3
"""
Evaluate MedSyn dialogue logs (baseline + interactive).

- Reads dialogue_*.jsonl from <base-output-dir>/dialogues
- Reads out_*.csv from <base-output-dir>
- Aggregates per-case metrics:
    * number of user turns (with and without final answer)
    * number of LLM turns
    * character counts (proxy for verbosity)
    * final answer from logs
    * CSV info (discharge diagnosis, real_phy_answer, time_spent_seconds)
- Produces:
    * <base-output-dir>/analysis/case_level_metrics.csv
    * <base-output-dir>/analysis/summary_by_mode.csv
"""

from pathlib import Path
import argparse
import json
import re

import pandas as pd


# ---------------------------------------------------------------------
# Loading dialogue JSONL and extracting per-case metrics
# ---------------------------------------------------------------------


def load_dialogue_jsonl(path: Path) -> pd.DataFrame:
    """Load one dialogue_*.jsonl file into a DataFrame."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def extract_cases_from_dialogue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given one session's dialogue DataFrame, extract per-case aggregates.

    We use system markers:
      - CASE_START: note_id=...
      - CASE_END:   final_answer=...

    For interactive mode:
      - Final user turn usually starts with "final answer" or "final diagnosis".

    For baseline mode:
      - Users type only the diagnosis (no prefix), so we treat the last user
        turn in the case as "final" if there is a CASE_END marker.

    Returns one row per case with metrics like:
      - n_user_turns_total
      - n_user_turns_nonfinal
      - n_llm_turns
      - user_chars_total, llm_chars_total
      - final_answer_from_log
    """
    required_cols = {
        "sender",
        "message",
        "note_id",
        "session_id",
        "experiment_mode",
        "case_index",
        "question_number",
    }
    if df.empty:
        return pd.DataFrame()
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dialogue dataframe missing required columns: {required_cols - set(df.columns)}")

    # Find all CASE_START markers
    is_case_start = (df["sender"] == "system") & df["message"].str.startswith("CASE_START")
    start_indices = list(df.index[is_case_start])

    if not start_indices:
        return pd.DataFrame()

    case_rows = []

    for i, start_idx in enumerate(start_indices):
        # Segment from this CASE_START up to (but not including) next CASE_START
        end_idx = start_indices[i + 1] if i + 1 < len(start_indices) else len(df)
        seg = df.iloc[start_idx:end_idx].copy()

        note_id = seg["note_id"].iloc[0]
        session_id = seg["session_id"].iloc[0]
        mode = seg["experiment_mode"].iloc[0]
        case_index = int(seg["case_index"].iloc[0])
        question_number = int(seg["question_number"].iloc[0])

        # CASE_END marker (if present)
        system_end = seg[
            (seg["sender"] == "system")
            & seg["message"].str.startswith("CASE_END")
        ]
        final_answer = None
        if not system_end.empty:
            msg = system_end["message"].iloc[0]
            m = re.search(r"CASE_END:\s*final_answer=(.*)", msg)
            if m:
                final_answer = m.group(1).strip()

        # User & LLM messages
        user_msgs = seg[seg["sender"] == "user"]["message"].astype(str)
        llm_msgs = seg[seg["sender"] == "llm"]["message"].astype(str)

        # Identify user turn(s) that are final answers.
        # 1) Interactive: explicit "final answer"/"final diagnosis" prefix
        user_is_final = user_msgs.str.lower().str.startswith(("final answer", "final diagnosis"))

        # 2) Baseline: no prefix; if we have a CASE_END and at least one user msg,
        #    treat the *last* user turn as final.
        if not user_is_final.any() and not system_end.empty and len(user_msgs) > 0:
            user_is_final.iloc[-1] = True

        user_nonfinal = user_msgs[~user_is_final]

        case_rows.append(
            {
                "experiment_mode": mode,
                "session_id": session_id,
                "case_index": case_index,
                "question_number": question_number,
                "note_id": note_id,
                "n_user_turns_total": int(len(user_msgs)),
                "n_user_turns_nonfinal": int(len(user_nonfinal)),
                "n_llm_turns": int(len(llm_msgs)),
                "user_chars_total": int(user_msgs.str.len().sum()),
                "llm_chars_total": int(llm_msgs.str.len().sum()),
                "final_answer_from_log": final_answer,
            }
        )

    return pd.DataFrame(case_rows)


# ---------------------------------------------------------------------
# Matching dialogue files to CSVs
# ---------------------------------------------------------------------


def parse_dialogue_filename(path: Path):
    """
    From dialogue_interactive__user__session.jsonl
    return (mode, user, session).
    """
    stem = path.stem  # e.g. "dialogue_interactive__sml-admin__sml-admin_20251209_001427"
    parts = stem.split("__")
    if len(parts) != 3 or not parts[0].startswith("dialogue_"):
        raise ValueError(f"Unexpected dialogue filename: {path.name}")
    mode = parts[0].replace("dialogue_", "", 1)
    user = parts[1]
    session = parts[2]
    return mode, user, session


def find_matching_csv(dialogue_path: Path, csv_dir: Path):
    """
    Given a dialogue_*.jsonl path, find the corresponding out_*.csv
    in csv_dir, or return None if no match.

    Pattern:
      dialogue_{mode}__{user}__{session}.jsonl  ->
      out_{mode}_ass_*__{user}__{session}*.csv
    """
    mode, user, session = parse_dialogue_filename(dialogue_path)

    pattern = f"out_{mode}_ass_*__{user}__{session}*.csv"
    candidates = list(csv_dir.glob(pattern))
    if not candidates:
        return None

    # Prefer non-partial (no "__partial" in name) as final full CSV
    full = [c for c in candidates if "__partial" not in c.name]
    if full:
        # If multiple, pick the last one lexicographically
        return sorted(full)[-1]

    # Fallback: pick the most recently modified
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_session(dialogue_path: Path, csv_dir: Path) -> pd.DataFrame:
    """
    Load a single session:
      - dialogue JSONL -> per-case metrics
      - join with CSV on note_id (if CSV exists)
    """
    df_d = load_dialogue_jsonl(dialogue_path)
    df_cases = extract_cases_from_dialogue(df_d)
    if df_cases.empty:
        return df_cases

    csv_path = find_matching_csv(dialogue_path, csv_dir)
    if csv_path is None:
        # No CSV found: return only dialogue metrics
        return df_cases

    df_csv = pd.read_csv(csv_path)

    merged = df_cases.merge(
        df_csv,
        how="left",
        on="note_id",
        suffixes=("", "_csv"),
    )

    # Convenience aliases
    if "discharge diagnosis" in merged.columns:
        merged["ground_truth_discharge"] = merged["discharge diagnosis"]

    if "real_phy_answer" in merged.columns:
        merged["final_answer_from_csv"] = merged["real_phy_answer"]

    # time_spent_seconds is already a column if your pipeline added it
    return merged


def load_all_sessions(base_output_dir: Path) -> pd.DataFrame:
    """
    Load all dialogue_*.jsonl under <base-output-dir>/dialogues and
    stack them into a single per-case DataFrame.
    """
    dialogues_dir = base_output_dir / "dialogues"
    csv_dir = base_output_dir

    if not dialogues_dir.exists():
        raise FileNotFoundError(f"Dialogue directory not found: {dialogues_dir}")

    all_rows = []

    for path in sorted(dialogues_dir.glob("dialogue_*.jsonl")):
        try:
            df_session = load_session(path, csv_dir)
            if df_session is None or df_session.empty:
                continue
            df_session = df_session.assign(dialogue_file=path.name)
            all_rows.append(df_session)
        except Exception as e:
            print(f"[ERROR] Failed to process {path.name}: {e}")

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------
# CLI + summary
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MedSyn baseline + interactive dialogue logs."
    )
    parser.add_argument(
        "--base-output-dir",
        "-o",
        type=str,
        default="output",
        help="Base output directory containing 'dialogues/' and 'out_*.csv' files (default: ./output)",
    )
    parser.add_argument(
        "--analysis-dir",
        "-a",
        type=str,
        default=None,
        help="Directory to write aggregated CSVs (default: <base-output-dir>/analysis)",
    )

    args = parser.parse_args()

    base_output_dir = Path(args.base_output_dir).resolve()
    analysis_dir = Path(args.analysis_dir).resolve() if args.analysis_dir else (base_output_dir / "analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base output dir: {base_output_dir}")
    print(f"Analysis dir:    {analysis_dir}")

    all_cases = load_all_sessions(base_output_dir)

    if all_cases.empty:
        print("No dialogue cases found.")
        return

    print(f"\nLoaded {len(all_cases)} cases from {all_cases['session_id'].nunique()} sessions.\n")

    has_time = "time_spent_seconds" in all_cases.columns

    # Summary by experiment_mode (baseline vs interactive)
    print("Summary by experiment_mode:\n")

    rows = []
    for mode in sorted(all_cases["experiment_mode"].dropna().unique()):
        sub = all_cases[all_cases["experiment_mode"] == mode]

        row = {
            "experiment_mode": mode,
            "n_sessions": int(sub["session_id"].nunique()),
            "n_cases": int(len(sub)),
            "avg_user_turns_nonfinal": sub["n_user_turns_nonfinal"].mean(),
            "avg_llm_turns": sub["n_llm_turns"].mean(),
            "avg_user_chars_total": sub["user_chars_total"].mean(),
            "avg_llm_chars_total": sub["llm_chars_total"].mean(),
        }
        if has_time:
            row["avg_time_spent_seconds"] = sub["time_spent_seconds"].mean()

        rows.append(row)

        print(f"- {mode}: {row['n_cases']} cases in {row['n_sessions']} sessions")
        print(f"    avg user turns (non-final): {row['avg_user_turns_nonfinal']:.2f}")
        print(f"    avg LLM turns:             {row['avg_llm_turns']:.2f}")
        if has_time:
            print(f"    avg time_spent_seconds:    {row['avg_time_spent_seconds']:.1f}")
        print()

    summary_df = pd.DataFrame(rows)

    # Save outputs
    all_cases_path = analysis_dir / "case_level_metrics.csv"
    summary_path = analysis_dir / "summary_by_mode.csv"

    all_cases.to_csv(all_cases_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved per-case metrics to:   {all_cases_path}")
    print(f"Saved summary-by-mode to:    {summary_path}")


if __name__ == "__main__":
    main()
