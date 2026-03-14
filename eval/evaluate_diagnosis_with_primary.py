import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def normalize_text(text: str) -> str:
    """Lowercase, strip and simplify whitespace/punctuation a bit."""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[\n\r]+", " ", text)
    # Keep only basic separators and alphanumerics
    text = re.sub(r"[^a-z0-9\s,;/\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_primary_block_gt(raw) -> str:
    """
    From the discharge_diagnosis field, return ONLY the primary diagnosis block.

    Assumptions:
    - If the text contains something like 'Primary diagnosis:' or 'Principal diagnoses:',
      we take everything after that, up to 'Secondary diagnosis(es)' if present.
    - Otherwise, we treat the whole field as primary diagnoses.
    """
    if pd.isna(raw):
        return ""
    text = str(raw).strip()
    if not text:
        return ""

    lower = text.lower()

    # Look for 'primary diagnosis' or 'principal diagnosis'
    m = re.search(r"(primary|principal)\s+(diagnosis|diagnoses?)", lower)
    if m:
        start = m.end()
        # Map index back into the original string
        primary_part = text[start:]

        # Cut off any 'secondary diagnosis/diagnoses' section
        sec = re.search(
            r"\bsecondary\s+(diagnosis|diagnoses?)", primary_part, flags=re.IGNORECASE
        )
        if sec:
            primary_part = primary_part[: sec.start()]

        return primary_part.strip()

    # Fallback: treat the entire string as primary diagnoses
    return text.strip()


def extract_pred_primary(raw) -> str:
    """
    Extract the predicted primary diagnosis from model_output.

    Handles:
    - JSON with a 'diagnosis' or 'primary_diagnosis' field
    - plain text patterns like 'Diagnosis: ...' or 'diagnosis = "..."'
    - otherwise, returns the whole string.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    if not s:
        return ""

    # Try JSON first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            if "diagnosis" in obj:
                return str(obj["diagnosis"]).strip()
            if "primary_diagnosis" in obj:
                return str(obj["primary_diagnosis"]).strip()
    except Exception:
        pass

    # diagnosis = "..."
    m = re.search(r"diagnosis\s*=\s*['\"](.+?)['\"]\s*$", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # Diagnosis: ...
    m = re.search(r"diagnosis\s*[:\-]\s*(.+)", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # Fallback: everything
    return s


def bag_of_words(text: str):
    """Convert text into a list of tokens for set-based metrics."""
    if not text:
        return []
    return [tok for tok in normalize_text(text).split() if tok]


def jaccard_similarity(tokens_a, tokens_b) -> float:
    a, b = set(tokens_a), set(tokens_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


def compute_metrics(df: pd.DataFrame, gt_col: str, pred_col: str, prefix: str = "primary"):
    """Compute exact-match, Jaccard, and token-level micro P/R/F1."""
    y_true = df[gt_col].fillna("").astype(str).tolist()
    y_pred = df[pred_col].fillna("").astype(str).tolist()

    # Exact match on normalized strings
    exact_matches = [
        normalize_text(t) == normalize_text(p) for t, p in zip(y_true, y_pred)
    ]
    df[f"{prefix}_exact_match"] = exact_matches

    # Jaccard over token sets
    jaccards = [
        jaccard_similarity(bag_of_words(t), bag_of_words(p))
        for t, p in zip(y_true, y_pred)
    ]
    df[f"{prefix}_jaccard"] = jaccards

    # Multi-label representation over token vocabulary
    all_tokens = sorted(
        set().union(
            *[set(bag_of_words(t)) for t in y_true],
            *[set(bag_of_words(p)) for p in y_pred],
        )
    )
    if not all_tokens:
        # Degenerate case: everything empty
        metrics = {
            f"{prefix}_exact_match": float(pd.Series(exact_matches).mean()),
            f"{prefix}_jaccard": float(pd.Series(jaccards).mean()),
            f"{prefix}_precision": 0.0,
            f"{prefix}_recall": 0.0,
            f"{prefix}_f1": 0.0,
        }
        return metrics

    token_index = {tok: i for i, tok in enumerate(all_tokens)}

    def to_multilabel(text: str):
        vec = [0] * len(all_tokens)
        for tok in set(bag_of_words(text)):
            vec[token_index[tok]] = 1
        return vec

    Y_true = [to_multilabel(t) for t in y_true]
    Y_pred = [to_multilabel(p) for p in y_pred]

    precision = precision_score(Y_true, Y_pred, average="micro", zero_division=0)
    recall = recall_score(Y_true, Y_pred, average="micro", zero_division=0)
    f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)

    metrics = {
        f"{prefix}_exact_match": float(pd.Series(exact_matches).mean()),
        f"{prefix}_jaccard": float(pd.Series(jaccards).mean()),
        f"{prefix}_precision": float(precision),
        f"{prefix}_recall": float(recall),
        f"{prefix}_f1": float(f1),
    }
    return metrics


def evaluate_single_run(gt_path: Path, pred_path: Path, out_dir: Path):
    """Evaluate a single CSV of model outputs against the ground truth."""
    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)

    if "note_id" not in gt.columns or "note_id" not in pred.columns:
        raise ValueError(
            "Both ground-truth and prediction files must contain a 'note_id' column."
        )

    if "discharge_diagnosis" not in gt.columns:
        raise ValueError(
            "Ground-truth file must contain a 'discharge_diagnosis' column."
        )

    if "model_output" not in pred.columns:
        raise ValueError("Prediction file must contain a 'model_output' column.")

    merged = gt.merge(
        pred[["note_id", "model_output"]],
        on="note_id",
        how="inner",
        suffixes=("", "_model"),
    )

    # Extract primary only
    merged["gt_primary"] = merged["discharge_diagnosis"].apply(extract_primary_block_gt)
    merged["pred_primary"] = merged["model_output"].apply(extract_pred_primary)

    metrics = compute_metrics(merged, "gt_primary", "pred_primary", prefix="primary")

    # Save merged rows for debugging / qualitative analysis
    merged_out = out_dir / f"{pred_path.stem}_merged_primary.csv"
    merged.to_csv(merged_out, index=False)

    print(f"\n=== {pred_path.name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return {
        "run": pred_path.stem,
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_file",
        type=str,
        default="final_cases_uex.csv",
        help="CSV with ground truth, must have note_id and discharge_diagnosis.",
    )
    parser.add_argument(
        "--pred_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more CSVs with columns note_id, model_output.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="eval_results",
        help="Directory to save merged CSVs and summary.",
    )
    args = parser.parse_args()

    gt_path = Path(args.gt_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for pred in args.pred_files:
        row = evaluate_single_run(gt_path, Path(pred), out_dir)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary_primary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
