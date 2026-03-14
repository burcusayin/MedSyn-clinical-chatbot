import pandas as pd
import numpy as np
import re
from pathlib import Path

# -----------------------
# 1. Helper functions
# -----------------------

def extract_pred_diag(raw):
    """Extract the diagnosis text from a model_output string.

    Handles patterns like:
      - "diagnosis='...'"
      - "Diagnosis: ..."
    and falls back to the raw text.
    """
    if pd.isna(raw):
        return ""
    text = str(raw).strip()
    # Pattern diagnosis='...'
    m = re.search(r"diagnosis\s*=\s*['\"](.+?)['\"]\s*$",
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)
    # Pattern Diagnosis: ...
    text = re.sub(r"^\s*diagnosis\s*[:=-]\s*", "",
                  text, flags=re.IGNORECASE)
    return text


def normalize_for_eval(text: str) -> str:
    """Light normalization for lexical comparison.

    - lowercases
    - removes obvious headers like 'primary diagnosis:'
    - removes non-alphanumeric chars
    - collapses whitespace
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # remove labels like primary/secondary/final diagnosis
    text = re.sub(
        r"\b(primary|secondary|principal|discharge|final)\s*(diagnosis|diagnoses)?\b[:\s]*",
        " ",
        text,
    )
    # replace explicit newlines
    text = text.replace("\\n", " ")
    # remove non-alphanumerics (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text: str):
    text = normalize_for_eval(text)
    return [t for t in text.split() if t]


def lexical_scores(gt, pred):
    """Token-set precision/recall/F1 + Jaccard for a single pair."""
    gt_tokens = tokens(gt)
    pred_tokens = tokens(pred)
    gt_set, pred_set = set(gt_tokens), set(pred_tokens)

    if not gt_set and not pred_set:
        return dict(precision=1.0, recall=1.0, f1=1.0, jaccard=1.0)

    inter = len(gt_set & pred_set)
    precision = inter / len(pred_set) if pred_set else 0.0
    recall = inter / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    jaccard = inter / len(gt_set | pred_set) if (gt_set | pred_set) else 0.0

    return dict(precision=precision, recall=recall, f1=f1, jaccard=jaccard)


def join_with_ground_truth(gt_path, pred_path):
    """Inner-join model outputs with ground truth on note_id."""
    gt = pd.read_csv(gt_path, usecols=["note_id", "discharge diagnosis"])
    pred = pd.read_csv(pred_path)
    if "note_id" not in pred.columns or "model_output" not in pred.columns:
        raise ValueError(f"{pred_path} must contain 'note_id' and 'model_output' columns.")
    pred = pred[["note_id", "model_output"]]
    df = gt.merge(pred, on="note_id", how="inner")
    return df


def evaluate_df(df, name="model"):
    """Compute aggregate lexical metrics for one model."""
    preds_clean = df["model_output"].apply(extract_pred_diag)
    gts = df["discharge diagnosis"]

    exact_strict = []
    exact_relaxed = []
    precs, recs, f1s, jaccs = [], [], [], []

    for gt, pred in zip(gts, preds_clean):
        n_gt = normalize_for_eval(gt)
        n_pred = normalize_for_eval(pred)

        # strict & relaxed exact match
        exact_strict.append(int(n_gt == n_pred))
        exact_relaxed.append(int(bool(n_gt) and (n_gt in n_pred or n_pred in n_gt)))

        # token overlap metrics
        scores = lexical_scores(gt, pred)
        precs.append(scores["precision"])
        recs.append(scores["recall"])
        f1s.append(scores["f1"])
        jaccs.append(scores["jaccard"])

    result = {
        "model": name,
        "n_cases": len(df),
        "exact_strict": float(np.mean(exact_strict)),
        "exact_relaxed": float(np.mean(exact_relaxed)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "f1": float(np.mean(f1s)),
        "jaccard": float(np.mean(jaccs)),
    }
    return result


def add_casewise_metrics(df):
    """Return a copy of df with per-case metrics for error analysis."""
    df = df.copy()
    df["gt_clean"] = df["discharge diagnosis"].apply(normalize_for_eval)
    df["pred_raw"] = df["model_output"]
    df["pred_extracted"] = df["model_output"].apply(extract_pred_diag)
    df["pred_clean"] = df["pred_extracted"].apply(normalize_for_eval)

    metrics = df.apply(
        lambda row: lexical_scores(row["discharge diagnosis"], row["pred_extracted"]), axis=1
    )
    df["precision"] = [m["precision"] for m in metrics]
    df["recall"] = [m["recall"] for m in metrics]
    df["f1"] = [m["f1"] for m in metrics]
    df["jaccard"] = [m["jaccard"] for m in metrics]

    df["exact_strict"] = (df["gt_clean"] == df["pred_clean"]).astype(int)
    df["exact_relaxed"] = df.apply(
        lambda r: int(
            bool(r["gt_clean"])
            and (r["gt_clean"] in r["pred_clean"] or r["pred_clean"] in r["gt_clean"])
        ),
        axis=1,
    )
    return df


# -----------------------
# 2. Optional: embeddings (semantic similarity)
# -----------------------

def embedding_scores(df, model_name="sentence-transformers/all-mpnet-base-v2"):
    """Average cosine similarity between GT and prediction.

    Requires:
        pip install sentence-transformers
    """
    from sentence_transformers import SentenceTransformer, util

    gt_texts = df["discharge diagnosis"].apply(normalize_for_eval).tolist()
    pred_texts = df["model_output"].apply(extract_pred_diag).apply(normalize_for_eval).tolist()

    encoder = SentenceTransformer(model_name)
    gt_emb = encoder.encode(gt_texts, convert_to_tensor=True, normalize_embeddings=True)
    pred_emb = encoder.encode(pred_texts, convert_to_tensor=True, normalize_embeddings=True)

    sims = util.cos_sim(pred_emb, gt_emb).diagonal().cpu().numpy()
    return float(np.mean(sims)), sims


# -----------------------
# 3. Example usage
# -----------------------

if __name__ == "__main__":
    GT_PATH = Path("final_cases.csv")

    # Map model names to their CSVs
    MODEL_OUTPUTS = {
        "baseline_gpt-oss-20b_free": Path("output_ass_baseline_gpt-oss-20b:free.csv"),
        "dialogue_gpt-oss-20b_free": Path("output_phy_gpt-oss-20b:free_ass_gpt-oss-20b:free.csv"),
        # add more here as you run other LLMs, e.g.:
        # "baseline_llama3": Path("output_ass_baseline_llama3.csv"),
    }

    summary_rows = []

    for name, pred_path in MODEL_OUTPUTS.items():
        df = join_with_ground_truth(GT_PATH, pred_path)
        res = evaluate_df(df, name)
        summary_rows.append(res)

        # also save per-case metrics if you like
        casewise = add_casewise_metrics(df)
        casewise.to_csv(f"casewise_{name}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    print(summary)

    # Optional: semantic similarity for one model
    # df_baseline = join_with_ground_truth(GT_PATH, MODEL_OUTPUTS["baseline_gpt-oss-20b_free"])
    # mean_sim, sims = embedding_scores(df_baseline)
    # print("Baseline mean embedding cosine similarity:", mean_sim)
