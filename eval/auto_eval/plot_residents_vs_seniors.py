import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Input: the exported table from the evaluation pipeline
TABLE_PATH = Path("session_by_expertise_summary.csv")

# Output folder
OUT_DIR = Path("plots_resident_vs_senior")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_ci(s):
    if pd.isna(s):
        return (np.nan, np.nan, np.nan)
    s = str(s).strip()
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*$", s)
    if not m:
        return (np.nan, np.nan, np.nan)
    return tuple(float(x) for x in m.groups())

def main():
    df = pd.read_csv(TABLE_PATH)

    metrics = {
        "Any-match accuracy": "any_match_acc_ci",
        "Item-F1": "f1_items_ci",
        "Exact-set accuracy": "exact_set_acc_ci",
        "Mean time (s)": "mean_time_s_ci",
    }

    rows = []
    for _, r in df.iterrows():
        for mname, col in metrics.items():
            mean, lo, hi = parse_ci(r[col])
            rows.append({
                "session": int(r["session"]),
                "condition": r["condition"],
                "expertise": r["expertise"],
                "metric": mname,
                "mean": mean,
                "lo": lo,
                "hi": hi,
            })
    tidy = pd.DataFrame(rows)

    sess_labels = (
        df[["session", "condition"]]
        .drop_duplicates()
        .sort_values("session")
    )
    label_map = {int(r.session): f"S{int(r.session)} ({r.condition})" for _, r in sess_labels.iterrows()}
    x = sorted(label_map.keys())
    x_labels = [label_map[i] for i in x]

    for metric_name in tidy["metric"].unique():
        sub = tidy[tidy.metric == metric_name].copy()
        sub["session"] = sub["session"].astype(int)
        sub = sub.sort_values("session")

        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        for exp in ["resident", "senior"]:
            s2 = sub[sub.expertise == exp].set_index("session").reindex(x)
            y = s2["mean"].values.astype(float)
            yerr = np.vstack([y - s2["lo"].values, s2["hi"].values - y])
            ax.errorbar(range(len(x)), y, yerr=yerr, marker="o", capsize=4, label=exp)

        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
        ax.set_title(f"{metric_name} by session (Residents vs Senior physicians)")
        ax.set_xlabel("Session")
        ax.set_ylabel(metric_name)
        ax.legend()
        fig.tight_layout()

        fname = f"{metric_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')}_by_session.png"
        fig.savefig(OUT_DIR / fname, dpi=200)
        plt.close(fig)

if __name__ == "__main__":
    main()
