# MedSyn 4-session evaluation

This package reproduces the quantitative evaluation for MedSyn Sessions 1-4:
- Session 1 (Baseline)
- Session 2 (Interactive)
- Session 3 (Baseline)
- Session 4 (Interactive)

## What it computes
Per participant × case × session:
- Any-match accuracy
- Diagnosis-set precision / recall / F1 (main concordance metric)
- Exact set match (fuzzy; set equality under the same fuzzy matching rule)
- Time per case (CSV seconds converted to minutes)

It also computes difficulty-standardized comparisons to correct for Session 3's different case mix.

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python medsyn_eval.py \
  --s1 auto_eval_session1_baseline.csv \
  --s2 auto_eval_session2_interactive.csv \
  --s3 auto_eval_session3_baseline.csv \
  --s4 auto_eval_session4_interactive.csv \
  --out_dir medsyn_outputs \
  --match_threshold 80 \
  --standard_weights 3,6,4
```

Outputs:
- `long_results_thrXX.csv` (long-format results)
- summary CSV tables
- `figures/` (PNG plots)
- `medsyn_4session_evaluation_report.docx`
- `medsyn_4session_evaluation_report.pdf`

## Notes
- Time in input CSVs is assumed to be **seconds**.
- Fuzzy matching uses `RapidFuzz.token_set_ratio` with a default match threshold of 80.
  See `sensitivity_thresholds.csv` for robustness across thresholds.
