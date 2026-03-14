# MedSyn Ablation Evaluation (Updated with gpt-5.2-chat)

This folder contains all inputs, code, and outputs used to evaluate model predictions on:
- baseline_scenario_outputs.csv
- interactive_scenario_outputs.csv

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python evaluate_ablation.py \
  --baseline inputs/baseline_scenario_outputs.csv \
  --interactive inputs/interactive_scenario_outputs.csv \
  --outdir results_reproduced \
  --threshold 62 \
  --sweep \
  --bootstrap 2000
```

## Outputs

- `results/` includes the exact CSV outputs generated in this run:
  - scenario summaries (micro metrics + top1 + avg_n_pred)
  - difficulty-stratified summaries
  - per-case TP/FP/FN values
  - threshold sweeps
  - bootstrap confidence intervals
  - newline audits proving outputs use literal `\n`

- `analysis_report.md` is the human-readable report corresponding to these outputs.

## Notes on trustworthy preprocessing

The model outputs in the CSVs frequently contain **literal** `\n` separators (two characters) and never contain real newline characters. The evaluator converts literal `\n` into real newlines and splits into a diagnosis list before scoring.
