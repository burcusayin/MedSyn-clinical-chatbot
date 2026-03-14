# MedSyn Dialogue Interaction Analysis (Session 2 & Session 4)

This package analyzes *interaction* in MedSyn interactive sessions:
- turns per case
- question-type patterns (detail-seeking vs suggestion-seeking)
- per-turn grounding proxies (context overlap) and optional LLM-judge metrics (faithfulness + answer relevancy)

## Inputs
- A directory of merged dialogue logs for Session 2 (one `.jsonl` per user)
- A directory of merged dialogue logs for Session 4 (one `.jsonl` per user)
- `session2_interactive.csv` and `session4_interactive.csv` (clinical notes + ground truth)

## Quick start (offline metrics only)
```bash
python medsyn_dialogue_analysis.py \
  --session2_dir /path/to/session2_merged \
  --session4_dir /path/to/session4_merged \
  --session2_notes_csv session2_interactive.csv \
  --session4_notes_csv session4_interactive.csv \
  --out_dir dialogue_analysis_out
```

Outputs:
- `case_level.csv`
- `turn_level.csv`
- `question_category_distribution.csv`
- `figures/*.png`

## Enable OpenRouter LLM-judge scoring (Faithfulness + Answer Relevancy)
Set:
- `OPENROUTER_API_KEY` (required)
Optionally:
- `OPENROUTER_MODEL` (judge model, e.g., `openai/gpt-4o-mini`)
- `OPENROUTER_HTTP_REFERER` and `OPENROUTER_X_TITLE` (optional headers)

Then:
```bash
python medsyn_dialogue_analysis.py ... --run_llm_judge
```

The script writes `judge_cache.jsonl` so repeated runs do not re-score identical turns.

## Build reports (DOCX + PDF)
After running analysis:
```bash
python build_report.py --out_dir dialogue_analysis_out
```

This regenerates:
- `MedSyn_Dialogue_Interaction_Report.docx`
- `MedSyn_Dialogue_Interaction_Report.pdf`
