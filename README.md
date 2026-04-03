# MedSyn – Clinical Decision Support Chatbot

MedSyn is a clinical decision-support chatbot for internal medicine that we use in controlled user studies with clinicians.  

It supports two experimental conditions:

- **Baseline:** clinicians receive the full clinical note and write diagnoses without AI assistance.
- **Interactive:** clinicians see only the chief complaint and interact with an LLM assistant; once confident, they write their final diagnoses.

This repository contains the **code and configuration** needed to reproduce the deployment and the analysis, but **does not include any clinical data or user logs**.

---

## 1. Repository structure

Key directories and files:

- `src/clinical_chatbot/`
  - `app.py` – main Chainlit app with baseline vs interactive logic.
  - `utils.py` – logging, timing, CSV saving, structured dialogue logging.
  - Agent / Langroid helpers under the same package.
- `prompts/`
  - Prompt files for physician instructions and assistant behavior  
    (baseline vs interactive variants).
- `scripts/`
  - `launch_chatbot.sh` – helper to run the app locally.
  - `simulate_baseline.sh`, `simulate_dialogue.sh` – utilities for scripted testing.
- `reverse-proxy/`
  - Nginx config used in the deployment.
- `eval/`
  - `run_evaluation.py` – unified diagnosis evaluation (automated metrics, manual alignment, concordance).
  - `run_dialogue_eval.py` – dialogue quality evaluation with optional LLM-as-judge scoring.
  - `ablation_eval/` – ablation study scripts.
- `public/`
  - `thank-you.html` – thank-you + survey page used in the interactive condition.
- `docker-compose.yml`, `Dockerfile`
  - Dockerized deployment.
- `.env.example`
  - Template for environment variables (no secrets).
- **Not tracked but used at runtime** (ignored by `.gitignore`):
  - `data/` – clinical cases (e.g., MIMIC-derived CSVs).
  - `output/` – per-session outputs, including CSVs and dialogue JSONL.
  - `logs/`, `db/`, `error/` – logs and SQLite auth DB.
  - `.env`, `.env.prod*` – local and production environment files with secrets.

---

## 2. Requirements

- Python **3.10+** (for local analysis / scripted evaluation).
- Docker + Docker Compose (for reproducible deployment).
- An OpenRouter account and API key (for interactive sessions with commercial LLMs).

Python dependencies are listed in `requirements.txt` (app) and `eval/requirements.txt` (evaluation pipeline):

```bash
# App dependencies (Chainlit, Langroid, etc.)
pip install -r requirements.txt

# Evaluation dependencies (rapidfuzz, matplotlib, scipy, etc.)
pip install -r eval/requirements.txt
```

---

## 3. Environment configuration

### 3.1 Local development (.env)

For running the Chainlit app locally (e.g. `chainlit run`):

1. Copy the example file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set:

   - `CHAINLIT_AUTH_SECRET` – random 64-character secret.
   - `ADMIN_USERNAME`, `ADMIN_PASSWORD` – credentials for the admin UI.
   - `DATASET_FOLDER`, `INPUT_DATA_FILE`, `SAMPLE_SIZE` – where your local dataset lives.
   - `LLM_BACKEND` – e.g. `openrouter` or `ollama`.
   - `ASS_MODEL`, `ASS_MODEL_NAME` – model settings for the assistant.
   - If `LLM_BACKEND=openrouter`, set `OPENROUTER_API_KEY` (but **never commit** this file).

`.env` is ignored by git.

### 3.2 Docker deployment env files (.env.prod.*)

For reproducible experiments we use two separate environment files (not tracked):

- `.env.prod.baseline` – configuration for the **baseline** session.
- `.env.prod.interactive` – configuration for the **interactive** session.

Both have the same structure as `.env.example`, but differ in:

- `EXPERIMENT_MODE`:
  - `baseline` or `interactive`.
- `INPUT_DATA_FILE` and `SAMPLE_SIZE`:
  - which CSV and how many cases to include.
- LLM settings:
  - Baseline sessions don’t use the assistant.
  - Interactive sessions set `LLM_BACKEND=openrouter` and choose `ASS_MODEL` accordingly.

Example (interactive):

```env
EXPERIMENT_MODE=interactive
LLM_BACKEND=openrouter

# Example OpenRouter model
ASS_MODEL=openai/gpt-4.1-mini
ASS_MODEL_NAME=gpt-4.1-mini-openrouter

# You must set this locally; do NOT commit it
OPENROUTER_API_KEY=sk-or-v1-...
```

Example (baseline):

```env
EXPERIMENT_MODE=baseline
LLM_BACKEND=openrouter  # or ollama; baseline does not call the assistant

INPUT_DATA_FILE=session3_baseline.csv
SAMPLE_SIZE=13
```

These `.env.prod.*` files are also ignored by git.

---

## 4. Running baseline vs interactive sessions

> **Important:** This repository does not ship any clinical data.  
> To run the experiments you must place your own CSVs under `data/` with the expected schema and set `INPUT_DATA_FILE` accordingly.

### 4.1 Baseline session (no AI assistant)

1. Ensure `.env.prod.baseline` is configured on your machine.
2. In the repo root:

   ```bash
   cp .env.prod.baseline .env.prod
   docker compose down
   docker compose up -d --build
   ```

3. Access the app (e.g. via your reverse proxy).

Behavior:

- Clinicians see the **full clinical note** (formatted).
- There is **no LLM assistant**.
- Their chat input is treated as the **final diagnosis** for each case.
- At the end:
  - They see an inline “All cases completed. Thank you!” message.
  - No survey / thank-you page is shown.

### 4.2 Interactive session (LLM assistant via OpenRouter)

1. Ensure `.env.prod.interactive` is configured with:
   - `EXPERIMENT_MODE=interactive`
   - `LLM_BACKEND=openrouter`
   - `ASS_MODEL` set to a valid OpenRouter model id.
   - `OPENROUTER_API_KEY` set in the environment.

2. In the repo root:

   ```bash
   cp .env.prod.interactive .env.prod
   docker compose down
   docker compose up -d --build
   ```

3. Behavior:

- Clinicians see **only the chief complaint**.
- They can interact with the LLM assistant (MedSyn) to ask for clarifications, labs, etc.
- When ready, they type their **final discharge diagnosis** in the chat.
- After the last case, they click **End session** and are redirected to:
  - A **thank-you / summary page**, which also links to the post-session survey.

---

## 5. Data and privacy

This repository **does not** contain:

- Original clinical notes (e.g. from MIMIC).
- Real clinician logs or session outputs.
- Any `.env*` files or API keys.

By design:

- `data/`, `logs/`, `output/`, `db/`, `error/`, `.env*` are all **git-ignored**.
- Additional ignore rules in `.gitignore` prevent committing evaluation CSVs that contain clinical text or clinician answers:

  ```gitignore
  eval/session_outputs/
  eval/ablation_eval/inputs/
  eval/dial-eval/*.csv
  ```

To reproduce the experiments you must:

1. Obtain access to the underlying dataset (e.g. MIMIC-IV) via the proper channels.
2. Generate the input CSVs with the expected columns  
   (e.g., `note_id`, `chief_complaint`, `history`, `physical_exam`, `results`, `discharge diagnosis`, …).
3. Place them under `data/` and point `INPUT_DATA_FILE` there in your env files.

---

## 6. Evaluation

All evaluations can be reproduced from the two scripts in `eval/`. The input CSVs (session outputs, manual labels, dialogue logs) are not tracked by git because they contain clinical text.

### 6.1 Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r eval/requirements.txt
```

### 6.2 Diagnosis evaluation (Parts A-C)

`eval/run_evaluation.py` runs three evaluation components in sequence:

- **Part A – Automated diagnosis metrics:** Compares clinician answers to ground-truth discharge diagnoses using fuzzy matching (RapidFuzz `token_set_ratio`, threshold=80). Computes exact match, any-match, precision, recall, F1, and time per case. Uses difficulty-weighted standardization (Easy=3/13, Medium=6/13, Hard=4/13) and paired bootstrap (20,000 replicates) for statistical testing. Reports Cohen's d effect sizes.
- **Part B – Manual evaluation alignment:** Compares automated fuzzy-match labels against expert clinician annotations (3-class: WRONG / PARTIALLY CORRECT / COMPLETELY CORRECT). Computes confusion matrices, binary/tri-class agreement, and Cohen's kappa.
- **Part C – Inter-user concordance:** Measures pairwise diagnosis-set F1 between participants (ground-truth independent) to quantify agreement within and across expertise groups.

**Usage:**

```bash
python eval/run_evaluation.py \
  --session_dir eval/session_outputs \
  --manual_dir eval/manual_eval/inputs \
  --out_dir eval/results \
  --threshold 80 \
  --bootstrap_n 20000 \
  --seed 42
```

**Required input files** (not tracked):
- `eval/session_outputs/auto_eval_session{1,2,3,4}_{baseline,interactive}.csv` — 13 cases × 7 participants per session.
- `eval/manual_eval/inputs/manual_eval_session{1,2,3,4}_{baseline,interactive}.csv` — same structure with `{participant}_correctness` columns.

**Required CSV columns** for session files:
- `note_id` — unique case identifier.
- `Difficulty` — one of `Easy`, `Medium`, `Hard`.
- `discharge diagnosis` — ground-truth diagnosis string (semicolon-separated if multiple).
- `{participant}_answer` — one column per participant (e.g. `phy1_answer`, `res3_answer`).
- `{participant}_time` — time in seconds for each participant-case.

Participants are hard-coded as `phy1, phy2, phy3` (Senior) and `res1, res3, res5, res6` (Resident). Sessions map as: S1/S3 = baseline, S2/S4 = interactive.

**Required CSV columns** for manual evaluation files:
- `note_id`, `Difficulty` — same as above.
- `{participant}_correctness` — one of `COMPLETELY CORRECT`, `PARTIALLY CORRECT`, or `WRONG`.

**Outputs** (saved to `--out_dir`):

CSVs:
- `a_case_level_long.csv` — per participant-case fuzzy-match metrics.
- `a_participant_aggregated.csv` — difficulty-standardised per-participant averages.
- `a_bootstrap_tests.csv` — 18 paired bootstrap tests (6 metrics × 3 groups).
- `a_difficulty_stratified.csv` — performance by difficulty × expertise × condition.
- `a_threshold_sensitivity.csv` — metrics across thresholds 60–90.
- `b_manual_per_session.csv` — blinded manual evaluation per session.
- `b_manual_detail_table.csv` — participant-level manual scores by difficulty × expertise.
- `b_manual_difficulty_interaction_tests.csv` — bootstrap tests per difficulty × expertise cell.
- `b_manual_strict_participant.csv` — difficulty-standardised completely-correct rate per participant.
- `b_manual_strict_bootstrap.csv` — paired bootstrap on strict binary manual scores (mirrors Table 1).
- `b_alignment_coefficients.csv` — Cohen's kappa, agreement statistics.
- `b_confusion_matrix_3class.csv` — 3-class confusion matrix (automated vs manual).
- `b_manual_label_distribution.csv` — manual label counts by expertise and condition.
- `c_concordance_case_level.csv` — pairwise F1 per participant-case.
- `c_concordance_bootstrap_tests.csv` — 6 concordance bootstrap tests.
- `c_concordance_by_difficulty.csv` — concordance by difficulty level.

Figures (400 DPI, saved to `--out_dir/figures/`):
- `fig_paired_trajectories.png` — slopegraph of individual participant changes (baseline → interactive).
- `fig_metrics_by_difficulty.png` — any-match, exact-match, F1, time by difficulty × expertise × condition.
- `fig_manual_distribution.png` — stacked bar of wrong/partial/complete by group and condition.
- `fig_manual_by_difficulty.png` — manual scores stratified by difficulty.
- `fig_manual_per_session.png` — ordinal score, binary, and completely-correct rate per session.
- `fig_confusion_matrices.png` — automated vs manual evaluation agreement.
- `fig_concordance_by_difficulty.png` — within- and cross-expertise concordance by difficulty.
- `fig_threshold_sensitivity.png` — metric robustness across fuzzy-matching thresholds.
- `fig_across_sessions.png` — S1→S2→S3→S4 trajectory by expertise group.
- `fig_expertise_gap.png` — Senior-minus-Resident gap by difficulty and condition.
- `fig_ablation_comparison.png` — 5-model precision/recall/F1 comparison (requires ablation results).
- `fig_forest_plot.png` — forest plot of all bootstrap effect sizes with 95% CIs.
- `fig_overall_comparison.png` — grouped bar of performance by expertise and condition.
- `fig_per_session_bars.png` — per-session performance by expertise.

### 6.3 Dialogue evaluation (Part D)

`eval/run_dialogue_eval.py` evaluates the quality of interactive dialogue turns from Sessions 2 and 4.

- **Heuristic metrics:** question categorization (detail/info/suggestion/other), answer specificity, context overlap.
- **LLM-as-judge** (optional): per-turn faithfulness and answer-relevancy scores via OpenRouter. Uses `google/gemini-2.5-flash` by default to avoid self-evaluation bias (the assistant uses GPT).

**Usage:**

```bash
# Without LLM judge (free, heuristic metrics only):
python eval/run_dialogue_eval.py \
  --turn_csv eval/dial-eval/turn_level.csv \
  --case_csv eval/dial-eval/case_level.csv \
  --session2_notes eval/session_outputs/auto_eval_session2_interactive.csv \
  --session4_notes eval/session_outputs/auto_eval_session4_interactive.csv \
  --out_dir eval/results/dialogue

# With LLM judge (~$0.70 via OpenRouter):
export OPENROUTER_API_KEY=sk-or-v1-...
python eval/run_dialogue_eval.py \
  --turn_csv eval/dial-eval/turn_level.csv \
  --case_csv eval/dial-eval/case_level.csv \
  --session2_notes eval/session_outputs/auto_eval_session2_interactive.csv \
  --session4_notes eval/session_outputs/auto_eval_session4_interactive.csv \
  --out_dir eval/results/dialogue \
  --run_llm_judge \
  --judge_model google/gemini-2.5-flash
```

**Required input files** (not tracked):
- `eval/dial-eval/turn_level.csv` — 829 dialogue turns (parsed from session dialogues).
- `eval/dial-eval/case_level.csv` — 182 case-level dialogue summaries.
- Session note CSVs (same as Part A) for building clinical context.

**Outputs** (saved to `--out_dir`):

CSVs:
- `case_summary.csv` — mean turns and duration per session × expertise group.
- `question_categories.csv` — question-type counts and proportions.
- `statistical_comparisons.csv` — Mann-Whitney U tests for group and session comparisons.
- `judge_summary.csv` — mean faithfulness and relevancy per group/session (if `--run_llm_judge`).

Figures (400 DPI, saved to `--out_dir/figures/`):
- `fig_dial_turns_per_case.png` — mean turns per case by expertise and session.
- `fig_dial_question_categories.png` — stacked bar of question types.
- `fig_dial_answer_quality.png` — box plots of answer specificity and context overlap.
- `fig_dial_judge_scores.png` — box plots of faithfulness and relevancy (if judge was run).
- `fig_dial_judge_by_category.png` — faithfulness/relevancy by question type.
- `fig_dial_duration_vs_turns.png` — scatter plot of case duration vs turn count.

### 6.4 Ablation study

`eval/ablation_eval/evaluate_ablation.py` evaluates multiple LLMs on the same 52 cases under both baseline and interactive scenarios to justify model selection.

It uses a lower fuzzy-matching threshold (62) than the physician evaluation (80) because LLM outputs tend to use more standardised medical terminology.

**Usage:**

```bash
python eval/ablation_eval/evaluate_ablation.py \
  --baseline eval/ablation_eval/inputs/baseline_outputs.csv \
  --interactive eval/ablation_eval/inputs/interactive_outputs.csv \
  --outdir eval/ablation_eval/results \
  --threshold 62 \
  --sweep \
  --bootstrap 2000
```

**Required input files** (not tracked):
- `eval/ablation_eval/inputs/baseline_outputs.csv` — one column per model with generated diagnoses for each of the 52 cases in the baseline (full-note) scenario.
- `eval/ablation_eval/inputs/interactive_outputs.csv` — same structure for the interactive (dialogue) scenario.

Both CSVs must contain `note_id`, `discharge diagnosis`, and `Difficulty` columns, plus one output column per model.

**Outputs** (saved to `--outdir`):
- `{scenario}_summary_primary.csv` — micro precision, recall, F1, top-1 accuracy, and average prediction count per model.
- `{scenario}_per_case_primary.csv` — per-case match details.
- `{scenario}_by_difficulty_primary.csv` — metrics stratified by Easy/Medium/Hard.
- `{scenario}_bootstrap_primary.csv` — bootstrap CIs (if `--bootstrap > 0`).
- `{scenario}_threshold_sweep_primary.csv` — metrics across thresholds (if `--sweep`).

