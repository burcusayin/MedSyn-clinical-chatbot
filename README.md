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

Python dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt

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
  eval/auto_eval/results/
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
pip install pandas numpy rapidfuzz matplotlib scipy python-docx scikit-learn openai
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

**Outputs** (saved to `--out_dir`):
- CSVs: case-level metrics, bootstrap tests, confusion matrices, concordance analysis, threshold sensitivity.
- Figures (400 DPI): paired trajectories, metrics by difficulty, manual label distributions, concordance, threshold sweep.

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

**Outputs:**
- CSVs: scored turns, judge summary, question categories, statistical comparisons.
- Figures (400 DPI): turns per case, question categories, answer quality, judge scores, judge by category.

### 6.4 Ablation studies

```bash
python eval/ablation_eval/evaluate_ablation.py --help
```

