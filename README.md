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
  - `evaluate_diagnosis*.py` – diagnosis accuracy evaluation scripts.
  - `evaluate_dialogues.py` – aggregates dialogue metrics (baseline vs interactive).
  - `auto_eval/`, `ablation_eval/` – more detailed evaluation tooling.
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

## 6. Dialogue and diagnosis evaluation

### 6.1 Dialogue metrics (baseline vs interactive)

After running clinician sessions, the app writes per-session CSVs and dialogue logs to `output/` (not tracked by git).

To analyze the conversation patterns, use:

```bash
pip install -r requirements.txt

python eval/evaluate_dialogues.py     --base-output-dir output
```

This script will:

- Read `output/dialogues/dialogue_*.jsonl` and the corresponding `out_*.csv` files.
- Aggregate per-case metrics:
  - `n_user_turns_nonfinal`, `n_llm_turns`
  - `user_chars_total`, `llm_chars_total`
  - `time_spent_seconds` (if present)
  - `final_answer_from_log`, `ground_truth_discharge`, etc.
- Save:
  - `output/analysis/case_level_metrics.csv`
  - `output/analysis/summary_by_mode.csv`

You can then perform further analysis (e.g., statistical tests, plotting) using these CSVs.

### 6.2 Diagnosis accuracy and multi-metric evaluation

The `eval/` folder also contains:

- `eval/evaluate_diagnosis.py`, `eval/evaluate_diagnosis_with_primary.py` – diagnosis-matching logic.
- `eval/auto_eval/` – scripts for computing:
  - exact match, any-match, precision/recall/F1, Jaccard, ROUGE, BERTScore, etc.
- `eval/ablation_eval/` – scripts and configs used for ablations.

All evaluation scripts rely only on:

- CSV outputs in `output/` plus
- The same input data that is *not* included in the repo.

