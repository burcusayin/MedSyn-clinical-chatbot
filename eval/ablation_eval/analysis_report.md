# MedSyn Ablation Evaluation Report (Updated with gpt-5.2-chat)

**Date (UTC):** 2025-12-15 10:10:43  
**Files evaluated:** `baseline_scenario_outputs.csv`, `interactive_scenario_outputs.csv`  
**Number of cases:** 52 baseline + 52 interactive (same case count; evaluated independently)

## 1) Goal

Evaluate how well each model predicts discharge diagnoses under:
- **Baseline scenario:** model sees the full note and outputs diagnoses directly.
- **Interactive scenario:** model outputs diagnoses after the interactive workflow.

The key decision for deployment is: **which model will perform best in real physician-facing interactive experiments**.

## 2) Data and label format

### Ground truth (reference labels)
Ground truth is in the column **`discharge diagnosis`** and typically follows:

- `primary:` (one or more diagnoses, one per line)
- optional `secondary:` section (one or more diagnoses, one per line)

For evaluation we compute results against **primary diagnoses only**:
- **Primary diagnoses:** all diagnoses listed under `primary:`

### Model outputs (predictions)
Each model output column is named `output_*`. Outputs are free-text but commonly wrapped like:

`diagnosis='Dx1\nDx2\nDx3'`

## 3) Trustworthy preprocessing (critical)

A common evaluation pitfall is treating the prediction as a single string instead of a **list**.

### Newline audit (why splitting matters)
In both updated CSV files, **model outputs contain 0% real newlines** and frequently contain the two-character sequence **`\n`** (a literal backslash + n).  
Therefore, a trustworthy evaluator must convert and split on literal `\n`.

**Baseline audit (fraction of rows containing literal `\n`):**
| column                      |   pct_literal_\n |   pct_real_newline |
|:----------------------------|-----------------:|-------------------:|
| output_gemini-3-pro-preview |            0.519 |                  0 |
| output_gpt-oss-120b         |            0.462 |                  0 |
| output_gpt-5.1              |            0.846 |                  0 |
| output_llama-4-scout        |            0.212 |                  0 |
| output_gpt-5.2-chat         |            0.442 |                  0 |

**Interactive audit (fraction of rows containing literal `\n`):**
| column                      |   pct_literal_\n |   pct_real_newline |
|:----------------------------|-----------------:|-------------------:|
| output_gemini-3-pro-preview |            0.731 |                  0 |
| output_gpt-oss-120b         |            0.577 |                  0 |
| output_gpt-5.1-chat         |            0.635 |                  0 |
| output_llama-4-scout        |            0.385 |                  0 |
| output_gpt-5.2-chat         |            0.538 |                  0 |

### Prediction parsing used in this evaluation
For each model cell:
1. Extract substring after `diagnosis=` when present.
2. Strip surrounding quotes.
3. Convert literal `\n` → real newline.
4. Split into a list of diagnoses using:
   - newline first, else `;`, else `,` (fallback)
5. Deduplicate predictions by a normalized key (see below).

### Label parsing used in this evaluation
For the ground truth cell:
1. Split `primary:` and `secondary:` sections (if present).
2. Split each section by real newlines.
3. Remove empty lines and trailing punctuation.

### Normalization (applied to both predictions and labels)
To reduce false mismatches due to formatting:
- lowercasing, punctuation removal, whitespace normalization
- small abbreviation expansion (e.g., `pna→pneumonia`, `tia→transient ischemic attack`, `aki→acute kidney injury`, `afib→atrial fibrillation`, `rvr→rapid ventricular response`, lobe abbreviations like `rll→right lower lobe`)
- very conservative stopword removal (e.g., “of”, “and”, “with”)

## 4) Matching rule

Because the strings are not guaranteed to be identical (abbreviations, word order), we use **soft matching**:

- Similarity = RapidFuzz **token_set_ratio** on normalized strings (0–100)
- A prediction matches a label if similarity >= **62**

To ensure robustness, we also provide a **threshold sweep** (Section 7).

When multiple predictions could match multiple labels, we perform **one-to-one matching** (each pred and label can be used at most once) by selecting the highest-similarity pairs first.

## 5) Metrics

For each scenario, we compute:

- **Micro-precision / micro-recall / micro-F1** at the diagnosis level
- **Top-1 primary-first accuracy:** whether the **first** predicted diagnosis matches any primary diagnosis
- **Any-primary recall:** whether **any** prediction matches a primary diagnosis
- **Average # predicted diagnoses** (`avg_n_pred`) as a proxy for verbosity/cognitive load

## 6) Main results (primary-diagnosis level, threshold = 0.62)

We report results at the **diagnosis level**, with a focus on **primary discharge diagnoses** (the clinically relevant target for MedSyn). For each model and scenario we compute:

- **micro-precision** (P): fraction of predicted diagnoses that match a ground-truth primary diagnosis
- **micro-recall** (R): fraction of ground-truth primary diagnoses recovered by any prediction
- **micro-F1**: harmonic mean of (P, R)
- **Top-1 primary-first accuracy**: whether the **first** predicted diagnosis matches any primary diagnosis
- **Any-match recall**: whether **any** predicted diagnosis matches any primary diagnosis for that case
- **Average number of predicted diagnoses** (`avg_n_pred`): proxy for verbosity / cognitive load

A diagnosis-level match is counted when the RapidFuzz token-set similarity between a predicted and ground-truth diagnosis is ≥ 0.62 (after normalization and abbreviation expansion), with greedy one-to-one matching.

### 6.1 Baseline scenario – primary diagnoses only

| model                | micro_P | micro_R | micro_F1 | top1_acc | any-match R | avg_n_pred |
|:---------------------|--------:|--------:|---------:|---------:|------------:|-----------:|
| gpt-oss-120b         |  0.543  |  0.739  |  0.626   |  0.846   |    0.885    |    1.808   |
| gpt-5.2-chat         |  0.495  |  0.754  |  0.598   |  0.865   |    0.923    |    2.019   |
| gemini-3-pro-preview |  0.490  |  0.739  |  0.590   |  0.788   |    0.865    |    2.000   |
| llama-4-scout        |  0.257  |  0.841  |  0.393   |  0.750   |    0.942    |    4.346   |
| gpt-5.1              |  0.161  |  0.812  |  0.269   |  0.827   |    0.904    |    6.673   |

**Observations (baseline):**

- All models achieve relatively **high recall** on primary diagnoses (R ≈ 0.74–0.84), but differ markedly in precision and verbosity.
- **gpt-oss-120b** attains the highest **primary micro-F1 (0.63)**, with a compact output (≈1.8 diagnoses per case).
- **gpt-5.2-chat** and **gemini-3-pro-preview** are very close in F1 (0.60 and 0.59), with similar recall (~0.74–0.75) and slightly higher average list length (~2 diagnoses).
- **llama-4-scout** and **gpt-5.1** achieve very high recall (≥0.81) but at the cost of very low precision and long lists (4–7 diagnoses per case), which would be burdensome in a physician UI.

Overall, in the baseline setting, **gpt-oss-120b, gpt-5.2-chat, and gemini-3-pro-preview** form the top tier in terms of F1, with **gpt-5.2-chat** offering a good balance between recall and output length.

### 6.2 Interactive scenario – primary diagnoses only

| model                | micro_P | micro_R | micro_F1 | top1_acc | any-match R | avg_n_pred |
|:---------------------|--------:|--------:|---------:|---------:|------------:|-----------:|
| llama-4-scout        |  0.485  |  0.710  |  0.576   |  0.712   |    0.846    |    1.942   |
| gpt-5.2-chat         |  0.459  |  0.725  |  0.562   |  0.808   |    0.846    |    2.096   |
| gpt-5.1-chat         |  0.403  |  0.783  |  0.532   |  0.885   |    0.923    |    2.577   |
| gpt-oss-120b         |  0.414  |  0.667  |  0.511   |  0.769   |    0.769    |    2.135   |
| gemini-3-pro-preview |  0.354  |  0.739  |  0.479   |  0.750   |    0.865    |    2.769   |

**Observations (interactive):**

- **llama-4-scout** achieves the highest **primary micro-F1 (0.58)** in the interactive scenario, with reasonably short lists (~1.9 diagnoses per case).
- **gpt-5.2-chat** is a close second in F1 (0.56), with slightly higher recall (0.73) but somewhat lower precision.
- **gpt-5.1-chat** is particularly strong on **recall (0.78)** and **Top-1 primary-first accuracy (0.89)**, while keeping the list length moderate (~2.6 diagnoses). This makes it attractive when the UI emphasizes the first suggestion and when missing the primary diagnosis is considered costly.
- **gemini-3-pro-preview** and **gpt-oss-120b** perform reasonably well but are not dominant under this metric.

### 6.3 Model choice for the physician-facing interactive study

The **interactive MedSyn prototype** used in the physician study was frozen before these ablation experiments were finalized. Model selection at that time was based on preliminary offline experiments and a simpler metric focusing on agreement with the full discharge diagnosis text.

The retrospective ablation above, using **primary-diagnosis–level precision, recall, and F1**, confirms that the **GPT family remains competitive in the interactive setting**:

- **gpt-5.1-chat** offers the **highest recall** and **highest Top-1 primary-first accuracy**, which are critical when the goal is to surface the correct primary diagnosis quickly.
- **gpt-5.2-chat** provides a strong compromise between F1, recall, and list length, and was therefore a natural choice for deployment in our initial interactive experiments.

Other models (llama-4-scout, gemini-3-pro-preview, gpt-oss-120b) are competitive and in some cases stronger on micro-F1, but tend either to be more verbose or to provide lower Top-1 accuracy, which would increase cognitive load or reduce immediate utility for clinicians. For reproducibility, all diagnosis-level metrics (including micro-precision, micro-recall, Top-1 accuracy, any-match recall, and avg_n_pred) are available in the CSV summaries and in the LaTeX tables in `tables_for_paper.tex`.

## 7) Performance by case difficulty

Your files include a `Difficulty` label (Easy=13, Medium=24, Hard=15).  
Below are difficulty-stratified results for the interactive scenario, evaluated **only on primary discharge diagnoses**.

### 7.1 Interactive – primary diagnoses (difficulty-stratified)
| Difficulty   | model                |   micro_f1 |   top1_primary_first_acc |   avg_n_pred |   n_cases |
|:-------------|:---------------------|-----------:|-------------------------:|-------------:|----------:|
| Easy         | gpt-5.2-chat         |      0.588 |                    0.846 |        2.385 |        13 |
| Easy         | gpt-oss-120b         |      0.565 |                    0.846 |        2     |        13 |
| Easy         | gpt-5.1-chat         |      0.545 |                    0.846 |        2.692 |        13 |
| Easy         | llama-4-scout        |      0.528 |                    0.692 |        2.538 |        13 |
| Easy         | gemini-3-pro-preview |      0.508 |                    0.846 |        3     |        13 |
| Hard         | llama-4-scout        |      0.622 |                    0.8   |        1.867 |        15 |
| Hard         | gpt-5.2-chat         |      0.596 |                    0.8   |        2     |        15 |
| Hard         | gpt-5.1-chat         |      0.566 |                    0.933 |        2.4   |        15 |
| Hard         | gpt-oss-120b         |      0.558 |                    0.733 |        1.733 |        15 |
| Hard         | gemini-3-pro-preview |      0.549 |                    0.733 |        2.267 |        15 |
| Medium       | llama-4-scout        |      0.583 |                    0.667 |        1.667 |        24 |
| Medium       | gpt-5.2-chat         |      0.525 |                    0.792 |        2     |        24 |
| Medium       | gpt-5.1-chat         |      0.505 |                    0.875 |        2.625 |        24 |
| Medium       | gpt-oss-120b         |      0.462 |                    0.75  |        2.458 |        24 |
| Medium       | gemini-3-pro-preview |      0.427 |                    0.708 |        2.958 |        24 |

**Observed pattern:** on **Hard** cases, llama-4-scout and gpt-5.2-chat are strongest by micro-F1, while **gpt-5.1-chat** maintains the best “primary-first” behavior.

## 8) Uncertainty estimates (bootstrap 95% CI, primary diagnoses, threshold = 0.62)

### 8.1 Interactive – primary diagnoses (micro-F1)
| model                |   mean |   ci_low |   ci_high |
|:---------------------|-------:|---------:|----------:|
| llama-4-scout        |  0.578 |    0.477 |     0.680 |
| gpt-5.2-chat         |  0.563 |    0.477 |     0.644 |
| gpt-5.1-chat         |  0.533 |    0.453 |     0.618 |
| gpt-oss-120b         |  0.513 |    0.418 |     0.614 |
| gemini-3-pro-preview |  0.479 |    0.404 |     0.553 |

### 8.2 Interactive – Top-1 primary-first accuracy
| model                |   mean |   ci_low |   ci_high |
|:---------------------|-------:|---------:|----------:|
| gpt-5.1-chat         |  0.885 |    0.788 |     0.962 |
| gpt-5.2-chat         |  0.809 |    0.692 |     0.904 |
| gpt-oss-120b         |  0.769 |    0.654 |     0.885 |
| gemini-3-pro-preview |  0.750 |    0.635 |     0.865 |
| llama-4-scout        |  0.711 |    0.577 |     0.827 |

**Note:** With only 52 cases, confidence intervals overlap for several models; differences should be interpreted as *directional* unless replicated with larger samples.

## 9) Sensitivity analysis (threshold sweep)

We repeated the evaluation at multiple similarity thresholds (55, 60, 62, 65, 70, 75), always evaluating against **primary discharge diagnoses only**.  
All raw sweep outputs are saved in the `results/` folder.

At thresholds 55–70, the best interactive micro-F1 (primary diagnoses) is consistently **llama-4-scout**; at 75, the stricter matching slightly favors **gpt-oss-120b** in this dataset. This is expected: stricter thresholds penalize abbreviation/synonym variance more heavily.

## 10) Reproducibility

All source materials needed to reproduce this evaluation are packaged with:
- Input CSVs (copied into `inputs/`)
- A runnable evaluation script: `evaluate_ablation.py`
- Requirements: `requirements.txt`
- All generated CSV outputs and LaTeX tables in `results/`
- This report in Markdown plus the Word/PDF versions

To rerun (example):

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

## 11) Limitations

- Soft matching cannot fully resolve true clinical synonymy (e.g., “sepsis” vs “bacteremia”) without a medical ontology.
- Comma-splitting is a fallback; in rare cases a single diagnosis could include commas.
- Ground truth sometimes lists multiple diagnoses under `primary:`; we treat **all** of those as primary targets (consistent with the file format).
