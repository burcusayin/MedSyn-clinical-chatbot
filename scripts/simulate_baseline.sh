#!/bin/bash

#!/usr/bin/env bash
set -euo pipefail

# Go to project root
[[ -d "$MAIN_DIR" ]] || die "MAIN_DIR not found: $MAIN_DIR"
cd "$MAIN_DIR"

[ -f src/__init__.py ] || touch src/__init__.py

pwd

BASELINE_MODEL="openrouter/openai/gpt-oss-20b:free"
SAFE_MODEL="${BASELINE_MODEL//\//_}"
# --- logging setup ---
LOG_DIR=${MAIN_DIR}logs/ablationLogs
LOG_FILE="$LOG_DIR/simulate_baseline_${SAFE_MODEL}_$(date +%Y%m%d_%H%M%S).log"

# Send everything (stdout + stderr) to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
# ------------------------

cd "$MAIN_DIR"
# ----------------------
#ollama commands below
#ollama serve &

# Wait until Ollama service has been started
#sleep 2

#echo "Server started"

#ollama run llama3:8b
#ollama run llama3:70b
#ollama run llama3.1:8b
#ollama run llama3.2:3b
#ollama run gemma2:27b

#BASELINE_MODEL=gemma2:27b

#ollama ps

# python -m src.simulate \
# --input_file ${MAIN_DIR}data/mimicIV_1000_random_sample_filtered_parsed_icd10.csv \
# --history_file ${MAIN_DIR}output/dialogues/assistant_agent/random/  \
# --discharge_data_file ${MAIN_DIR}output/discharge_texts/random/ \
# --mode ass_baseline \
# --baseline_model $BASELINE_MODEL \
# --baseline_system_prompt ${MAIN_DIR}prompts/baseline_system.txt \
# --baseline_prompt_template ${MAIN_DIR}prompts/baseline_user.txt 

python -m src.simulate \
  --input_file ${MAIN_DIR}data/patient_cases.csv \
  --history_file ${MAIN_DIR}output/dialogues/assistant_agent/ \
  --discharge_data_file ${MAIN_DIR}output/discharge_texts/ \
  --phy_history_file ${MAIN_DIR}output/dialogues/physician_agent/ \
  --ass_history_file ${MAIN_DIR}output/dialogues/assistant_agent/ \
  --mode ass_baseline \
  --baseline_model  "$BASELINE_MODEL" \
  --baseline_system_prompt ${MAIN_DIR}prompts/baseline_system.txt \
  --baseline_prompt_template ${MAIN_DIR}prompts/baseline_user.txt \
  --assistant_model "$BASELINE_MODEL" \
  --physician_model "$BASELINE_MODEL" \
  --phy_system_prompt ${MAIN_DIR}prompts/phy_system_prompt.txt \
  --phy_prompt_template ${MAIN_DIR}prompts/phy_user_prompt.txt \
  --ass_system_prompt ${MAIN_DIR}prompts/ass_system_prompt.txt \
  --ass_prompt_template ${MAIN_DIR}prompts/ass_user_prompt.txt \
  --num_rows 5
