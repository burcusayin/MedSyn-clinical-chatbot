#!/bin/bash

#!/usr/bin/env bash
set -euo pipefail

MAIN_DIR="/home/gunel/medSyn/"

# Go to project root
[[ -d "$MAIN_DIR" ]] || die "MAIN_DIR not found: $MAIN_DIR"
cd "$MAIN_DIR"

[ -f src/__init__.py ] || touch src/__init__.py

pwd

PHYSICIAN_MODEL="openrouter/openai/gpt-oss-20b:free"
ASSISTANT_MODEL="openrouter/openai/gpt-oss-20b:free"
SAFE_MODEL="${PHYSICIAN_MODEL//\//_}"
# --- logging setup ---
LOG_DIR=${MAIN_DIR}logs/ablationLogs
LOG_FILE="$LOG_DIR/simulate_two_agent_${SAFE_MODEL}_$(date +%Y%m%d_%H%M%S).log"

# Send everything (stdout + stderr) to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
# ------------------------

cd "$MAIN_DIR"


# ollama serve &

# # Wait until Ollama service has been started
# sleep 2

# echo "Server started"


# ollama pull llama3:8b
# #ollama run llama3:70b
# #ollama run llama3.1:8b
# #ollama run llama3.2:3b
# ollama pull gemma2:27b

# sleep 10

# PHYSICIAN_MODEL=gemma2:27b
# ASS_MODEL=llama3:8b

# ollama ps

# python -m src.simulate \
# --input_file ${MAIN_DIR}data/mimicIV_1000_random_sample_filtered_parsed_icd10.csv \
# --history_file ${MAIN_DIR}output/dialogues/ \
# --discharge_data_file ${MAIN_DIR}output/discharge_texts/random/ \
# --mode interactive \
# --assistant_model $ASS_MODEL \
# --physician_model $PHYSICIAN_MODEL \
# --ass_history_file ${MAIN_DIR}output/dialogues/assistant_agent/random/ \
# --phy_history_file ${MAIN_DIR}output/dialogues/physician_agent/random/ \
# --phy_system_prompt ${MAIN_DIR}prompts/phy_system_prompt.txt \
# --phy_prompt_template ${MAIN_DIR}prompts/phy_user_prompt.txt \
# --ass_system_prompt ${MAIN_DIR}prompts/ass_system_prompt.txt \
# --ass_prompt_template ${MAIN_DIR}prompts/ass_user_prompt.txt  

python -m src.simulate \
  --input_file ${MAIN_DIR}data/patient_cases.csv \
  --history_file ${MAIN_DIR}output/dialogues/two_agents/ \
  --discharge_data_file ${MAIN_DIR}output/discharge_texts/ \
  --phy_history_file ${MAIN_DIR}output/dialogues/two_agents/phy/ \
  --ass_history_file ${MAIN_DIR}output/dialogues/two_agents/ass/ \
  --mode two-agent \
  --baseline_model "$PHYSICIAN_MODEL" \
  --baseline_system_prompt ${MAIN_DIR}prompts/baseline_system.txt \
  --baseline_prompt_template ${MAIN_DIR}prompts/baseline_user.txt \
  --assistant_model "$ASSISTANT_MODEL" \
  --physician_model "$PHYSICIAN_MODEL" \
  --phy_system_prompt ${MAIN_DIR}prompts/phy_system_prompt.txt \
  --phy_prompt_template ${MAIN_DIR}prompts/phy_user_prompt.txt \
  --ass_system_prompt ${MAIN_DIR}prompts/ass_system_prompt.txt \
  --ass_prompt_template ${MAIN_DIR}prompts/ass_user_prompt.txt \
  --num_rows 5
