#!/bin/bash

MAIN_DIR="/home/gunel/medSyn/"

# Go to project root
[[ -d "$MAIN_DIR" ]] || die "MAIN_DIR not found: $MAIN_DIR"
cd "$MAIN_DIR"
log "PWD=$(pwd)"

[ -f src/__init__.py ] || touch src/__init__.py

ollama serve &

# Wait until Ollama service has been started
sleep 2

echo "Server started"

#ollama run llama3:8b
#ollama run llama3:70b
#ollama run llama3.1:8b
#ollama run llama3.2:3b
#ollama run gemma2:27b

BASELINE_MODEL=gemma2:27b

ollama ps

python -m src.simulate \
--input_file ${MAIN_DIR}data/mimicIV_1000_random_sample_filtered_parsed_icd10.csv \
--history_file ${MAIN_DIR}output/dialogues/assistant_agent/random/  \
--discharge_data_file ${MAIN_DIR}output/discharge_texts/random/ \
--mode ass_baseline \
--baseline_model $BASELINE_MODEL \
--baseline_system_prompt ${MAIN_DIR}prompts/baseline_system.txt \
--baseline_prompt_template ${MAIN_DIR}prompts/baseline_user.txt 
