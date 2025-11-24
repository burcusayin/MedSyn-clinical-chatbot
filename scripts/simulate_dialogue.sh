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


ollama pull llama3:8b
#ollama run llama3:70b
#ollama run llama3.1:8b
#ollama run llama3.2:3b
ollama pull gemma2:27b

sleep 10

PHYSICIAN_MODEL=gemma2:27b
ASS_MODEL=llama3:8b

ollama ps

python -m src.simulate \
--input_file ${MAIN_DIR}data/mimicIV_1000_random_sample_filtered_parsed_icd10.csv \
--history_file ${MAIN_DIR}output/dialogues/ \
--discharge_data_file ${MAIN_DIR}output/discharge_texts/random/ \
--mode interactive \
--assistant_model $ASS_MODEL \
--physician_model $PHYSICIAN_MODEL \
--ass_history_file ${MAIN_DIR}output/dialogues/assistant_agent/random/ \
--phy_history_file ${MAIN_DIR}output/dialogues/physician_agent/random/ \
--phy_system_prompt ${MAIN_DIR}prompts/phy_system_prompt.txt \
--phy_prompt_template ${MAIN_DIR}prompts/phy_user_prompt.txt \
--ass_system_prompt ${MAIN_DIR}prompts/ass_system_prompt.txt \
--ass_prompt_template ${MAIN_DIR}prompts/ass_user_prompt.txt  
