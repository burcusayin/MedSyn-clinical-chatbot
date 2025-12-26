python -m eval.rag_metrics \
--clinic_notes eval/test_data/mimicIV_1000_random_sample_filtered_parsed_icd10.csv \
--log_file eval/session_2/dialogue_interactive__phy1__phy1_20251219_163456.jsonl \
--output_file_prefix eval/test_data/dialogue_interactive__phy1__phy1_20251219_163456 \
--openai_model gpt-4o-mini