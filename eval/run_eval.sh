python -m eval.rag_metrics \
--clinic_notes eval/test_data/mimicIV_1000_random_sample_filtered_parsed_icd10.csv \
--log_file eval/test_data/chat__20251126__sml-admin__sml-admin_20251126_092502.log \
--output_file_prefix eval/test_data/20251126_092502 \
--openai_model gpt-4o-mini