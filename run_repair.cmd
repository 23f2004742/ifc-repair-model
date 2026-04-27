@echo off
python src\ifc_repair_ai.py run --input-dir data\input --output-dir data\output\repaired_ifc --report-out data\output\fault_report.jsonl --metrics-out data\output\metrics.json --model-out model\repair_profile.json --classifier-out model\fault_classifier.json --fast
