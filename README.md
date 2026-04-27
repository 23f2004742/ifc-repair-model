# IFC Repair AI

CPU-friendly IFC repair pipeline for the Hackathon Problem 2 dataset. It combines:

- a learned repair profile from IFC files
- a scikit-learn `RandomForestClassifier` fault classifier
- deterministic IFC repair rules for schema-safe output
- JSONL fault reporting and metrics

The project is designed to run on a normal laptop CPU, without a billion-parameter LLM.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Folder Layout

Put corrupted/test IFC files here:

```text
data/input/
```

Generated outputs are written here:

```text
data/output/repaired_ifc/
data/output/fault_report.jsonl
data/output/metrics.json
model/repair_profile.json
model/fault_classifier.json
model/fault_classifier.joblib
```

Large datasets, repaired IFCs, ZIPs, and binary model artifacts are ignored by Git.

## One-Command Run

```powershell
.\run_repair.cmd
```

This runs fast bulk mode:

```powershell
python src\ifc_repair_ai.py run --input-dir data\input --output-dir data\output\repaired_ifc --report-out data\output\fault_report.jsonl --metrics-out data\output\metrics.json --model-out model\repair_profile.json --classifier-out model\fault_classifier.json --fast
```

Remove `--fast` for full before/after IfcOpenShell validation. It is slower but gives stricter validation metrics.

## Train Only

```powershell
python src\ifc_repair_ai.py train --input-dir data\input --model-out model\repair_profile.json --classifier-out model\fault_classifier.json
```

## Repair Only

```powershell
python src\ifc_repair_ai.py repair --input-dir data\input --output-dir data\output\repaired_ifc --report-out data\output\fault_report.jsonl --model model\repair_profile.json --classifier model\fault_classifier.json --fast
```

## Fault Report Format

`data/output/fault_report.jsonl` uses the required submission shape:

```json
{"id": "tp_0501", "faults": [{"entity_id": "#386", "fault_type": "missing_required_attribute", "description": "Detected missing_required_attribute at entity #386"}]}
```

Only `id` and `faults` are written to the JSONL submission report. Internal ML predictions remain in `metrics.json`.

## Metrics

Open:

```powershell
Get-Content data\output\metrics.json
```

Useful fields:

- `schema_validity_rate`
- `repair_efficiency_percent`
- `repair_loss_proxy`
- `validation_errors_after`
- `fault_classifier.accuracy`
- `fault_classifier.loss`
- `ml_predicted_faults`

## Repair Classes

Current repair logic handles:

- missing or duplicate `GlobalId`
- invalid required attributes such as bad window height
- invalid `IfcCartesianPoint` coordinates
- truncated `IfcPolyline` geometry
- invalid product representation references
- short column extrusion outliers
- missing `IfcSpace` when wall models have no spaces
- unreadable IFC files as reportable failures

## Colab

Upload a ZIP containing `src`, `model`, `requirements.txt`, and `data/input`, then:

```python
!pip install -r requirements.txt
!python "src/ifc_repair_ai.py" run \
  --input-dir "data/input" \
  --output-dir "data/output/repaired_ifc" \
  --report-out "data/output/fault_report.jsonl" \
  --metrics-out "data/output/metrics.json" \
  --model-out "model/repair_profile.json" \
  --classifier-out "model/fault_classifier.json" \
  --fast
```

