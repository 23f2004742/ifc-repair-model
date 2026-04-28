from __future__ import annotations

import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))

from ifc_repair_ai import load_classifier, load_model, repair_dir, train_fault_classifier, train_model, build_metrics


ROOT = Path(__file__).parent
INPUT_DIR = ROOT / "data" / "input"
OUTPUT_DIR = ROOT / "data" / "output" / "repaired_ifc"
REPORT_PATH = ROOT / "data" / "output" / "fault_report.jsonl"
METRICS_PATH = ROOT / "data" / "output" / "metrics.json"
PROFILE_PATH = ROOT / "model" / "repair_profile.json"
CLASSIFIER_PATH = ROOT / "model" / "fault_classifier.json"


def ensure_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)


def clear_folder(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for path in folder.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def save_upload(uploaded_file) -> list[Path]:
    saved: list[Path] = []
    name = uploaded_file.name
    target = INPUT_DIR / name
    target.write_bytes(uploaded_file.getbuffer())
    if name.lower().endswith(".zip"):
        extract_dir = INPUT_DIR / target.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "r") as archive:
            archive.extractall(extract_dir)
        for ifc_path in extract_dir.rglob("*.ifc"):
            flat_target = INPUT_DIR / ifc_path.name
            if flat_target.exists():
                flat_target = INPUT_DIR / f"{ifc_path.stem}_{len(saved)}{ifc_path.suffix}"
            shutil.copy2(ifc_path, flat_target)
            saved.append(flat_target)
        target.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)
    elif name.lower().endswith(".ifc"):
        saved.append(target)
    return saved


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def make_output_zip() -> Path:
    archive_base = ROOT / "data" / "output" / "ifc_repair_outputs"
    zip_path = shutil.make_archive(str(archive_base), "zip", ROOT / "data" / "output")
    return Path(zip_path)


def run_repair(validate: bool) -> dict:
    profile = train_model(INPUT_DIR, PROFILE_PATH)
    train_fault_classifier(INPUT_DIR, CLASSIFIER_PATH)
    classifier = load_classifier(CLASSIFIER_PATH)
    start = time.perf_counter()
    results = repair_dir(INPUT_DIR, OUTPUT_DIR, REPORT_PATH, profile, classifier=classifier, validate=validate)
    metrics = build_metrics(results, elapsed_seconds=time.perf_counter() - start, profile=profile, classifier=classifier, validate=validate)
    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump({"metrics": metrics, "files": results}, fh, indent=2)
    return {"metrics": metrics, "files": results}


def main() -> None:
    st.set_page_config(page_title="IFC Repair AI", layout="wide")
    ensure_dirs()

    st.title("IFC Repair AI")
    st.caption("Upload corrupted IFC files or a ZIP, run repair, download repaired IFCs and the JSONL fault report.")

    left, right = st.columns([1, 1])
    with left:
        uploaded_files = st.file_uploader("Upload .ifc files or a .zip folder", type=["ifc", "zip"], accept_multiple_files=True)
        clear_input = st.checkbox("Clear existing input before saving upload", value=True)
        full_validation = st.checkbox("Run full IfcOpenShell validation", value=False)
        run_clicked = st.button("Run repair", type="primary")

    with right:
        st.subheader("Current model parameters")
        profile = load_json(PROFILE_PATH)
        if profile:
            st.json(
                {
                    "training_files": profile.get("training_files"),
                    "window_overall_height": profile.get("window_overall_height"),
                    "default_extrusion_depth": profile.get("default_extrusion_depth"),
                    "polyline_min_thickness": profile.get("polyline_min_thickness"),
                }
            )
        else:
            st.info("No trained repair profile yet.")

    if run_clicked:
        if clear_input:
            clear_folder(INPUT_DIR)
            clear_folder(OUTPUT_DIR)
        saved_files: list[Path] = []
        for uploaded in uploaded_files or []:
            saved_files.extend(save_upload(uploaded))
        if not list(INPUT_DIR.glob("*.ifc")):
            st.error("No .ifc files found. Upload at least one IFC file or a ZIP containing IFC files.")
            return

        with st.spinner("Training classifier and repairing IFC files..."):
            result = run_repair(validate=full_validation)
        st.success("Repair completed.")
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result") or load_json(METRICS_PATH)
    if result:
        metrics = result.get("metrics", result)
        st.subheader("Run metrics")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Files", metrics.get("files_processed", 0))
        metric_cols[1].metric("Valid outputs", metrics.get("valid_outputs", 0))
        metric_cols[2].metric("Schema rate", metrics.get("schema_validity_rate", 0))
        metric_cols[3].metric("Repair loss", metrics.get("repair_loss_proxy", 0))
        metric_cols[4].metric("Efficiency %", metrics.get("repair_efficiency_percent", 0))

        st.subheader("Classifier")
        st.json(metrics.get("fault_classifier", {}))

    st.subheader("Outputs")
    report_exists = REPORT_PATH.exists()
    if report_exists:
        st.download_button("Download fault_report.jsonl", REPORT_PATH.read_bytes(), file_name="fault_report.jsonl")
        st.text_area("Fault report preview", REPORT_PATH.read_text(encoding="utf-8")[:5000], height=220)

    repaired_files = sorted(OUTPUT_DIR.glob("*.ifc"))
    if repaired_files:
        st.write(f"Repaired IFC files: {len(repaired_files)}")
        for path in repaired_files[:20]:
            st.write(path.name)
        zip_path = make_output_zip()
        st.download_button("Download all outputs as ZIP", zip_path.read_bytes(), file_name="ifc_repair_outputs.zip")


if __name__ == "__main__":
    main()
