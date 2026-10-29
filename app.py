from pathlib import Path
from html import escape
import hashlib
import json
import tempfile
import numpy as np
import pandas as pd
import pymupdf
import streamlit as st
from src.inference import predict_volume,measurements,save_mask
from src.visualization import overlay

st.set_page_config(
    page_title="Liver Tumor Segmentation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    :root {
        --bg: #0b1220;
        --panel: #111b2b;
        --panel-2: #13233d;
        --line: rgba(148, 163, 184, 0.2);
        --text: #e2e8f0;
        --muted: #94a3b8;
        --primary: #38bdf8;
        --accent: #22c55e;
        --warning: #f59e0b;
        --danger: #f87171;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(14, 165, 233, 0.14), transparent 35%),
            radial-gradient(ellipse at 92% 16%, rgba(99, 102, 241, 0.12), transparent 32%),
            linear-gradient(180deg, #08111d 0%, #0b1424 55%, #0c1728 100%);
        color: var(--text);
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
        width: 0 !important;
        min-width: 0 !important;
    }
    [data-testid="stAppViewContainer"] > .main {
        margin-left: 0 !important;
    }
    h1, h2, h3, h4 {
        color: #f8fafc;
        letter-spacing: -0.03em;
    }
    .stMetric {
        background: linear-gradient(180deg, rgba(17,27,43,0.95), rgba(19,35,61,0.95));
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.28);
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.72), rgba(15, 23, 42, 0.76));
        border: 1px solid rgba(186, 230, 253, 0.2);
        border-radius: 18px;
        padding: 0.7rem 0.9rem;
        box-shadow: 0 12px 28px rgba(2, 8, 23, 0.3), inset 0 1px rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        animation: metric-enter 520ms ease both;
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        border-color: rgba(125, 211, 252, 0.48);
        box-shadow: 0 16px 34px rgba(2, 8, 23, 0.4), 0 0 20px rgba(56, 189, 248, 0.08);
    }
    @keyframes metric-enter {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }
    div[data-testid="metric-container"] > div {
        background: transparent;
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .stAlert {
        border-radius: 12px;
        background: rgba(15,23,42,0.9);
        border: 1px solid var(--line);
        color: var(--text);
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.9);
        color: var(--text);
        border: 1px solid var(--line);
        border-radius: 10px;
    }
    .stFileUploader > div {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid var(--line);
        border-radius: 12px;
    }
    .stTabs [role="tablist"] {
        gap: 0.5rem;
    }
    .stTabs [role="tab"] {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid var(--line);
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1rem;
    }
    .stDataFrame, .stJson {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid var(--line);
        border-radius: 12px;
    }
    .topbar {
        background: linear-gradient(90deg, rgba(12, 22, 36, 0.98), rgba(18, 34, 58, 0.98));
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 18px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.2);
    }
    .brand {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: 0.02em;
    }
    .status-chip {
        display: inline-block;
        background: rgba(34,197,94,0.15);
        color: #bbf7d0;
        border: 1px solid rgba(34,197,94,0.3);
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .record-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.78));
        border: 1px solid rgba(186, 230, 253, 0.16);
        border-radius: 20px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 14px 34px rgba(2, 6, 23, 0.28), inset 0 1px rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
    }
    .hero-panel {
        background: linear-gradient(135deg, rgba(14, 116, 144, 0.22), rgba(59, 130, 246, 0.08), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(125, 211, 252, 0.22);
        border-radius: 22px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 32px rgba(14, 116, 144, 0.12);
    }
    .hero-title {
        font-size: clamp(1.8rem, 2.5vw, 2.7rem);
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0;
        color: #f8fafc;
    }
    .hero-sub {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-top: 0.45rem;
    }
    .timeline-card, .report-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.68), rgba(15, 23, 42, 0.78));
        border: 1px solid rgba(186, 230, 253, 0.16);
        border-radius: 20px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 14px 34px rgba(2, 6, 23, 0.26), inset 0 1px rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
    }
    .timeline-title, .report-kicker {
        color: #7dd3fc;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.95rem;
    }
    .timeline-step {
        position: relative;
        display: flex;
        gap: 0.75rem;
        padding: 0 0 1rem;
        color: #e2e8f0;
    }
    .timeline-step:not(:last-child)::before {
        content: "";
        position: absolute;
        left: 0.49rem;
        top: 1.05rem;
        bottom: 0.05rem;
        width: 1px;
        background: linear-gradient(rgba(34, 211, 238, 0.65), rgba(34, 211, 238, 0.12));
    }
    .timeline-dot {
        z-index: 1;
        width: 1rem;
        height: 1rem;
        flex: 0 0 1rem;
        margin-top: 0.08rem;
        border-radius: 50%;
        background: #22d3ee;
        border: 3px solid rgba(8, 47, 73, 0.9);
        box-shadow: 0 0 14px rgba(34, 211, 238, 0.44);
    }
    .timeline-step strong { display: block; font-size: 0.86rem; }
    .timeline-step small { color: #94a3b8; font-size: 0.74rem; }
    .report-card { margin-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.48);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 16px;
        padding: 0.35rem;
        backdrop-filter: blur(14px);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 11px;
        padding: 0.6rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(14, 165, 233, 0.16);
    }
    .stButton > button {
        width: 100%;
        min-height: 2.8rem;
        border-radius: 12px;
        font-weight: 700;
        transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        border-color: rgba(125, 211, 252, 0.6);
    }
    .record-header {
        font-size: 0.8rem;
        color: #7dd3fc;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .record-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding: 0.65rem 0;
        border-bottom: 1px solid rgba(148,163,184,0.12);
    }
    .record-row:last-child {
        border-bottom: none;
    }
    .record-label {
        color: var(--muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .record-value {
        color: #f8fafc;
        font-weight: 700;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
  <div style="display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;">
    <div class="brand">Liver Tumor AI Suite</div>
    <div class="status-chip">Clinical prototype</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-panel">
    <div class="hero-title">Computer-Aided Liver Tumor Segmentation</div>
    <div class="hero-sub">Clinical analysis dashboard for CT volume review, segmentation insights, and structured reporting.</div>
</div>
""", unsafe_allow_html=True)

if "patient_confirmed" not in st.session_state:
    st.session_state.patient_confirmed = False
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "patient_number" not in st.session_state:
    st.session_state.patient_number = 0
if "patient_id" not in st.session_state:
    st.session_state.patient_id = ""
if "workspace_section" not in st.session_state:
    st.session_state.workspace_section = "AI - Assisted"

if not st.session_state.patient_confirmed:
    st.markdown("### Step 1 · Patient details")
    st.caption("Enter the patient record to continue to the CT scan workspace.")
    with st.form("patient_details_form"):
        name_col, number_col, id_col = st.columns(3)
        with name_col:
            patient_name_input = st.text_input(
                "Patient name", value=st.session_state.patient_name,
                placeholder="Enter full patient name",
            )
        with number_col:
            patient_number_input = st.number_input(
                "Patient number", min_value=0, step=1, format="%d",
                value=int(st.session_state.patient_number),
            )
        with id_col:
            patient_id_input = st.text_input(
                "Patient ID", value=st.session_state.patient_id,
                placeholder="Enter patient ID",
            )
        continue_to_scan = st.form_submit_button(
            "Save patient details · Continue to CT workspace →",
            type="primary", use_container_width=True,
        )

    if continue_to_scan:
        if not patient_name_input.strip() or not patient_id_input.strip():
            st.error("Enter both a patient name and patient ID to continue.")
        else:
            st.session_state.patient_name = patient_name_input.strip()
            st.session_state.patient_number = int(patient_number_input)
            st.session_state.patient_id = patient_id_input.strip()
            st.session_state.patient_confirmed = True
            st.session_state.workspace_section = "AI - Assisted"
            st.rerun()
    st.stop()

patient_name = st.session_state.patient_name
patient_number = st.session_state.patient_number
patient_id = st.session_state.patient_id

patient_col, return_col = st.columns([4, 1])
with patient_col:
    st.markdown(
        f"**Current patient:** {escape(patient_name)} · ID: {escape(patient_id)} · "
        f"Record no. {int(patient_number)}"
    )
with return_col:
    if st.button("← Edit patient", use_container_width=True):
        st.session_state.patient_confirmed = False
        st.rerun()

active_section = st.segmented_control(
    "Choose a workspace page",
    ["AI - Assisted", "CT Review", "Segmentation", "Treatment Discussion", "Clinical Reporting"],
    selection_mode="single",
    default=st.session_state.workspace_section,
    key="workspace_section",
    label_visibility="collapsed",
)
st.warning("Educational research prototype only. Not for diagnosis or treatment decisions.")

def resolve_checkpoint_path():
    candidates = [
        Path("checkpoints/best_model_msd_extended.pt"),
        Path("checkpoints/best_model_msd.pt"),
        Path("checkpoints/best_model_task03.pt"),
        Path("checkpoints/best_model.pt"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]

with st.expander("Dataset, training, and evaluation status", expanded=True):
    dataset_path = Path("data/Task03_Liver/Task_liver_dataset/Task03_Liver/Task03_Liver")
    inspection_path = Path("outputs/dataset_inspection.json")
    split_path = Path("data/msd_task03_liver_splits.json")
    status_candidates = [
        (
            Path("outputs/msd_extended_training_status.json"),
            Path("outputs/msd_extended_training_history.csv"),
        ),
        (
            Path("outputs/msd_task03_training_status.json"),
            Path("outputs/msd_task03_training_history.csv"),
        ),
    ]
    training_status_path, training_history_path = next(
        (
            (status_path, history_path)
            for status_path, history_path in status_candidates
            if status_path.is_file()
        ),
        status_candidates[-1],
    )
    evaluation_path = Path("outputs/validation_metrics_aggregate.json")
    training_status = {}
    if inspection_path.is_file():
        try:
            inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
            inspected_path = Path(inspection.get("dataset_path", "")).resolve()
            if inspected_path == dataset_path.resolve():
                st.write("**Inspected dataset path:**", inspected_path)
                st.metric(
                    "Valid NIfTI image-mask pairs",
                    int(inspection.get("valid_nifti_pair_count", 0)),
                )
                st.write(
                    "**Observed mask values:**",
                    inspection.get("observed_mask_label_values", "Not reported"),
                )
                st.write(
                    "**Label validation:**",
                    inspection.get("label_validation", {}),
                )
            else:
                st.write(f"**Dataset path:** `{dataset_path}`")
                st.info(
                    "The existing inspection report belongs to a different dataset "
                    "and is not used for this Task03 workflow."
                )
        except (OSError, json.JSONDecodeError) as status_error:
            st.error(f"Could not read dataset inspection report: {status_error}")
    else:
        st.write(f"**Dataset path:** `{dataset_path}`")
        st.info("Selected dataset has not yet been checksum-verified and inspected.")

    if split_path.is_file():
        try:
            split_status = json.loads(split_path.read_text(encoding="utf-8"))
            split_cols = st.columns(3)
            split_cols[0].metric("Training cases", len(split_status.get("train", [])))
            split_cols[1].metric("Validation cases", len(split_status.get("val", [])))
            split_cols[2].metric(
                "External imagesTs cases",
                split_status.get("external_test_case_count", 0),
            )
            if split_status.get("semantic_label_mapping_confirmed_by_user"):
                st.write(
                    "**User-verified raw label map:**",
                    split_status.get("raw_label_map", {}),
                )
        except (OSError, json.JSONDecodeError) as status_error:
            st.error(f"Could not read selected-dataset split file: {status_error}")
    else:
        st.info("Selected-dataset splits have not been generated.")

    if training_status_path.is_file():
        try:
            training_status = json.loads(
                training_status_path.read_text(encoding="utf-8")
            )
            state = training_status.get("status", "unknown")
            st.write(
                f"**Training:** {state} "
                f"({training_status.get('completed_epochs', 0)}/"
                f"{training_status.get('total_epochs', 0)} epochs)"
            )
            total_epochs = int(training_status.get("total_epochs", 0))
            completed_epochs = int(training_status.get("completed_epochs", 0))
            if total_epochs > 0:
                st.progress(min(completed_epochs / total_epochs, 1.0))
            if training_history_path.is_file():
                history = pd.read_csv(training_history_path)
                if not history.empty and {"train_loss", "val_loss"}.issubset(history.columns):
                    if len(history) > 1:
                        st.line_chart(
                            history.set_index("epoch")[["train_loss", "val_loss"]]
                        )
                    else:
                        st.dataframe(history, hide_index=True)
        except (OSError, json.JSONDecodeError, pd.errors.ParserError) as status_error:
            st.error(f"Could not read training status/history: {status_error}")
    else:
        st.info("No Task03 training run has been recorded.")

    checkpoint_path = resolve_checkpoint_path()
    if checkpoint_path.is_file():
        if training_status_path.is_file() and training_status.get("status") == "completed":
            st.success(f"Checkpoint exists: `{checkpoint_path}`")
        else:
            st.warning(
                f"Checkpoint exists at `{checkpoint_path}`, but training provenance is unverified."
            )
    else:
        st.info(f"No checkpoint saved at `{checkpoint_path}`.")

    if evaluation_path.is_file():
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            st.markdown("**Validation evaluation**")
            st.json(evaluation)
        except (OSError, json.JSONDecodeError) as status_error:
            st.error(f"Could not read evaluation results: {status_error}")

ck = resolve_checkpoint_path()
if not ck.exists():
    st.error("No trained checkpoint was found. Train a model before inference.")
    st.code(
        "python -m src.train --splits data/msd_task03_liver_splits.json "
        "--epochs 1 --batch-size 1 --slices-per-case 4 --patch-size 256 "
        "--workers 0 --checkpoint checkpoints/best_model_msd.pt"
    )
    st.stop()

st.markdown("### Uploads")
ct_col, pdf_col = st.columns(2)
with ct_col:
    f=st.file_uploader("Upload CT NIfTI (.nii or .nii.gz)",type=["nii","gz"])
with pdf_col:
    pdf_report = st.file_uploader("Upload PDF report", type=["pdf"])

if pdf_report is not None:
    st.success(f"PDF report uploaded: {pdf_report.name} ({pdf_report.size} bytes)")
    st.caption("PDF preview is for viewing only. To run AI segmentation, upload a 3D NIfTI volume in the CT uploader.")
    with st.expander("Open CT PDF preview", expanded=True):
        try:
            pdf_document = pymupdf.open(stream=pdf_report.getvalue(), filetype="pdf")
            if pdf_document.page_count == 0:
                st.warning("This PDF contains no pages to preview.")
            else:
                selected_page = st.number_input(
                    "PDF page", min_value=1, max_value=pdf_document.page_count,
                    value=1, step=1, key="pdf_preview_page",
                )
                page = pdf_document.load_page(int(selected_page) - 1)
                page_image = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                st.image(page_image.tobytes("png"), caption=f"Page {int(selected_page)} of {pdf_document.page_count}", use_container_width=True)
                pdf_document.close()
        except Exception as pdf_error:
            st.error(f"Could not preview this PDF. It may be damaged, encrypted, or unsupported: {pdf_error}")
    st.download_button(
        "Download uploaded PDF", pdf_report.getvalue(), pdf_report.name,
        "application/pdf", key="download_uploaded_pdf",
    )

if pdf_report is not None and f is None:
    st.info("PDF preview is ready. Upload a .nii or .nii.gz CT volume separately to run segmentation.")

def render_treatment_discussion():
    st.markdown("### Treatment discussion")
    st.warning(
        "This section lists general treatment categories for discussion only. "
        "It does not recommend a treatment for this patient and does not interpret the AI segmentation as a diagnosis. "
        "Treatment decisions must be made by the patient's qualified oncology and liver-care team."
    )
    st.markdown("#### General options a specialist team may consider")
    st.caption(
        "The appropriate options depend on the confirmed tumor type and stage, location and extent of disease, "
        "liver function, overall health, prior treatments, and the patient's goals. Some options may not apply."
    )
    with st.expander("Surgery or transplant evaluation"):
        st.write(
            "Specialists may assess whether a tumor can be removed surgically or whether transplant evaluation "
            "is appropriate. Eligibility depends on the diagnosis, disease extent, liver health, and other clinical criteria."
        )
    with st.expander("Local tumor-directed treatments"):
        st.write(
            "Depending on the confirmed diagnosis and clinical circumstances, a team may discuss ablation, "
            "embolization-based procedures, or radiation-based approaches. These treatments are not suitable for everyone."
        )
    with st.expander("Medication-based treatment"):
        st.write(
            "For some tumor types or stages, specialists may discuss systemic medicines such as targeted therapy, "
            "immunotherapy, or chemotherapy. The choice requires review of the exact diagnosis, test results, "
            "expected benefits, and risks."
        )
    with st.expander("Monitoring, supportive care, or clinical trials"):
        st.write(
            "Depending on the diagnosis and patient preferences, a care team may discuss active monitoring, "
            "symptom-focused supportive care, or eligibility for a clinical trial."
        )
    st.markdown("#### Questions to take to the care team")
    st.markdown(
        "- What is the confirmed diagnosis and stage, and is additional testing or biopsy needed?\n"
        "- Which specialists should review the case together?\n"
        "- What options apply to this diagnosis, and what are their expected benefits, risks, and alternatives?\n"
        "- How do liver function, other health conditions, and personal goals affect the choices?\n"
        "- Is a second opinion or clinical trial worth discussing?"
    )
    st.info(
        "The CT segmentation shown in this research prototype is not clinically validated and must not be used "
        "to select, start, stop, or change treatment."
    )

if f is None:
    st.session_state.pop("segmentation_result",None)
    if active_section == "AI - Assisted":
        st.info("AI-assisted analysis is ready. Upload a 3D .nii or .nii.gz CT volume above to run the segmentation model.")
    elif active_section == "CT Review":
        st.info("CT Review needs a 3D NIfTI volume for slice navigation. An uploaded PDF can be viewed in the PDF preview above.")
    elif active_section == "Segmentation":
        st.info("Segmentation results appear here after you upload a supported 3D NIfTI CT volume.")
    elif active_section == "Treatment Discussion":
        render_treatment_discussion()
    else:
        st.info("Clinical Reporting is available after segmentation. You can preview and download an uploaded PDF above.")

if f:
    with tempfile.TemporaryDirectory() as td:
        try:
            scan_bytes=f.getvalue()
            checkpoint_stat=ck.stat()
            cache_key=(
                hashlib.sha256(scan_bytes).hexdigest(),
                checkpoint_stat.st_mtime_ns,
                checkpoint_stat.st_size,
            )
            cached_result=st.session_state.get("segmentation_result")
            if cached_result is not None and cached_result[0] == cache_key:
                _,vol,pred,obj=cached_result
                st.caption("Segmentation is ready. Reusing the result for this scan.")
            else:
                st.session_state["segmentation_result"]=None
                with st.status("Preparing CT scan...",expanded=True) as inference_status:
                    progress=st.progress(0,text="Loading CT volume...")
                    def update_inference_progress(completed,total):
                        progress.progress(
                            completed/total,
                            text=f"Analyzing slices {completed:,} of {total:,}...",
                        )
                        inference_status.update(
                            label=f"Analyzing CT scan ({completed:,}/{total:,} slices)",
                            state="running",
                        )
                    vol,pred,obj=predict_volume(
                        scan_bytes,
                        ck,
                        progress_callback=update_inference_progress,
                    )
                    st.session_state["segmentation_result"]=(cache_key,vol,pred,obj)
                    inference_status.update(
                        label=f"Segmentation complete · {vol.shape[2]:,} slices",
                        state="complete",
                        expanded=False,
                    )
            patient_meta={
                "patient_name": patient_name,
                "patient_number": int(patient_number),
                "patient_id": patient_id,
            }
            metrics=measurements(pred,obj)
            out=Path(td)/"predicted_segmentation.nii.gz"
            save_mask(pred,obj,out)
            tumor_pct = metrics.get("tumor_percent_of_liver_plus_tumor")
            report_text = f"""
LIVER TUMOR SEGMENTATION — RESEARCH SUMMARY
Educational research prototype only. Not for diagnosis or treatment decisions.

Patient: {patient_name or 'Unknown'}
Patient Number: {int(patient_number)}
Patient ID: {patient_id or 'N/A'}
Uploaded CT: {f.name}

Segmentation Summary:
- Tumor voxels: {metrics.get('tumor_voxels', 0)}
- Liver + tumor voxels: {metrics.get('liver_plus_tumor_voxels', 0)}
- Tumor % of liver: {f'{tumor_pct:.2f}%' if tumor_pct is not None else 'N/A'}
- Predicted tumor volume: {metrics.get('predicted_tumor_volume_cm3')} cm^3
- Predicted liver+tumor volume: {metrics.get('predicted_liver_plus_tumor_volume_cm3')} cm^3

Imaging Summary:
- Volume shape: {vol.shape}
- Spacing: {obj.header.get_zooms()[:3]}
- Spatial unit: {obj.header.get_xyzt_units()[0]}
"""

            if active_section == "CT Review":
                st.markdown("### CT Scan Viewer")
                z=st.slider("Axial slice",0,vol.shape[2]-1,vol.shape[2]//2)
                window_col, level_col = st.columns(2)
                with window_col:
                    window = st.slider("Window width", min_value=50, max_value=2000, value=400, step=10)
                with level_col:
                    level = st.slider("Window level", min_value=-200, max_value=400, value=50, step=10)

                ct_slice = np.rot90(vol[:,:,z])
                ct_windowed = np.clip(ct_slice, level - window/2, level + window/2)
                ct_windowed = (ct_windowed - (level - window/2)) / max(window, 1)
                ct_windowed = np.clip(ct_windowed, 0, 1)
                scan_col, overlay_col = st.columns(2)
                with scan_col:
                    st.markdown("#### CT Scan")
                    st.image(ct_windowed, clamp=True, use_container_width=True)
                with overlay_col:
                    st.markdown("#### Predicted overlay")
                    st.pyplot(overlay(vol[:,:,z],pred[:,:,z]),clear_figure=True)

            elif active_section == "AI - Assisted":
                st.markdown("### Case dashboard")
                timeline_col, overview_col = st.columns([0.85, 2.15], gap="large")
                with timeline_col:
                    st.markdown("""
                    <div class="timeline-card">
                      <div class="timeline-title">Patient timeline</div>
                      <div class="timeline-step"><span class="timeline-dot"></span><div><strong>Patient record</strong><small>Information entered</small></div></div>
                      <div class="timeline-step"><span class="timeline-dot"></span><div><strong>CT scan uploaded</strong><small>Volume loaded for review</small></div></div>
                      <div class="timeline-step"><span class="timeline-dot"></span><div><strong>Segmentation complete</strong><small>Model output generated</small></div></div>
                      <div class="timeline-step"><span class="timeline-dot"></span><div><strong>Review package ready</strong><small>Analysis and exports available</small></div></div>
                    </div>
                    """, unsafe_allow_html=True)
                with overview_col:
                    dash_col1, dash_col2 = st.columns(2)
                    dash_col3, dash_col4 = st.columns(2)
                    with dash_col1:
                        st.metric("Patient", patient_name if patient_name else "Unknown")
                    with dash_col2:
                        st.metric("Patient ID", patient_id if patient_id else "N/A")
                    with dash_col3:
                        st.metric("Volume dimensions", f"{vol.shape[0]} × {vol.shape[1]} × {vol.shape[2]}")
                    with dash_col4:
                        st.metric("Tumor volume", f"{metrics.get('predicted_tumor_volume_cm3', 0):.2f} cm³")
                    st.markdown("#### Segmentation snapshot")
                    result_col1, result_col2, result_col3 = st.columns(3)
                    with result_col1:
                        st.metric("Tumor voxels", f"{int(metrics.get('tumor_voxels', 0)):,}")
                    with result_col2:
                        st.metric("Liver + tumor voxels", f"{int(metrics.get('liver_plus_tumor_voxels', 0)):,}")
                    with result_col3:
                        st.metric("Tumor share", f"{tumor_pct:.2f}%" if tumor_pct is not None else "N/A")
                    st.caption(f"Voxel spacing: {tuple(obj.header.get_zooms()[:3])} · Spatial unit: {obj.header.get_xyzt_units()[0]}")

            elif active_section == "Segmentation":
                st.markdown("### Segmentation results and measurement analysis")
                seg_col1, seg_col2, seg_col3 = st.columns(3)
                with seg_col1:
                    st.metric("Tumor voxels", f"{int(metrics.get('tumor_voxels', 0)):,}")
                with seg_col2:
                    st.metric("Liver + tumor voxels", f"{int(metrics.get('liver_plus_tumor_voxels', 0)):,}")
                with seg_col3:
                    st.metric("Tumor share", f"{tumor_pct:.2f}%" if tumor_pct is not None else "N/A")
                analysis_col1, analysis_col2 = st.columns(2, gap="large")
                with analysis_col1:
                    st.markdown("#### Segmentation measurements")
                    st.json({
                        "tumor_voxels": metrics.get("tumor_voxels", 0),
                        "liver_plus_tumor_voxels": metrics.get("liver_plus_tumor_voxels", 0),
                        "tumor_percent_of_liver_plus_tumor": tumor_pct,
                        "predicted_tumor_volume_cm3": metrics.get("predicted_tumor_volume_cm3"),
                        "predicted_liver_plus_tumor_volume_cm3": metrics.get("predicted_liver_plus_tumor_volume_cm3"),
                    })
                with analysis_col2:
                    st.markdown("#### Image and patient metadata")
                    st.json({
                        "patient": patient_meta,
                        "study_shape": list(vol.shape),
                        "voxel_spacing": list(map(float, obj.header.get_zooms()[:3])),
                        "spatial_unit": obj.header.get_xyzt_units()[0],
                        "uploaded_ct": f.name,
                    })
                st.info("Measurements are model-generated research outputs and require qualified review before any clinical use.")

            elif active_section == "Treatment Discussion":
                render_treatment_discussion()

            elif active_section == "Clinical Reporting":
                st.markdown("### Reports & exports")
                st.markdown("""
                <div class="report-card">
                  <div class="report-kicker">Export center</div>
                  <div>Review the generated research summary and download the available case files.</div>
                </div>
                """, unsafe_allow_html=True)
                st.text_area("Research report preview", report_text.strip(), height=300)
                export_col1, export_col2 = st.columns(2)
                with export_col1:
                    st.download_button(
                        "Download report (.txt)", report_text.strip(),
                        f"liver_segmentation_report_{patient_id or 'case'}.txt",
                        "text/plain", use_container_width=True,
                    )
                with export_col2:
                    st.download_button(
                        "Download segmentation mask (.nii.gz)", out.read_bytes(),
                        "predicted_segmentation.nii.gz", "application/gzip",
                        use_container_width=True,
                    )
                if pdf_report is not None:
                    st.download_button(
                        f"Download attached PDF ({pdf_report.name})", pdf_report.getvalue(),
                        pdf_report.name, "application/pdf", use_container_width=True,
                    )
        except Exception as e:
            st.error(f"Could not process file: {e}")
