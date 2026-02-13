"""OncoEdge Streamlit Application

Main web interface for oral cancer screening.
Uses configuration-driven pipeline initialization.
"""
import streamlit as st
import numpy as np
from PIL import Image

from src.config import load_config
from src.pipeline.inference_pipeline import OncoEdgePipeline
from src.utils.feedback import FeedbackStore


@st.cache_resource
def load_pipeline():
    """Load pipeline once and cache it.

    Device and model selection driven entirely by config/default.yaml
    and environment variable overrides.

    Returns:
        OncoEdgePipeline: Initialized pipeline
    """
    config = load_config()
    return OncoEdgePipeline(config=config)


@st.cache_resource
def get_feedback_store():
    """Load feedback store once and cache it."""
    config = load_config()
    return FeedbackStore(feedback_dir=config.paths.feedback_dir)


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="OncoEdge - Oral Cancer Screening",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 OncoEdge: Oral Cancer Screening System")
    st.markdown("AI-powered screening using YOLO11n-seg and BiomedCLIP")

    config = load_config()
    device = config.device.resolved

    # Sidebar: Patient Information
    with st.sidebar:
        st.header("📋 Patient Information")
        tobacco_years = st.number_input(
            "Years of Tobacco Use (chewing/gutka/paan)",
            min_value=0, max_value=60, value=0,
            help=f"Smokeless tobacco. Risk normalizes to 1.0 at "
                 f"{config.clinical.tobacco_significant_years} years."
        )
        smoking_years = st.number_input(
            "Years of Smoking (cigarettes/bidis)",
            min_value=0, max_value=60, value=0,
            help=f"Cigarette/bidi smoking. Risk normalizes to 1.0 at "
                 f"{config.clinical.smoking_significant_years} years."
        )

        st.markdown("---")
        st.markdown("### ℹ️ About OncoEdge")
        st.info("This system uses YOLO11n-seg for lesion detection and BiomedCLIP for medical classification.")

        # Show device and model info from config
        st.markdown(f"**Device:** {device.upper()}")
        if device in ['cuda', 'mps']:
            st.success(f"🚀 GPU acceleration enabled ({device.upper()})")
        else:
            st.warning("⚠️ Running on CPU")

        st.markdown(f"**Detector:** `{config.detector.model_path.split('/')[-1]}`")
        classifier_name = config.classifier.model_path.split('/')[-1]
        if len(classifier_name) > 30:
            classifier_name = classifier_name[:27] + "..."
        st.markdown(f"**Classifier:** `{classifier_name}`")

    # Main area: Image Upload
    st.header("📤 Upload Oral Cavity Image")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of the oral cavity"
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, width='stretch')

        # Analysis button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            try:
                with st.spinner("🔄 Loading models and analyzing image..."):
                    # Load pipeline
                    pipeline = load_pipeline()

                    # Prepare patient metadata
                    patient_metadata = {
                        'tobacco_years': tobacco_years,
                        'smoking_years': smoking_years,
                    }

                    # Run inference
                    results = pipeline.process_image(
                        image=np.array(image),
                        patient_metadata=patient_metadata
                    )

                # Display results
                with col2:
                    display_results(results)

                # Feedback section
                display_feedback_form(results, patient_metadata)

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)


def display_results(results):
    """Display analysis results.

    Args:
        results: Dictionary containing analysis results
    """
    # Risk Assessment Header
    risk_colors = {
        'HIGH': '🔴',
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }

    risk_emoji = risk_colors.get(results['risk_level'], '⚪')

    st.subheader(f"{risk_emoji} Risk Assessment: {results['risk_level']}")

    # Risk color for boxes
    risk_box_colors = {
        'HIGH': 'error',
        'MEDIUM': 'warning',
        'LOW': 'success'
    }

    box_type = risk_box_colors.get(results['risk_level'], 'info')

    # Display recommendation
    if box_type == 'error':
        st.error(f"**Recommendation:** {results['recommendation']}")
    elif box_type == 'warning':
        st.warning(f"**Recommendation:** {results['recommendation']}")
    else:
        st.success(f"**Recommendation:** {results['recommendation']}")

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Risk Score", f"{results['risk_score']:.1f}")

    with col2:
        st.metric("Lesions Detected", len(results['detections']))

    with col3:
        if results['detections']:
            max_conf = max(d['fusion_score'] for d in results['detections'])
            st.metric("CLIP Score (max)", f"{max_conf:.1%}")
        else:
            st.metric("CLIP Score (max)", "N/A")

    # Score breakdown
    breakdown = results.get('breakdown')
    if breakdown:
        with st.expander("📐 Risk Score Breakdown"):
            st.write(f"**CLIP base** (clip_score x 10): {breakdown['clip_base']}")
            st.write(f"**Tobacco** (α={breakdown['tobacco_norm']:.2f}): "
                     f"+{breakdown['tobacco_contribution']}")
            st.write(f"**Smoking** (β={breakdown['smoking_norm']:.2f}): "
                     f"+{breakdown['smoking_contribution']}")

    # Visualization
    if results['detections']:
        st.subheader("🎯 Detected Lesions")
        st.image(results['visualization'], width='stretch',
                caption="Annotated image with detected lesions")

        # Detailed detection information
        st.subheader("📊 Detection Details")

        for i, det in enumerate(results['detections']):
            with st.expander(f"Lesion {i+1}: {det['class']} (Confidence: {det['fusion_score']:.1%})"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Classification:** {det['class']}")
                    st.write(f"**Fusion Score:** {det['fusion_score']:.1%}")
                    st.write(f"**YOLO Confidence:** {det['yolo_conf']:.1%}")

                with col2:
                    st.write("**Class Probabilities:**")
                    for cls, score in det['class_scores'].items():
                        st.write(f"- **{cls}:** {score:.1%}")
                        st.progress(score)
    else:
        st.info("ℹ️ No lesions detected in the image.")

    # Performance metrics
    if results.get('timing'):
        with st.expander("⏱️ Performance Metrics"):
            timing = results['timing']
            cols = st.columns(3)
            if 'detection_ms' in timing:
                cols[0].metric("Detection", f"{timing['detection_ms']:.0f}ms")
            if 'classification_ms' in timing:
                cols[1].metric("Classification", f"{timing['classification_ms']:.0f}ms")
            if 'pipeline_total_ms' in timing:
                cols[2].metric("Total Pipeline", f"{timing['pipeline_total_ms']:.0f}ms")

            if results.get('model_info'):
                st.caption("Model Info:")
                for role, info in results['model_info'].items():
                    if isinstance(info, dict):
                        fmt = info.get('format', 'unknown')
                        st.caption(f"  {role}: {fmt}")


def display_feedback_form(results, patient_metadata):
    """Doctor feedback form -- shown after inference results."""
    st.markdown("---")
    st.subheader("📝 Clinical Feedback")
    st.caption("Help improve the system by providing your assessment")

    with st.form("feedback_form"):
        agrees = st.radio(
            "Do you agree with the AI assessment?",
            ["Yes", "No"],
            horizontal=True,
        )

        doctor_class = None
        if agrees == "No":
            doctor_class = st.selectbox(
                "Your classification:",
                ["OSCC", "OPMD", "Normal"],
            )

        doctor_notes = st.text_area("Clinical notes (optional)")

        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            store = get_feedback_store()

            top_detection = results['detections'][0] if results['detections'] else None
            ai_prediction = top_detection['class'] if top_detection else "No detection"
            ai_score = top_detection['fusion_score'] if top_detection else 0.0

            store.record_clinical_feedback(
                inference_id=results.get('inference_id', ''),
                ai_prediction=ai_prediction,
                ai_risk_level=results['risk_level'],
                ai_fusion_score=ai_score,
                doctor_agrees=(agrees == "Yes"),
                doctor_classification=doctor_class,
                doctor_notes=doctor_notes,
                patient_metadata=patient_metadata,
            )
            st.success("✅ Feedback recorded. Thank you!")


if __name__ == "__main__":
    main()
