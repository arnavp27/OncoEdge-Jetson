"""
OncoEdge Streamlit Application

Main web interface for oral cancer screening.
"""
import streamlit as st
import numpy as np
import torch
from PIL import Image

from src.pipeline.inference_pipeline import OncoEdgePipeline


@st.cache_resource
def load_pipeline():
    """
    Load pipeline once and cache it.

    Returns:
        OncoEdgePipeline: Initialized pipeline
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return OncoEdgePipeline(device=device)


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="OncoEdge - Oral Cancer Screening",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 OncoEdge: Oral Cancer Screening System")
    st.markdown("AI-powered screening using YOLO11n-seg and BiomedCLIP")

    # Sidebar: Patient Information
    with st.sidebar:
        st.header("📋 Patient Information")
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        tobacco_years = st.number_input("Years of Tobacco Use", min_value=0, value=0)
        lesion_duration = st.selectbox(
            "Lesion Duration",
            ["< 2 weeks", "2-6 weeks", "> 6 weeks"]
        )

        st.markdown("---")
        st.markdown("### ℹ️ About OncoEdge")
        st.info("This system uses YOLO11n-seg for lesion detection and BiomedCLIP for medical classification.")

        # Show device info
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.markdown(f"**Device:** {device.upper()}")
        if device == 'cuda':
            st.success("GPU acceleration enabled")
        else:
            st.warning("Running on CPU")

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
                        'age': age,
                        'tobacco_years': tobacco_years,
                        'lesion_duration': lesion_duration
                    }

                    # Run inference
                    results = pipeline.process_image(
                        image=np.array(image),
                        patient_metadata=patient_metadata
                    )

                # Display results
                with col2:
                    display_results(results)

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)


def display_results(results):
    """
    Display analysis results.

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
        st.metric("Risk Score", f"{results['risk_score']:.1f}/10")

    with col2:
        st.metric("Lesions Detected", len(results['detections']))

    with col3:
        if results['detections']:
            max_conf = max(d['fusion_score'] for d in results['detections'])
            st.metric("Max Confidence", f"{max_conf:.1%}")
        else:
            st.metric("Max Confidence", "N/A")

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
                        # Create a simple progress bar
                        st.write(f"- **{cls}:** {score:.1%}")
                        st.progress(score)
    else:
        st.info("ℹ️ No lesions detected in the image.")


if __name__ == "__main__":
    main()
