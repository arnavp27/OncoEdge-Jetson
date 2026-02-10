# See the main claude.md file for complete implementation instructions
# OncoEdge: AI-Powered Oral Cancer Detection System
## Instructions for Claude Code Agent

---

## Project Context

You are building **OncoEdge**, a real-time oral cancer screening system for NVIDIA Jetson Orin deployment. This is a **2-day prototype** focusing on zero-shot inference using pretrained models.

**Critical Constraints:**
- ✅ Use ONLY the models and technologies specified below
- ✅ No model training - zero-shot inference only
- ✅ Follow the exact file structure provided
- ✅ Optimize for Jetson Orin edge deployment
- ❌ DO NOT add features beyond the core MVP
- ❌ DO NOT suggest model training or fine-tuning
- ❌ DO NOT use models other than YOLO26n-seg and BiomedCLIP

---

## System Architecture

### Pipeline Flow
```
Smartphone Photo Input
    ↓
[YOLO26n-seg] → Lesion Detection & Segmentation
    ↓
[BiomedCLIP] → Medical Classification (per lesion)
    ↓
[Fusion Module] → Combine detection + classification scores
    ↓
[Decision Tree] → Clinical risk stratification
    ↓
Risk Report + Visualization Output
```

### Core Components

**1. YOLO26n-seg (Detection & Segmentation)**
- **Model**: `yolo26n-seg.pt` (Nano variant for edge devices)
- **Purpose**: Detect and segment oral lesions
- **Input**: RGB image (any size, resized to 640x640)
- **Output**: 
  - Bounding boxes (xyxy format)
  - Segmentation masks (pixel-level)
  - Confidence scores (0-1)
- **Why Nano**: 2.7M params, 53ms CPU inference, optimized for Jetson Orin

**2. BiomedCLIP (Medical Classification)**
- **Model**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **Purpose**: Zero-shot medical image classification
- **Input**: Cropped lesion patches from YOLO detections
- **Output**: Text-image similarity scores for each class
- **Prompts** (use exactly these):
  ```python
  BIOMEDCLIP_PROMPTS = [
      "clinical photograph of oral squamous cell carcinoma",
      "clinical photograph of oral leukoplakia white patch",
      "clinical photograph of normal oral mucosa"
  ]
  ```

**3. Fusion Module**
- **Formula**: `final_score = yolo_confidence × biomedclip_max_score`
- **Logic**: Combine spatial confidence with semantic classification
- **Output**: Single score per lesion (0-1 range)

**4. Clinical Decision Tree**
- **Inputs**: 
  - Vision score (from fusion module)
  - Patient metadata (age, tobacco use, lesion duration)
- **Output**: Risk level (HIGH/MEDIUM/LOW) + referral recommendation
- **Rules**:
  ```python
  risk_score = vision_score * 10
  if age > 40: risk_score += 1.5
  if tobacco_years > 10: risk_score += 2.0
  if lesion_duration == "> 6 weeks": risk_score += 1.5
  
  if risk_score > 8.0: return "HIGH"
  elif risk_score > 5.0: return "MEDIUM"
  else: return "LOW"
  ```

---

## File Structure

**IMPLEMENT EXACTLY THIS STRUCTURE:**

```
oncoedge/
├── .venv/                          # uv virtual environment (auto-generated)
├── .python-version                 # Python 3.10
├── pyproject.toml                  # uv project config
├── README.md                       # Setup and usage instructions
├── claude.md                       # This file
│
├── data/
│   ├── test_images/                # Sample oral cavity photos for testing
│   └── sample_metadata.json        # Example patient metadata
│
├── models/                         # Downloaded models (auto-cached)
│   ├── yolo26n-seg.pt             # Auto-downloaded by ultralytics
│   └── biomedclip/                 # Auto-cached from HuggingFace
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py        # YOLO26n-seg wrapper
│   │   └── biomedclip_classifier.py # BiomedCLIP wrapper
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── inference_pipeline.py   # Main orchestration logic
│   │   ├── fusion.py               # Score fusion module
│   │   └── decision_tree.py        # Clinical risk rules
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py        # Overlay masks, bounding boxes
│   │   ├── image_processing.py     # Preprocessing utilities
│   │   └── metrics.py              # Performance tracking
│   │
│   └── config.py                   # Centralized configuration
│
├── streamlit_app.py                # Main Streamlit UI (SINGLE FILE)
│
└── tests/
    ├── __init__.py
    ├── test_yolo.py                # Unit test for YOLO module
    ├── test_biomedclip.py          # Unit test for BiomedCLIP
    └── test_pipeline.py            # Integration test
```

---

## Technical Specifications

### 1. Dependencies (via uv)

**Core ML Libraries:**
```toml
# pyproject.toml dependencies - LATEST VERSIONS (Feb 2026)
torch = ">=2.10.0"              # Latest: 2.10.0 (Jan 2026)
torchvision = ">=0.25.0"        # Latest: 0.25.0 (Jan 2026)
ultralytics = ">=8.4.8"         # Latest: 8.4.8 (YOLO26 support)
open-clip-torch = ">=3.2.0"     # Latest: 3.2.0 (BiomedCLIP)
```

**UI & Utilities:**
```toml
streamlit = ">=1.54.0"          # Latest: 1.54.0 (Feb 2026)
opencv-python = ">=4.12.0"      # Latest: 4.12.0.88
pillow = ">=12.1.0"             # Latest: 12.1.0 (Jan 2026)
numpy = ">=2.3.0"               # Latest: 2.3.1 (supports Python 3.10+)
pandas = ">=2.2.0"              # Latest stable
matplotlib = ">=3.9.0"          # Latest stable
```

**Installation Command:**
```bash
# Install PyTorch with CUDA 11.8 support (use cu121 for latest CUDA support)
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
uv pip install ultralytics==8.4.8 open-clip-torch==3.2.0 streamlit==1.54.0 opencv-python==4.12.0.88 pillow==12.1.0 numpy pandas matplotlib
```

### 2. Model Configuration

**YOLO26n-seg Settings:**
```python
# src/models/yolo_detector.py
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path='yolo26n-seg.pt', device='0'):
        self.model = YOLO(model_path)
        self.device = device
    
    def detect(self, image, conf_threshold=0.25, imgsz=640):
        """
        Args:
            image: numpy array (H, W, 3) or PIL Image
            conf_threshold: confidence threshold (0-1)
            imgsz: input image size
        
        Returns:
            results: ultralytics Results object
                - boxes: xyxy format bounding boxes
                - masks: segmentation masks
                - conf: confidence scores
        """
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            imgsz=imgsz,
            device=self.device,
            end2end=True,  # NMS-free (faster)
            verbose=False
        )
        return results[0]  # Return first result
```

**BiomedCLIP Settings:**
```python
# src/models/biomedclip_classifier.py
import open_clip
import torch
from PIL import Image

class BiomedCLIPClassifier:
    PROMPTS = [
        "clinical photograph of oral squamous cell carcinoma",
        "clinical photograph of oral leukoplakia white patch",
        "clinical photograph of normal oral mucosa"
    ]
    
    CLASS_NAMES = ["OSCC", "OPMD", "Normal"]
    
    def __init__(self, device='cuda'):
        self.device = device
        # Using open-clip-torch 3.2.0 syntax
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.model.to(device)
        self.model.eval()
        
        # Tokenize prompts once
        self.tokenizer = open_clip.get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.text_features = self._encode_text_prompts()
    
    def _encode_text_prompts(self):
        """Precompute text embeddings"""
        with torch.no_grad():
            text_tokens = self.tokenizer(self.PROMPTS).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def classify(self, image_patch):
        """
        Args:
            image_patch: PIL Image or numpy array of lesion crop
        
        Returns:
            dict: {'class': str, 'scores': dict, 'max_score': float}
        """
        if not isinstance(image_patch, Image.Image):
            image_patch = Image.fromarray(image_patch)
        
        # Preprocess and encode image
        image_tensor = self.preprocess(image_patch).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            scores = similarity[0].cpu().numpy()
        
        # Format output
        class_scores = {name: float(score) for name, score in zip(self.CLASS_NAMES, scores)}
        max_idx = scores.argmax()
        
        return {
            'class': self.CLASS_NAMES[max_idx],
            'scores': class_scores,
            'max_score': float(scores[max_idx])
        }
```

### 3. Inference Pipeline

**Pipeline Architecture:**
```python
# src/pipeline/inference_pipeline.py
class OncoEdgePipeline:
    def __init__(self, device='cuda'):
        self.yolo = YOLODetector(device=device)
        self.biomedclip = BiomedCLIPClassifier(device=device)
        self.fusion = FusionModule()
        self.decision_tree = ClinicalDecisionTree()
    
    def process_image(self, image, patient_metadata):
        """
        Main inference pipeline
        
        Args:
            image: numpy array or PIL Image
            patient_metadata: dict with keys:
                - age: int
                - tobacco_years: int
                - lesion_duration: str ("< 2 weeks", "2-6 weeks", "> 6 weeks")
        
        Returns:
            dict: {
                'risk_level': str,
                'recommendation': str,
                'detections': list of detection dicts,
                'visualization': annotated image
            }
        """
        # Step 1: YOLO detection
        yolo_results = self.yolo.detect(image)
        
        if len(yolo_results.boxes) == 0:
            return self._no_detection_output(image)
        
        # Step 2: BiomedCLIP classification per lesion
        detections = []
        for i, (box, mask, conf) in enumerate(zip(
            yolo_results.boxes.xyxy,
            yolo_results.masks.data,
            yolo_results.boxes.conf
        )):
            # Crop lesion patch
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            lesion_crop = image[y1:y2, x1:x2]
            
            # Classify with BiomedCLIP
            classification = self.biomedclip.classify(lesion_crop)
            
            # Fuse scores
            fusion_score = self.fusion.combine(
                yolo_conf=float(conf),
                biomedclip_score=classification['max_score']
            )
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'mask': mask.cpu().numpy(),
                'yolo_conf': float(conf),
                'class': classification['class'],
                'class_scores': classification['scores'],
                'fusion_score': fusion_score
            })
        
        # Step 3: Clinical decision
        max_fusion_score = max(d['fusion_score'] for d in detections)
        risk_output = self.decision_tree.assess_risk(
            vision_score=max_fusion_score,
            **patient_metadata
        )
        
        # Step 4: Visualization
        annotated_image = self._create_visualization(image, detections)
        
        return {
            'risk_level': risk_output['level'],
            'recommendation': risk_output['recommendation'],
            'detections': detections,
            'visualization': annotated_image,
            'risk_score': risk_output['score']
        }
```

**Fusion Module:**
```python
# src/pipeline/fusion.py
class FusionModule:
    def combine(self, yolo_conf, biomedclip_score):
        """
        Simple multiplicative fusion
        
        Args:
            yolo_conf: YOLO confidence (0-1)
            biomedclip_score: BiomedCLIP max class score (0-1)
        
        Returns:
            float: fused score (0-1)
        """
        return yolo_conf * biomedclip_score
```

**Decision Tree:**
```python
# src/pipeline/decision_tree.py
class ClinicalDecisionTree:
    def assess_risk(self, vision_score, age, tobacco_years, lesion_duration):
        """
        Clinical risk stratification
        
        Args:
            vision_score: fusion score (0-1)
            age: patient age
            tobacco_years: years of tobacco use
            lesion_duration: "< 2 weeks" | "2-6 weeks" | "> 6 weeks"
        
        Returns:
            dict: {'level': str, 'recommendation': str, 'score': float}
        """
        risk_score = vision_score * 10
        
        # Add clinical risk factors
        if age > 40:
            risk_score += 1.5
        if tobacco_years > 10:
            risk_score += 2.0
        if lesion_duration == "> 6 weeks":
            risk_score += 1.5
        
        # Classify
        if risk_score > 8.0:
            return {
                'level': 'HIGH',
                'recommendation': 'Urgent referral to specialist within 2 weeks',
                'score': risk_score
            }
        elif risk_score > 5.0:
            return {
                'level': 'MEDIUM',
                'recommendation': 'Non-urgent referral within 4 weeks',
                'score': risk_score
            }
        else:
            return {
                'level': 'LOW',
                'recommendation': 'Routine monitoring recommended',
                'score': risk_score
            }
```

---

## Streamlit Application

**Single-page UI with 3 sections:**

```python
# streamlit_app.py
import streamlit as st
from src.pipeline.inference_pipeline import OncoEdgePipeline

def main():
    st.set_page_config(page_title="OncoEdge", layout="wide")
    
    st.title("🔬 OncoEdge: Oral Cancer Screening System")
    
    # Sidebar: Patient Information
    with st.sidebar:
        st.header("Patient Information")
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        tobacco_years = st.number_input("Years of Tobacco Use", min_value=0, value=0)
        lesion_duration = st.selectbox(
            "Lesion Duration",
            ["< 2 weeks", "2-6 weeks", "> 6 weeks"]
        )
    
    # Main area: Image Upload
    st.header("Upload Oral Cavity Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=400)
        
        # Run inference on button click
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing..."):
                # Initialize pipeline
                pipeline = OncoEdgePipeline(device='cuda')
                
                # Run inference
                results = pipeline.process_image(
                    image=np.array(image),
                    patient_metadata={
                        'age': age,
                        'tobacco_years': tobacco_years,
                        'lesion_duration': lesion_duration
                    }
                )
                
                # Display results
                display_results(results)

def display_results(results):
    """Display risk assessment and detections"""
    
    # Risk Level (color-coded)
    risk_colors = {
        'HIGH': '🔴',
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }
    
    st.header(f"{risk_colors[results['risk_level']]} Risk Assessment: {results['risk_level']}")
    st.info(results['recommendation'])
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Risk Score", f"{results['risk_score']:.1f}/10")
    with col2:
        st.metric("Lesions Detected", len(results['detections']))
    with col3:
        max_conf = max(d['fusion_score'] for d in results['detections']) if results['detections'] else 0
        st.metric("Max Confidence", f"{max_conf:.2%}")
    
    # Visualization
    st.image(results['visualization'], caption='Detected Lesions', use_column_width=True)
    
    # Detailed detections
    if results['detections']:
        st.subheader("Detection Details")
        for i, det in enumerate(results['detections']):
            with st.expander(f"Lesion {i+1}: {det['class']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Classification:** {det['class']}")
                    st.write(f"**Fusion Score:** {det['fusion_score']:.2%}")
                with col2:
                    st.write("**Class Probabilities:**")
                    for cls, score in det['class_scores'].items():
                        st.write(f"- {cls}: {score:.2%}")

if __name__ == "__main__":
    main()
```

---

## Configuration Management

```python
# src/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    """Model configuration"""
    yolo_model: str = 'yolo26n-seg.pt'
    yolo_conf_threshold: float = 0.25
    yolo_imgsz: int = 640
    
    biomedclip_model: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    
    device: str = 'cuda'  # 'cuda' or 'cpu'

@dataclass
class PathConfig:
    """Path configuration"""
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / 'data'
    models_dir: Path = project_root / 'models'
    test_images_dir: Path = data_dir / 'test_images'

@dataclass
class ClinicalConfig:
    """Clinical decision thresholds"""
    high_risk_threshold: float = 8.0
    medium_risk_threshold: float = 5.0
    
    age_risk_threshold: int = 40
    age_risk_weight: float = 1.5
    
    tobacco_risk_threshold: int = 10
    tobacco_risk_weight: float = 2.0
    
    long_duration_weight: float = 1.5

# Global config instance
CONFIG = {
    'model': ModelConfig(),
    'paths': PathConfig(),
    'clinical': ClinicalConfig()
}
```

---

## Testing Requirements

**Unit Tests:**
```python
# tests/test_yolo.py
def test_yolo_detection():
    """Test YOLO26n-seg detection"""
    detector = YOLODetector()
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = detector.detect(test_image)
    assert results is not None

# tests/test_biomedclip.py
def test_biomedclip_classification():
    """Test BiomedCLIP classification"""
    classifier = BiomedCLIPClassifier()
    test_patch = Image.new('RGB', (224, 224), color='red')
    result = classifier.classify(test_patch)
    assert 'class' in result
    assert 'scores' in result
    assert result['max_score'] >= 0 and result['max_score'] <= 1

# tests/test_pipeline.py
def test_end_to_end_pipeline():
    """Test complete pipeline"""
    pipeline = OncoEdgePipeline()
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    metadata = {'age': 50, 'tobacco_years': 15, 'lesion_duration': '> 6 weeks'}
    results = pipeline.process_image(test_image, metadata)
    assert 'risk_level' in results
```

---

## Performance Targets

**Inference Time (Jetson Orin NX):**
- YOLO26n-seg: < 60ms per image
- BiomedCLIP: < 150ms per lesion
- Total pipeline: < 500ms for typical case (1-3 lesions)

**Memory Usage:**
- Model loading: ~1.5GB GPU memory
- Inference: ~500MB GPU memory
- Total: < 2GB GPU memory

**Accuracy (Zero-Shot):**
- Expected baseline: 65-70% on test dataset
- Primary goal: Working system, not benchmark accuracy

---

## Important Constraints & Guidelines

### DO:
✅ Use YOLO26n-seg (nano variant) for edge optimization
✅ Use BiomedCLIP with exact prompts specified
✅ Follow the file structure exactly
✅ Implement decision tree with specified thresholds
✅ Create single-page Streamlit interface
✅ Add error handling for edge cases
✅ Document all functions with docstrings
✅ Test on sample images before deployment

### DON'T:
❌ Don't train or fine-tune models (zero-shot only)
❌ Don't use other YOLO variants (must be YOLO26n-seg)
❌ Don't modify BiomedCLIP prompts without justification
❌ Don't add complex features beyond MVP
❌ Don't use additional ML models
❌ Don't implement database or user authentication
❌ Don't add PDF generation (stretch goal only)

---

## Deployment Checklist

**Day 1 Evening:**
- [ ] Environment setup with uv complete
- [ ] YOLO26n-seg loading and inference working
- [ ] BiomedCLIP loading and inference working
- [ ] Fusion module implemented
- [ ] Decision tree implemented
- [ ] Unit tests passing

**Day 2 Morning:**
- [ ] Streamlit app created
- [ ] Patient metadata form functional
- [ ] Image upload working
- [ ] Pipeline integrated into UI
- [ ] Visualization working

**Day 2 Afternoon:**
- [ ] Tested on sample images
- [ ] Deployed to Jetson Orin
- [ ] Accessible via browser at `http://<jetson-ip>:8501`
- [ ] Documentation complete

---

## README Template

```markdown
# OncoEdge: Oral Cancer Detection System

AI-powered screening system for oral cancer detection using YOLO26n-seg and BiomedCLIP.

## Setup (Jetson Orin NX)

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and Install
```bash
git clone <repo-url>
cd oncoedge
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies with latest versions
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu118
uv pip install ultralytics==8.4.8 open-clip-torch==3.2.0 streamlit==1.54.0 opencv-python==4.12.0.88 pillow==12.1.0 numpy pandas matplotlib
```

### Run Application
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Access at: `http://<jetson-ip>:8501`

## Usage

1. Enter patient information in sidebar
2. Upload oral cavity image
3. Click "Analyze Image"
4. Review risk assessment and detected lesions

## System Requirements

- NVIDIA Jetson Orin NX (16GB RAM)
- CUDA 11.8+
- Python 3.10
```

---

## Final Notes

**This is a prototype for rapid deployment.** Focus on:
1. Getting the core pipeline working end-to-end
2. Clean, modular code structure
3. Basic error handling
4. Clear user interface

**Not required for MVP:**
- Training or fine-tuning
- Advanced visualizations
- PDF reports
- Database integration
- Multi-user support

**Success = Working Streamlit app on Jetson Orin that can analyze oral cavity images and provide risk assessments using zero-shot inference.**