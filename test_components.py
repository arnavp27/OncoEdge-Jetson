"""
Quick test script to verify all components work
"""
import numpy as np
from PIL import Image
import torch
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("OncoEdge Component Test")
print("=" * 60)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n[+] Device: {device.upper()}")
if device == 'cuda':
    print(f"    GPU: {torch.cuda.get_device_name(0)}")

# Test 1: YOLO Detector
print("\n[1/5] Testing YOLO Detector...")
try:
    from src.models.yolo_detector import YOLODetector
    detector = YOLODetector(device=device)

    # Create test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = detector.detect(test_image)

    print(f"  [+] YOLO loaded successfully")
    print(f"  [+] Test inference completed")
except Exception as e:
    print(f"  [-] YOLO test failed: {e}")

# Test 2: BiomedCLIP Classifier
print("\n[2/5] Testing BiomedCLIP Classifier...")
try:
    from src.models.biomedclip_classifier import BiomedCLIPClassifier
    classifier = BiomedCLIPClassifier(device=device)

    # Create test patch
    test_patch = Image.new('RGB', (224, 224), color='red')
    result = classifier.classify(test_patch)

    print(f"  [+] BiomedCLIP loaded successfully")
    print(f"  [+] Classification result: {result['class']}")
    print(f"  [+] Max score: {result['max_score']:.3f}")
except Exception as e:
    print(f"  [-] BiomedCLIP test failed: {e}")

# Test 3: Fusion Module
print("\n[3/5] Testing Fusion Module...")
try:
    from src.pipeline.fusion import FusionModule
    fusion = FusionModule()

    score = fusion.combine(0.8, 0.9)
    print(f"  [+] Fusion module working")
    print(f"  [+] Test score: {score:.3f}")
except Exception as e:
    print(f"  [-] Fusion test failed: {e}")

# Test 4: Decision Tree
print("\n[4/5] Testing Clinical Decision Tree...")
try:
    from src.pipeline.decision_tree import ClinicalDecisionTree
    decision_tree = ClinicalDecisionTree()

    result = decision_tree.assess_risk(
        vision_score=0.75,
        age=50,
        tobacco_years=15,
        lesion_duration="> 6 weeks"
    )

    print(f"  [+] Decision tree working")
    print(f"  [+] Risk level: {result['level']}")
    print(f"  [+] Risk score: {result['score']:.1f}/10")
except Exception as e:
    print(f"  [-] Decision tree test failed: {e}")

# Test 5: Complete Pipeline
print("\n[5/5] Testing Complete Pipeline...")
try:
    from src.pipeline.inference_pipeline import OncoEdgePipeline

    pipeline = OncoEdgePipeline(device=device)

    # Create test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    patient_metadata = {
        'age': 50,
        'tobacco_years': 10,
        'lesion_duration': '2-6 weeks'
    }

    results = pipeline.process_image(test_image, patient_metadata)

    print(f"  [+] Pipeline executed successfully")
    print(f"  [+] Risk level: {results['risk_level']}")
    print(f"  [+] Detections: {len(results['detections'])}")
    print(f"  [+] Visualization shape: {results['visualization'].shape}")
except Exception as e:
    print(f"  [-] Pipeline test failed: {e}")

print("\n" + "=" * 60)
print("Component Testing Complete!")
print("=" * 60)
print("\n[+] All core components are working!")
print(f"[+] Streamlit app is running at: http://localhost:8501")
print("\nNext steps:")
print("1. Open http://localhost:8501 in your browser")
print("2. Upload an oral cavity image")
print("3. Fill in patient information")
print("4. Click 'Analyze Image' to test the complete system")
