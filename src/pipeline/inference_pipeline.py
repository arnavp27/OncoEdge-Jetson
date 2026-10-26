"""
OncoEdge Inference Pipeline

Main orchestration logic combining YOLO detection, BiomedCLIP classification,
score fusion, and clinical risk assessment.

Auto-detects Jetson Nano and uses TensorRT backend when available.
"""
import numpy as np
from PIL import Image
from pathlib import Path

from src.models.yolo_detector import YOLODetector
from src.config import CONFIG
from src.pipeline.fusion import FusionModule
from src.pipeline.decision_tree import ClinicalDecisionTree
from src.pipeline.demo_readout import DemoOptions, build_demo_readout
from src.utils.visualization import draw_detections


def is_jetson_nano():
    """
    Detect if running on Jetson Nano.

    Returns:
        bool: True if running on Jetson Nano, False otherwise
    """
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            content = f.read()
            return 'Nano' in content or 'tegra' in content.lower()
    except FileNotFoundError:
        return False


def has_tensorrt_engine(engine_path='models/biomedclip/tensorrt/vision_encoder_int8.engine'):
    """
    Check if TensorRT engine exists.

    Args:
        engine_path: Path to TensorRT engine file

    Returns:
        bool: True if engine exists, False otherwise
    """
    return Path(engine_path).exists()


class OncoEdgePipeline:
    """
    Complete inference pipeline for oral cancer screening.

    Orchestrates the full workflow:
    1. YOLO detection and segmentation
    2. BiomedCLIP classification per lesion
    3. Score fusion
    4. Clinical risk assessment
    5. Visualization generation
    """

    def __init__(self, device='cuda', force_backend=None):
        """
        Initialize pipeline with all components.

        Auto-detects Jetson Nano and uses TensorRT backend when available.
        Falls back to PyTorch if TensorRT not available.

        Args:
            device: Device for inference ('cuda' or 'cpu') - only used for PyTorch backend
            force_backend: Force specific backend:
                - None: Auto-detect (recommended)
                - 'tensorrt': Force TensorRT (raises error if not available)
                - 'pytorch': Force PyTorch (ignores TensorRT even if available)
        """
        print(f"Initializing OncoEdge Pipeline...")

        # Initialize YOLO detector
        self.yolo = YOLODetector(device=device)

        # Auto-select or force backend
        use_tensorrt = False

        if force_backend == 'tensorrt':
            # Force TensorRT - raise error if not available
            if not has_tensorrt_engine():
                raise RuntimeError(
                    "TensorRT engine not found. Build it first:\n"
                    "  python3 src/models/build_tensorrt_engine.py"
                )
            use_tensorrt = True
            print("[OK] Forcing TensorRT backend (user override)")

        elif force_backend == 'pytorch':
            # Force PyTorch - ignore TensorRT even if available
            use_tensorrt = False
            print("[OK] Forcing PyTorch backend (user override)")

        else:
            # Auto-detect
            use_tensorrt = is_jetson_nano() and has_tensorrt_engine()

            if use_tensorrt:
                print(f"[OK] Detected Jetson Nano with TensorRT engine")
            elif is_jetson_nano():
                print(f"[WARNING] Jetson Nano detected but no TensorRT engine found")
                print(f"         Falling back to PyTorch. Build engine for better performance:")
                print(f"         python3 src/models/build_tensorrt_engine.py")
            else:
                print(f"[OK] Running on development machine")

        # Load appropriate BiomedCLIP backend
        if use_tensorrt:
            print("[OK] Loading TensorRT INT8 backend...")
            from src.models.biomedclip_tensorrt import BiomedCLIPTensorRT
            self.biomedclip = BiomedCLIPTensorRT()
            self.backend = 'tensorrt'
        else:
            print(f"[OK] Loading PyTorch FP32 backend (device: {device})...")
            from src.models.biomedclip_classifier import BiomedCLIPClassifier
            self.biomedclip = BiomedCLIPClassifier(device=device)
            self.backend = 'pytorch'

        # Initialize fusion and decision modules
        self.fusion = FusionModule()
        self.decision_tree = ClinicalDecisionTree()

        print(f"[OK] OncoEdge pipeline initialized (backend: {self.backend})")
        print()

    def process_image(self, image, patient_metadata, *, demo_options=None):
        """
        Process an image through the complete pipeline.

        Args:
            image: Input image (numpy array HxWx3 or PIL Image)
            patient_metadata: Dictionary containing:
                - age: Patient age (int)
                - tobacco_years: Years of tobacco use (int)
                - lesion_duration: Duration category (str)
            demo_options: Optional DemoOptions for provisional model readouts.
                The conservative assessment never bypasses its review gates.

        Returns:
            dict: Results containing:
                - 'risk_level': Risk level (str)
                - 'recommendation': Clinical recommendation (str)
                - 'detections': List of detection dicts
                - 'visualization': Annotated image (numpy array)
                - 'risk_score': Demo score, or None if inconclusive
                - 'assessment_issues': Reasons the assessment is inconclusive
        """
        if demo_options is not None:
            # A cached Streamlit pipeline can outlive a module reload. Copy the
            # values into the current validated type instead of comparing class
            # identity across reloads or retaining mutable caller settings.
            demo_options = DemoOptions(
                detector_threshold=demo_options.detector_threshold,
                min_classifier_score=demo_options.min_classifier_score,
                min_classifier_margin=demo_options.min_classifier_margin,
            )

        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))

        # Ensure RGB format
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]

        # Step 1: YOLO detection
        print("Running YOLO detection...")
        yolo_results = self.yolo.detect(
            image,
            conf_threshold=(demo_options.detector_threshold if demo_options
                            else CONFIG['model'].yolo_conf_threshold),
            imgsz=CONFIG['model'].yolo_imgsz,
        )

        # Check if any detections found
        if yolo_results.boxes is None or len(yolo_results.boxes) == 0:
            return self._no_detection_output(image, patient_metadata, demo_options=demo_options)

        # Step 2: Process each detection
        print(f"Processing {len(yolo_results.boxes)} detections...")
        detections = []
        assessment_issues = []

        for i in range(len(yolo_results.boxes)):
            # Extract detection info
            box = yolo_results.boxes.xyxy[i].cpu().numpy()
            conf = float(yolo_results.boxes.conf[i].cpu().numpy())
            class_id = int(yolo_results.boxes.cls[i].item())
            yolo_class = yolo_results.names[class_id]

            # Get mask if available
            mask = None
            if yolo_results.masks is not None and i < len(yolo_results.masks):
                mask = yolo_results.masks.data[i].cpu().numpy()

            # Crop lesion patch for classification
            x1, y1, x2, y2 = map(int, box)
            # Ensure valid crop coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                assessment_issues.append(f'Region {i+1}: detector crop is invalid.')
                continue  # Skip invalid boxes

            lesion_crop = image[y1:y2, x1:x2]

            # Skip if crop is too small
            if lesion_crop.size == 0 or min(lesion_crop.shape[:2]) < 10:
                assessment_issues.append(f'Region {i+1}: detector crop is too small.')
                continue

            # Step 3: Classify with BiomedCLIP
            print(f"  Classifying lesion {i+1}...")
            classification = self.biomedclip.classify(lesion_crop)

            # Step 4: Fuse scores
            fusion_score = self.fusion.combine(
                yolo_conf=conf,
                biomedclip_score=classification['max_score']
            )

            # Preserve both labels: high relative confidence cannot resolve a
            # disagreement between the localizer and zero-shot classifier.
            model_disagreement = yolo_class != classification['class'] and not (
                yolo_class == 'Benign' and classification['class'] == 'Normal'
            )
            ranked_scores = sorted(classification['scores'].values(), reverse=True)
            margin = ranked_scores[0] - ranked_scores[1]
            classification_uncertain = (
                classification['max_score'] < CONFIG['model'].min_classifier_score
                or margin < CONFIG['model'].min_classifier_margin
            )
            detection_uncertain = conf < CONFIG['model'].yolo_conf_threshold
            if detection_uncertain:
                assessment_issues.append(
                    f'Region {i+1}: detector confidence is below the conservative limit.'
                )
            if model_disagreement:
                assessment_issues.append(
                    f"Region {i+1}: models disagree (YOLO {yolo_class}, "
                    f"classifier {classification['class']})."
                )
            if classification_uncertain:
                assessment_issues.append(
                    f'Region {i+1}: classifier scores are ambiguous.'
                )
            concern_score = self.fusion.combine_concern(conf, classification['scores'])

            # Store detection
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'mask': mask,
                'yolo_conf': conf,
                'yolo_class': yolo_class,
                'class': classification['class'],
                'class_scores': classification['scores'],
                'fusion_score': fusion_score,
                'concern_score': concern_score,
                'classifier_margin': margin,
                'model_disagreement': model_disagreement,
                'classification_uncertain': classification_uncertain,
                'detection_uncertain': detection_uncertain,
            })

        # Handle case where all detections were invalid
        if not detections:
            return self._no_detection_output(
                image, patient_metadata,
                reason='The detector returned no usable regions to classify.',
                demo_options=demo_options,
            )

        # Step 5: Clinical risk assessment
        print("Assessing clinical risk...")
        max_concern_score = max(d['concern_score'] for d in detections)
        agreed_classes = {
            d['class'] for d in detections
            if not d['model_disagreement'] and not d['classification_uncertain']
            and not d['detection_uncertain']
        }
        concerning_class = ('OCA' if 'OCA' in agreed_classes else
                            'OPMD' if 'OPMD' in agreed_classes else None)
        risk_output = self.decision_tree.assess_risk(
            vision_score=max_concern_score,
            age=patient_metadata['age'],
            tobacco_years=patient_metadata['tobacco_years'],
            lesion_duration=patient_metadata['lesion_duration'],
            concerning_class=concerning_class,
            assessment_issues=assessment_issues,
        )

        # Step 6: Create visualization
        print("Creating visualization...")
        annotated_image = draw_detections(image, detections)

        output = {
            'risk_level': risk_output['level'],
            'recommendation': risk_output['recommendation'],
            'detections': detections,
            'visualization': annotated_image,
            'risk_score': risk_output['score'],
            'assessment_issues': assessment_issues,
        }
        if demo_options is not None:
            output['demo_readout'] = build_demo_readout(detections, demo_options)
        return output

    def _no_detection_output(self, image, patient_metadata,
                             reason='No lesion regions were detected; the image was not assessed.',
                             *, demo_options=None):
        """
        Generate output when no lesions are detected.

        Args:
            image: Original image
            patient_metadata: Patient metadata

        Returns:
            dict: Results with no detections
        """
        risk_output = self.decision_tree.assess_risk(
            vision_score=0.0,
            age=patient_metadata['age'],
            tobacco_years=patient_metadata['tobacco_years'],
            lesion_duration=patient_metadata['lesion_duration'],
            detection_available=False,
        )

        output = {
            'risk_level': risk_output['level'],
            'recommendation': risk_output['recommendation'],
            'detections': [],
            'visualization': image,
            'risk_score': risk_output['score'],
            'assessment_issues': [reason],
        }
        if demo_options is not None:
            output['demo_readout'] = build_demo_readout([], demo_options)
        return output
