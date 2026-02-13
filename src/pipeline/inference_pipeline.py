"""OncoEdge Inference Pipeline -- Model-Agnostic Orchestrator.

Does NOT import any concrete model classes. Uses the backend registry
to create detector and classifier instances from configuration.
Swapping model formats requires ONLY config changes, not code changes.
"""
import time
import uuid

import numpy as np
from PIL import Image
from typing import Optional

from src.config import OncoEdgeConfig, load_config
from src.interfaces.detector import BaseDetector, DetectorOutput
from src.interfaces.classifier import BaseClassifier
from src.backends.registry import create_detector, create_classifier
from src.backends.load_backends import load_all_backends
from src.pipeline.fusion import FusionModule
from src.pipeline.decision_tree import ClinicalDecisionTree
from src.utils.visualization import draw_detections


class OncoEdgePipeline:
    """Model-agnostic inference pipeline for oral cancer screening.

    Orchestrates the full workflow:
    1. Detection via BaseDetector (any format: .pt, .onnx, .engine)
    2. Classification via BaseClassifier (any format: hf-hub, .onnx, .engine)
    3. Score fusion
    4. Clinical risk assessment
    5. Visualization generation

    All model-specific decisions are driven by configuration.
    Swapping YOLO versions or BiomedCLIP formats requires
    ZERO code changes -- only config changes.
    """

    def __init__(self, config: Optional[OncoEdgeConfig] = None, device: str = None):
        """Initialize pipeline from configuration.

        Args:
            config: OncoEdgeConfig instance. If None, loads from default YAML.
            device: Deprecated. Use config.device.preferred instead.
                    Kept for backward compatibility.
        """
        self.config = config or load_config()

        # Backward compatibility: if device is passed directly, override config
        if device is not None:
            self.config.device.preferred = device

        resolved_device = self.config.device.resolved
        print(f"Initializing OncoEdge Pipeline on device: {resolved_device}")

        # Register all available backends
        load_all_backends()

        # Create detector and classifier via registry (format auto-detected)
        self.detector: BaseDetector = create_detector(
            model_path=self.config.detector.model_path,
            device=resolved_device,
            imgsz=self.config.detector.imgsz,
        )
        self.classifier: BaseClassifier = create_classifier(
            model_path=self.config.classifier.model_path,
            device=resolved_device,
            prompts=self.config.classifier.prompts,
            class_names=self.config.classifier.class_names,
            text_embeddings_path=self.config.classifier.text_embeddings_path,
            logit_scale_path=self.config.classifier.logit_scale_path,
        )

        self.fusion = FusionModule(
            method=self.config.fusion.method,
            gate_threshold=self.config.fusion.gate_threshold,
        )
        self.decision_tree = ClinicalDecisionTree(config=self.config.clinical)

        # Lazy-load metrics tracker
        self._metrics = None
        if self.config.metrics.enabled:
            try:
                from src.utils.metrics import MetricsTracker
                self._metrics = MetricsTracker(config=self.config.metrics)
            except Exception:
                pass

        print("Pipeline initialization complete")

    def process_image(self, image, patient_metadata):
        """Process an image through the complete pipeline.

        Args:
            image: Input image (numpy array HxWx3 or PIL Image)
            patient_metadata: Dictionary containing:
                - tobacco_years: Years of smokeless tobacco use (int)
                - smoking_years: Years of smoking (int)
                - age: Patient age (int, optional)
                - lesion_duration: Duration category (str, optional)

        Returns:
            dict: Results containing:
                - 'risk_level': Risk level (str)
                - 'recommendation': Clinical recommendation (str)
                - 'detections': List of detection dicts
                - 'visualization': Annotated image (numpy array)
                - 'risk_score': Computed risk score (float)
                - 'inference_id': Unique ID for this inference (str)
                - 'timing': Dict of timing info (dict)
                - 'model_info': Dict of model metadata (dict)
        """
        pipeline_start = time.perf_counter()

        # Normalize input
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        # Step 1: Detection (format-agnostic)
        detector_output: DetectorOutput = self.detector.detect(
            image, conf_threshold=self.config.detector.conf_threshold
        )

        if not detector_output.detections:
            return self._no_detection_output(image, patient_metadata, pipeline_start)

        # Step 2: Classification per lesion + gate fusion
        detections = []
        classification_time_ms = 0.0

        for det in detector_output.detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            lesion_crop = image[y1:y2, x1:x2]
            if lesion_crop.size == 0 or min(lesion_crop.shape[:2]) < 10:
                continue

            # Gate check: skip classification if YOLO conf below gate threshold
            # (saves compute — don't run CLIP on low-confidence detections)
            if (self.fusion.method == "gate"
                    and det.confidence < self.fusion.gate_threshold):
                continue

            classification = self.classifier.classify(lesion_crop)
            classification_time_ms += classification.inference_time_ms

            fusion_score = self.fusion.combine(
                yolo_conf=det.confidence,
                biomedclip_score=classification.max_score,
            )

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'mask': det.mask,
                'yolo_conf': det.confidence,
                'class': classification.predicted_class,
                'class_scores': classification.class_scores,
                'fusion_score': fusion_score,
            })

        if not detections:
            return self._no_detection_output(image, patient_metadata, pipeline_start)

        # Step 3: Clinical risk assessment
        # With gate fusion, fusion_score IS the CLIP score
        max_fusion_score = max(d['fusion_score'] for d in detections)
        risk_output = self.decision_tree.assess_risk(
            vision_score=max_fusion_score,
            tobacco_years=patient_metadata.get('tobacco_years', 0),
            smoking_years=patient_metadata.get('smoking_years', 0),
        )

        # Step 4: Visualization
        annotated_image = draw_detections(image, detections)

        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000

        result = {
            'risk_level': risk_output['level'],
            'recommendation': risk_output['recommendation'],
            'detections': detections,
            'visualization': annotated_image,
            'risk_score': risk_output['score'],
            'breakdown': risk_output.get('breakdown', {}),
            'inference_id': str(uuid.uuid4())[:8],
            'timing': {
                'detection_ms': detector_output.inference_time_ms,
                'classification_ms': classification_time_ms,
                'pipeline_total_ms': pipeline_ms,
            },
            'model_info': {
                'detector': self.detector.get_model_info(),
                'classifier': self.classifier.get_model_info(),
            },
        }

        # Log metrics if enabled
        if self._metrics:
            self._metrics.log_inference(result, patient_metadata)

        return result

    def _no_detection_output(self, image, patient_metadata, pipeline_start=None):
        """Generate output when no lesions are detected."""
        risk_output = self.decision_tree.assess_risk(
            vision_score=0.0,
            tobacco_years=patient_metadata.get('tobacco_years', 0),
            smoking_years=patient_metadata.get('smoking_years', 0),
        )

        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000 if pipeline_start else 0.0

        return {
            'risk_level': risk_output['level'],
            'recommendation': 'No lesions detected. ' + risk_output['recommendation'],
            'detections': [],
            'visualization': image,
            'risk_score': risk_output['score'],
            'breakdown': risk_output.get('breakdown', {}),
            'inference_id': str(uuid.uuid4())[:8],
            'timing': {'pipeline_total_ms': pipeline_ms},
            'model_info': {},
        }
