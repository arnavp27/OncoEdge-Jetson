"""
OncoEdge Inference Pipeline

Main orchestration logic combining YOLO detection, BiomedCLIP classification,
score fusion, and clinical risk assessment.
"""
import numpy as np
from PIL import Image

from src.models.yolo_detector import YOLODetector
from src.models.biomedclip_classifier import BiomedCLIPClassifier
from src.pipeline.fusion import FusionModule
from src.pipeline.decision_tree import ClinicalDecisionTree
from src.utils.visualization import draw_detections


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

    def __init__(self, device='cuda'):
        """
        Initialize pipeline with all components.

        Args:
            device: Device for inference ('cuda' or 'cpu')
        """
        print(f"Initializing OncoEdge Pipeline on device: {device}")

        # Initialize all components
        self.yolo = YOLODetector(device=device)
        self.biomedclip = BiomedCLIPClassifier(device=device)
        self.fusion = FusionModule()
        self.decision_tree = ClinicalDecisionTree()

        print("Pipeline initialization complete")

    def process_image(self, image, patient_metadata):
        """
        Process an image through the complete pipeline.

        Args:
            image: Input image (numpy array HxWx3 or PIL Image)
            patient_metadata: Dictionary containing:
                - age: Patient age (int)
                - tobacco_years: Years of tobacco use (int)
                - lesion_duration: Duration category (str)

        Returns:
            dict: Results containing:
                - 'risk_level': Risk level (str)
                - 'recommendation': Clinical recommendation (str)
                - 'detections': List of detection dicts
                - 'visualization': Annotated image (numpy array)
                - 'risk_score': Computed risk score (float)
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure RGB format
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]

        # Step 1: YOLO detection
        print("Running YOLO detection...")
        yolo_results = self.yolo.detect(image)

        # Check if any detections found
        if yolo_results.boxes is None or len(yolo_results.boxes) == 0:
            return self._no_detection_output(image, patient_metadata)

        # Step 2: Process each detection
        print(f"Processing {len(yolo_results.boxes)} detections...")
        detections = []

        for i in range(len(yolo_results.boxes)):
            # Extract detection info
            box = yolo_results.boxes.xyxy[i].cpu().numpy()
            conf = float(yolo_results.boxes.conf[i].cpu().numpy())

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
                continue  # Skip invalid boxes

            lesion_crop = image[y1:y2, x1:x2]

            # Skip if crop is too small
            if lesion_crop.size == 0 or min(lesion_crop.shape[:2]) < 10:
                continue

            # Step 3: Classify with BiomedCLIP
            print(f"  Classifying lesion {i+1}...")
            classification = self.biomedclip.classify(lesion_crop)

            # Step 4: Fuse scores
            fusion_score = self.fusion.combine(
                yolo_conf=conf,
                biomedclip_score=classification['max_score']
            )

            # Store detection
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'mask': mask,
                'yolo_conf': conf,
                'class': classification['class'],
                'class_scores': classification['scores'],
                'fusion_score': fusion_score
            })

        # Handle case where all detections were invalid
        if not detections:
            return self._no_detection_output(image, patient_metadata)

        # Step 5: Clinical risk assessment
        print("Assessing clinical risk...")
        max_fusion_score = max(d['fusion_score'] for d in detections)
        risk_output = self.decision_tree.assess_risk(
            vision_score=max_fusion_score,
            age=patient_metadata['age'],
            tobacco_years=patient_metadata['tobacco_years'],
            lesion_duration=patient_metadata['lesion_duration']
        )

        # Step 6: Create visualization
        print("Creating visualization...")
        annotated_image = draw_detections(image, detections)

        return {
            'risk_level': risk_output['level'],
            'recommendation': risk_output['recommendation'],
            'detections': detections,
            'visualization': annotated_image,
            'risk_score': risk_output['score']
        }

    def _no_detection_output(self, image, patient_metadata):
        """
        Generate output when no lesions are detected.

        Args:
            image: Original image
            patient_metadata: Patient metadata

        Returns:
            dict: Results with no detections
        """
        # Still assess risk based on patient factors only
        risk_output = self.decision_tree.assess_risk(
            vision_score=0.0,
            age=patient_metadata['age'],
            tobacco_years=patient_metadata['tobacco_years'],
            lesion_duration=patient_metadata['lesion_duration']
        )

        return {
            'risk_level': risk_output['level'],
            'recommendation': 'No lesions detected. ' + risk_output['recommendation'],
            'detections': [],
            'visualization': image,
            'risk_score': risk_output['score']
        }
