"""Score Fusion Module

Combines YOLO detection confidence with BiomedCLIP classification scores.
Supports configurable fusion methods.
"""


class FusionModule:
    """Fuses detection and classification scores.

    Supports multiple fusion methods controlled via configuration:
    - "gate": YOLO as binary gate; if conf >= gate_threshold, pass CLIP score
              through unchanged. CLIP drives the risk score. (default)
    - "multiplicative": final_score = yolo_conf x biomedclip_score (legacy)
    - "weighted": final_score = 0.4 * yolo_conf + 0.6 * biomedclip_score (legacy)
    """

    def __init__(self, method: str = "gate", gate_threshold: float = 0.5):
        """Initialize fusion module.

        Args:
            method: Fusion method name. One of "gate", "multiplicative", "weighted".
            gate_threshold: YOLO confidence threshold for gate method.
                Detections below this are discarded entirely.
        """
        self.method = method
        self.gate_threshold = gate_threshold

    def combine(self, yolo_conf, biomedclip_score):
        """Combine YOLO confidence with BiomedCLIP classification score.

        Args:
            yolo_conf: YOLO detection confidence (0-1)
            biomedclip_score: BiomedCLIP max class score (0-1)

        Returns:
            float: Fused score (0-1), or -1.0 if gated out (below threshold)
        """
        if self.method == "gate":
            if yolo_conf >= self.gate_threshold:
                return float(biomedclip_score)
            return -1.0  # Signal: detection rejected by gate
        if self.method == "weighted":
            return float(0.4 * yolo_conf + 0.6 * biomedclip_score)
        # Legacy: multiplicative
        return float(yolo_conf * biomedclip_score)

    def combine_batch(self, yolo_confs, biomedclip_scores):
        """Combine scores for multiple detections.

        Args:
            yolo_confs: List of YOLO confidences
            biomedclip_scores: List of BiomedCLIP scores

        Returns:
            list: List of fused scores (-1.0 for gated-out detections)
        """
        return [self.combine(yc, bs) for yc, bs in zip(yolo_confs, biomedclip_scores)]
