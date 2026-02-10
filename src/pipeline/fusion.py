"""
Score Fusion Module

Combines YOLO detection confidence with BiomedCLIP classification scores.
"""


class FusionModule:
    """
    Fuses detection and classification scores.

    Uses simple multiplicative fusion to combine spatial confidence from YOLO
    with semantic classification confidence from BiomedCLIP.
    """

    def combine(self, yolo_conf, biomedclip_score):
        """
        Combine YOLO confidence with BiomedCLIP classification score.

        Uses multiplicative fusion: final_score = yolo_conf × biomedclip_score

        Args:
            yolo_conf: YOLO detection confidence (0-1)
            biomedclip_score: BiomedCLIP max class score (0-1)

        Returns:
            float: Fused score (0-1)
        """
        return float(yolo_conf * biomedclip_score)

    def combine_batch(self, yolo_confs, biomedclip_scores):
        """
        Combine scores for multiple detections.

        Args:
            yolo_confs: List of YOLO confidences
            biomedclip_scores: List of BiomedCLIP scores

        Returns:
            list: List of fused scores
        """
        return [self.combine(yc, bs) for yc, bs in zip(yolo_confs, biomedclip_scores)]
