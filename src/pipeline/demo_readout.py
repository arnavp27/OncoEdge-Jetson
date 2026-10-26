"""Opt-in exploratory readouts, separate from the conservative assessment."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class DemoOptions:
    """Per-request display gates; these do not modify model weights or CONFIG."""

    detector_threshold: float = 0.10
    min_classifier_score: float = 0.35
    min_classifier_margin: float = 0.02

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be a finite value between 0 and 1.')
        if self.detector_threshold == 0:
            raise ValueError('detector_threshold must be greater than zero.')


def build_demo_readout(detections, options):
    """Expose real classifier outputs even when the models disagree.

    The headline uses the highest YOLO-confidence × classifier-score match,
    rather than selecting the most concerning class. Other regions and every
    conservative assessment issue remain available for inspection.
    """
    accepted, excluded = [], []
    for number, detection in enumerate(detections, 1):
        reasons = []
        if detection['yolo_conf'] < options.detector_threshold:
            reasons.append('Detector confidence is below the demo limit.')
        if max(detection['class_scores'].values()) < options.min_classifier_score:
            reasons.append('Classifier relative score is below the demo limit.')
        if detection['classifier_margin'] < options.min_classifier_margin:
            reasons.append('Classifier margin is below the demo limit.')
        if reasons:
            excluded.append({'region': number, 'reasons': reasons})
        else:
            accepted.append((number, detection))

    primary = max(accepted, key=lambda pair: pair[1]['fusion_score']) if accepted else None
    return {
        'status': 'provisional' if primary else 'no_accepted_regions',
        'provisional': True,
        'settings': asdict(options),
        'selection_rule': 'Highest YOLO confidence × classifier relative score among eligible regions.',
        'primary_region': primary[0] if primary else None,
        'class': primary[1]['class'] if primary else None,
        'yolo_class': primary[1]['yolo_class'] if primary else None,
        'classifier_score': max(primary[1]['class_scores'].values()) if primary else None,
        'fusion_score': primary[1]['fusion_score'] if primary else None,
        'concern_index': 10 * max(d['concern_score'] for _, d in accepted) if primary else None,
        'concern_index_formula': '10 × max(YOLO confidence × (OCA + OPMD relative scores)) over eligible regions; no patient-context weighting.',
        'accepted_regions': [number for number, _ in accepted],
        'excluded_regions': excluded,
        'model_disagreement': any(d['model_disagreement'] for _, d in accepted),
        'note': 'Unvalidated exploratory model output. Not a diagnosis or disease probability. Lower display limits do not improve accuracy.',
    }
