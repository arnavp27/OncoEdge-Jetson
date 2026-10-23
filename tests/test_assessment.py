"""Regressions for the failures observed in the published-photo evaluation."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.yolo_detector import YOLODetector
from src.pipeline.decision_tree import ClinicalDecisionTree
from src.pipeline.fusion import FusionModule
from src.pipeline.inference_pipeline import OncoEdgePipeline


METADATA = {"age": 45, "tobacco_years": 0, "lesion_duration": "< 2 weeks"}
HIGH_FACTORS = {"age": 65, "tobacco_years": 20, "lesion_duration": "> 6 weeks"}


def make_pipeline(yolo_classes, classifications, confidences=None, boxes=None):
    pipeline = OncoEdgePipeline.__new__(OncoEdgePipeline)
    confidences = confidences or [0.9] * len(yolo_classes)
    boxes = boxes or [[0, 0, 24, 24]] * len(yolo_classes)
    result = SimpleNamespace(
        names={0: "OCA", 1: "OPMD", 2: "Benign"}, masks=None,
    )
    # A small real tensor-backed box object supports the pipeline's normal API.
    from ultralytics.engine.results import Boxes
    values = [[*box, conf, cls] for box, conf, cls in zip(boxes, confidences, yolo_classes)]
    result.boxes = Boxes(torch.tensor(values, dtype=torch.float32).reshape(-1, 6), (24, 24))
    pipeline.yolo = Mock(detect=Mock(return_value=result))
    pipeline.biomedclip = Mock(classify=Mock(side_effect=classifications))
    pipeline.fusion = FusionModule()
    pipeline.decision_tree = ClinicalDecisionTree()
    return pipeline


def classification(label, scores):
    return {"class": label, "scores": scores, "max_score": scores[label]}


@pytest.mark.parametrize("pil", [False, True])
def test_detector_supplies_bgr_without_changing_rgb_input(pil):
    rgb = np.full((24, 24, 3), [220, 90, 30], dtype=np.uint8)
    original = rgb.copy()
    detector = YOLODetector.__new__(YOLODetector)
    detector.device = "cpu"
    detector.model = Mock(predict=Mock(return_value=[object()]))
    detector.detect(Image.fromarray(rgb) if pil else rgb)
    supplied = detector.model.predict.call_args.kwargs["source"]
    np.testing.assert_array_equal(supplied, original[:, :, ::-1])
    np.testing.assert_array_equal(rgb, original)
    assert supplied.flags.c_contiguous


@pytest.mark.parametrize("metadata", [METADATA, HIGH_FACTORS])
def test_no_detection_is_inconclusive_and_has_no_risk_score(metadata):
    pipeline = make_pipeline([], [])
    result = pipeline.process_image(Image.new("RGB", (24, 24)), metadata)
    assert result["risk_level"] == "INCONCLUSIVE"
    assert result["risk_score"] is None
    assert result["assessment_issues"]
    assert "routine monitoring" not in result["recommendation"].lower()
    pipeline.biomedclip.classify.assert_not_called()


def test_invalid_crop_is_inconclusive_instead_of_low():
    pipeline = make_pipeline([0], [], boxes=[[0, 0, 5, 5]])
    result = pipeline.process_image(Image.new("RGB", (24, 24)), METADATA)
    assert result["risk_level"] == "INCONCLUSIVE"
    assert result["risk_score"] is None
    pipeline.biomedclip.classify.assert_not_called()


def test_confident_normal_crop_does_not_drive_cancer_risk():
    scores = {"OCA": 0.001, "OPMD": 0.002, "Benign": 0.007, "Normal": 0.99}
    pipeline = make_pipeline([2], [classification("Normal", scores)])
    result = pipeline.process_image(Image.new("RGB", (24, 24)), METADATA)
    assert result["risk_level"] == "LOW"
    assert result["risk_score"] < 1.6
    detection = result["detections"][0]
    assert detection["fusion_score"] > 0.8  # Strong semantic match is still recorded.
    assert detection["concern_score"] < 0.01
    assert detection["yolo_class"] == "Benign"


def test_cancer_detector_and_benign_classifier_disagreement_is_not_reassuring():
    scores = {"OCA": 0.001, "OPMD": 0.002, "Benign": 0.99, "Normal": 0.007}
    pipeline = make_pipeline([0], [classification("Benign", scores)])
    result = pipeline.process_image(Image.new("RGB", (24, 24)), METADATA)
    assert result["risk_level"] == "INCONCLUSIVE"
    assert result["risk_score"] is None
    assert result["detections"][0]["model_disagreement"]
    assert any("disagree" in issue.lower() for issue in result["assessment_issues"])


def test_ambiguous_classifier_abstains_even_with_high_patient_factors():
    scores = {"OCA": 0.3, "OPMD": 0.4, "Benign": 0.2, "Normal": 0.1}
    pipeline = make_pipeline([1], [classification("OPMD", scores)])
    result = pipeline.process_image(Image.new("RGB", (24, 24)), HIGH_FACTORS)
    assert result["risk_level"] == "INCONCLUSIVE"
    assert result["risk_score"] is None
    assert result["detections"][0]["classification_uncertain"]


@pytest.mark.parametrize("label,class_id,expected", [("OCA", 0, "HIGH"), ("OPMD", 1, "MEDIUM")])
def test_agreed_concerning_class_cannot_return_low(label, class_id, expected):
    scores = {"OCA": 0.03, "OPMD": 0.03, "Benign": 0.02, "Normal": 0.02}
    scores[label] = 0.93
    pipeline = make_pipeline([class_id], [classification(label, scores)], confidences=[0.26])
    result = pipeline.process_image(Image.new("RGB", (24, 24)), METADATA)
    assert result["risk_level"] == expected
    assert result["assessment_issues"] == []


def test_rgb_crop_reaches_classifier_unchanged():
    rgb = np.full((24, 24, 3), [220, 90, 30], dtype=np.uint8)
    scores = {"OCA": 0.001, "OPMD": 0.002, "Benign": 0.99, "Normal": 0.007}
    pipeline = make_pipeline([2], [classification("Benign", scores)])
    pipeline.process_image(rgb, METADATA)
    np.testing.assert_array_equal(pipeline.biomedclip.classify.call_args.args[0], rgb)


def test_score_is_bounded_to_the_displayed_scale():
    result = ClinicalDecisionTree().assess_risk(
        1.0, **HIGH_FACTORS, concerning_class="OCA"
    )
    assert result["score"] == 10.0


def test_any_unresolved_region_prevents_a_complete_image_assessment():
    benign = {"OCA": 0.001, "OPMD": 0.002, "Benign": 0.99, "Normal": 0.007}
    pipeline = make_pipeline([2, 0], [classification("Benign", benign)] * 2)
    result = pipeline.process_image(Image.new("RGB", (24, 24)), METADATA)
    assert result["risk_level"] == "INCONCLUSIVE"
    assert len(result["detections"]) == 2


def test_inconclusive_ui_displays_no_numeric_score_or_success_message():
    from streamlit.testing.v1 import AppTest

    result = make_pipeline([], []).process_image(Image.new("RGB", (24, 24)), HIGH_FACTORS)
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from streamlit_app import display_results\n"
        "display_results(st.session_state['assessment'])\n"
    )
    app.session_state['assessment'] = result
    app.run(timeout=15)
    assert not app.exception
    assert any('INCONCLUSIVE' in heading.value for heading in app.subheader)
    assert app.metric[0].value == 'Not assessed'
    assert app.warning
    assert not app.success


def test_detected_region_ui_displays_both_labels_and_relative_scores():
    from streamlit.testing.v1 import AppTest

    scores = {"OCA": 0.001, "OPMD": 0.002, "Benign": 0.99, "Normal": 0.007}
    result = make_pipeline([0], [classification('Benign', scores)]).process_image(
        Image.new('RGB', (24, 24)), METADATA
    )
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from streamlit_app import display_results\n"
        "display_results(st.session_state['assessment'])\n"
    )
    app.session_state['assessment'] = result
    app.run(timeout=15)
    assert not app.exception
    text = '\n'.join(item.value for item in app.markdown)
    assert 'Detector label:** OCA' in text
    assert 'Classifier label:** Benign' in text
    assert 'not disease probabilities' in text
