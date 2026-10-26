"""User-flow regressions for the local sample workspace."""
import hashlib
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError
from streamlit.testing.v1 import AppTest

from src.ui import app as ui
from src.pipeline.demo_readout import DemoOptions, build_demo_readout


@pytest.fixture
def demo(monkeypatch):
    result = {
        'risk_level': 'INCONCLUSIVE', 'risk_score': None,
        'recommendation': 'Unable to assess reliably; disease has not been ruled out.',
        'assessment_issues': ['No regions were available for classification.'],
        'detections': [], 'visualization': np.zeros((40, 40, 3), dtype=np.uint8),
    }
    pipeline = SimpleNamespace(backend='pytorch', process_image=Mock(return_value=result))
    monkeypatch.setattr(ui, 'load_pipeline', lambda: pipeline)
    app = AppTest.from_file(str(ui.ROOT / 'streamlit_app.py')).run(timeout=20)
    assert not app.exception
    return app, pipeline


def click(app, key):
    next(button for button in app.button if button.key == key).click().run(timeout=20)
    assert not app.exception


def test_result_survives_reruns_without_repeating_inference(demo):
    app, pipeline = demo
    click(app, 'run_analysis')
    assert app.metric[0].value == 'Not assessed'
    app.run(timeout=20)
    assert not app.exception
    assert app.metric[1].value == '0'
    pipeline.process_image.assert_called_once()


def test_selecting_another_photo_removes_the_previous_result(demo):
    app, pipeline = demo
    click(app, 'run_analysis')
    click(app, 'pick_benign-01')
    assert app.session_state['selected_sample'] == 'benign-01'
    assert 'analysis' not in app.session_state
    assert not app.metric
    assert any('Inputs changed' in message.value for message in app.info)
    pipeline.process_image.assert_called_once()


def test_changing_context_clears_old_result_and_rejects_invalid_years(demo):
    app, pipeline = demo
    click(app, 'run_analysis')
    next(field for field in app.number_input if field.key == 'demo_tobacco').set_value(80).run()
    assert not app.exception
    assert 'analysis' not in app.session_state
    assert next(button for button in app.button if button.key == 'run_analysis').disabled
    assert any('cannot exceed age' in warning.value for warning in app.warning)
    pipeline.process_image.assert_called_once()


def test_upload_mode_requires_a_photo_and_clears_sample_result(demo):
    app, pipeline = demo
    click(app, 'run_analysis')
    app.session_state['image_source'] = 'Upload'
    app.run(timeout=20)
    assert not app.exception
    assert 'analysis' not in app.session_state
    assert next(button for button in app.button if button.key == 'run_analysis').disabled
    assert not app.metric
    pipeline.process_image.assert_called_once()


def test_phone_orientation_is_applied_and_unreadable_upload_is_rejected():
    output = io.BytesIO()
    source = Image.new('RGB', (40, 64), '#a3516c')
    exif = source.getexif()
    exif[274] = 6  # A portrait camera image displayed after a 90-degree rotation.
    source.save(output, format='JPEG', exif=exif)
    assert ui.read_image(output.getvalue()).size == (64, 40)
    with pytest.raises(UnidentifiedImageError):
        ui.read_image(b'this is not an image')


def test_demo_samples_preserve_reviewed_inputs_and_attribution():
    samples = ui.load_samples()
    assert len(samples) == 6
    for sample in samples:
        path = ui.ASSETS / 'samples' / sample['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sample['sha256']
        assert sample['license'] == 'CC BY 4.0'
        assert sample['source_url'].startswith('https://')
        assert ui.read_image(path.read_bytes()).mode == 'RGB'


def test_mode_and_limit_changes_clear_previous_results(demo):
    app, pipeline = demo
    click(app, 'run_analysis')
    next(field for field in app.slider if field.key == 'demo_detector_limit').set_value(15).run()
    assert not app.exception
    assert 'analysis' not in app.session_state
    assert not app.metric
    click(app, 'run_analysis')
    assert pipeline.process_image.call_args.kwargs['demo_options'].detector_threshold == 0.15
    app.toggle(key='require_model_agreement').set_value(True).run()
    assert not app.exception
    assert 'analysis' not in app.session_state
    assert not app.metric
    click(app, 'run_analysis')
    assert pipeline.process_image.call_args.kwargs['demo_options'] is None


def test_simple_model_results_keep_conflict_and_exported_settings(demo):
    app, pipeline = demo
    detection = {
        'bbox': [0, 0, 30, 30], 'mask': np.ones((40, 40)),
        'yolo_conf': 0.5, 'yolo_class': 'OCA', 'class': 'Benign',
        'class_scores': {'OCA': 0.01, 'OPMD': 0.02, 'Benign': 0.95, 'Normal': 0.02},
        'fusion_score': 0.475, 'concern_score': 0.015, 'classifier_margin': 0.93,
        'model_disagreement': True, 'classification_uncertain': False,
    }
    result = pipeline.process_image.return_value.copy()
    result.update(detections=[detection], assessment_issues=['Models disagree.'],
                  demo_readout=build_demo_readout([detection], DemoOptions()))
    pipeline.process_image.return_value = result
    click(app, 'run_analysis')
    assert any('Model label: Benign' in heading.value for heading in app.subheader)
    assert any('different labels' in warning.value for warning in app.warning)
    assert any('Models disagree' in message.value for message in app.markdown)
    assert any('Model checks' == panel.label for panel in app.expander)
    assert app.metric[0].label == 'Text match'
    assert app.metric[0].value == '95.0%'
    assert not any('demo' in panel.label.lower() or 'provisional' in panel.label.lower()
                   for panel in app.expander)
    assert not app.success
    report = json.loads(ui.report_json(app.session_state['analysis']))
    assert report['risk_level'] == 'INCONCLUSIVE'
    assert report['risk_score'] is None
    assert report['demo_readout']['provisional'] is True
    assert report['demo_readout']['model_disagreement'] is True
    assert report['inference_settings']['mode'] == 'exploratory_demo'
    assert report['inference_settings']['detector_threshold'] == 0.10
    assert 'mask' not in report['detections'][0]
