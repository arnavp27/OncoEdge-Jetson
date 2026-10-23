"""User-flow regressions for the local sample workspace."""
import hashlib
import io
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError
from streamlit.testing.v1 import AppTest

from src.ui import app as ui


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
