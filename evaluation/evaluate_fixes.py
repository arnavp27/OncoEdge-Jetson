"""Replay the saved photographs and compare fixes with the immutable baseline.

Run from the repo root with the existing uv-created .venv and offline model cache:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
    .venv/bin/python evaluation/evaluate_fixes.py
"""
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from src.config import CONFIG
from src.pipeline.inference_pipeline import OncoEdgePipeline


def main():
    directory = ROOT / 'evaluation/2026-10-05'
    baseline_bytes = (directory / 'results.json').read_bytes()
    baseline = json.loads(baseline_bytes)
    manifest = json.loads((directory / 'manifest.json').read_text())
    previous = {case['id']: case for case in baseline['cases']}
    overlay_dir = directory / 'fixed-overlays'
    overlay_dir.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    pipeline = OncoEdgePipeline(device='cpu', force_backend='pytorch')
    cases = []
    for case in manifest['cases']:
        path = directory / case['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case['sha256']
        image = Image.open(path).convert('RGB')
        started = time.perf_counter()
        result = pipeline.process_image(image, manifest['patient_metadata'])
        elapsed = time.perf_counter() - started
        detections = [{k: v for k, v in d.items() if k != 'mask'} for d in result['detections']]
        Image.fromarray(result['visualization']).save(overlay_dir / f"{case['id']}.png")
        expected = previous[case['id']]['bgr_input_diagnostic']['detections']
        # The previous controlled experiment provides a numeric check on the
        # channel fix. Scores/classes must not improve through hidden tuning.
        assert len(detections) == len(expected), case['id']
        for actual, old in zip(detections, expected):
            assert actual['bbox'] == old['bbox'], case['id']
            assert actual['class'] == old['class'], case['id']
            assert abs(actual['fusion_score'] - old['fusion_score']) < 1e-5, case['id']
        if not detections or result['assessment_issues']:
            assert result['risk_level'] == 'INCONCLUSIVE', case['id']
            assert result['risk_score'] is None, case['id']
        if case['expected_class'] == 'OCA':
            assert result['risk_level'] != 'LOW', case['id']
        row = {
            'id': case['id'], 'reference': case['expected_class'],
            'old_regions': len(previous[case['id']]['baseline']['detections']),
            'old_risk_level': previous[case['id']]['baseline']['risk_level'],
            'regions': len(detections), 'detections': detections,
            'risk_level': result['risk_level'], 'risk_score': result['risk_score'],
            'assessment_issues': result['assessment_issues'],
            'recommendation': result['recommendation'], 'pipeline_seconds': elapsed,
        }
        cases.append(row)
        print(json.dumps({k: row[k] for k in ['id', 'regions', 'risk_level', 'risk_score']}), flush=True)
    samples = []
    for filename in ['027.jpeg', 'photo.webp']:
        result = pipeline.process_image(Image.open(ROOT / filename).convert('RGB'), manifest['patient_metadata'])
        samples.append({'file': filename, 'regions': len(result['detections']),
                        'risk_level': result['risk_level'], 'risk_score': result['risk_score'],
                        'assessment_issues': result['assessment_issues']})
    summary = {
        'photos': len(cases),
        'old_photos_with_regions': sum(c['old_regions'] > 0 for c in cases),
        'fixed_photos_with_regions': sum(c['regions'] > 0 for c in cases),
        'fixed_risk_levels': dict(Counter(c['risk_level'] for c in cases)),
        'photos_with_reference_class_crop': sum(
            any(d['class'] == c['reference'] for d in c['detections']) for c in cases
        ),
        'source_labeled_cancers_returning_low': sum(
            c['reference'] == 'OCA' and c['risk_level'] == 'LOW' for c in cases
        ),
    }
    report = {
        'date': '2026-10-05', 'backend': 'cpu/pytorch',
        'baseline_results_sha256': hashlib.sha256(baseline_bytes).hexdigest(),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_hashes': {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / 'src').rglob('*.py'))
        },
        'abstention_gates': {
            'min_classifier_score': CONFIG['model'].min_classifier_score,
            'min_classifier_margin': CONFIG['model'].min_classifier_margin,
            'calibrated': False,
        },
        'summary': summary, 'cases': cases, 'bundled_samples': samples,
        'limitations': baseline['limitations'],
    }
    (directory / 'fixed-results.json').write_text(json.dumps(report, indent=2) + '\n')
    # Replaying must not replace the historical report with fixed-code outputs.
    assert (directory / 'results.json').read_bytes() == baseline_bytes
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
