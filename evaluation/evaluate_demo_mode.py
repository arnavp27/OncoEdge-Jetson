"""Replay conservative and exploratory modes without overwriting earlier results.

This convenience set was already inspected while developing the demo. Coverage
counts are software behavior checks, not an independent accuracy evaluation.

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
    .venv/bin/python evaluation/evaluate_demo_mode.py
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from src.pipeline.demo_readout import DemoOptions
from src.pipeline.inference_pipeline import OncoEdgePipeline


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'evaluation/2026-10-23/demo-readout-results.json')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists. Choose a new --output to preserve earlier results.')
    directory = ROOT / 'evaluation/2026-10-05'
    manifest = json.loads((directory / 'manifest.json').read_text())
    baseline = json.loads((directory / 'fixed-results.json').read_text())
    historical = {c['id']: c for c in baseline['cases']}
    torch.set_num_threads(2)
    pipeline = OncoEdgePipeline(device='cpu', force_backend='pytorch')
    options = DemoOptions()
    cases = []
    for case in manifest['cases']:
        path = directory / case['file']
        assert sha256(path) == case['sha256'], f'Input changed: {case["id"]}'
        image = Image.open(path).convert('RGB')
        modes = {}
        for mode, setting in [('conservative', None), ('exploratory', options)]:
            started = time.perf_counter()
            result = pipeline.process_image(image, manifest['patient_metadata'], demo_options=setting)
            modes[mode] = {
                'elapsed_seconds': round(time.perf_counter() - started, 3),
                **{key: value for key, value in result.items()
                   if key not in ('visualization', 'detections')},
                'detections': [{key: value for key, value in region.items() if key != 'mask'}
                               for region in result['detections']],
            }
        assert modes['conservative']['risk_level'] == historical[case['id']]['risk_level']
        assert len(modes['conservative']['detections']) == historical[case['id']]['regions']
        for actual, previous in zip(modes['conservative']['detections'], historical[case['id']]['detections']):
            assert actual['class'] == previous['class']
            assert actual['bbox'] == previous['bbox']
            assert abs(actual['fusion_score'] - previous['fusion_score']) < 1e-4
        cases.append({'id': case['id'], 'reference': case['expected_class'], **modes})
        readout = modes['exploratory']['demo_readout']
        print(f'{case["id"]}: {len(modes["exploratory"]["detections"])} regions; '
              f'provisional={readout["class"]}', flush=True)

    readings = [case for case in cases if case['exploratory']['demo_readout']['class'] is not None]
    sources = [directory / 'manifest.json', directory / 'fixed-results.json',
               ROOT / 'models/best.pt', ROOT / 'models/biomedclip/open_clip_pytorch_model.bin',
               *sorted((ROOT / 'src').rglob('*.py'))]
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Coverage and behavior replay on an already inspected convenience set; not accuracy validation.',
        'backend': pipeline.backend + '/cpu',
        'source_hashes': {str(path.relative_to(ROOT)): sha256(path) for path in sources if path.is_file()},
        'summary': {
            'photos': len(cases),
            'conservative_risk_levels': dict(Counter(c['conservative']['risk_level'] for c in cases)),
            'conservative_matches_historical_baseline': True,
            'exploratory_photos_with_regions': sum(bool(c['exploratory']['detections']) for c in cases),
            'exploratory_photos_with_provisional_reading': len(readings),
            'provisional_class_counts': dict(Counter(c['exploratory']['demo_readout']['class'] for c in readings)),
            'no_reading_ids': [c['id'] for c in cases if c not in readings],
            'provisional_primary_label_differs_from_photo_reference': [
                {'id': c['id'], 'reference': c['reference'],
                 'provisional_class': c['exploratory']['demo_readout']['class']}
                for c in readings if c['reference'] != c['exploratory']['demo_readout']['class']
            ],
        },
        'cases': cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
