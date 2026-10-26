# Exploratory demo readouts — 23 October 2026

The local demo can now display real classifier readings while retaining the
conservative review result separately. No weights, prompts, crop preprocessing
or medical assessment recommendations were changed. This has not yet been
deployed to the public AWS demo at the time of this experiment. The simplified
results interface was subsequently [deployed on 26 October](../2026-10-26/AWS_RELEASE.md).

| Limit or rule | Conservative request | Default exploratory request |
|---|---:|---:|
| Minimum YOLO confidence | 0.25 | 0.10 |
| Minimum classifier relative score | 0.60 | 0.35 for provisional display |
| Minimum top-two score gap | 0.15 | 0.02 for provisional display |
| Incompatible model labels | Abstain | Display provisionally; retain both labels and warning |
| Missing or unusable regions | No assessment or score | No reading or score |

`OncoEdgePipeline.process_image` remains conservative by default. Callers opt
into readouts with a per-request immutable `DemoOptions`; the UI enables this
mode by default and exposes the three limits in an expander. Settings are not
written into shared CONFIG or model resources. Changing a mode or limit clears
stale UI results. JSON exports retain the settings, provisional output, both
labels, omitted regions and conservative assessment issues.

The headline is the eligible region with the highest YOLO-confidence ×
classifier-relative-score match. It is not selected by the most concerning
label or the published reference. Other regions remain inspectable. The
provisional index is `10 × max(YOLO confidence × (OCA + OPMD relative scores))`
over eligible regions. It excludes age, tobacco and duration inputs, and is
neither a calibrated probability nor a clinical risk score. Conservative review
retains its original classifier/agreement gates and also flags candidates
below the original YOLO limit.

## Real-model replay

[Retained results](demo-readout-results.json) record CPU/PyTorch inference on
all 17 existing published photographs, using the same `best.pt` and BioMedCLIP
weights. The script checks photo hashes and verifies that a conservative rerun
reproduces historical region counts, boxes, labels, fused matches within 1e-4
and all 17 INCONCLUSIVE assessments.

- Conservative requests: regions on 11/17 photos; 0 completed assessments.
- Exploratory requests: regions and provisional readings on 15/17 photos.
- Gallery: all six samples receive provisional readings.
- No reading: `benign-04`, `benign-07`; no whole-image fallback is used.
- Provisional headline labels: 7 Benign, 4 OPMD, 2 OCA, 2 Normal.
- Eight headline labels differ from the publication's whole-photo category,
  including source-described OCA images returning Benign. This demonstrates
  that lower display limits have not fixed classification errors. It is not
  region-level accuracy because reference boxes and crop labels are absent.

This convenience set was already inspected during development and tuning.
The result establishes software behavior and increased provisional coverage,
not independent medical performance. No image overlays or copied restricted
source photos are published by this replay.

To reproduce without overwriting the retained result:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  .venv/bin/python evaluation/evaluate_demo_mode.py --output /tmp/oncoedge-demo-replay.json
.venv/bin/python -m pytest tests/test_assessment.py tests/test_demo_ui.py tests/test_deployment.py -q
```

The automated suite passes 39 checks, including cached pipeline compatibility
with reloaded UI settings. The live local browser was also checked
for real inference, mode switching, stale-result clearing and displayed limits.
