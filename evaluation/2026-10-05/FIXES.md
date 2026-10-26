# Assessment integration fixes

The integration and assessment bugs are fixed locally. The trained `best.pt`,
BioMedCLIP weights, four prompts, detector threshold and input size are unchanged.
This report retains the local verification completed on 5 October 2026.
Later deployment verification is recorded separately in the deployment guide.

## Changes

- **Correct detector colors.** The application retains RGB for classifier crops
  and display, and supplies contiguous BGR arrays to Ultralytics. Previously,
  passing RGB arrays into its BGR preprocessing reversed the model's colors.
- **Abstain on failed detection.** Missing or unusable regions return
  `INCONCLUSIVE` with no numeric score, including when all patient factors apply.
  They no longer produce a reassuring LOW result or routine-monitoring advice.
- **Retain both models' labels.** Disagreement or ambiguous classification makes
  the image assessment inconclusive. YOLO `Benign` and classifier `Normal` are
  treated as compatible non-concerning labels. One unresolved region prevents a
  completed image assessment.
- **Separate semantic match from concern.** A strong Normal or Benign match no
  longer raises cancer concern. Semantic fusion remains detector confidence
  times the winning classifier score; concern evidence uses detector confidence
  times the sum of the OCA and OPMD relative scores.
- **Prevent dilution of concerning labels.** Resolved, agreed OCA receives a HIGH
  research flag; agreed OPMD receives at least MEDIUM. These class floors and the
  patient-factor weights are unvalidated software heuristics.
- **Correct presentation.** The score is bounded to 0–10 and called a demo
  concern score; inconclusive cases display “Not assessed.” The UI shows both
  labels and calls classifier scores relative matches rather than disease
  probabilities. LOW is qualified as not a negative screening result. Overlay
  colors are corrected, review flags are visible, and overlapping region labels
  are separated where space allows. JPEG, PNG and WebP uploads are supported.

The classifier review gates are a maximum relative score of at least 0.60 and a
top-two margin of at least 0.15, configured in `src/config.py`. They are
engineering abstention choices, not calibrated clinical thresholds. No
whole-image classifier fallback was added.

## Retained replay

The same 17 visually reviewed published photographs and synthetic patient inputs
were replayed through the fixed CPU/PyTorch pipeline. Image hashes were checked;
the historical baseline's `results.json` was preserved byte for byte. The fixed
boxes, classifier labels and fused match scores match the earlier isolated BGR
diagnostic, confirming that the result did not come from hidden weight or prompt
tuning.

| Observation | Original baseline | Fixed local pipeline |
|---|---:|---:|
| Photos tested | 17 | 17 |
| Photos with detected regions | 0 | 11 |
| LOW assessments | 17 | 0 |
| INCONCLUSIVE assessments | 0 | 17 |
| Source-labeled cancer photos receiving LOW | 4 | 0 |
| Numeric score when detection fails | 1.5 with default inputs | Not assessed |

Both bundled sample images also return no regions and INCONCLUSIVE. They are
demonstration inputs, not evidence of successful lesion classification.

**The code fixes do not establish model accuracy.** Six of 17 photos have at
least one classified crop matching the publication's broad category; other
crops may disagree. That count is not a diagnostic accuracy estimate. Models
still miss regions and misclassify or disagree. All 17 assessments abstain under
the current gates. Independent patient-level data and model evaluation remain
necessary before making performance claims. Source labels, image quality,
possible training overlap and missing box/mask ground truth retain the limits
described in the [baseline report](REPORT.md).

Results and reproducibility details: [fixed-results.json](fixed-results.json),
[fixed overlays](fixed-overlays/),
[input manifest](manifest.json), and [source attribution](ATTRIBUTION.md).
The run log and the two CC BY-NC input photos and their derivatives stay local.
To replay all 17 cases, obtain those two source photos under the license in the
attribution file and restore their manifest paths and hashes first.

## Verification

The assessment and existing deployment regression suite has **19 passing
checks**. It covers channel order and preserving RGB input, failed detection,
unusable crops, misleading benign confidence, model disagreement, ambiguous
classification, concerning class floors, score bounds, unresolved multiple
regions and Streamlit rendering. The local Docker app was rebuilt on port 8503
with its existing model caches. Its running UI, detector, pipeline and overlay
source hashes match the local files. TensorRT and physical Jetson execution were
not tested on this machine.

The actual local upload and sample interfaces were checked in the browser:

- The bundled lip sample shows zero regions, INCONCLUSIVE and “Not assessed”:
  [DOM record](local-sample-fixed-dom.txt), [screenshot](local-sample-fixed.png).
- Published cancer photo `oca-04.png` shows two regions, INCONCLUSIVE and “Not
  assessed,” with YOLO OPMD versus classifier OCA/Benign disagreements explained:
  [DOM record](local-cancer-fixed-dom.txt). Its screenshot is retained locally
  because the photograph has a noncommercial license.
- The uploader advertises JPEG, PNG and WebP support.

From the repository root, using the existing uv-created environment:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests/test_assessment.py tests/test_deployment.py -q

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  .venv/bin/python evaluation/evaluate_fixes.py
```

`evaluation/evaluate_cases.py` is pinned to the clean historical baseline commit
`1f195a099c4bdde1341375fc1368ace8f9408e9c` and refuses to run against fixed source,
so it cannot accidentally replace the retained baseline with post-fix outputs.
