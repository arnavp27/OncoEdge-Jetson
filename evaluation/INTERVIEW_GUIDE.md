# OncoEdge evaluation: interview preparation

Training and replay numbers were verified against retained project files on
5 October 2026; UI regression coverage was updated on 23 October 2026. Use these
as talking points and describe your own contribution accurately. Deployment
verification is recorded separately in the deployment guide.

## A short answer to “How did you evaluate your project?”

> I separate detection performance, classifier behavior, end-to-end decisions,
> and deployment performance. The fine-tuning branch retains YOLO validation
> results, including box mAP50 of about 0.512. Separately, we tested the complete
> pipeline on 17 source-labeled published photographs. That exposed an RGB/BGR
> integration bug and a decision rule that returned LOW when detection failed.
> Correcting the colors produced regions in 11 photos. We also added abstention
> for missing or unresolved outputs and regression checks for those failures.
> All 17 conservative assessments abstain. The separate exploratory demo shows
> explicitly provisional readings on 15 photos with lower display limits. I
> present this as a research and edge
> engineering prototype, not a clinically validated screening system.

Do not turn “regions in 11 photos” into a claim of correct localization. Do not
turn “no more LOW cancer results” into a claim that every cancer was detected.

## Numbers to remember

| Evidence | Verified result | What it measures |
|---|---|---|
| Training metadata | 3,000 image records, 714 patient records | Source metadata, not proof that every record was used in the training export |
| Metadata class mix | OPMD 1,394; Benign 748; Healthy 729; OCA 129 | Strong class imbalance; OCA is 4.3% of image records |
| Logged training run | 72 epochs; best combined box/mask mAP50–95 at epoch 71 | Retained validation log, not an independent clinical test |
| YOLO box precision / recall | 0.617 / 0.448 | Logged detector validation metrics |
| YOLO box mAP50 / mAP50–95 | 0.512 / 0.388 | Logged detector validation metrics |
| YOLO mask mAP50 / mAP50–95 | 0.477 / 0.287 | Logged segmentation validation metrics |
| Exploratory photo set | 17: 4 OCA, 6 OPMD, 7 Benign | Small convenience sample from three publications |
| Photos with regions, before → after | 0/17 → 11/17 | Detection presence, **not accuracy or recall** |
| Conservative pipeline decision coverage | 0/17 = 0% | Fraction receiving a completed, non-INCONCLUSIVE assessment |
| Conservative pipeline abstention rate | 17/17 = 100% | Unresolved conservative assessments on this set |
| Exploratory demo readout coverage | 15/17; all 6 gallery samples | Provisional display coverage, not diagnostic accuracy |
| Assessment/UI/deployment regression checks | 39 passing | Code behavior; not clinical accuracy |

Training numbers were checked directly against the cached training branch at
commit `cf745d5785f9422659928701dca7bb56501db9c2`. Metrics are rounded to three
decimals. That checkpoint's validation record and the web pipeline's 17-photo
replay are different evaluations and must be described separately.

## Likely follow-up questions

### 1. “Where did the 17 test images and their labels come from?”

The images came from three published oral-lesion articles. Their broad categories
came from figure descriptions, not our visual diagnosis. Each input was visually
checked to confirm that it was a single clinical photograph without diagnostic
text, a collage, histology or a radiograph. Source URLs, figure IDs, licenses,
dimensions and hashes are retained in the [manifest](2026-10-05/manifest.json).

This was a convenience sample for investigating failures. Some source diagnoses
are mixed, source labels are not uniformly confirmed by biopsy, and the set has
no healthy controls. It is not representative of clinical prevalence.

### 2. “Is 11/17 your accuracy?”

No. It means a detector returned one or more regions on 11 photos. Some boxes
were broad or overlapping. Without reference boxes or masks, we cannot calculate
localization precision, recall, mAP or segmentation overlap on these photos.
An unrelated or incorrectly localized box would still count in that 11/17.

### 3. “Then what accuracy can you report?”

The retained YOLO validation log supports reporting its detection/segmentation
metrics with their split limitations. The 17-photo replay supports describing
specific failures and the effect of a controlled input fix. It does not support
a reliable end-to-end diagnostic accuracy claim.

Six photos had at least one crop whose classifier category matched the source
category. Other crops in the same photo could disagree, and undetected photos
had no classified crops. That “any matching crop” count is not a standard
classification accuracy metric.

### 4. “What does mAP50 mean?”

Intersection over union, or IoU, measures predicted/reference region overlap.
Average precision summarizes a class's precision–recall curve under a matching
criterion. mAP averages AP over classes. mAP50 uses IoU 0.50; mAP50–95 averages
over IoU thresholds from 0.50 to 0.95 in steps of 0.05. Box and mask metrics use
their respective overlap representations.

Box mAP50 of 0.512 does **not** mean “51.2% of patients were diagnosed correctly.”
The logged box recall of 0.448 is also a limitation worth acknowledging, rather
than quoting only the stronger-looking aggregate metric.

### 5. “How did you avoid train/test leakage?”

The original converter shuffles individual images and assigns a configured 20%
to validation. It does not group by patient. With multiple images per patient,
leakage is possible; the available split code does not establish patient-level
independence. Do not claim that leakage was ruled out.

The published-photo set is not verified independent either: the full YOLO
training image set was unavailable locally for overlap checks, and biomedical
pretraining overlap is also possible. These photos were inspected during bug
fixing, so they are now development evidence. A future independent test set
must be kept separate from tuning.

### 6. “What exactly did the RGB/BGR fix prove?”

The pipeline used RGB arrays, but Ultralytics interprets NumPy images as BGR and
reverses their channels during preprocessing. That swapped red and blue at the
model input. A controlled experiment changed only the detector's channel order,
keeping the photos, checkpoint, prompts, threshold and inference size fixed.
Photos with regions increased from zero to 11. Classifier crops and displayed
images continued to use RGB.

The production fix reproduced the diagnostic boxes, labels and match scores.
This isolates an integration defect; it does not establish correct localization
or improved classifier accuracy.

### 7. “If every result is inconclusive, is the system useful?”

It demonstrates a runnable engineering pipeline, deployment and reproducible
failure analysis. On this set, its completed-decision coverage is **0%**. It does
not currently demonstrate dependable screening performance. The improvement is
that failed or unresolved inference no longer creates unsupported reassurance.
Abstention is a guardrail, not a solution to weak model performance.

For a model that can abstain, report coverage alongside error rates on completed
decisions. With zero completed decisions here, accuracy on completed decisions
is undefined. Abstentions must not be counted as correct negatives or detected
cancers. Lowering gates simply to make the demo look successful would not fix
the underlying errors.

The opt-in pipeline demo readout (enabled by default in the local UI) is a
separate display policy: YOLO 0.10, classifier score 0.35, margin 0.02, with
conflicts permitted for provisional display. It produces readings on 15/17
photos; eight headline labels differ from the whole-photo source category.
That is not a region-level error rate because reference regions are unavailable.
The [demo-mode report](2026-10-23/DEMO_MODE.md) records all cases and settings.
Describe this as a coverage demonstration, not a trained or validated improvement.

### 8. “Why use 0.60 confidence and a 0.15 margin?”

They are provisional engineering gates, not clinically calibrated thresholds.
No held-out calibration study establishes that they are optimal. BioMedCLIP's
four-prompt softmax outputs are relative scores, not disease probabilities.
Model agreement does not guarantee correctness either.

There is also a useful mathematical detail: with normalized four-way scores,
a top score of at least 0.60 leaves at most 0.40 for the runner-up, so the margin
is at least 0.20. The current 0.15 margin gate adds no further restriction once
the 0.60 gate passes; it becomes relevant under other configurations. Do not
describe these as two independently validated confidence tests.

Threshold selection should use a separate validation/calibration set and a
declared error-versus-coverage objective, followed by evaluation with fixed gates
on an untouched test set.

### 9. “Why do YOLO and BioMedCLIP disagree?”

YOLO was fine-tuned for three lesion classes; healthy training images are
background negatives rather than a fourth detector class. BioMedCLIP remains
zero-shot here and compares crops to four fixed text descriptions, including a
Normal prototype. The OPMD prompt describes
leukoplakia, and the Benign prompt describes aphthous ulcers; these descriptions
cover less variety than their broad categories. Detection crops, visual domain
differences, prompt wording and model limitations may contribute to disagreement.
The present evaluation does not isolate which contributes most.

The [23 October abstention investigation](2026-10-23/ABSTENTION_INVESTIGATION.md)
compared crop padding, surrounding context, broader prompts, prompt averaging,
the checkpoint's learned score scale and the detector's training image size.
Broader prompts completed three assessments, but one labelled a source-described
benign photograph OPMD with a relative score of 0.9906. None of those alternatives
was selected for deployment. This illustrates why completed-decision coverage
must be considered alongside errors, rather than treating fewer abstentions as
an improvement by itself.

We retain both labels and abstain on incompatible outputs. We did not add a
whole-image fallback: the separate whole-photo classifier experiment matched
only 7/17 source categories, so it did not establish a dependable fallback.

### 10. “Why did you change the risk calculation?”

The original formula used the maximum classifier score regardless of class. A
strong Normal match could increase concern just like a cancer match. It also
returned LOW after no detections, when classification had never run.

The fixed code separates semantic match from concern evidence: concern uses
detector confidence times the sum of the OCA and OPMD relative scores. Missing
or unresolved inference has no numeric score. Completed scores are bounded to
0–10; resolved OCA has a HIGH flag and OPMD at least MEDIUM. These weights and
class floors remain demo heuristics, not validated clinical risk estimates.

### 11. “What do the 39 passing tests prove?”

They check failure handling, channel order, label disagreement, class-aware
concern, bounds, Streamlit rendering, model/checkpoint provisioning, session
persistence, stale results, upload decoding and gallery integrity. Most
assessment tests use controlled detector/classifier outputs to exercise edge
cases. Separately, the real-model replay checks the saved photographs and the
actual browser upload/sample flows were verified.

The additional demo checks cover provisional disagreement, adjustable score
and margin gates, missing regions, settings isolation between requests, clearing
results after mode/threshold changes, and preserving uncertainty in JSON exports.

Passing tests establish expected software behavior. They do not establish model
generalization, medical safety or diagnostic accuracy. Retained input and source
hashes, model provenance, baseline outputs and replay outputs make the observed
results inspectable and reproducible.

### 12. “How would you design a stronger evaluation?”

1. Establish consented/licensed, expert-reviewed labels, reference regions and
   patient identifiers, including healthy and benign controls.
2. Split by patient before fitting, tune only on training/validation data, check
   duplicates and keep an untouched test set. If possible, test a separate site
   or camera population too.
3. Evaluate detection and segmentation separately with per-class precision,
   recall and mAP; report segmentation overlap with an explicit matching rule.
4. Evaluate classification on reference crops to isolate the classifier, then
   on detected crops to measure errors introduced by detection.
5. Define the image/patient-level decision rule and handling of multiple regions
   and abstentions in advance. Report per-class confusion results and, for a
   declared positive/negative target, sensitivity and specificity with confidence
   intervals, plus coverage and abstention counts.
6. Assess score calibration before making probability claims. Report performance
   across relevant image-quality and population groups where sample sizes allow.
7. Compare simple baselines and controlled component changes on the same split,
   retaining failure examples and all experiment settings.

These are proposed next steps, not completed experiments. Prioritize independent
labels/splits and classifier evaluation before tuning demo thresholds.

### 13. “How did you evaluate edge performance?”

The current retained replay is CPU/PyTorch. The recruiter demo is hosted on AWS
CPU infrastructure. Jetson/TensorRT export, calibration and benchmark code exists,
but this checkout has no retained target-device engine/benchmark proving the
documented latency or engine-size expectations. Do not present estimates as
measured Jetson results.

For an edge benchmark, retain the actual device, power mode, software versions,
engine hash/size, peak memory, initialization cost, warmed latency percentiles
and timing boundaries. Compare PyTorch and TensorRT outputs and task metrics
before and after quantization on independent images. Test the complete pipeline,
not only an encoder.

One backend-parity detail still needs review: the current PyTorch classifier
multiplies similarities by a fixed 100 before softmax; the TensorRT classifier
uses an extracted model logit scale. Backend score equivalence has not been
established, and score-dependent gates can behave differently if these differ.

## Claims to avoid

| Avoid saying | Say instead |
|---|---|
| “The model has 64.7% accuracy.” | “Regions were returned on 11 of 17 exploratory photos.” |
| “We detected every cancer after the fix.” | “No cancer photo received LOW; all four assessments were inconclusive.” |
| “Abstention made it clinically safe.” | “Abstention addresses demonstrated misleading outputs; clinical safety is unvalidated.” |
| “The softmax score is the chance of cancer.” | “It is a relative score within four prompts.” |
| “All 25 tests passed, so the model is accurate.” | “The software regressions pass; model quality is evaluated separately.” |
| “The risk thresholds are medically validated.” | “They are provisional demo rules.” |
| “We have an independent external benchmark.” | “Published photos supplied diagnostic cases; independence is unverified.” |
| “Quantization preserved accuracy and achieved a measured Jetson speedup.” | “That requires retained target-device measurements; current size/latency figures are estimates.” |

## Evidence you can open in an interview

- [Baseline evaluation](2026-10-05/REPORT.md): sources, controlled experiments,
  case results and limitations.
- [Local fixes and replay](2026-10-05/FIXES.md): exact changes, before/after
  observations and actual browser evidence.
- [Fixed numeric results](2026-10-05/fixed-results.json): per-region labels,
  scores, abstention reasons and source hashes.
- [Manifest](2026-10-05/manifest.json) and [attribution](2026-10-05/ATTRIBUTION.md):
  source categories, provenance and licenses.
- [Assessment regression tests](../tests/test_assessment.py) and
  [deployment checks](../tests/test_deployment.py).
- Pinned training artifacts: [validation log](https://github.com/arnavp27/OncoEdge-Jetson/blob/cf745d5785f9422659928701dca7bb56501db9c2/zenodo_DOAOC/yolo_finetune/oncoedge_v1/results.csv),
  [image metadata](https://github.com/arnavp27/OncoEdge-Jetson/blob/cf745d5785f9422659928701dca7bb56501db9c2/zenodo_DOAOC/Imagewise_Data.csv),
  [patient metadata](https://github.com/arnavp27/OncoEdge-Jetson/blob/cf745d5785f9422659928701dca7bb56501db9c2/zenodo_DOAOC/Patientwise_Data.csv),
  and [split converter](https://github.com/arnavp27/OncoEdge-Jetson/blob/cf745d5785f9422659928701dca7bb56501db9c2/zenodo_DOAOC/convert_coco_to_yolo_seg.py).

Use `evaluation/evaluate_fixes.py` to replay the current local pipeline. The
historical evaluator is pinned to the old baseline and refuses to run against
fixed source. Tests and replay commands are in the fixes report.
