# Investigation of frequent inconclusive assessments

23 October 2026. Local CPU/PyTorch experiments; application code, model weights,
thresholds and the AWS deployment were not changed.

The current system cannot be made reliably decisive by a configuration change
demonstrated in this experiment. Several alternatives changed predictions, but
none established a trustworthy replacement for the existing assessment policy.
The next model improvement needs labelled oral photographs and an independent
evaluation, rather than weaker abstention gates.

## What causes the current results?

The saved 17-photo replay has 19 detector regions across 11 photographs. All 17
image assessments are INCONCLUSIVE. The image-level reasons are:

| Reason | Photographs |
|---|---:|
| No detected regions | 6 |
| Incompatible detector/classifier labels only | 6 |
| Ambiguous classifier scores only | 2 |
| Both incompatible labels and ambiguous scores | 3 |

There are 11 region-level label disagreements and six region-level ambiguous
scores; those counts overlap. A single unresolved region currently prevents a
completed image assessment. Agreement alone is not evidence of correctness:
some resolved crop labels differ from their photograph's source category.

YOLO is the fine-tuned three-class model in `models/best.pt`. BioMedCLIP remains
zero-shot, with one text description per class. In particular, leukoplakia and
aphthous ulcers stand in for the much broader OPMD and Benign categories.

## Controlled comparisons

The [replay script](../diagnose_abstention.py) preserves the historical files,
verifies every photograph's SHA-256, and checks that baseline classifier labels
and scores reproduce the saved results on the saved detector crops. The five
classifier alternatives below each change one setup choice from that baseline;
the detector-size comparison runs YOLO again on the original photographs.

| Setup | Photos with detector regions | Photos with all regions resolved | Photos with any crop matching the photo's source category |
|---|---:|---:|---:|
| Deployed baseline | 11/17 | 0/17 | 6/17 |
| Checkpoint's learned logit scale | 11/17 | 0/17 | 6/17 |
| Preserve the full crop with square padding | 11/17 | 0/17 | 2/17 |
| Add 20% surrounding crop context | 11/17 | 1/17 | 4/17 |
| Broader class descriptions | 11/17 | 3/17 | 6/17 |
| Average two text templates for broader descriptions | 11/17 | 3/17 | 5/17 |

These are diagnostic counts, **not accuracy measurements**. An entire photograph's
source category is not a reference label for every region, and the collection
has no ground-truth localization annotations or healthy controls. This small
set was already inspected during development; it is not an untouched test set.

The broader descriptions resolved `opmd-01`, `opmd-05` and `benign-02` under the
existing gates. The first two matched the source OPMD category. However,
`benign-02`, described by its source as a major aphthous ulcer, received OPMD
with a relative classifier score of **0.9906**. The two-template version also
resolved those same three photographs. Fewer abstentions would therefore not
establish better decisions.

Adding context resolved `opmd-01` with the source-matching OPMD category, but
increased region disagreements from 11 to 16 and reduced photographs with any
source-matching crop from six to four. It is not a consistent fix on this replay.

Changing only YOLO's inference size from 640 to its training size of 512 still
produced regions on 11/17 photographs. It gained regions on `opmd-03` but lost
them on `benign-03`; region count changed from 19 to 16. Without reference boxes,
these differences cannot establish better localization.

The CPU classifier currently hardcodes a similarity scale of 100. The checkpoint
contains a learned scale of **85.2322769165039**, which matches the parameter
used by the Jetson implementation and the approach in Microsoft's
[official inference example](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224#model-use).
Using that parameter did not change winning labels or resolve an image. It
increased ambiguous regions from six to seven. Backend consistency is a separate
engineering improvement; it is not a demonstrated remedy for abstention.

Removing both classifier ambiguity checks (minimum score and margin), while
retaining the disagreement check, would
complete two baseline images: `opmd-01` as OPMD and `opmd-06` as Benign. Both have
source category OPMD. This is a concrete reason not to lower the gate to improve
the demo's apparent success rate. No threshold sweep was used to choose a new
operating point.

All per-region scores, variants, hashes and detector-size outputs are retained
in [abstention-experiment.json](abstention-experiment.json). No variant was
selected for deployment.

## Practical next step with only best.pt available

The user confirmed that the labelled images and original split are unavailable.
The repository's training branch retains metadata and annotation/conversion
code, but these do not replace the original photographs or establish patient
independence of the original split.

The original dataset is still listed by its authors on
[Zenodo](https://zenodo.org/records/10664056), with 3,000 oral photographs and
annotations. Its files are restricted and require an access request. No request
was submitted, and no dataset was downloaded during this investigation.

A scoped next experiment would keep YOLO as the localizer and compare its
predictions and the present zero-shot classifier against a supervised classifier
trained on frozen BioMedCLIP image features. This preserves the edge pipeline
while replacing the unsupported assumption that four prompt matches are a
reliable oral-lesion classifier. It is a proposed experiment, not a promise of
improvement.

Before fitting, recover permitted photographs, reference lesion regions,
categories and patient IDs; create patient-separated development/calibration/test
manifests; and check overlap with the earlier training data. A test patient must
be excluded from **all** component training for an independent end-to-end claim.
The existing detector's original image-level split does not establish this.

Compare crop classification separately from detection, then measure completed
decision coverage alongside per-class errors and concerning lesions assigned
benign/normal outputs. Select the uncertainty policy using development data and
evaluate it once on the untouched test set. The current 17 publication images
remain failure-analysis examples, not training or calibration data.

## Reproduce

Use the existing uv-created environment and cached weights; no installation or
cloud resources are needed:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python evaluation/diagnose_abstention.py \
  --output /tmp/oncoedge-abstention-replay.json
```

The script refuses to replace an existing output. It leaves the historical
manifest, baseline replay, fixed replay, model weights and app configuration
unchanged.
