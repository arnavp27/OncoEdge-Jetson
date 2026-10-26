# Local sample workspace

The local Streamlit UI was redesigned on 23 October 2026. It runs at
**http://127.0.0.1:8504/** using the existing uv-created `.venv`. Another local
project occupies port 8503. This records local UI verification before the
subsequent AWS release; deployment verification is recorded separately.

## Interface

- An ivory and deep-green workspace with a compact pipeline explanation,
  selectable photo gallery, large input preview and separate analysis panel.
- Six real published sample photographs, all CC BY 4.0, copied unchanged from
  the reviewed evaluation set. [Source attribution](../../assets/samples/ATTRIBUTION.md)
  and per-file SHA-256 hashes are retained. Gallery thumbnails alone are reduced
  for display; inference uses the original photo.
- Samples and upload are explicit alternatives. Empty uploads disable analysis.
  JPEG, PNG and WebP are accepted, camera EXIF orientation is respected, and
  corrupt, tiny or oversized images produce a readable error.
- Synthetic patient inputs are optional under **Demo context**. Tobacco years
  greater than age disable analysis and show a validation message.
- Results remain in the current session across reruns, including downloads.
  Changing the photo, source or patient context removes the prior result.
- Outputs explain inconclusive assessments, show both models' labels, expose
  per-class relative scores, display actual elapsed time and offer a JSON export.
  Image previews preserve the complete image rather than cropping to fill a box.
- Presentation code is in `src/ui/app.py`; styles and sample assets are in
  `assets/`. The Docker build includes these assets and the Streamlit theme.

## Verification

The existing assessment/deployment checks and six new interaction/input checks
pass: **25 total**. New checks cover session persistence without repeated
inference, stale results after sample/context/source changes, invalid context,
upload orientation/corrupt bytes and sample integrity/attribution.

The real local browser flow was also checked: selecting the default tongue
sample runs the actual models and produces one region with INCONCLUSIVE,
explained by ambiguous classifier scores. The same WebP was uploaded through
the file chooser and analyzed successfully; exporting its JSON report preserved
the visible result. Switching back to samples clears the uploaded-photo result.
Inference, model weights, prompts and assessment
rules are unchanged by this redesign; it does not improve clinical accuracy.

Run the focused checks with:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests/test_assessment.py tests/test_deployment.py tests/test_demo_ui.py -q
```

## Running locally

The current app runs as the transient user service `oncoedge-local.service`,
with CPU thread limits and offline model caches. It survives closing the
terminal, but is not configured to start on boot.

```bash
systemctl --user status oncoedge-local.service
journalctl --user -u oncoedge-local.service -n 40 --no-pager
systemctl --user stop oncoedge-local.service
```

To run in a terminal instead, stop the service first, then use the command from
the repository root:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.venv/bin/python -m streamlit run streamlit_app.py \
  --server.address=127.0.0.1 --server.port=8504 --server.headless=true
```

Use a free port if 8504 is occupied. The default Docker host port is still 8503;
choose a free `ONCOEDGE_PORT` when running the container alongside other apps.
