# AWS release — 26 October 2026

Live application: <https://oncoedge.54-253-96-120.sslip.io/>

Application commit: `4f071ccb925c62873a18c579e3243fe2f23599c4`, pushed through
the personal GitHub SSH remote on `codex/deploy-demo`. Subsequent documentation
commits do not change this application image. Its OCI revision label records
the application commit.

## Release

The existing Lightsail instance (2 vCPUs / 4 GB RAM), static IP, HTTPS gateway,
trained YOLO checkpoint and named model/certificate volumes were reused. No new
AWS resources were created. Dependency layers were cached; only application
layers needed rebuilding. All model checks used the existing offline caches.

The interface now shows a classifier label, relative text match, detected
regions and timing. Analysis settings and model checks remain in collapsed
details. The separate provisional banner and research-use strip were removed;
model limitations remain documented in About OncoEdge and the repository.
Machine-readable exports retain settings, provisional flags, both model labels
and unresolved assessment issues. No weights or prompts were replaced.

- Image tag: `oncoedge:release-4f071cc`, also tagged `oncoedge:local`.
- Build manifest list: `sha256:ede87ce9f43360b1e0cfe76f2a7f38b6cd5b1b9c1d9d88e548f7b10743177f4b`.
- Previous application image retained as `oncoedge:rollback-before-4f071cc`.
- Cutover: **11:50:59–11:52:03 UTC**, 64 seconds. The app was stopped for model
  checks to avoid two loaded model processes on the 4 GB host. Caddy stayed up.
- The replacement was started only after model checks passed. The release
  command included automatic restoration of the previous image on failure;
  rollback was not needed.

## Verification

- **39 local regression checks passed** for assessment, deployment and UI flows.
- **Eight real-image AWS Docker checks**, each exercising conservative assessment,
  default model-result settings and an additional classifier call. All eight
  produced a classifier readout with the default display settings.
- Model initialization: **18.2 seconds**; process peak RSS **1,973.6 MiB**;
  smoke-container peak **2,231.7 MiB**.
- **37 runtime and model checksums match** the tested local release, including
  Python sources, UI assets, Streamlit settings and both saved model weights.
- Public HTTPS gallery: **Cheek · fine markings** produced **OPMD**, **94.2%**
  relative text match, **2 regions**, in **17.546 seconds** including first model
  loading. Its JSON download matches the displayed label, score and regions.
- Switching to Upload cleared the result and disabled analysis until a photo
  was supplied.
- Uploaded bundled `benign-01.png` through public HTTPS: **Benign**, **97.5%**
  relative text match, **4 regions**, **1.973 seconds** warmed. The downloaded
  report preserves the underlying INCONCLUSIVE assessment and model disagreement.
- The simplified public page has no provisional/demo banner or research-use strip.
- After the browser checks: healthy, **0 restarts**, `OOMKilled=false`, approximately
  **1.306 GiB / 3 GiB** in Docker's memory display. Web-container peak **2,292.1 MiB**.
- Public health endpoint returned `ok` with normal TLS verification.

These checks establish execution and deployment behavior. They are not clinical
validation, localization accuracy measurements or a concurrency benchmark.

Evidence:

- [AWS smoke results](aws-smoke-summary.json)
- [Runtime/model hashes](runtime-hashes.json)
- [Public sample report](hosted-sample-report.json) and [page text](hosted-sample-text.txt)
- [Public sample screenshot](hosted-sample.png)
- [Public upload report](hosted-upload-report.json) and [page text](hosted-upload-text.txt)
- [Public upload screenshot](hosted-upload.png)
- [Container health and memory observation](web-runtime.json)

Screenshots use bundled CC BY 4.0 photographs; see the
[sample attribution](../../assets/samples/ATTRIBUTION.md).

## Rollback

Run on the existing server in `/home/ubuntu/OncoEdge-Jetson`:

```bash
sudo docker tag oncoedge:rollback-before-4f071cc oncoedge:local
sudo docker compose -f compose.yaml -f compose.production.yaml \
  up -d --no-deps --wait oncoedge
curl --fail https://oncoedge.54-253-96-120.sslip.io/_stcore/health
```

This restores the previous application image while preserving model and
certificate volumes. The Git checkout stays at the newer revision; record a
rollback before rebuilding so that image and source state are not confused.
