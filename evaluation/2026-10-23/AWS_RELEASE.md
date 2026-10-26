# AWS demo release — 23 October 2026

Live URL: <https://oncoedge.54-253-96-120.sslip.io/>

Application revision: `b9fc1bcfcb103fb4dff4d91ea6973974c65345a6` on
`codex/deploy-demo`. This release was committed locally, pushed through the
personal GitHub SSH remote, pulled with `--ff-only` on AWS and built there.
Subsequent documentation commits do not change the application image.

## Infrastructure and release

- Existing `oncoedge-demo` Lightsail server, 2 vCPUs / 4 GB RAM, reused.
- Existing static IP, Caddy HTTPS gateway, `.env`, trained checkpoint and named
  model/certificate volumes retained. No additional AWS resources were created.
- CPU dependencies reused from Docker's cached layers; model smoke checks used
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- Current image manifest: `sha256:bbbdd4988dfba3baf081952f4a671563fb57dbdae2f5fc662d51a65f2479126e`.
- OCI `org.opencontainers.image.revision` label matches the application revision.
- Prior image retained as `oncoedge:rollback-1f195a0-20261023`
  (image ID `068734e85015`).
- The app was stopped for model verification to avoid running two model processes
  on the 4 GB host. Cutover started at 06:23:53 UTC; the replacement became healthy
  and its public health endpoint returned `ok` at 06:25:22 UTC. Caddy stayed up.

The local Docker socket required privileges unavailable to this session. The
replacement image was therefore built and checked on AWS before being exposed
to visitors. A smoke-run command consumed the remaining stdin of the initial
cutover script, so the already-verified web image was started in a separate SSH
command. The local cutover script was corrected to redirect the smoke command's
stdin from `/dev/null`; the extra pause is included in the times above.

## Verification

- **25 passing local regression checks**: assessment failures, model reuse,
  checkpoint provisioning, session/input handling and sample integrity.
- **Eight real-model Docker smoke cases**: both original photos and all six new
  gallery photos. Trained YOLO classes and classifier score validity were checked.
  All eight assessments were INCONCLUSIVE; no clinical performance claim follows.
- Model initialization: **18.8 seconds**. Process peak RSS: **1,972.5 MiB**;
  container peak memory: **2,220.5 MiB**.
- Public browser: selected **Oral cavity · red patch**, then ran actual inference.
  One region, YOLO OPMD versus BioMedCLIP Benign, INCONCLUSIVE, no numeric score.
  First analysis: **20.2 seconds**, including loading models into the web process.
- Downloaded JSON matches the visible result and preserves it across the rerun.
- Switching to Upload clears the old assessment and disables Run without a file.
- Uploaded original `benign-01.png` through public HTTPS: **3 regions**, INCONCLUSIVE,
  explanation for the third region's label disagreement, **1.6 seconds** warmed.
- Runtime UI/detector/pipeline/CSS/theme SHA-256 hashes match the local release.
- Observed healthy, zero restarts, `OOMKilled=false`; about **1.215 GiB** in Docker's
  memory display after the initial browser analysis. No multi-user load test.

Evidence:

- [AWS model smoke output summary](aws-smoke-summary.json)
- [Downloaded public sample report](hosted-sample-report.json)
- [Public upload accessibility record](hosted-upload-dom.txt)
- [Public upload screenshot](hosted-upload.png)
- [Public gallery screenshot](hosted-gallery.png)

## Rollback

From `/home/ubuntu/OncoEdge-Jetson` on the server:

```bash
sudo docker tag oncoedge:rollback-1f195a0-20261023 oncoedge:local
sudo docker compose -f compose.yaml -f compose.production.yaml \
  up -d --no-deps --wait oncoedge
curl --fail https://oncoedge.54-253-96-120.sslip.io/_stcore/health
```

This restores the previous runtime image without deleting any model or
certificate volumes. The source checkout remains at the newer commit, so record
the rollback explicitly before rebuilding or deploying again. The saved image's
presence was verified; restoring it was not necessary during this release.
