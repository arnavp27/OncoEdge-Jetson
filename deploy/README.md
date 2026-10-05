# CPU demo deployment

This Docker deployment runs the existing PyTorch pipeline and sample picker on a
Linux x86_64 server. The Jetson/TensorRT implementation is still available in the
repository. Docker uses Python 3.12, uv, and a CPU dependency lock derived from the
working local environment. The detector, prompts, fusion and risk rules are unchanged.

## Local verification

Run these commands from the repository root:

```bash
python3 deploy/prepare_checkpoint.py
docker compose build oncoedge
docker compose run --rm oncoedge python deploy/warm_models.py --smoke-test
docker compose up -d --wait oncoedge
```

Open `http://127.0.0.1:8503`. If another local Streamlit process is already using
8503, stop that process first or use `ONCOEDGE_PORT=8504 docker compose up -d --wait
oncoedge` for a temporary test. On this development machine, the previous launcher
can be stopped with `systemctl --user stop oncoedge-local`.

The warm-up command checks the trained detector's classes, initializes BioMedCLIP,
and optionally runs both samples and the classifier. It exercises the classifier
even if YOLO finds no regions. These checks verify execution, not clinical accuracy.
The web health check verifies the Streamlit server; it does not initialize the models.
Visitors' first analysis still loads the models into the web process's memory, using
the prepared files. Model downloads may take several minutes on the first warm-up.

`models/best.pt` is a read-only host file. The preparation script verifies its
SHA-256 checksum and extracts the original checkpoint from the pinned training
commit if the file is missing. If Git reports that the training commit is absent,
run `git fetch origin develop_yolo_ft` and retry. The script leaves an existing
checkpoint with a different checksum untouched.

BioMedCLIP weights and Hugging Face configuration/tokenizer files use named volumes.
BioMedCLIP loads the saved checkpoint directly rather than downloading another copy
into the Hugging Face hub cache. These volumes survive container replacement and
ordinary `docker compose down`. Do not use `docker compose down --volumes` unless
you intend to delete the model caches and HTTPS certificates.

The container runs as a non-root user, caps CPU/RAM use, rotates logs, and binds its
host port to localhost. The defaults are 2 CPUs, 2 inference threads and a 3 GiB
memory cap. Confirm actual peak memory with the real container before sizing a
server for all three projects; the other applications and host also need memory.

### Verification on 5 October 2026

The image was built locally and the real CPU pipeline ran inside Docker with
Hugging Face network access disabled. Both samples and the classifier executed;
the detector found no regions in these two samples at the existing threshold.
The local-checkpoint loader produced the same prompt embeddings and sample
classification scores as the original hub loader. Four focused regression tests
passed, and the checkpoint was successfully extracted into a fresh directory and
verified against its checksum.

The container ran as UID 10001, became healthy, and retained its model files after
replacement. Sample selection and analysis worked through a temporary Caddy HTTP
proxy in the browser, including Streamlit's WebSocket connection. Caddy's HTTPS
configuration also validated; public certificate issuance has not been tested.

With the 2-CPU/3-GiB limits, the standalone model smoke check peaked at 2,744.5 MiB.
The web container reached its 3-GiB memory cap during startup/analysis and later
used about 1.4 GiB in Docker's memory display. The peak includes filesystem cache;
these observations come from this development machine and are not an AWS load or
concurrency benchmark. Budget memory for the other two applications before
choosing between a 4-GB and an 8-GB shared server.

## OncoEdge AWS deployment

The demo uses the `codex/deploy-demo` branch on the `oncoedge-demo` Lightsail
instance in Sydney (`ap-southeast-2a`): Ubuntu 24.04, 2 vCPUs, 4 GB RAM and
80 GB SSD. The plan is $24/month. The attached static IP is `54.253.96.120`,
with the live demo at [https://oncoedge.54-253-96-120.sslip.io/](https://oncoedge.54-253-96-120.sslip.io/).

The server checkout is `/home/ubuntu/OncoEdge-Jetson`. The private `.env`,
trained detector and named model/certificate volumes stay on the server.
SSH permits the deployment connection's IPv4 address and Lightsail browser SSH;
update that source rule if your public IP changes. Public web ports are 80/443,
while the Streamlit port binds to `127.0.0.1:8503`.

On 5 October 2026, the real model smoke test passed on this instance. Model
preparation including downloads took 29.2 seconds; the pipeline analyzed the
bundled samples in 1.11 and 0.71 seconds. Both had zero detections at the unchanged
threshold. The separate classifier check returned Benign for the lip photograph
and OPMD for the annotated tongue photograph. Peak container memory was
2,740.3 MiB. These are technical smoke checks, not a clinical evaluation or
multi-user benchmark.

The public health endpoint returned HTTP 200 with certificate verification enabled,
and HTTP redirected to HTTPS. Caddy obtained a Let's Encrypt certificate. The first
web analysis loaded the cached model files successfully; the web container later
used about 1.2 GiB and peaked at 2,538,381,312 bytes (about 2.36 GiB).
Both sample choices and a JPEG upload completed analysis in the browser through
public HTTPS, including the Streamlit WebSocket connection.

## AWS server setup

Use one Ubuntu 24.04 x86_64 EC2 or Lightsail instance. Install Git and Docker Engine
with the Compose plugin using [Docker's Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/).
`deploy/bootstrap-ubuntu.sh` packages those repository-based installation steps for
a fresh Ubuntu 24.04 server; run it there with `sudo bash deploy/bootstrap-ubuntu.sh`.
It refuses other operating systems/architectures and leaves an existing Docker
installation in place. Prefix the Docker commands below with `sudo` if your server
account does not have access to its Docker daemon.
Clone the repository after the deployment changes have been pushed, provision the
checkpoint, and run the local verification commands above on the server.

When using a Lightsail launch script, invoke Bash explicitly with a heredoc:

```sh
/usr/bin/bash <<'BOOTSTRAP'
# Put the bootstrap commands here.
BOOTSTRAP
```

Lightsail prepends its own `sh` wrapper, so a Bash shebang inside the entered
script does not select Bash. Running the saved script over SSH with `sudo bash`
also works.

No private SSH key belongs in the image. This is a public repository, so the server
can clone over HTTPS. Use a read-only deploy key if the repository becomes private.

```bash
git clone --single-branch --branch codex/deploy-demo https://github.com/arnavp27/OncoEdge-Jetson.git
cd OncoEdge-Jetson
git fetch origin develop_yolo_ft
python3 deploy/prepare_checkpoint.py
cp .env.example .env
```

Give the instance a stable public IPv4 address and point the demo hostname's DNS A
record at it. Without a purchased domain, `oncoedge.<static-ip-with-dashes>.sslip.io`
can provide an IP-based hostname; verify that it resolves before starting Caddy.
Set `ONCOEDGE_DOMAIN` in `.env` to that hostname. In the AWS firewall,
allow inbound TCP 80 and 443; restrict SSH 22 to your own IP. Port 8503 stays private.
Then launch the HTTPS gateway:

```bash
docker compose -f compose.yaml -f compose.production.yaml up -d --build --wait
```

Caddy obtains and renews the certificate and proxies Streamlit WebSocket connections.
The hostname must resolve to the instance and ports 80/443 must be reachable before
certificate issuance will work. The Compose health status alone does not confirm
that the public DNS and certificate are working; open the actual HTTPS URL afterward.

Only Caddy publishes public web ports. It can later route the other projects by
hostname. Their Compose files can join the existing `demo_edge` Docker network as
an external network, using distinct service names, while keeping their own model
or database volumes. Do not run an additional gateway on the same host ports.

## Code updates

Edit and test locally, commit and push, then run these commands in the server checkout:

```bash
git pull --ff-only
docker compose -f compose.yaml -f compose.production.yaml up -d --build --no-deps --wait oncoedge
```

Dependency installation is an earlier Docker layer, so UI and pipeline code edits
reuse it while the dependency lock and base image remain unchanged. Updating OncoEdge
restarts its web process and resets active browser sessions. The gateway and other
projects keep running, and model caches stay available. The checkpoint verification
script is needed again only when provisioning a new checkout or deliberately
changing the supplied model artifact.

If application dependencies change, regenerate the CPU lock with uv before rebuilding:

```bash
uv pip compile requirements.txt --python-version 3.12 --torch-backend cpu \
  --python-platform x86_64-unknown-linux-gnu --generate-hashes \
  --output-file deploy/requirements.cpu.lock
```

For local Docker without the HTTPS gateway, use the same update command with only
`docker compose up -d --build --no-deps --wait oncoedge`.

Check operation with:

```bash
docker compose ps
docker compose logs --tail=100 oncoedge
docker stats --no-stream
```
