#!/usr/bin/env bash
# Run only on the new Ubuntu AWS server: sudo bash deploy/bootstrap-ubuntu.sh
set -euo pipefail

if [[ "$(id -u)" != "0" ]]; then
  echo "Run this on the AWS server with sudo bash deploy/bootstrap-ubuntu.sh" >&2
  exit 1
fi

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  echo "This bootstrap is for a fresh Ubuntu 24.04 server." >&2
  exit 1
fi
if [[ "$(dpkg --print-architecture)" != "amd64" ]]; then
  echo "The CPU dependency lock targets Linux x86_64 (amd64)." >&2
  exit 1
fi

# Existing Docker installations should be reviewed rather than replaced.
if command -v docker >/dev/null 2>&1; then
  docker version
  docker compose version
  exit 0
fi

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl git
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: noble
Components: stable
Architectures: amd64
Signed-By: /etc/apt/keyrings/docker.asc
EOF

apt-get update
apt-get install -y --no-install-recommends \
  docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
docker version
docker compose version
