# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3

COPY --from=ghcr.io/astral-sh/uv:0.12.10@sha256:2bb3ebca0a796a155094a27773d290c4b074572e6107f171d88d086682fd2500 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/oncoedge/.cache/huggingface \
    MPLCONFIGDIR=/home/oncoedge/.cache/matplotlib \
    YOLO_CONFIG_DIR=/home/oncoedge/.config/ultralytics \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/requirements.cpu.lock /tmp/requirements.cpu.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --torch-backend cpu --require-hashes \
        --requirement /tmp/requirements.cpu.lock \
    && uv pip check --python /opt/venv/bin/python

RUN useradd --create-home --uid 10001 oncoedge \
    && mkdir -p /app/models/biomedclip /home/oncoedge/.cache/huggingface \
        /home/oncoedge/.config/ultralytics/Ultralytics \
    && chown -R oncoedge:oncoedge /app/models /home/oncoedge

COPY --chown=oncoedge:oncoedge src /app/src
COPY --chown=oncoedge:oncoedge streamlit_app.py 027.jpeg photo.webp /app/
COPY --chown=oncoedge:oncoedge deploy/warm_models.py /app/deploy/warm_models.py

USER oncoedge
EXPOSE 8503
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8503/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8503", "--server.headless=true", "--server.maxUploadSize=20", "--browser.gatherUsageStats=false"]
