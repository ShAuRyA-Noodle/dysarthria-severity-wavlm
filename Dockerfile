# Reproducibility container for the ATD project (T9).
#
# Two build modes via the WITH_TORCH build-arg:
#
#   docker build -t atd:repro .                     # full image (default)
#       Installs the complete stack incl. torch/torchaudio, so SSL
#       extraction (experiments/04_extract_ssl.py) works inside the image.
#
#   docker build --build-arg WITH_TORCH=0 -t atd:analysis .
#       Lean analysis-only image: installs requirements-core.txt with
#       torch/torchaudio filtered out. Use when you mount pre-extracted
#       embeddings from Zenodo / HF Hub and only need to recompute ATD,
#       ablations, stats and figures (reproduce_all.py --skip-extract).
#
# Pre-extracted embeddings + reference manifests are expected to be mounted
# at /workspace/data (downloaded from Zenodo / HF Hub at run-time).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# System libs: libsndfile for soundfile, fontconfig for matplotlib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 fontconfig git build-essential \
    && rm -rf /var/lib/apt/lists/*

# WITH_TORCH=1 (default): install the full stack (core+llm+demo, incl. torch).
# WITH_TORCH=0: install core only, with torch/torchaudio stripped out for a
# smaller analysis-only image.
ARG WITH_TORCH=1

# Copy the (small) requirements files first for layer caching. The
# .dockerignore keeps data/, embeddings/ and .env out of the build context.
COPY requirements*.txt /workspace/
RUN if [ "$WITH_TORCH" = "0" ]; then \
        grep -vE '^(torch|torchaudio)==' requirements-core.txt > /tmp/req-core-notorch.txt && \
        pip install --no-cache-dir -r /tmp/req-core-notorch.txt ; \
    else \
        pip install --no-cache-dir -r requirements.txt ; \
    fi

# Copy the repo (data + embeddings + .env excluded via .dockerignore).
COPY . /workspace

# Default to reproducing the analysis stack with cached embeddings.
ENV PYTHONPATH=/workspace
CMD ["python", "reproduce_all.py", "--smoke"]
