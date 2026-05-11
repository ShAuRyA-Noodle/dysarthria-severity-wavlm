# Reproducibility container for the ATD project (T9).
# Builds the analysis-only stack (no SSL extraction). Pre-extracted
# embeddings + reference manifests are expected to be mounted at
# /workspace/data, downloaded from Zenodo / HF Hub at run-time.
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

# Project requirements. We pin the analysis subset; torch/torchaudio are
# heavy and installed via wheel if the user wants the full extraction
# path (override CMD or rebuild with --build-arg WITH_TORCH=1).
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the repo (skip data + embeddings via .dockerignore).
COPY . /workspace
RUN pip install --no-cache-dir praatio pydantic

# Default to reproducing the analysis stack with cached embeddings.
ENV PYTHONPATH=/workspace
CMD ["python", "reproduce_all.py", "--smoke"]
