# =============================================================================
# Dockerfile  –  fVDB Reality Capture 파이프라인
#
# 빌드:   docker build -t fvdb-reality-capture .
# 실행:   bash run_docker.sh <input_video.mp4> [output_dir] [fps]
# =============================================================================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV IN_DOCKER=1
ENV PYTHONUNBUFFERED=1

# ── 시스템 패키지 설치 ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    ffmpeg \
    colmap \
    git \
    curl \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# python3 → python3.11 링크
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip --quiet

WORKDIR /workspace

# ── PyTorch 2.10.0 + CUDA 12.8 ───────────────────────────────────────────────
RUN pip install \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --quiet

# ── fvdb-core 0.4.2 (pt210 / cu128) ─────────────────────────────────────────
RUN pip install \
    "fvdb-core==0.4.2+pt210.cu128" \
    --extra-index-url https://d36m13axqqhiit.cloudfront.net/simple \
    --quiet

# ── fvdb-reality-capture 소스 설치 ───────────────────────────────────────────
COPY fvdb-reality-capture /workspace/fvdb-reality-capture
RUN pip install \
    --extra-index-url https://d36m13axqqhiit.cloudfront.net/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    /workspace/fvdb-reality-capture \
    --quiet

# ── SAM2 (객체 제거 모듈) ─────────────────────────────────────────────────────
RUN pip install sam2 --quiet

# ── 파이프라인 스크립트 복사 ──────────────────────────────────────────────────
COPY pipeline.sh run_colmap.py mesh_to_usdz.py remove_objects.py download_sam2_checkpoint.sh /workspace/

# SAM2 체크포인트 디렉토리 (호스트에서 마운트하거나 빌드 후 직접 다운로드)
RUN mkdir -p /workspace/checkpoints

# 출력 마운트 포인트
RUN mkdir -p /workspace/output

ENTRYPOINT ["bash", "/workspace/pipeline.sh"]
