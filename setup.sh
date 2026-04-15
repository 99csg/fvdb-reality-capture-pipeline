#!/usr/bin/env bash
# =============================================================================
# setup.sh  –  fVDB Reality Capture 환경 설치 스크립트
#
# 사용법: bash setup.sh
#
# 이 스크립트는 다음을 수행합니다:
#   1. Python 가상환경 생성 (./venv)
#   2. PyTorch 2.10.0 + CUDA 12.8 설치
#   3. fvdb-core 0.4.2 (pt210 / cu128) 설치
#   4. fvdb-reality-capture 소스 설치
#   5. 기타 의존성 (ffmpeg, point-cloud-utils 등)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/fvdb-reality-capture"
VENV_DIR="$SCRIPT_DIR/venv"

echo "============================================================"
echo " fVDB Reality Capture 환경 설치"
echo " 작업 디렉토리: $SCRIPT_DIR"
echo "============================================================"

# --- 1. 필수 시스템 패키지 확인 ---
echo ""
echo "[1/5] 시스템 의존성 확인..."
for cmd in ffmpeg colmap python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "  오류: '$cmd' 를 찾을 수 없습니다."
        if [ "$cmd" = "ffmpeg" ]; then
            echo "  설치: sudo apt-get install -y ffmpeg"
        elif [ "$cmd" = "colmap" ]; then
            echo "  설치: sudo apt-get install -y colmap"
        fi
        exit 1
    fi
    echo "  ✓ $cmd: $(command -v $cmd)"
done

# --- 2. 가상환경 생성 ---
echo ""
echo "[2/5] Python 가상환경 생성: $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
    echo "  이미 존재합니다. 재사용합니다."
else
    python3 -m venv "$VENV_DIR"
    echo "  ✓ 가상환경 생성 완료"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# --- 2b. fvdb-reality-capture를 v0.4.0 으로 고정 (main 브랜치는 미출시된 fvdb-core 필요) ---
if [ -d "$REPO_DIR" ]; then
    pushd "$REPO_DIR" >/dev/null
    CURRENT_TAG=$(git describe --exact-match --tags HEAD 2>/dev/null || true)
    if [ "$CURRENT_TAG" != "v0.4.0" ]; then
        echo "  fvdb-reality-capture 를 v0.4.0 으로 전환합니다..."
        git checkout v0.4.0 2>&1 | head -3
    else
        echo "  ✓ fvdb-reality-capture 이미 v0.4.0"
    fi
    popd >/dev/null
fi

# --- 3. PyTorch 2.10.0 + CUDA 12.8 설치 ---
echo ""
echo "[3/5] PyTorch 2.10.0 (CUDA 12.8) 설치..."
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

INSTALLED_TORCH=$(pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}' || true)
if [[ "$INSTALLED_TORCH" == "2.10.0+cu128" ]]; then
    echo "  ✓ PyTorch 2.10.0+cu128 이미 설치됨"
else
    pip install \
        torch==2.10.0+cu128 \
        torchvision==0.25.0+cu128 \
        --extra-index-url "$TORCH_INDEX" \
        --quiet
    echo "  ✓ PyTorch 2.10.0+cu128 설치 완료"
fi

# --- 4. fvdb-core 설치 ---
echo ""
echo "[4/5] fvdb-core 0.4.2 (pt210/cu128) 설치..."
FVDB_INDEX="https://d36m13axqqhiit.cloudfront.net/simple"

INSTALLED_FVDB=$(pip show fvdb-core 2>/dev/null | grep "^Version:" | awk '{print $2}' || true)
if [[ "$INSTALLED_FVDB" == "0.4.2+pt210.cu128" ]]; then
    echo "  ✓ fvdb-core 0.4.2+pt210.cu128 이미 설치됨"
else
    pip install \
        "fvdb-core==0.4.2+pt210.cu128" \
        --extra-index-url "$FVDB_INDEX" \
        --quiet
    echo "  ✓ fvdb-core 설치 완료"
fi

# --- 5. fvdb-reality-capture 및 나머지 의존성 설치 ---
echo ""
echo "[5/5] fvdb-reality-capture 소스 설치..."
if [ ! -d "$REPO_DIR" ]; then
    echo "  오류: fvdb-reality-capture 소스가 없습니다: $REPO_DIR"
    echo "  먼저 'git clone https://github.com/openvdb/fvdb-reality-capture.git' 를 실행하세요."
    exit 1
fi

pip install \
    --extra-index-url "$FVDB_INDEX" \
    --extra-index-url "$TORCH_INDEX" \
    -e "$REPO_DIR" \
    --quiet

echo "  ✓ fvdb-reality-capture 설치 완료"

# --- 설치 검증 ---
echo ""
echo "============================================================"
echo " 설치 검증"
echo "============================================================"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'  CUDA 사용 가능: {torch.cuda.is_available()}')"
python3 -c "import fvdb; print(f'  fvdb: OK')"
python3 -c "from pxr import Usd; print(f'  USD (usd-core): OK')"
python3 -c "import fvdb_reality_capture; print(f'  fvdb-reality-capture: OK')"
echo "  frgs 실행 경로: $(which frgs 2>/dev/null || echo '없음')"

echo ""
echo "============================================================"
echo " 설치 완료!"
echo ""
echo " 파이프라인 실행:"
echo "   source venv/bin/activate"
echo "   bash pipeline.sh <입력_비디오.mp4> [출력_디렉토리]"
echo "============================================================"
