#!/usr/bin/env bash
# =============================================================================
# download_sam2_checkpoint.sh  –  SAM2.1 체크포인트 다운로드
#
# 사용법:
#   bash download_sam2_checkpoint.sh [모델크기]
#
# 모델크기 옵션:
#   large     (기본) – 896MB, 가장 정확
#   base_plus          – 311MB
#   small              – 184MB, 빠름
#   tiny               – 155MB, 가장 빠름
#
# 예시:
#   bash download_sam2_checkpoint.sh           # large 다운로드
#   bash download_sam2_checkpoint.sh small     # small 다운로드
# =============================================================================
set -euo pipefail

MODEL="${1:-large}"
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

case "$MODEL" in
    large)     CKPT="sam2.1_hiera_large.pt" ;;
    base_plus) CKPT="sam2.1_hiera_base_plus.pt" ;;
    small)     CKPT="sam2.1_hiera_small.pt" ;;
    tiny)      CKPT="sam2.1_hiera_tiny.pt" ;;
    *)
        echo "오류: 알 수 없는 모델 크기: $MODEL"
        echo "사용 가능: large (기본) | base_plus | small | tiny"
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="$SCRIPT_DIR/checkpoints"
CKPT_PATH="$CKPT_DIR/$CKPT"

mkdir -p "$CKPT_DIR"

if [ -f "$CKPT_PATH" ]; then
    echo "✓ 이미 존재합니다: $CKPT_PATH"
    exit 0
fi

echo "SAM2.1 체크포인트 다운로드 중..."
echo "  모델  : $MODEL ($CKPT)"
echo "  저장  : $CKPT_PATH"
echo ""

wget -c --show-progress -P "$CKPT_DIR" "$BASE_URL/$CKPT"

echo ""
echo "✓ 다운로드 완료: $CKPT_PATH"
echo ""
echo "사용 방법:"
echo "  python remove_objects.py \\"
echo "      --frames_dir ./output/images_raw \\"
echo "      --out_dir    ./output/images_clean \\"
echo "      --interactive"
