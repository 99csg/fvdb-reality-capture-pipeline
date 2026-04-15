#!/usr/bin/env bash
# =============================================================================
# run_docker.sh  –  Docker로 fVDB Reality Capture 파이프라인 실행
#
# 사용법:
#   bash run_docker.sh <input_video.mp4> [output_dir] [fps]
#
# 인수:
#   input_video   : 입력 비디오 파일 경로 (필수)
#   output_dir    : 결과물 저장 디렉토리 (기본값: ./output)
#   fps           : 프레임 추출 FPS (기본값: 2)
#
# 환경 변수:
#   IMAGE_NAME          : 이미지 이름 (기본값: fvdb-reality-capture)
#   COLMAP_MAX_FRAMES   : COLMAP 최대 프레임 수 (기본값: 400)
#   SKIP_BUILD          : 1로 설정하면 이미지 빌드 건너뜀
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-fvdb-reality-capture}"

INPUT_VIDEO="${1:-}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/output}"
EXTRACT_FPS="${3:-2}"

# ── 입력 검증 ─────────────────────────────────────────────────────────────────
if [ -z "$INPUT_VIDEO" ]; then
    echo "사용법: $0 <input_video.mp4> [output_dir] [fps]"
    echo "  예시: $0 my_scene.mp4"
    echo "  예시: $0 my_scene.mp4 ./results 3"
    exit 1
fi

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "오류: 비디오 파일이 없습니다: $INPUT_VIDEO"
    exit 1
fi

# 절대 경로 변환
INPUT_ABS="$(realpath "$INPUT_VIDEO")"
OUTPUT_ABS="$(realpath -m "$OUTPUT_DIR")"
INPUT_FILENAME="$(basename "$INPUT_ABS")"

mkdir -p "$OUTPUT_ABS"

# ── Docker 이미지 빌드 ────────────────────────────────────────────────────────
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    echo "============================================================"
    echo " Docker 이미지 빌드: $IMAGE_NAME"
    echo "============================================================"
    docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
    echo ""
fi

# ── 파이프라인 실행 ───────────────────────────────────────────────────────────
echo "============================================================"
echo " 파이프라인 실행 (Docker)"
echo "  이미지      : $IMAGE_NAME"
echo "  입력 비디오 : $INPUT_ABS"
echo "  출력 디렉토리: $OUTPUT_ABS"
echo "  FPS         : $EXTRACT_FPS"
echo "============================================================"

docker run --rm \
    --gpus all \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e COLMAP_MAX_FRAMES="${COLMAP_MAX_FRAMES:-400}" \
    -v "$INPUT_ABS":/input/"$INPUT_FILENAME":ro \
    -v "$OUTPUT_ABS":/workspace/output \
    "$IMAGE_NAME" \
    /input/"$INPUT_FILENAME" \
    /workspace/output \
    "$EXTRACT_FPS"
