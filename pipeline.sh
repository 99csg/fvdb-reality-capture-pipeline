#!/usr/bin/env bash
# =============================================================================
# pipeline.sh  –  비디오 → USDZ 메쉬 완전 자동화 파이프라인
#
# 사용법:
#   bash pipeline.sh <input_video.mp4> [output_dir] [fps]
#
# 인수:
#   input_video   : 입력 비디오 파일 경로 (필수)
#   output_dir    : 출력 디렉토리 (기본값: ./output/<비디오이름>_<timestamp>)
#   fps           : 프레임 추출 FPS (기본값: 2, 더 촘촘한 캡처는 3-5 권장)
#
# 파이프라인 단계:
#   1.  프레임 추출              (ffmpeg)
#   1.5 객체 제거 [선택 사항]    (SAM2 + OpenCV inpainting)
#   2.  SfM 재구성               (COLMAP)
#   3.  Gaussian Splat 학습      (frgs reconstruct)
#   4.  삼각 메쉬 추출           (frgs mesh-basic)
#   5.  USDZ 변환                (mesh_to_usdz.py)
#
# 객체 제거 활성화:
#   REMOVE_OBJECTS=1 REMOVE_POINTS="320,240" REMOVE_LABELS="1" \
#       bash pipeline.sh video.mp4
#
#   환경 변수:
#     REMOVE_OBJECTS    : 1 이면 활성화 (기본: 0)
#     REMOVE_POINTS     : 포인트 좌표 (예: "320,240" 또는 "320,240 410,310")
#     REMOVE_LABELS     : 포인트 레이블 (예: "1" 또는 "1 0")
#     SAM2_CHECKPOINT   : SAM2 체크포인트 경로 (기본: ./checkpoints/sam2.1_hiera_large.pt)
#     SAM2_MODEL_CFG    : SAM2 모델 설정 (기본: configs/sam2.1/sam2.1_hiera_large.yaml)
#
# 전제조건:
#   - bash setup.sh 를 먼저 실행해 가상환경(./venv)을 구성하세요.
#   - ffmpeg, colmap 이 시스템에 설치되어 있어야 합니다.
#   - 객체 제거 사용 시: bash download_sam2_checkpoint.sh 로 체크포인트를 받으세요.
# =============================================================================
set -euo pipefail

# ── 인수 파싱 ─────────────────────────────────────────────────────────────────
INPUT_VIDEO="${1:-}"
OUTPUT_DIR="${2:-}"
EXTRACT_FPS="${3:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
REPO_DIR="$SCRIPT_DIR/fvdb-reality-capture"
MESH_TO_USDZ="$SCRIPT_DIR/mesh_to_usdz.py"
REMOVE_OBJECTS_SCRIPT="$SCRIPT_DIR/remove_objects.py"

# ── 객체 제거 설정 (환경 변수로 제어) ────────────────────────────────────────
REMOVE_OBJECTS="${REMOVE_OBJECTS:-0}"
REMOVE_POINTS="${REMOVE_POINTS:-}"
REMOVE_LABELS="${REMOVE_LABELS:-1}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-$SCRIPT_DIR/checkpoints/sam2.1_hiera_large.pt}"
SAM2_MODEL_CFG="${SAM2_MODEL_CFG:-configs/sam2.1/sam2.1_hiera_large.yaml}"

# ── 입력 검증 ─────────────────────────────────────────────────────────────────
if [ -z "$INPUT_VIDEO" ]; then
    echo "사용법: $0 <input_video.mp4> [output_dir] [fps]"
    echo "  예시: $0 my_scene.mp4"
    echo "  예시: $0 my_scene.mp4 ./output/my_scene 3"
    exit 1
fi

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "오류: 비디오 파일이 없습니다: $INPUT_VIDEO"
    exit 1
fi

# 가상환경 활성화 (Docker 환경에서는 시스템 Python 사용)
if [ -z "${IN_DOCKER:-}" ]; then
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo "오류: 가상환경이 없습니다: $VENV_DIR"
        echo "먼저 'bash setup.sh' 를 실행하세요."
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

# ── 출력 디렉토리 설정 ────────────────────────────────────────────────────────
VIDEO_BASENAME="$(basename "${INPUT_VIDEO%.*}")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$SCRIPT_DIR/output/${VIDEO_BASENAME}_${TIMESTAMP}"
fi
mkdir -p "$OUTPUT_DIR"

# 각 단계 디렉토리
FRAMES_DIR="$OUTPUT_DIR/images_raw"        # 원본 추출 프레임
CLEAN_DIR="$OUTPUT_DIR/images_clean"       # 객체 제거 후 프레임 (사용 시)
COLMAP_DIR="$OUTPUT_DIR"                    # COLMAP workspace
SPLAT_PATH="$OUTPUT_DIR/splat.ply"         # Gaussian splat (PLY 포맷)
MESH_PATH="$OUTPUT_DIR/mesh.ply"           # 추출된 메쉬
USDZ_PATH="$OUTPUT_DIR/output.usdz"        # 최종 USDZ 파일
LOG_PATH="$OUTPUT_DIR/pipeline.log"        # 파이프라인 로그
COLMAP_SCRIPT="$SCRIPT_DIR/run_colmap.py"  # COLMAP 파이프라인 스크립트

# ── 로깅 설정 ─────────────────────────────────────────────────────────────────
exec > >(tee -a "$LOG_PATH") 2>&1

log() { echo "[$(date '+%H:%M:%S')] $*"; }
step() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo " $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

echo ""
echo "============================================================"
echo " fVDB Reality Capture 파이프라인"
echo "============================================================"
echo " 입력 비디오  : $INPUT_VIDEO"
echo " 출력 디렉토리: $OUTPUT_DIR"
echo " 추출 FPS     : $EXTRACT_FPS"
if [ "$REMOVE_OBJECTS" -eq 1 ]; then
echo " 객체 제거    : 활성화 (포인트: $REMOVE_POINTS / 레이블: $REMOVE_LABELS)"
else
echo " 객체 제거    : 비활성화 (REMOVE_OBJECTS=1 로 활성화)"
fi
echo " 로그 파일    : $LOG_PATH"
echo "============================================================"

# ────────────────────────────────────────────────────────────────────────────
# STEP 1: 프레임 추출
# ────────────────────────────────────────────────────────────────────────────
step "STEP 1/5: 프레임 추출 (ffmpeg, ${EXTRACT_FPS} fps)"

mkdir -p "$FRAMES_DIR"
FRAME_COUNT=$(find "$FRAMES_DIR" -maxdepth 1 -name "*.jpg" | wc -l)

if [ "$FRAME_COUNT" -gt 10 ]; then
    log "이미 추출된 프레임 ${FRAME_COUNT}개 발견 – 건너뜁니다."
else
    log "비디오에서 프레임 추출 중..."
    ffmpeg -i "$INPUT_VIDEO" \
        -vf "fps=${EXTRACT_FPS}" \
        -q:v 2 \
        "$FRAMES_DIR/frame_%05d.jpg" \
        -hide_banner -loglevel warning

    FRAME_COUNT=$(find "$FRAMES_DIR" -maxdepth 1 -name "*.jpg" | wc -l)
    log "✓ 프레임 ${FRAME_COUNT}개 추출 완료"
fi

if [ "$FRAME_COUNT" -lt 20 ]; then
    echo "경고: 프레임 수가 적습니다 ($FRAME_COUNT 개). FPS를 높이거나 더 긴 비디오를 사용하세요."
fi

# COLMAP 입력 프레임 서브샘플링 (너무 많으면 자동으로 줄임)
COLMAP_MAX_FRAMES="${COLMAP_MAX_FRAMES:-400}"
if [ "$FRAME_COUNT" -gt "$COLMAP_MAX_FRAMES" ]; then
    log "프레임 ${FRAME_COUNT}개 → COLMAP용 ${COLMAP_MAX_FRAMES}개로 서브샘플링..."
    STEP=$((FRAME_COUNT / COLMAP_MAX_FRAMES))
    IDX=0
    for f in $(find "$FRAMES_DIR" -maxdepth 1 -name "*.jpg" | sort); do
        if [ $((IDX % STEP)) -ne 0 ]; then
            rm -f "$f"
        fi
        IDX=$((IDX + 1))
    done
    FRAME_COUNT=$(find "$FRAMES_DIR" -maxdepth 1 -name "*.jpg" | wc -l)
    log "✓ 서브샘플링 후 프레임: ${FRAME_COUNT}개"
fi

# ────────────────────────────────────────────────────────────────────────────
# STEP 1.5: SAM2 객체 제거 (선택 사항)
# ────────────────────────────────────────────────────────────────────────────
# COLMAP이 사용할 이미지 경로 – 기본은 images_raw/, 객체 제거 시 images_clean/
COLMAP_IMAGE_PATH="$FRAMES_DIR"

if [ "${REMOVE_OBJECTS:-0}" -eq 1 ]; then
    step "STEP 1.5: SAM2 객체 제거"

    if [ -z "$REMOVE_POINTS" ]; then
        echo "오류: REMOVE_OBJECTS=1 이지만 REMOVE_POINTS 가 설정되지 않았습니다."
        echo "  예시: REMOVE_POINTS=\"320,240\" REMOVE_LABELS=\"1\" bash pipeline.sh ..."
        exit 1
    fi

    if [ ! -f "$SAM2_CHECKPOINT" ]; then
        echo "오류: SAM2 체크포인트를 찾을 수 없습니다: $SAM2_CHECKPOINT"
        echo "  다운로드: bash download_sam2_checkpoint.sh"
        exit 1
    fi

    CLEAN_DONE_MARKER="$OUTPUT_DIR/.remove_objects_done"
    if [ -f "$CLEAN_DONE_MARKER" ] && [ -d "$CLEAN_DIR" ]; then
        log "이미 완료된 객체 제거 결과 발견 – 건너뜁니다."
    else
        log "SAM2 마스크 생성 및 인페인팅 중..."
        log "  포인트  : $REMOVE_POINTS"
        log "  레이블  : $REMOVE_LABELS"
        log "  체크포인트: $SAM2_CHECKPOINT"

        python3 "$REMOVE_OBJECTS_SCRIPT" \
            --frames_dir "$FRAMES_DIR" \
            --out_dir    "$CLEAN_DIR" \
            --points     "$REMOVE_POINTS" \
            --labels     "$REMOVE_LABELS" \
            --checkpoint "$SAM2_CHECKPOINT" \
            --model_cfg  "$SAM2_MODEL_CFG"

        touch "$CLEAN_DONE_MARKER"
        log "✓ 객체 제거 완료: $CLEAN_DIR"
    fi

    COLMAP_IMAGE_PATH="$CLEAN_DIR"
fi

# ────────────────────────────────────────────────────────────────────────────
# STEP 2: COLMAP 구조-운동 복원 (SfM)
# ────────────────────────────────────────────────────────────────────────────
step "STEP 2/5: COLMAP SfM 재구성"

# COLMAP 결과가 이미 있으면 건너뜀
COLMAP_DONE_MARKER="$COLMAP_DIR/.colmap_done"
if [ -f "$COLMAP_DONE_MARKER" ]; then
    log "이미 완료된 COLMAP 결과 발견 – 건너뜁니다."
else
    log "COLMAP feature extraction & matching & mapping 실행 중..."
    log "(이 단계는 프레임 수에 따라 수십 분 ~ 수 시간이 걸릴 수 있습니다)"
    log "  이미지 경로: $COLMAP_IMAGE_PATH"

    python3 "$COLMAP_SCRIPT" \
        --source_path "$COLMAP_DIR" \
        --image_path  "$COLMAP_IMAGE_PATH" \
        --no_gpu \
        --camera OPENCV \
        --max_sift_features 8192 \
        --sift_threads 8

    touch "$COLMAP_DONE_MARKER"
    log "✓ COLMAP 완료"
fi

# COLMAP 결과 검증
if [ ! -d "$COLMAP_DIR/sparse/0" ]; then
    echo "오류: COLMAP sparse 재구성 결과가 없습니다: $COLMAP_DIR/sparse/0"
    echo "프레임이 너무 적거나 장면이 COLMAP으로 재구성하기 어려울 수 있습니다."
    exit 1
fi

# ────────────────────────────────────────────────────────────────────────────
# STEP 3: Gaussian Splat 학습
# ────────────────────────────────────────────────────────────────────────────
step "STEP 3/5: Gaussian Splat 학습 (frgs reconstruct)"

if [ -f "$SPLAT_PATH" ]; then
    log "이미 학습된 splat 체크포인트 발견: $SPLAT_PATH – 건너뜁니다."
else
    log "Gaussian Splat 학습 중..."
    log "(RTX PRO 5000 24GB 기준 약 10-30분 소요)"

    frgs reconstruct "$COLMAP_DIR" \
        --out-path "$SPLAT_PATH" \
        --dataset-type colmap

    log "✓ Gaussian Splat 학습 완료: $SPLAT_PATH"
fi

# ────────────────────────────────────────────────────────────────────────────
# STEP 4: 삼각 메쉬 추출
# ────────────────────────────────────────────────────────────────────────────
step "STEP 4/5: 삼각 메쉬 추출 (frgs mesh-basic)"

if [ -f "$MESH_PATH" ]; then
    log "이미 추출된 메쉬 발견: $MESH_PATH – 건너뜁니다."
else
    log "TSDF 기반 메쉬 추출 중..."

    frgs mesh-basic "$SPLAT_PATH" 0.05 \
        --output-path "$MESH_PATH" \
        --image-downsample-factor 2

    log "✓ 메쉬 추출 완료: $MESH_PATH"
fi

# ────────────────────────────────────────────────────────────────────────────
# STEP 5: USDZ 변환
# ────────────────────────────────────────────────────────────────────────────
step "STEP 5/5: USDZ 변환"

if [ -f "$USDZ_PATH" ]; then
    log "이미 변환된 USDZ 발견: $USDZ_PATH – 건너뜁니다."
else
    log "PLY 메쉬 → USDZ 변환 중..."
    python3 "$MESH_TO_USDZ" "$MESH_PATH" "$USDZ_PATH"
    log "✓ USDZ 변환 완료: $USDZ_PATH"
fi

# ────────────────────────────────────────────────────────────────────────────
# 완료
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " 파이프라인 완료!"
echo "============================================================"
echo " 출력 파일:"
echo "   메쉬 (PLY) : $MESH_PATH"
echo "   USDZ       : $USDZ_PATH"
echo ""
echo " 파일 크기:"
if [ -f "$MESH_PATH" ]; then
    du -sh "$MESH_PATH" | awk '{print "   메쉬: " $1}'
fi
if [ -f "$USDZ_PATH" ]; then
    du -sh "$USDZ_PATH" | awk '{print "   USDZ: " $1}'
fi
echo "============================================================"
