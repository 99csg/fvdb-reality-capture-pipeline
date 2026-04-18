#!/usr/bin/env python3
"""
remove_objects.py – SAM2 기반 객체 제거 모듈

첫 프레임에서 클릭(또는 좌표 지정)으로 물체를 선택하면,
SAM2 video predictor가 모든 프레임에서 마스크를 추적하고
OpenCV inpainting으로 해당 영역을 배경으로 채웁니다.

필요 패키지:
    pip install sam2 opencv-python-headless numpy torch
    pip install matplotlib  # 대화형 모드에서만 필요

SAM2 체크포인트 다운로드:
    bash download_sam2_checkpoint.sh          # large 모델 (기본)
    bash download_sam2_checkpoint.sh small    # 소형 모델 (빠름)

사용법:
  # 대화형 모드: matplotlib 창에서 좌클릭으로 제거할 물체 선택, Enter로 확정
  python remove_objects.py \\
      --frames_dir ./output/images_raw \\
      --out_dir    ./output/images_clean \\
      --interactive

  # 비대화형 모드: 좌표 직접 지정
  python remove_objects.py \\
      --frames_dir ./output/images_raw \\
      --out_dir    ./output/images_clean \\
      --points "320,240" --labels "1"

  # 여러 포인트: 포함(1) / 배경 힌트(0)
  python remove_objects.py \\
      --frames_dir ./output/images_raw \\
      --out_dir    ./output/images_clean \\
      --points "320,240 410,310" --labels "1 0"

파이프라인 통합:
  pipeline.sh 에서 STEP 1 (프레임 추출) 이후 자동 실행됩니다.
  환경 변수로 제어:
      REMOVE_OBJECTS=1               # 활성화
      REMOVE_POINTS="320,240"        # 포인트 좌표
      REMOVE_LABELS="1"              # 레이블
      SAM2_CHECKPOINT=<경로>         # 체크포인트 경로 (선택)
      SAM2_MODEL_CFG=<설정파일명>    # 모델 설정 (선택)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# SAM2.1 hiera_large – 기본 체크포인트 및 모델 설정
_SCRIPT_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT = _SCRIPT_DIR / "checkpoints" / "sam2.1_hiera_large.pt"
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_large.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def collect_frames(frames_dir: Path) -> list[Path]:
    """디렉토리에서 프레임 파일을 정렬된 순서로 수집."""
    frames: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        frames.extend(frames_dir.glob(ext))
    return sorted(frames)


# ─────────────────────────────────────────────────────────────────────────────
# 대화형 포인트 선택
# ─────────────────────────────────────────────────────────────────────────────

def interactive_point_selection(frame_path: Path) -> tuple[list[list[int]], list[int]]:
    """
    첫 프레임을 matplotlib 창으로 보여 주고 포인트를 클릭으로 선택합니다.

    좌클릭  → 포함 포인트 (녹색 별)  – 제거할 물체 위에 클릭
    우클릭  → 배경 힌트 포인트 (빨강 별) – 보존할 배경 위에 클릭
    Enter 키 → 선택 완료
    """
    try:
        import matplotlib
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "대화형 모드에는 matplotlib 이 필요합니다: pip install matplotlib"
        )

    # DISPLAY 환경 변수 확인 (헤드리스 환경 경고)
    import os
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        logger.warning(
            "DISPLAY 환경 변수가 없습니다. 헤드리스 환경에서는 --points/--labels 를 사용하세요."
        )

    img_bgr = cv2.imread(str(frame_path))
    if img_bgr is None:
        raise FileNotFoundError(f"프레임을 읽을 수 없습니다: {frame_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    points: list[list[int]] = []
    labels: list[int] = []

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img_rgb)
    ax.set_title(
        "좌클릭: 제거할 물체 선택 (녹색)  |  우클릭: 배경 힌트 (빨강)  |  Enter: 완료",
        fontsize=12,
    )

    def _onclick(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        if event.button == 1:  # 좌클릭 – 포함
            points.append([x, y])
            labels.append(1)
            ax.plot(x, y, "g*", markersize=18, zorder=5)
        elif event.button == 3:  # 우클릭 – 배경 힌트
            points.append([x, y])
            labels.append(0)
            ax.plot(x, y, "r*", markersize=18, zorder=5)
        fig.canvas.draw_idle()

    def _onkey(event):
        if event.key in ("enter", "return"):
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _onclick)
    fig.canvas.mpl_connect("key_press_event", _onkey)

    legend_handles = [
        mpatches.Patch(color="green", label="포함 – 제거할 물체 (좌클릭)"),
        mpatches.Patch(color="red",   label="배경 힌트 – 보존 (우클릭)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.show()

    return points, labels


# ─────────────────────────────────────────────────────────────────────────────
# SAM2 추론
# ─────────────────────────────────────────────────────────────────────────────

def _build_predictor(checkpoint: Path, model_cfg: str, device: str):
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        raise ImportError(
            "SAM2 패키지가 설치되지 않았습니다: pip install sam2\n"
            "또는: pip install git+https://github.com/facebookresearch/sam2.git"
        )
    return build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)


def generate_masks(
    predictor,
    frames_dir: Path,
    points: list[list[int]],
    labels: list[int],
) -> dict[int, np.ndarray]:
    """
    SAM2 video predictor로 모든 프레임의 이진 마스크를 생성합니다.

    반환값: {frame_index: uint8 마스크 (0 또는 255, shape [H, W])}
    """
    pts_np = np.array(points, dtype=np.float32)   # [N, 2]
    lbl_np = np.array(labels, dtype=np.int32)      # [N]

    masks: dict[int, np.ndarray] = {}

    with torch.inference_mode():
        state = predictor.init_state(video_path=str(frames_dir))
        predictor.reset_state(state)

        # 0번 프레임에 포인트 프롬프트 등록
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=pts_np,
            labels=lbl_np,
        )

        # 전체 비디오에 마스크 전파
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            # mask_logits shape: [num_objects, 1, H, W]
            binary = (mask_logits[0] > 0.0).squeeze(0).cpu().numpy()  # [H, W] bool
            masks[frame_idx] = (binary * 255).astype(np.uint8)

    return masks


# ─────────────────────────────────────────────────────────────────────────────
# 인페인팅
# ─────────────────────────────────────────────────────────────────────────────

def _inpaint(img_bgr: np.ndarray, mask: np.ndarray, dilation_px: int) -> np.ndarray:
    """OpenCV TELEA 인페인팅으로 마스크 영역을 배경으로 채웁니다."""
    if dilation_px > 0:
        r = dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        mask = cv2.dilate(mask, kernel)
    return cv2.inpaint(img_bgr, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

def remove_objects(
    frames_dir: Path,
    out_dir: Path,
    points: list[list[int]],
    labels: list[int],
    checkpoint: Path,
    model_cfg: str,
    device: str,
    dilation_px: int,
) -> None:
    """SAM2로 마스크를 생성하고 모든 프레임에서 물체를 인페인팅합니다."""
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = collect_frames(frames_dir)
    if not frames:
        raise ValueError(f"프레임을 찾을 수 없습니다: {frames_dir}")

    logger.info(f"총 {len(frames)}개 프레임 | 포인트: {points} | 레이블: {labels}")
    logger.info(f"장치: {device} | 마스크 팽창: {dilation_px}px")

    logger.info("SAM2 모델 로드 중...")
    predictor = _build_predictor(checkpoint, model_cfg, device)

    logger.info("SAM2 마스크 생성 및 전파 중...")
    masks = generate_masks(predictor, frames_dir, points, labels)
    logger.info(f"  마스크 생성 완료: {len(masks)}개 프레임")

    logger.info("프레임 인페인팅 중...")
    removed_count = 0
    for idx, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None:
            logger.warning(f"읽기 실패: {frame_path}, 건너뜁니다.")
            continue

        mask = masks.get(idx)
        if mask is not None and mask.max() > 0:
            result = _inpaint(img, mask, dilation_px)
            removed_count += 1
        else:
            result = img  # 마스크 없으면 원본 그대로

        cv2.imwrite(str(out_dir / frame_path.name), result)

        if (idx + 1) % 50 == 0 or idx == len(frames) - 1:
            logger.info(f"  {idx + 1}/{len(frames)} 완료 (인페인팅 적용: {removed_count}개)")

    logger.info(f"✓ 완료 → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_points(s: str) -> list[list[int]]:
    """'320,240 400,300' → [[320,240],[400,300]]"""
    return [[int(v) for v in pt.split(",")] for pt in s.strip().split()]


def _parse_labels(s: str) -> list[int]:
    """'1 0 1' → [1, 0, 1]"""
    return [int(v) for v in s.strip().split()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM2 기반 객체 제거: 첫 프레임에서 선택한 물체를 모든 프레임에서 제거",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--frames_dir", type=Path, required=True,
        help="입력 프레임 디렉토리 (images_raw/)",
    )
    parser.add_argument(
        "--out_dir", type=Path, required=True,
        help="출력 프레임 디렉토리 (images_clean/)",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help=f"SAM2 체크포인트 경로 (기본: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--model_cfg", type=str, default=DEFAULT_MODEL_CFG,
        help=f"SAM2 모델 설정 파일 (기본: {DEFAULT_MODEL_CFG})",
    )
    parser.add_argument(
        "--points", type=str, default=None,
        help="포인트 좌표 예: '320,240 400,300' (공백 구분)",
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="포인트 레이블 예: '1 0' (1=포함, 0=배경 힌트)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="대화형 모드: matplotlib 창에서 클릭으로 포인트 선택",
    )
    parser.add_argument(
        "--dilation", type=int, default=5,
        help="마스크 팽창 픽셀 반경 (기본: 5). 경계 인페인팅 품질 향상",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="실행 장치 (cuda / cpu)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── 포인트 확보 ────────────────────────────────────────────────────────────
    if args.interactive:
        frames = collect_frames(args.frames_dir)
        if not frames:
            logger.error(f"프레임 없음: {args.frames_dir}")
            sys.exit(1)
        logger.info(f"대화형 선택 모드 – 첫 프레임: {frames[0].name}")
        points, labels = interactive_point_selection(frames[0])
        if not points:
            logger.error("포인트가 선택되지 않았습니다. 종료합니다.")
            sys.exit(1)
    elif args.points and args.labels:
        points = _parse_points(args.points)
        labels = _parse_labels(args.labels)
    else:
        parser.error("--interactive 또는 (--points + --labels) 중 하나를 지정해야 합니다.")

    # ── 체크포인트 확인 ────────────────────────────────────────────────────────
    if not args.checkpoint.exists():
        logger.error(f"SAM2 체크포인트를 찾을 수 없습니다: {args.checkpoint}")
        logger.error("다운로드 방법:")
        logger.error("  bash download_sam2_checkpoint.sh")
        sys.exit(1)

    remove_objects(
        frames_dir=args.frames_dir,
        out_dir=args.out_dir,
        points=points,
        labels=labels,
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
        dilation_px=args.dilation,
    )


if __name__ == "__main__":
    main()
