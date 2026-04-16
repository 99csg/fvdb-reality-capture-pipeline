#!/usr/bin/env python3
"""
render_train_views.py  –  학습 뷰를 Gaussian Splat PLY로 렌더링하여 이미지로 저장

노트북 (radiance_field_and_mesh_reconstruction.ipynb) 의 plot_reconstruction_results 패턴을 그대로 사용.

사용법:
    python render_train_views.py <colmap_dir> <splat.ply> [--out-dir <dir>] [--save-comparison]

예시:
    python render_train_views.py output/IMG_0017 output/IMG_0017/splat.ply
    python render_train_views.py output/IMG_0017 output/IMG_0017/splat.ply --save-comparison
"""

import argparse
import logging
import pathlib

import cv2
import fvdb
import fvdb_reality_capture as frc
import fvdb_reality_capture.transforms as fvt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def render_one_view(
    model: fvdb.GaussianSplat3d,
    image_meta: frc.sfm_scene.SfmPosedImageMetadata,
    near: float = 0.1,
    far: float = 10000.0,
):
    """노트북의 plot_reconstruction_results와 동일한 방식으로 한 뷰를 렌더링."""
    camera_meta: frc.sfm_scene.SfmCameraMetadata = image_meta.camera_metadata

    camera_to_world_matrix = torch.from_numpy(image_meta.camera_to_world_matrix).to(
        device=model.device, dtype=torch.float32
    )
    projection_matrix = torch.from_numpy(camera_meta.projection_matrix).to(
        device=model.device, dtype=torch.float32
    )
    image_height, image_width = image_meta.image_size

    rendered_rgbd, alphas = model.render_images_and_depths(
        world_to_camera_matrices=camera_to_world_matrix.inverse().unsqueeze(0).contiguous(),
        projection_matrices=projection_matrix.unsqueeze(0),
        image_width=image_width,
        image_height=image_height,
        near=near,
        far=far,
    )

    rgb = rendered_rgbd[0, ..., :3].cpu().numpy()
    depth_raw = rendered_rgbd[0, ..., 3].cpu().numpy()     # (H, W)
    alpha = alphas[0].squeeze().cpu().numpy()               # (H, W)

    rendered_image = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    # depth: alpha가 0인 픽셀 처리 (divide-by-zero 방지)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(alpha > 1e-6, depth_raw / alpha, 0.0)

    return rendered_image, depth


def load_gt_image(image_path: str) -> np.ndarray | None:
    """GT 이미지 로드 (BGR→RGB)."""
    gt = cv2.imread(image_path)
    if gt is None:
        return None
    return cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    """depth를 turbo colormap으로 변환하여 uint8 RGB로 반환 (cv2 사용)."""
    valid = depth[depth > 0]
    if len(valid) == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    d_min, d_max = valid.min(), valid.max()
    normalized = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0, 1)
    gray = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)  # BGR
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def make_comparison_image(
    gt: np.ndarray | None,
    rendered: np.ndarray,
    depth_color: np.ndarray,
    title: str,
) -> Image.Image:
    """GT / 렌더링 / Depth를 가로로 이어붙인 PIL 이미지 생성."""
    H, W = rendered.shape[:2]
    pad = 4
    label_h = 24
    n = 3
    canvas_w = W * n + pad * (n + 1)
    canvas_h = H + label_h + pad * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    panels = [
        ("Ground Truth", gt if gt is not None else np.zeros_like(rendered)),
        ("Rendered", rendered),
        ("Depth", depth_color),
    ]
    for idx, (label, img) in enumerate(panels):
        x = pad + idx * (W + pad)
        y = pad
        canvas.paste(Image.fromarray(img.astype(np.uint8)), (x, y))
        draw.text((x + 4, y + H + 4), label, fill=(200, 200, 200))

    draw.text((pad, 4), title, fill=(255, 255, 100))
    return canvas


def render_all_views(
    colmap_dir: pathlib.Path,
    splat_ply: pathlib.Path,
    out_dir: pathlib.Path,
    device: str = "cuda",
    near: float = 0.1,
    far: float = 10000.0,
    save_comparison: bool = False,
    downsample_factor: int = 4,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_comparison:
        (out_dir / "comparison").mkdir(parents=True, exist_ok=True)

    # ── 1. SfmScene 로드 (노트북과 동일: SfmScene.from_colmap) ────────────────
    logger.info(f"SfmScene 로드 중: {colmap_dir}")
    sfm_scene: frc.sfm_scene.SfmScene = frc.sfm_scene.SfmScene.from_colmap(colmap_dir)
    logger.info(f"  로드된 뷰 수: {len(sfm_scene.images)}")

    # ── 1b. frgs reconstruct 와 동일한 scene transform 적용 ──────────────────
    # frgs reconstruct 기본값: NormalizeScene(pca) + PercentileFilterPoints(0%,100%)
    #   + DownsampleImages(4) + FilterImagesWithLowPoints(5)
    # PLY는 이 transformed space에서 학습됐으므로 동일하게 변환해야 pose가 일치함
    logger.info("  scene transform 적용 중 (NormalizeScene pca + DownsampleImages 4)...")
    scene_transform = fvt.Compose(
        fvt.NormalizeScene(normalization_type="pca"),
        fvt.PercentileFilterPoints(
            percentile_min=np.zeros(3),
            percentile_max=np.full(3, 100.0),
        ),
        fvt.DownsampleImages(image_downsample_factor=downsample_factor),
        fvt.FilterImagesWithLowPoints(min_num_points=5),
    )
    sfm_scene = scene_transform(sfm_scene)
    num_views = len(sfm_scene.images)
    logger.info(f"  transform 후 뷰 수: {num_views}")

    # ── 2. PLY 로드 ────────────────────────────────────────────────────────────
    logger.info(f"Gaussian Splat 로드 중: {splat_ply}")
    model, _ = fvdb.GaussianSplat3d.from_ply(splat_ply, device=device)
    logger.info(f"  Gaussians: {model.num_gaussians:,}  device: {model.device}")

    # ── 3. 뷰별 렌더링 ────────────────────────────────────────────────────────
    logger.info(f"렌더링 시작 → {out_dir}")

    for i in tqdm(range(num_views), desc="Rendering"):
        image_meta: frc.sfm_scene.SfmPosedImageMetadata = sfm_scene.images[i]
        name = pathlib.Path(image_meta.image_path).stem

        rendered, depth = render_one_view(model, image_meta, near=near, far=far)

        # 렌더링 이미지 저장
        Image.fromarray(rendered).save(out_dir / f"{name}_render.png")

        # GT + Rendered + Depth 비교 이미지 저장
        if save_comparison:
            gt = load_gt_image(image_meta.image_path)
            depth_color = depth_to_colormap(depth)
            comp = make_comparison_image(gt, rendered, depth_color, f"View {i:03d} — {name}")
            comp.save(out_dir / "comparison" / f"{name}_compare.jpg", quality=85)

    logger.info(f"✓ 완료: {num_views}개 렌더링 → {out_dir}")
    if save_comparison:
        logger.info(f"  비교 이미지 → {out_dir / 'comparison'}")


def main():
    parser = argparse.ArgumentParser(description="학습 뷰를 Gaussian Splat PLY로 렌더링")
    parser.add_argument("colmap_dir", type=pathlib.Path, help="COLMAP 출력 디렉토리")
    parser.add_argument("splat_ply", type=pathlib.Path, help="splat.ply 경로")
    parser.add_argument(
        "--out-dir", "-o", type=pathlib.Path, default=None,
        help="결과 저장 디렉토리 (기본값: <colmap_dir>/renders)",
    )
    parser.add_argument("--device", default="cuda", help="연산 장치 (기본값: cuda)")
    parser.add_argument("--near", type=float, default=0.1, help="Near plane (기본값: 0.1)")
    parser.add_argument("--far", type=float, default=10000.0, help="Far plane (기본값: 10000.0)")
    parser.add_argument(
        "--downsample-factor", type=int, default=4,
        help="frgs reconstruct 에 사용된 image_downsample_factor (기본값: 4)",
    )
    parser.add_argument(
        "--save-comparison", action="store_true",
        help="GT / 렌더링 / Depth 비교 이미지 함께 저장",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.colmap_dir / "renders"

    if not args.colmap_dir.exists():
        parser.error(f"COLMAP 디렉토리가 없습니다: {args.colmap_dir}")
    if not args.splat_ply.exists():
        parser.error(f"PLY 파일이 없습니다: {args.splat_ply}")

    render_all_views(
        colmap_dir=args.colmap_dir,
        splat_ply=args.splat_ply,
        out_dir=args.out_dir,
        device=args.device,
        near=args.near,
        far=args.far,
        save_comparison=args.save_comparison,
        downsample_factor=args.downsample_factor,
    )


if __name__ == "__main__":
    main()
