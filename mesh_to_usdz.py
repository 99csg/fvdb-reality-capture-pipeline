#!/usr/bin/env python3
"""
mesh_to_usdz.py  –  PLY 삼각형 메쉬를 USDZ 파일로 변환

사용법:
    python mesh_to_usdz.py <input.ply> <output.usdz>

의존성:
    - point-cloud-utils (pip install point-cloud-utils)
    - usd-core          (pip install usd-core)
"""
import sys
import pathlib
import logging
import tempfile
import zipfile

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_ply_mesh(ply_path: pathlib.Path):
    """PLY 파일에서 vertices, faces, colors 를 로드합니다."""
    try:
        import point_cloud_utils as pcu
        v, f, n, c = pcu.load_mesh_vfnc(str(ply_path))
    except Exception:
        # pcu.load_mesh_vfc 로 재시도 (normal 없는 경우)
        import point_cloud_utils as pcu
        v, f, c = pcu.load_mesh_vfc(str(ply_path))
    return v, f, c


def ply_mesh_to_usdz(ply_path: pathlib.Path, usdz_path: pathlib.Path) -> None:
    """PLY 삼각형 메쉬를 USDZ 파일로 변환합니다."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    logger.info(f"PLY 로드 중: {ply_path}")
    v, f, c = load_ply_mesh(ply_path)
    logger.info(f"  정점: {len(v):,}  면: {len(f):,}")

    # --- USD Stage 생성 ---
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Y")

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    mesh_prim = UsdGeom.Mesh.Define(stage, "/World/Mesh")

    # 정점 위치 설정
    points = Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in v])
    mesh_prim.GetPointsAttr().Set(points)

    # 면 인덱스 설정 (삼각형)
    face_vertex_counts = Vt.IntArray([3] * len(f))
    face_vertex_indices = Vt.IntArray(f.flatten().tolist())
    mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh_prim.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    # 정점 색상 설정
    if c is not None and len(c) == len(v):
        # 색상 범위 정규화 (0-1)
        c_arr = np.asarray(c, dtype=np.float32)
        if c_arr.max() > 1.0:
            c_arr = c_arr / 255.0
        c_arr = np.clip(c_arr, 0.0, 1.0)

        display_color = UsdGeom.PrimvarsAPI(mesh_prim).CreatePrimvar(
            "displayColor",
            Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.vertex,
        )
        colors = Vt.Vec3fArray([Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in c_arr[:, :3]])
        display_color.Set(colors)

    # 스무딩 방향 설정
    mesh_prim.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    # --- USDZ 로 저장 ---
    logger.info(f"USDZ 저장 중: {usdz_path}")
    usdz_path.parent.mkdir(parents=True, exist_ok=True)

    # 임시 USDC 파일 생성 후 USDZ 로 패키징
    with tempfile.TemporaryDirectory() as tmpdir:
        usdc_path = pathlib.Path(tmpdir) / "mesh.usdc"
        stage.GetRootLayer().Export(str(usdc_path))

        with zipfile.ZipFile(str(usdz_path), "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(str(usdc_path), "mesh.usdc")

    logger.info(f"✓ USDZ 저장 완료: {usdz_path}  ({usdz_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"사용법: {sys.argv[0]} <input.ply> <output.usdz>")
        sys.exit(1)

    in_path = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])

    if not in_path.exists():
        print(f"오류: 입력 파일이 없습니다: {in_path}")
        sys.exit(1)
    if in_path.suffix.lower() != ".ply":
        print(f"오류: 입력 파일은 .ply 이어야 합니다: {in_path}")
        sys.exit(1)
    if out_path.suffix.lower() != ".usdz":
        print(f"오류: 출력 파일은 .usdz 이어야 합니다: {out_path}")
        sys.exit(1)

    ply_mesh_to_usdz(in_path, out_path)
