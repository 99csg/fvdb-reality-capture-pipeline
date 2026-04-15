# fVDB Reality Capture Pipeline

비디오 파일 하나를 입력받아 **USDZ 3D 메쉬**를 자동으로 생성하는 완전 자동화 파이프라인입니다.  
Docker 컨테이너로 패키징되어 NVIDIA GPU가 있는 어떤 Linux 머신에서도 동일하게 동작합니다.

```
Video (.mp4/.mov)
    │
    ├─ [1] Frame Extraction  (ffmpeg)
    ├─ [2] SfM Reconstruction (COLMAP)
    ├─ [3] Gaussian Splatting (frgs reconstruct)
    ├─ [4] Mesh Extraction    (frgs mesh-basic)
    └─ [5] USDZ Export        (mesh_to_usdz.py)
                │
            output.usdz
```

## 요구사항

| 항목 | 사양 |
|------|------|
| OS | Ubuntu 20.04+ (Linux) |
| GPU | NVIDIA GPU (VRAM 8GB+ 권장) |
| CUDA | 12.8 |
| Driver | 570+ |
| Docker | 20.10+ |
| NVIDIA Container Toolkit | 최신 버전 |

## 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/<your-username>/fvdb-reality-capture-pipeline.git
cd fvdb-reality-capture-pipeline
```

### 2. fvdb-reality-capture 소스 클론 (필수)

```bash
git clone https://github.com/openvdb/fvdb-reality-capture.git
cd fvdb-reality-capture && git checkout v0.4.0 && cd ..
```

### 3. Docker 그룹 설정 (최초 1회)

```bash
sudo usermod -aG docker $USER
# 이후 터미널 재시작 또는 VS Code 재시작 필요
```

### 4. 파이프라인 실행

```bash
bash run_docker.sh <input_video.mp4> [output_dir] [fps]
```

**예시:**
```bash
# 기본 실행 (FPS=2, 출력: ./output/<이름>_<타임스탬프>/)
bash run_docker.sh data/my_scene.mp4

# 출력 디렉토리 지정 및 FPS 높이기
bash run_docker.sh data/my_scene.mp4 ./results 3
```

첫 실행 시 Docker 이미지를 자동으로 빌드합니다 (약 20~30분, 이후 캐시 사용).

## 파이프라인 단계

| 단계 | 도구 | 설명 |
|------|------|------|
| 1. 프레임 추출 | ffmpeg | 영상에서 JPG 프레임 추출 (기본 2fps) |
| 2. SfM 재구성 | COLMAP | 카메라 포즈 및 희소 포인트 클라우드 생성 |
| 3. Gaussian Splat 학습 | frgs reconstruct | 3D Gaussian Splatting 학습 |
| 4. 메쉬 추출 | frgs mesh-basic | TSDF 기반 삼각형 메쉬 생성 |
| 5. USDZ 변환 | mesh_to_usdz.py | PLY → USDZ 변환 (AR Quick Look 호환) |

## 출력 결과

```
output/<video_name>_<timestamp>/
├── images_raw/      # 추출된 원본 프레임
├── images/          # COLMAP undistorted 프레임
├── sparse/          # COLMAP sparse 재구성
├── database.db      # COLMAP feature database
├── splat.ply        # Gaussian Splat
├── mesh.ply         # 삼각형 메쉬
├── output.usdz      # 최종 결과물 (AR 호환)
└── pipeline.log     # 실행 로그
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `SKIP_BUILD` | `0` | `1`로 설정 시 Docker 이미지 빌드 건너뜀 |
| `COLMAP_MAX_FRAMES` | `400` | COLMAP에 사용할 최대 프레임 수 |

```bash
# 예시: 이미 빌드된 이미지 재사용, 프레임 200개 제한
SKIP_BUILD=1 COLMAP_MAX_FRAMES=200 bash run_docker.sh my_scene.mp4
```

## 기존 방식 (venv)

Docker를 사용하지 않고 로컬 환경에서 직접 실행하려면:

```bash
# 환경 설치 (최초 1회)
bash setup.sh

# 파이프라인 실행
source venv/bin/activate
bash pipeline.sh <input_video.mp4>
```

> **요구사항:** Python 3.11+, ffmpeg, colmap, CUDA 12.8

## Docker 이미지 구성

| 컴포넌트 | 버전 |
|----------|------|
| Base | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` |
| Python | 3.11 |
| PyTorch | 2.10.0+cu128 |
| fvdb-core | 0.4.2+pt210.cu128 |
| fvdb-reality-capture | v0.4.0 |
| ffmpeg / COLMAP | 시스템 패키지 |

## 테스트 결과

**환경:** NVIDIA RTX PRO 5000 (24GB), CUDA 12.8  
**입력:** iPhone 촬영 영상 (159MB, `.mov`)

| 단계 | 소요 시간 |
|------|----------|
| STEP 1 프레임 추출 | ~1분 |
| STEP 2 COLMAP SfM | ~수분 |
| STEP 3 Gaussian Splat | **약 7분** |
| STEP 4 메쉬 추출 | 11초 |
| STEP 5 USDZ 변환 | 35초 |
| **총** | **약 8분** |

출력: `splat.ply` 301MB / `mesh.ply` 179MB / `output.usdz` **52MB**

## 파일 구조

```
.
├── Dockerfile           # Docker 이미지 빌드 설정
├── run_docker.sh        # Docker 실행 래퍼 (빌드 + 실행)
├── pipeline.sh          # 메인 파이프라인 스크립트
├── setup.sh             # 로컬 venv 설치 스크립트
├── run_colmap.py        # COLMAP 파이프라인
├── mesh_to_usdz.py      # PLY → USDZ 변환
├── fvdb-reality-capture/ # (별도 클론 필요)
├── data/                # 입력 영상 (gitignore)
└── output/              # 결과물 (gitignore)
```

## 트러블슈팅

**`permission denied` (docker 소켓)**
```bash
sudo usermod -aG docker $USER
# 터미널이 아닌 VS Code / 세션 전체 재시작 필요
```

**COLMAP 재구성 실패**
- 프레임 수가 너무 적을 경우: FPS를 3~5로 높이기
- 장면이 너무 어둡거나 움직임이 많은 경우 재촬영 권장

**GPU 메모리 부족 (OOM)**
```bash
COLMAP_MAX_FRAMES=200 bash run_docker.sh my_scene.mp4
```

## 라이선스

Apache-2.0 (fvdb-reality-capture 기반)
