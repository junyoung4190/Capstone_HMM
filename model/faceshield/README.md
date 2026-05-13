# FaceShield Multi-branch (v4)

원본 FaceShield 논문 기반 적대적 노이즈 보호 시스템에
**Multi-branch 영역별 분리 구조**를 추가한 개선판.

## 주요 변경사항 (v4)

1. **Contrast-aware noise scaling** (v2)
   - `compute_contrast_weight`: 픽셀별 텍스처/평탄 가중치
   - 평탄 영역 약화로 LPIPS 개선

2. **Multi-branch 영역별 가중치** (v4 ★ 새로 추가)
   - `generate_face_mask`: dlib 68 landmark로 얼굴 영역 마스크 생성
   - PGD step에 영역별 가중치 곱셈 (얼굴 1.5배, 배경 1.0배)
   - arc_sim: 0.097 (baseline) → **-0.10** (v4)

## 두 가지 사용 방식

### 1. 연구용 (CLI)
```bash
sh run.sh                                  # 폴더 안 사진 일괄 처리
python compare.py --tag <실험명>             # 평가
```

### 2. 백엔드 통합용 (Python API)
```python
from pipeline import protect_image

# 다양한 입력 형식 지원
result = protect_image(image_bytes)       # 백엔드 업로드
result = protect_image("/path/to/img.jpg") # 파일 경로
result = protect_image(numpy_array)       # numpy
result = protect_image(pil_image)         # PIL

# 결과
# {
#   "success": True,
#   "protected_bytes": <PNG bytes>,
#   "metrics": {"px_diff": 2.7},
#   "error": None
# }
```

## 의존성

```bash
pip install dlib opencv-python imutils scipy
# FaceShield 원본 의존성도 필요
```

## 필요 모델 파일 (별도 다운로드)

```bash
# dlib landmark 모델
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# 기타 모델 (Stable Diffusion, IP-Adapter, ArcFace)
# HuggingFace에서 자동 다운로드 또는 수동 다운로드
```

## 환경변수 설정

```bash
export MODEL_PATH="runwayml/stable-diffusion-v1-5"
export UNET_CONFIG="./utils/unet/unet_config15.json"
export IP_ADAPTER_PATH="./utils/unet/ip_adapter/ip-adapter_sd15.bin"
export IMAGE_ENCODER_PATH="h94/IP-Adapter"
export ARCFACE50_PATH="./models/arcface50_checkpoint.tar"
export ARCFACE100_PATH="./models/arcface100_checkpoint.tar"
export LANDMARK_PATH="./shape_predictor_68_face_landmarks.dat"
```

## 실험 결과 (n=3 이미지)

| Tag | arc_sim ↓ | clip_sim ↓ | LPIPS ↓ |
| --- | --- | --- | --- |
| baseline (원본) | 0.097 | 0.592 | 0.057 |
| contrast_v2 | 0.32 | 0.71 | 0.038 |
| **multibranch_v4** | **-0.10** | **0.58** | **0.069** |

→ 보호력: baseline 초과 (arc_sim -0.10)
→ 화질: 살짝 손해 (LPIPS +0.012, 추후 λ_face 튜닝 필요)
