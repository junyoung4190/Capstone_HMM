#!/usr/bin/env bash
# 로컬 개발용 AI 서버 실행 스크립트
# 사용법: bash run_local.sh
#
# 환경변수 기본값 (변경 필요 시 아래에서 수정)
#   GPU_ID=-1        → CPU 모드 (맥 로컬)
#   GPU_ID=0         → CUDA GPU (서버 배포 시)

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

GPU_ID=-1 \
YOLO_WEIGHTS_PATH="$PROJECT_ROOT/weights/yolov8n-face.pt" \
  uvicorn app.main:app --reload
