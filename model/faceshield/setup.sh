#!/bin/bash
# FaceShield+ Setup — 필수 의존성 자동 다운로드

set -e

echo "=== FaceShield+ Setup ==="

# 1) dlib landmark 모델 (95MB) — git에 안 들어가있음
if [ ! -f shape_predictor_68_face_landmarks.dat ]; then
    echo "Downloading dlib face landmark model (95MB)..."
    if command -v wget &> /dev/null; then
        wget -q http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    else
        curl -sLO http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    fi
    bunzip2 shape_predictor_68_face_landmarks.dat.bz2
    echo "✓ shape_predictor downloaded"
fi

# 2) CNN 가중치 확인
if [ ! -f fs_predictor_model.pth ]; then
    echo "⚠️  fs_predictor_model.pth missing — git에서 받아야 함"
    exit 1
fi
echo "✓ fs_predictor_model.pth found ($(du -h fs_predictor_model.pth | cut -f1))"

# 3) Python 패키지 설치 안내
echo ""
echo "Python deps: pip install -r requirements.txt"
echo "환경변수 셋업: see README.md"
echo ""
echo "=== Setup 완료 ==="
