import os
import sys
import tempfile

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.risk_analyzer.detect_face import detect_face
from model.risk_analyzer.head_pose import get_head_pose


def _calculate_risk(face_ratio: float, yaw: float) -> dict:
    exposure  = min(face_ratio * 2.0, 1.0)          # 얼굴 비율 (0.5 이상이면 만점)
    frontal   = max(0.0, 1.0 - abs(yaw) / 90.0)     # 정면일수록 높음

    score = round(0.5 * exposure + 0.5 * frontal, 4)

    if score >= 0.7:
        level = "HIGH"
    elif score >= 0.4:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {"score": score, "level": level}


def analyze_image(contents: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        detection = detect_face(tmp_path)
    finally:
        os.unlink(tmp_path)

    if detection is None:
        return {
            "face_count": 0,
            "faces": [],
            "risk": {"score": 0.0, "level": "LOW"},
        }

    face_crop, face_ratio = detection
    pose = get_head_pose(face_crop)

    yaw   = round(float(pose[0]), 2) if pose else 0.0
    pitch = round(float(pose[1]), 2) if pose else 0.0

    return {
        "face_count": 1,
        "faces": [
            {
                "face_ratio": round(float(face_ratio), 4),
                "head_pose": {"yaw": yaw, "pitch": pitch},
            }
        ],
        "risk": _calculate_risk(float(face_ratio), yaw),
    }
