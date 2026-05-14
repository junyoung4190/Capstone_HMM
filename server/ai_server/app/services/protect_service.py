import os
import sys


def _get_faceshield():
    faceshield_dir = os.environ.get(
        "FACESHIELD_DIR",
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "model", "faceshield")
        ),
    )
    if faceshield_dir not in sys.path:
        sys.path.insert(0, faceshield_dir)

    from pipeline import protect_image as _fn  # noqa: E402
    return _fn


def protect_image(contents: bytes) -> bytes:
    _faceshield_protect = _get_faceshield()
    result = _faceshield_protect(contents)
    if not result["success"]:
        raise RuntimeError(result.get("error", "FaceShield protection failed"))
    return result["protected_bytes"]
