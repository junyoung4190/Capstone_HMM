import os

ALLOWED_TYPES: set[str] = {"image/jpeg", "image/png"}

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
PROTECTED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "protected")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROTECTED_DIR, exist_ok=True)
