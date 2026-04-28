import os
import uuid

from app.core.config import UPLOAD_DIR


def save_upload(contents: bytes, original_filename: str) -> dict:
    filename = f"{uuid.uuid4()}_{original_filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(contents)
    return {"filename": filename, "size": len(contents), "path": filepath}
