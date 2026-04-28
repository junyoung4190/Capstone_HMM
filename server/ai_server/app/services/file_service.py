from fastapi import HTTPException, UploadFile

from app.core.config import ALLOWED_TYPES


async def validate_and_read(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="JPEG/PNG 이미지만 허용됩니다.")
    return await file.read()
