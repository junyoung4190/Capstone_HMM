from fastapi import APIRouter, File, UploadFile

from app.services.file_service import validate_and_read
from app.services.upload_service import save_upload

router = APIRouter(tags=["upload"])


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await validate_and_read(file)
    result = save_upload(contents, file.filename)
    return {"message": "업로드 성공", **result}
