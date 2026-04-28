from fastapi import APIRouter, File, UploadFile

from app.services.analyze_service import analyze_image
from app.services.file_service import validate_and_read

router = APIRouter(tags=["analyze"])


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await validate_and_read(file)
    return analyze_image(contents)
