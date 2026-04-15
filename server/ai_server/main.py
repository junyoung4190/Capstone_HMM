from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import uuid

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_TYPES = ["image/jpeg", "image/png"]

@app.get("/")
def status():
    return {"status": "AI Server Running"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="이미지 파일만 업로드 가능합니다."
        )

    contents = await file.read()

    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(contents)

    return {
        "message": "업로드 성공",
        "filename": filename,
        "size": len(contents),
        "path": filepath
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Mock AI 응답
    return {
        "result": "deepfake",
        "confidence": 0.87,
        "message": "딥페이크 의심 이미지입니다."
    }