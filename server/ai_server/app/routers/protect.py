import io

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse

from app.services.file_service import validate_and_read
from app.services.protect_service import protect_image

router = APIRouter(tags=["protect"])


@router.post("/protect")
async def protect(file: UploadFile = File(...)):
    contents = await validate_and_read(file)
    protected = protect_image(contents)
    return StreamingResponse(
        io.BytesIO(protected),
        media_type=file.content_type,
        headers={"Content-Disposition": f'attachment; filename="protected_{file.filename}"'},
    )
