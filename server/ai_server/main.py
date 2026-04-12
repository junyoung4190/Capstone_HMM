from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.get("/")
def status():
    return "TEST"

@app.post("/uploadfile/")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents)
    }