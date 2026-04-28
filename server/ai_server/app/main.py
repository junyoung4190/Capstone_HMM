from fastapi import FastAPI

from app.routers import analyze, protect, upload

app = FastAPI(title="AI Server")

app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(protect.router)


@app.get("/")
def status():
    return {"status": "AI Server Running"}
