from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="SMS Spam Detector API",
    version="0.1"
)

app.include_router(router)