from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import health, predict

app = FastAPI(title="PulmoScan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(predict.router, prefix="/api")

@app.get("/")
def root():
    return {"message": "PulmoScan API is running"}
