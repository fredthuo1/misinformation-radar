from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from database import Base, engine
from services import build_narratives, get_claims, get_narratives, ingest_feed

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Misinformation Radar")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/ingest")
def api_ingest(limit: int = Query(default=3, ge=1, le=20)):
    return ingest_feed(limit=limit)


@app.post("/api/build")
def api_build():
    return build_narratives()


@app.get("/api/claims")
def api_claims():
    return get_claims()


@app.get("/api/narratives")
def api_narratives():
    return get_narratives()