from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from database import Base, engine, SessionLocal
from models import Claim, Narrative
from services import build_narratives, get_claims, get_narratives, ingest_feed

Base.metadata.drop_all(bind=engine)
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
def api_ingest(
    target_new: int = Query(default=3, ge=1, le=20),
    max_scan: int = Query(default=30, ge=1, le=100),
):
    return ingest_feed(target_new=target_new, max_scan=max_scan)


@app.post("/api/build")
def api_build():
    return build_narratives()


@app.post("/api/refresh")
def api_refresh(
    target_new: int = Query(default=3, ge=1, le=20),
    max_scan: int = Query(default=30, ge=1, le=100),
):
    ingest_result = ingest_feed(target_new=target_new, max_scan=max_scan)
    build_result = build_narratives()
    return {
        "ingest": ingest_result,
        "build": build_result,
    }


@app.get("/api/claims")
def api_claims():
    return get_claims()


@app.get("/api/narratives")
def api_narratives():
    return get_narratives()


@app.get("/api/debug")
def api_debug():
    claims = get_claims()
    narratives = get_narratives()
    return {
        "claims_count": len(claims),
        "narratives_count": len(narratives),
        "latest_claim": claims[0] if claims else None,
        "latest_narrative": narratives[0] if narratives else None,
    }


@app.post("/api/reset")
def api_reset():
    db = SessionLocal()
    try:
        db.query(Claim).delete()
        db.query(Narrative).delete()
        db.commit()
        return {"status": "ok", "message": "Database cleared"}
    finally:
        db.close()