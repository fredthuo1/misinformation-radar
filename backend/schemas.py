from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel


class ClaimOut(BaseModel):
    id: int
    source: str
    title: str
    url: str
    published_at: datetime | None = None
    normalized_claim: str
    category: str
    risk_domain: str
    confidence: float
    notes: str
    narrative_id: int | None = None

    class Config:
        from_attributes = True


class NarrativeOut(BaseModel):
    id: int
    title: str
    summary: str
    topic: str
    risk_level: str
    risk_score: float
    claim_count: int
    latest_claim_at: datetime | None = None
    keywords: str

    class Config:
        from_attributes = True
