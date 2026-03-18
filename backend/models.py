from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from database import Base


class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(100), nullable=False, default="PolitiFact")
    title = Column(Text, nullable=False)
    url = Column(Text, nullable=False, unique=True, index=True)
    published_at = Column(DateTime, nullable=True)

    normalized_claim = Column(Text, nullable=False)
    category = Column(String(100), nullable=False, default="other misinformation")
    risk_domain = Column(String(50), nullable=False, default="general")
    confidence = Column(Float, nullable=False, default=0.0)
    notes = Column(Text, nullable=False, default="")

    narrative_id = Column(Integer, ForeignKey("narratives.id"), nullable=True)

    narrative = relationship("Narrative", back_populates="claims")


class Narrative(Base):
    __tablename__ = "narratives"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    summary = Column(Text, nullable=False, default="")
    topic = Column(String(100), nullable=False, default="general")
    risk_score = Column(Float, nullable=False, default=0.0)
    risk_level = Column(String(20), nullable=False, default="low")
    claim_count = Column(Integer, nullable=False, default=0)
    latest_claim_at = Column(DateTime, nullable=True)
    top_keywords = Column(Text, nullable=False, default="")

    claims = relationship("Claim", back_populates="narrative")