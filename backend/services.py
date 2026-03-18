import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import requests
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from database import SessionLocal
from models import Claim, Narrative

load_dotenv()

FEED_URL = "https://www.politifact.com/rss/factchecks/"
AGENT_ENDPOINT = os.getenv("AGENT_ENDPOINT", "").rstrip("/")
AGENT_ACCESS_KEY = os.getenv("AGENT_ACCESS_KEY", "").strip()


def fetch_feed(limit: int | None = None):
    feed = feedparser.parse(FEED_URL)
    entries = feed.entries or []
    return entries[:limit] if limit else entries


def _safe_datetime(entry: Any):
    published = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if not published:
        return None
    try:
        dt = parsedate_to_datetime(published)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


def _extract_json_block(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Empty model response.")

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)

    if cleaned.startswith("{") and cleaned.endswith("}"):
        return json.loads(cleaned)

    match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise ValueError(f"Could not extract JSON from model response: {text[:300]}")


def call_agent(prompt: str) -> dict[str, Any]:
    if not AGENT_ENDPOINT or not AGENT_ACCESS_KEY:
        raise RuntimeError("Missing AGENT_ENDPOINT or AGENT_ACCESS_KEY in .env")

    url = f"{AGENT_ENDPOINT}/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AGENT_ACCESS_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "n/a",
        "messages": [{"role": "user", "content": prompt}],
    }

    response = requests.post(url, headers=headers, json=payload, timeout=90)
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return _extract_json_block(content)


def extract_claim(entry: Any) -> dict[str, Any]:
    title = getattr(entry, "title", "")
    summary = getattr(entry, "summary", "")

    prompt = f"""
Task: CLAIM_EXTRACTION

Input title:
{title}

Input snippet:
{summary}

Return valid JSON only.
Do not include markdown.
Do not include explanations.
Do not include <think> tags.

Use this exact schema:
{{
  "task": "claim_extraction",
  "normalized_claim": "<underlying claim only, no prefix like 'claims allege'>",
  "category": "<one category>",
  "risk_domain": "<health|disaster|politics|finance|science|crime|general>",
  "confidence": <0-1>,
  "notes": "<brief explanation>"
}}
""".strip()

    result = call_agent(prompt)
    normalized_claim = str(result.get("normalized_claim", title)).strip()
    normalized_claim = re.sub(
        r"^(Claims allege|Posts claim|Post claims|Claim|Claims that)\s+",
        "",
        normalized_claim,
        flags=re.I,
    ).strip(" .")

    return {
        "task": "claim_extraction",
        "normalized_claim": normalized_claim or title,
        "category": str(result.get("category", "other misinformation")).strip(),
        "risk_domain": str(result.get("risk_domain", "general")).strip(),
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "notes": str(result.get("notes", "")).strip(),
    }


def ingest_feed(target_new: int = 3, max_scan: int = 30) -> dict[str, Any]:
    db = SessionLocal()
    created = 0
    skipped = 0
    scanned = 0
    errors: list[str] = []

    try:
        entries = fetch_feed(limit=max_scan)
        print(f"[INGEST] Loaded {len(entries)} feed entries, target_new={target_new}, max_scan={max_scan}")

        for idx, entry in enumerate(entries, start=1):
            if created >= target_new:
                break

            scanned += 1
            title = getattr(entry, "title", "(no title)")
            link = getattr(entry, "link", "")
            print(f"[INGEST] {idx}/{len(entries)} Processing: {title}")

            existing = db.query(Claim).filter(Claim.url == link).first()
            if existing:
                skipped += 1
                print("[INGEST] Skipped existing")
                continue

            try:
                extracted = extract_claim(entry)
                print(f"[INGEST] Extracted claim: {extracted['normalized_claim']}")

                claim = Claim(
                    source="PolitiFact",
                    title=title,
                    url=link,
                    published_at=_safe_datetime(entry),
                    normalized_claim=extracted["normalized_claim"],
                    category=extracted["category"],
                    risk_domain=extracted["risk_domain"],
                    confidence=extracted["confidence"],
                    notes=extracted["notes"],
                )
                db.add(claim)
                db.commit()
                db.refresh(claim)
                created += 1
                print(f"[INGEST] Saved claim id={claim.id}")

            except Exception as e:
                db.rollback()
                error_msg = f"{title}: {str(e)}"
                print(f"[INGEST] ERROR {error_msg}")
                errors.append(error_msg)

    finally:
        db.close()

    return {
        "created": created,
        "skipped": skipped,
        "scanned": scanned,
        "errors": errors,
        "message": (
            "No new unseen fact-checks found in the current feed window."
            if created == 0
            else f"Added {created} new fact-check(s)."
        ),
    }


def _risk_level(score: float) -> str:
    if score >= 70:
        return "high"
    if score >= 40:
        return "medium"
    return "low"


def _compute_risk_score(claims: list[Claim], topic: str) -> float:
    recurrence = min(len(claims) * 18, 40)
    avg_conf = sum(c.confidence for c in claims) / max(len(claims), 1)
    confidence_component = avg_conf * 20

    topic_bonus_map = {
        "health": 18,
        "disaster": 16,
        "politics": 15,
        "finance": 14,
        "crime": 12,
        "science": 10,
        "general": 8,
    }
    topic_bonus = topic_bonus_map.get(topic, 8)

    recent_bonus = 0
    dated = [c.published_at for c in claims if c.published_at]
    if dated:
        latest = max(dated)
        days_old = (datetime.utcnow() - latest).days
        if days_old <= 7:
            recent_bonus = 20
        elif days_old <= 30:
            recent_bonus = 10

    score = min(recurrence + confidence_component + topic_bonus + recent_bonus, 100)
    return round(score, 1)


def build_narratives() -> dict[str, Any]:
    db = SessionLocal()
    try:
        claims = db.query(Claim).all()
        print(f"[BUILD] Claims found: {len(claims)}")

        if not claims:
            return {
                "narratives_created": 0,
                "message": "No claims available. Ingest feed first.",
            }

        for narrative in db.query(Narrative).all():
            db.delete(narrative)
        db.commit()

        for claim in claims:
            claim.narrative_id = None
        db.commit()

        texts = [c.normalized_claim for c in claims]
        print(f"[BUILD] Texts: {texts}")

        if len(claims) == 1:
            labels = [0]
        else:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            X = vectorizer.fit_transform(texts)

            n_clusters = min(max(2, len(claims) // 2), len(claims))
            print(f"[BUILD] Using n_clusters={n_clusters}")

            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(X.toarray())

        grouped: dict[int, list[Claim]] = {}
        for label, claim in zip(labels, claims):
            grouped.setdefault(int(label), []).append(claim)

        created = 0

        for label, cluster_claims in grouped.items():
            print(f"[BUILD] Cluster {label} size={len(cluster_claims)}")

            sorted_claims = sorted(
                cluster_claims,
                key=lambda c: (c.published_at or datetime.min),
                reverse=True,
            )
            representative = sorted_claims[0]

            words: list[str] = []
            for c in cluster_claims:
                words.extend(re.findall(r"\b[a-zA-Z]{4,}\b", c.normalized_claim.lower()))

            stop = {
                "claim", "claims", "posts", "false", "falsely", "says", "said",
                "about", "with", "from", "that", "this", "they", "have", "after",
                "will", "there", "their", "what", "when", "where", "does", "said"
            }
            top_words = [w for w, _ in Counter(w for w in words if w not in stop).most_common(6)]

            topic = Counter(c.risk_domain for c in cluster_claims).most_common(1)[0][0]
            risk_score = _compute_risk_score(cluster_claims, topic)
            risk_level = _risk_level(risk_score)

            title = " ".join(w.capitalize() for w in top_words[:4]) if top_words else representative.normalized_claim[:80]
            summary = representative.normalized_claim
            keywords = ", ".join(top_words)

            narrative = Narrative(
                title=title,
                summary=summary,
                topic=topic,
                risk_score=risk_score,
                risk_level=risk_level,
                claim_count=len(cluster_claims),
                latest_claim_at=max((c.published_at for c in cluster_claims if c.published_at), default=None),
                top_keywords=keywords,
            )
            db.add(narrative)
            db.commit()
            db.refresh(narrative)

            for c in cluster_claims:
                c.narrative_id = narrative.id
            db.commit()
            created += 1
            print(f"[BUILD] Saved narrative id={narrative.id}")

        return {
            "narratives_created": created,
            "message": f"Built {created} narrative(s).",
        }

    finally:
        db.close()


def get_claims() -> list[dict[str, Any]]:
    db = SessionLocal()
    try:
        claims = db.query(Claim).order_by(Claim.published_at.desc().nullslast()).all()
        return [
            {
                "id": c.id,
                "title": c.title,
                "url": c.url,
                "source": c.source,
                "published_at": c.published_at.isoformat() if c.published_at else None,
                "normalized_claim": c.normalized_claim,
                "category": c.category,
                "risk_domain": c.risk_domain,
                "confidence": c.confidence,
                "notes": c.notes,
                "narrative_id": c.narrative_id,
            }
            for c in claims
        ]
    finally:
        db.close()


def get_narratives() -> list[dict[str, Any]]:
    db = SessionLocal()
    try:
        narratives = db.query(Narrative).order_by(Narrative.risk_score.desc()).all()
        output = []
        for n in narratives:
            output.append(
                {
                    "id": n.id,
                    "title": n.title,
                    "summary": n.summary,
                    "topic": n.topic,
                    "risk_score": n.risk_score,
                    "risk_level": n.risk_level,
                    "claim_count": n.claim_count,
                    "latest_claim_at": n.latest_claim_at.isoformat() if n.latest_claim_at else None,
                    "top_keywords": n.top_keywords,
                    "claims": [
                        {
                            "id": c.id,
                            "title": c.title,
                            "url": c.url,
                            "normalized_claim": c.normalized_claim,
                            "category": c.category,
                            "risk_domain": c.risk_domain,
                            "confidence": c.confidence,
                        }
                        for c in n.claims
                    ],
                }
            )
        return output
    finally:
        db.close()