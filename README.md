# Misinformation Radar

An end-to-end hackathon MVP that:
- pulls fact-check items from the PolitiFact RSS feed
- sends each title to your DigitalOcean agent for claim extraction
- stores normalized claims in SQLite
- clusters related claims into misinformation narratives
- shows everything in a lightweight dashboard

## Stack
- FastAPI backend
- SQLite database via SQLAlchemy
- PolitiFact RSS ingestion with `feedparser`
- DigitalOcean agent inference using your existing DeepSeek-powered agent
- TF-IDF + agglomerative clustering for narrative detection
- Simple HTML dashboard

## Project structure
```
misinformation-radar/
├── backend/
│   ├── .env.example
│   ├── main.py
│   ├── models.py
│   ├── requirements.txt
│   ├── schemas.py
│   └── services.py
├── frontend/
│   └── index.html
└── README.md
```

## Setup
1. Create and activate a virtual environment.
2. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your values:
   ```env
   AGENT_ENDPOINT=https://ysyotpnqiaqde47sbdfsx6je.agents.do-ai.run
   AGENT_ACCESS_KEY=your_key_here
   RSS_URL=https://www.politifact.com/rss/factchecks/
   DATABASE_URL=sqlite:///./radar.db
   MAX_FEED_ITEMS=30
   ```
4. Run the API:
   ```bash
   uvicorn main:app --reload
   ```
5. Open `http://127.0.0.1:8000`

## How to use
1. Click **Ingest feed** to fetch recent fact-check items and run claim extraction through the agent.
2. Click **Build narratives** to cluster claims into narrative groups.
3. Inspect the narrative cards and detail panel.

## Notes
- The agent prompt strips reasoning and expects JSON output only.
- If DeepSeek still emits reasoning, the backend extracts the JSON object from the response.
- For production, swap SQLite for Postgres and serve the frontend with a proper React app.

## Next upgrades
- Add embeddings from a dedicated model instead of TF-IDF
- Persist raw retrieval and article snippets
- Add filters by topic, date, and risk level
- Add scheduled ingestion and deployment to DigitalOcean App Platform
