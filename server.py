"""Thin web layer for the mimOE triage agent.

Separation of concerns: this file owns *only* HTTP. All clinical logic lives in
agent.py / schemas.py / tools.py and is reused unchanged. The browser frontend
in web/ owns presentation. This server just:

    - serves the static frontend at  /
    - exposes the agent at           POST /api/triage
    - reports endpoint config at     GET  /api/config

Run:  uvicorn server:app --reload --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import BASE_URL, MODEL_NAME, TEMPERATURE, build_llm, run_triage

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="mimOE Health Triage Agent", version="1.0.0")

# One reusable LLM client for the process (cheap, avoids re-reading env per call).
_llm = build_llm()


class TriageRequest(BaseModel):
    symptom: str = Field(min_length=1, description="Patient symptom report.")
    patient_id: str = Field(default="demo-patient")


@app.get("/api/config")
def config() -> dict:
    """Expose the (non-secret) on-device endpoint config for the UI header."""
    return {
        "base_url": BASE_URL,
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "on_device": True,
    }


@app.post("/api/triage")
def triage(req: TriageRequest) -> dict:
    """Run one triage loop and return the full trace as JSON."""
    run = run_triage(req.symptom, req.patient_id, llm=_llm, verbose=False)
    return run.to_dict()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


# Static assets (styles.css, app.js). Mounted last so it doesn't shadow routes.
app.mount("/", StaticFiles(directory=WEB_DIR), name="static")
