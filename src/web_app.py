"""
Local web API for the Text-DIRE detector.

The app serves the static website and exposes /api/analyze. It shares scoring
logic with the Vercel serverless handler in api/analyze.py.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .scoring import build_analysis_response, provider_name, resolve_mask_ratios, score_text


ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT_DIR / "web"

load_dotenv(ROOT_DIR / ".env")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=40, max_length=8000)
    mode: Literal["fast", "balanced", "careful"] = "balanced"
    mask_ratios: list[float] | None = None


class ScoreBreakdown(BaseModel):
    mask_ratio: float
    reconstruction_error: float
    token_accuracy: float | None = None


class AnalyzeResponse(BaseModel):
    prediction: Literal["human", "ai", "uncertain"]
    confidence: float
    score: float
    threshold: float
    provider: str
    elapsed_seconds: float
    breakdown: list[ScoreBreakdown]
    notes: list[str]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Text-DIRE Detector",
        description="AI text detection using Diffusion Reconstruction Error.",
        version="0.1.0",
    )

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "provider": provider_name()}

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
        text = payload.text.strip()
        if len(text.split()) < 20:
            raise HTTPException(
                status_code=422,
                detail="Please provide at least 20 words for a meaningful signal.",
            )

        try:
            mask_ratios = resolve_mask_ratios(payload.mode, payload.mask_ratios)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        started = perf_counter()
        try:
            provider_result = score_text(text, mask_ratios)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return AnalyzeResponse(**build_analysis_response(provider_result, perf_counter() - started))

    return app


app = create_app()
