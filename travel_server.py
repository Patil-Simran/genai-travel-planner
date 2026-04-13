from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import app as planner

_static = Path(__file__).resolve().parent / "travel_ui"


class PlanBody(BaseModel):
    query: str = Field(default="", max_length=8000)
    options: str = Field(default="", max_length=8000)


web_app = FastAPI(title="Travel Planner", version="0.1.0")
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@web_app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@web_app.get("/")
def index() -> FileResponse:
    path = _static / "index.html"
    if not path.is_file():
        raise HTTPException(404, "travel_ui/index.html missing")
    return FileResponse(path)


@web_app.post("/api/plan")
def plan(body: PlanBody) -> JSONResponse:
    q = body.query.strip()
    o = body.options.strip()
    if not q and not o:
        raise HTTPException(status_code=400, detail="query or options required")
    try:
        payload = planner.plan_trip_query(q, options=o or None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return JSONResponse(payload)
