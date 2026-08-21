"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI application entry-point.

Start with:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST   /api/projects              create a new movie project
    GET    /api/projects              list all projects
    GET    /api/projects/{id}         project detail + episode list
    POST   /api/projects/{id}/ingest  upload book PDF → trigger digestion
    GET    /api/projects/{id}/ingest/stream  SSE: live digestion progress
    GET    /api/projects/{id}/episodes       list all episodes (chapters)
    GET    /api/projects/{id}/episodes/{ep}  episode detail + scene list
    POST   /api/projects/{id}/episodes/{ep}/generate  trigger generation
    GET    /api/projects/{id}/episodes/{ep}/stream    SSE: live gen progress
    GET    /api/projects/{id}/scenes/{scene_key}      scene detail
    GET    /api/library               completed movies listing
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from api.db import create_db_and_tables
from api.routers import episodes, ingest, library, projects, scenes, stream

app = FastAPI(
    title="Agentic Cinema API",
    description="Book digestion → LangGraph episode generation → full movie",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup() -> None:
    create_db_and_tables()

# Allow the Next.js frontend dev server and production build
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://cinema.yourdomain.com",  # update for production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated assets (storyboards, video clips, final movies)
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")

# Register routers
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(ingest.router,   prefix="/api/projects", tags=["ingest"])
app.include_router(episodes.router, prefix="/api/projects", tags=["episodes"])
app.include_router(scenes.router,   prefix="/api/projects", tags=["scenes"])
app.include_router(library.router,  prefix="/api/library",  tags=["library"])
app.include_router(stream.router,   prefix="/api",          tags=["stream"])


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "agentic-cinema"}
