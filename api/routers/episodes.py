"""
api/routers/episodes.py
─────────────────────────────────────────────────────────────────────────────
Episode (chapter) endpoints.

  GET  /api/projects/{id}/episodes               — list all episodes
  GET  /api/projects/{id}/episodes/{ep}          — episode detail + scene list
  POST /api/projects/{id}/episodes/{ep}/generate — trigger LangGraph generation
  GET  /api/projects/{id}/episodes/{ep}/stream   — SSE: live generation progress
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from api.db import Episode, Project, Scene, get_session
from api.services.event_bus import generate_channel, subscribe
from api.services.generation_service import run_generate

router = APIRouter()


# ── Response models ────────────────────────────────────────────────────────────

class SceneSummary(BaseModel):
    id: int
    scene_number: int
    global_scene_number: int
    slug: str
    location_name: str
    emotional_beat: str
    status: str
    storyboard_path: str
    video_path: str
    quality_score: float
    characters: list[str]


class EpisodeDetail(BaseModel):
    id: int
    project_id: int
    chapter_number: int
    title: str
    summary: str
    act: int
    emotional_arc: str
    total_scenes: int
    status: str
    episode_video_path: str
    thumbnail_path: str
    created_at: datetime
    updated_at: datetime
    scenes: list[SceneSummary] = []


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/{project_id}/episodes", response_model=list[EpisodeDetail])
def list_episodes(
    project_id: int,
    session: Session = Depends(get_session),
) -> list[EpisodeDetail]:
    _require_project(project_id, session)
    episodes = session.exec(
        select(Episode)
        .where(Episode.project_id == project_id)
        .order_by(Episode.chapter_number)
    ).all()
    return [_episode_detail(ep, session) for ep in episodes]


@router.get("/{project_id}/episodes/{episode_number}", response_model=EpisodeDetail)
def get_episode(
    project_id: int,
    episode_number: int,
    session: Session = Depends(get_session),
) -> EpisodeDetail:
    ep = _get_episode(project_id, episode_number, session)
    return _episode_detail(ep, session)


@router.post("/{project_id}/episodes/{episode_number}/generate", status_code=202)
def generate_episode(
    project_id: int,
    episode_number: int,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> dict:
    """
    Trigger LangGraph generation for a single episode.

    Returns 202 immediately.  Use the /stream endpoint to watch progress.
    """
    project = _require_project(project_id, session)

    if project.status not in ("ready", "generating", "done"):
        raise HTTPException(
            status_code=409,
            detail=f"Project must be in 'ready' status to generate (currently '{project.status}'). Run ingest first.",
        )

    ep = _get_episode(project_id, episode_number, session)

    if ep.status == "generating":
        raise HTTPException(status_code=409, detail="Episode is already generating")

    background_tasks.add_task(run_generate, project_id, episode_number)

    return {
        "status": "accepted",
        "project_id": project_id,
        "episode_number": episode_number,
        "stream_url": f"/api/projects/{project_id}/episodes/{episode_number}/stream",
    }


@router.get("/{project_id}/episodes/{episode_number}/stream")
async def episode_generation_stream(
    project_id: int,
    episode_number: int,
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """
    SSE stream for live episode generation progress.

    Event types:
      status       — overall status changed (status field)
      progress     — percentage update (pct, step, scenes_done, scenes_total)
      log          — raw node log line (node, message)
      storyboard   — storyboard image ready (scene_number, path)
      video_clip   — video clip ready (scene_number, path, quality_score)
      scene_accuracy — Parallel Web Search accuracy result (scene_number, accuracy_answer)
      warning      — non-fatal issue
      error        — fatal error
      done         — stream closed
    """
    _require_project(project_id, session)
    _get_episode(project_id, episode_number, session)

    channel = generate_channel(project_id, episode_number)
    return StreamingResponse(
        subscribe(channel),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_project(project_id: int, session: Session) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _get_episode(project_id: int, episode_number: int, session: Session) -> Episode:
    ep = session.exec(
        select(Episode).where(
            Episode.project_id == project_id,
            Episode.chapter_number == episode_number,
        )
    ).first()
    if ep is None:
        raise HTTPException(status_code=404, detail=f"Episode {episode_number} not found")
    return ep


def _episode_detail(ep: Episode, session: Session) -> EpisodeDetail:
    scenes = session.exec(
        select(Scene)
        .where(Scene.episode_id == ep.id)
        .order_by(Scene.scene_number)
    ).all()
    scene_summaries = [
        SceneSummary(
            id=sc.id,
            scene_number=sc.scene_number,
            global_scene_number=sc.global_scene_number,
            slug=sc.slug,
            location_name=sc.location_name,
            emotional_beat=sc.emotional_beat,
            status=sc.status,
            storyboard_path=sc.storyboard_path,
            video_path=sc.video_path,
            quality_score=sc.quality_score,
            characters=json.loads(sc.characters_json or "[]"),
        )
        for sc in scenes
    ]
    return EpisodeDetail(
        id=ep.id,
        project_id=ep.project_id,
        chapter_number=ep.chapter_number,
        title=ep.title,
        summary=ep.summary,
        act=ep.act,
        emotional_arc=ep.emotional_arc,
        total_scenes=ep.total_scenes,
        status=ep.status,
        episode_video_path=ep.episode_video_path,
        thumbnail_path=ep.thumbnail_path,
        created_at=ep.created_at,
        updated_at=ep.updated_at,
        scenes=scene_summaries,
    )
