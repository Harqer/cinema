"""
api/routers/stream.py
─────────────────────────────────────────────────────────────────────────────
Generic SSE re-export endpoints — convenience aliases that forward directly
to the per-resource stream routes already wired in ingest.py and episodes.py.

These exist so the Next.js frontend can use a single URL pattern:
  /api/stream/ingest/{project_id}
  /api/stream/generate/{project_id}/{episode_number}

Both delegate to the event_bus.subscribe() generator on the matching channel.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from api.db import Episode, Project, get_session
from api.services.event_bus import generate_channel, ingest_channel, subscribe

router = APIRouter()


@router.get("/stream/ingest/{project_id}")
async def stream_ingest(
    project_id: int,
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """SSE stream alias for book ingestion progress."""
    if session.get(Project, project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    return StreamingResponse(
        subscribe(ingest_channel(project_id)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/stream/generate/{project_id}/{episode_number}")
async def stream_generate(
    project_id: int,
    episode_number: int,
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """SSE stream alias for episode generation progress."""
    from sqlmodel import select
    ep = session.exec(
        select(Episode).where(
            Episode.project_id == project_id,
            Episode.chapter_number == episode_number,
        )
    ).first()
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")

    return StreamingResponse(
        subscribe(generate_channel(project_id, episode_number)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
