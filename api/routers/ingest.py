"""
api/routers/ingest.py
─────────────────────────────────────────────────────────────────────────────
Book upload and live ingestion progress endpoints.

  POST /api/projects/{id}/ingest
      Upload a PDF (multipart/form-data field: file).
      Saves the file to disk, launches run_ingest() as a background task,
      returns 202 immediately.

  GET  /api/projects/{id}/ingest/stream
      SSE stream — yields events until ingestion completes.
      Event types: status | progress | research | warning | error | done

Parallel Web Search:
  The ingest_service calls research_comparable_films() and
  research_adaptation_rights() during ingestion.  The results flow back
  here as "research" SSE events so the UI can display them live.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from api.db import Project, get_session
from api.services.event_bus import ingest_channel, subscribe
from api.services.ingest_service import run_ingest

router = APIRouter()

UPLOADS_DIR = Path(os.environ.get("UPLOADS_DIR", "uploads"))
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/{project_id}/ingest", status_code=202)
async def upload_book(
    project_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> dict:
    """
    Accept a book PDF, save it, and kick off background ingestion.

    Returns immediately with 202 Accepted.  The client should open the
    SSE stream endpoint to watch progress.
    """
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status not in ("created", "error"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot re-ingest project in status '{project.status}'",
        )

    # Save uploaded file
    suffix = Path(file.filename or "book.pdf").suffix or ".pdf"
    dest = UPLOADS_DIR / f"project_{project_id}{suffix}"
    content = await file.read()
    dest.write_bytes(content)

    project.book_path = str(dest)
    session.add(project)
    session.commit()

    background_tasks.add_task(run_ingest, project_id, str(dest))

    return {
        "status": "accepted",
        "project_id": project_id,
        "book_path": str(dest),
        "stream_url": f"/api/projects/{project_id}/ingest/stream",
    }


@router.get("/{project_id}/ingest/stream")
async def ingest_stream(
    project_id: int,
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """
    Server-Sent Events stream for live ingestion progress.

    Yields until a ``{"type":"done"}`` event is published or the client
    disconnects.
    """
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    channel = ingest_channel(project_id)

    return StreamingResponse(
        subscribe(channel),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
