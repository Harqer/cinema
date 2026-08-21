"""
api/services/ingest_service.py
─────────────────────────────────────────────────────────────────────────────
Background task: upload PDF → process_book() → populate Episode + Scene rows.

Flow:
  1. Mark project status = "ingesting"
  2. Run process_book() (long-context Gemini call, can take 2–5 min)
  3. Persist BookAnalysis JSON next to the uploaded file
  4. Create one Episode row per Chapter + one Scene row per BookScene
  5. Mark project status = "ready"
  6. Publish SSE events throughout so the browser shows live progress

Parallel Web Search integration:
  After the book is analysed, `research_comparable_films()` and
  `research_adaptation_rights()` are called to enrich the project with
  market intelligence.  These results are stored in the project's analysis
  notes and emitted as SSE events so the UI can show them in real time.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlmodel import Session, select

from api.db import Episode, Project, Scene, engine
from api.services.event_bus import ingest_channel, publish
from cinema_studio.agents.document_processor import BookAnalysis, process_book
from cinema_studio.agents.research import (
    research_adaptation_rights,
    research_comparable_films,
)


def _emit(project_id: int, event_type: str, **kwargs) -> None:
    publish(
        ingest_channel(project_id),
        {"type": event_type, "project_id": project_id, "ts": datetime.utcnow().isoformat(), **kwargs},
    )


def run_ingest(project_id: int, book_path: str) -> None:
    """
    Execute the full ingestion pipeline for *project_id*.

    Designed to run inside a FastAPI BackgroundTask (sync, runs in a
    thread-pool worker so it does not block the event loop).
    """
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project is None:
            return

        # ── 1. Mark ingesting ────────────────────────────────────────────────
        project.status = "ingesting"
        project.updated_at = datetime.utcnow()
        session.add(project)
        session.commit()
        _emit(project_id, "status", status="ingesting", message="Starting book analysis…")

        # ── 2. Run Gemini long-context analysis ──────────────────────────────
        try:
            _emit(project_id, "progress", step="book_analysis", pct=5,
                  message="Sending book to Gemini 2.5 Pro for chapter/scene breakdown…")
            analysis: BookAnalysis = process_book(book_path, title=project.title)
        except Exception as exc:
            project.status = "error"
            project.updated_at = datetime.utcnow()
            session.add(project)
            session.commit()
            _emit(project_id, "error", message=str(exc))
            _emit(project_id, "done")
            return

        _emit(project_id, "progress", step="book_analysis", pct=30,
              message=f"Book analysed: {len(analysis.chapters)} chapters, {analysis.total_scenes} scenes")

        # ── 3. Persist BookAnalysis JSON ─────────────────────────────────────
        analysis_path = Path(book_path).with_suffix(".analysis.json")
        analysis_path.write_text(analysis.model_dump_json(indent=2))
        project.book_analysis_path = str(analysis_path)
        project.total_chapters = len(analysis.chapters)
        project.total_scenes = analysis.total_scenes
        session.add(project)
        session.commit()

        # ── 4. Parallel Web Search: market intelligence ──────────────────────
        _emit(project_id, "progress", step="parallel_research", pct=40,
              message="Running Parallel Web Search: comparable films + adaptation rights…")
        try:
            genre_str = ", ".join(analysis.genre) if analysis.genre else "drama"
            comps = research_comparable_films(analysis.title, genre_str)
            rights = research_adaptation_rights(analysis.title, analysis.author)
            _emit(project_id, "research", step="comparable_films",
                  answer=comps.answer[:1000], sources=comps.sources[:5])
            _emit(project_id, "research", step="adaptation_rights",
                  answer=rights.answer[:1000], sources=rights.sources[:5])
        except Exception as exc:
            # Research is non-fatal — log and continue
            _emit(project_id, "warning", message=f"Parallel research failed (non-fatal): {exc}")

        _emit(project_id, "progress", step="parallel_research", pct=55,
              message="Market research complete")

        # ── 5. Create Episode + Scene rows ───────────────────────────────────
        _emit(project_id, "progress", step="db_populate", pct=60,
              message="Creating episode and scene records…")

        global_scene_num = 0
        for chapter in sorted(analysis.chapters, key=lambda c: c.chapter_number):
            episode = Episode(
                project_id=project_id,
                chapter_number=chapter.chapter_number,
                title=chapter.title,
                summary=chapter.summary,
                act=chapter.act,
                emotional_arc=chapter.emotional_arc,
                total_scenes=len(chapter.scenes),
                status="pending",
            )
            session.add(episode)
            session.flush()  # get episode.id

            for book_scene in sorted(chapter.scenes, key=lambda s: s.scene_number):
                global_scene_num += 1
                chars = [a.character_name for a in book_scene.characters_present]
                scene = Scene(
                    episode_id=episode.id,
                    project_id=project_id,
                    chapter_number=chapter.chapter_number,
                    scene_number=book_scene.scene_number,
                    global_scene_number=global_scene_num,
                    slug=book_scene.slug,
                    location_name=book_scene.location_name,
                    description=book_scene.description,
                    emotional_beat=book_scene.emotional_beat,
                    camera_suggestion=book_scene.camera_suggestion,
                    dialogue_excerpt=book_scene.dialogue_excerpt,
                    characters_json=json.dumps(chars),
                    status="pending",
                )
                session.add(scene)

        project.status = "ready"
        project.updated_at = datetime.utcnow()
        session.add(project)
        session.commit()

        _emit(project_id, "progress", step="db_populate", pct=100,
              message=f"Ready — {len(analysis.chapters)} episodes, {global_scene_num} scenes")
        _emit(project_id, "status", status="ready")
        _emit(project_id, "done")
