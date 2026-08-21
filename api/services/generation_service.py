"""
api/services/generation_service.py
─────────────────────────────────────────────────────────────────────────────
Background task: trigger LangGraph produce_movie() scoped to a single episode
(chapter), write live progress events to SSE, and sync outcomes back to DB.

The key connection:
  produce_movie() → LangGraph streaming events → event_bus.publish() →
  SSE endpoint → browser GenerationStatusBar component

Parallel Web Search integration:
  Before kicking off generation, `verify_scene_accuracy()` is called for each
  scene in the episode.  Accuracy metadata is stored in Scene.script_json and
  emitted as an SSE event.  This is the second Parallel touchpoint (after
  ingest_service) in the pipeline.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from sqlmodel import Session, select

from api.db import Episode, Project, Scene, engine
from api.services.event_bus import generate_channel, publish
from cinema_studio.agents.research import verify_scene_accuracy
from cinema_studio.workflows.movie_graph import build_movie_graph
from cinema_studio.memory import get_memory


def _emit(project_id: int, ep_num: int, event_type: str, **kwargs) -> None:
    publish(
        generate_channel(project_id, ep_num),
        {
            "type": event_type,
            "project_id": project_id,
            "episode": ep_num,
            "ts": datetime.utcnow().isoformat(),
            **kwargs,
        },
    )


def _sync_scene_to_db(
    session: Session,
    project_id: int,
    chapter_number: int,
    node_state: dict,
) -> None:
    """Write per-scene artefact paths + status back to the Scene row."""
    scene_number = node_state.get("current_scene", {}).get("scene_number")
    if scene_number is None:
        return

    stmt = select(Scene).where(
        Scene.project_id == project_id,
        Scene.chapter_number == chapter_number,
        Scene.scene_number == scene_number,
    )
    scene = session.exec(stmt).first()
    if scene is None:
        return

    if node_state.get("current_script"):
        scene.script_json = json.dumps(node_state["current_script"])
        scene.status = "scripting"
    if node_state.get("current_storyboard_path"):
        scene.storyboard_path = node_state["current_storyboard_path"]
        scene.status = "storyboarding"
    if node_state.get("current_video_path"):
        scene.video_path = node_state["current_video_path"]
        scene.status = "generating"
    if node_state.get("current_quality_score", 0.0) > 0:
        scene.quality_score = node_state["current_quality_score"]
        if node_state["current_quality_score"] >= 0.68:
            scene.status = "done"
        elif node_state.get("current_retry_count", 0) >= node_state.get("max_clip_retries", 2):
            scene.status = "skipped"
    scene.retry_count = node_state.get("current_retry_count", 0)
    scene.updated_at = datetime.utcnow()
    session.add(scene)


def run_generate(
    project_id: int,
    episode_number: int,  # 1-based chapter number
    chapter_only: bool = True,
) -> None:
    """
    Generate a single episode (chapter) for *project_id*.

    The LangGraph graph is built with the chapter filter baked into the
    initial state.  We stream node events and write progress to SSE +
    back to the DB.
    """
    postgres_uri = os.environ.get("DATABASE_URL") if os.environ.get("DATABASE_URL", "").startswith("postgresql") else None

    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project is None:
            return

        stmt_ep = select(Episode).where(
            Episode.project_id == project_id,
            Episode.chapter_number == episode_number,
        )
        episode = session.exec(stmt_ep).first()
        if episode is None:
            _emit(project_id, episode_number, "error", message="Episode not found")
            _emit(project_id, episode_number, "done")
            return

        # ── Mark generating ──────────────────────────────────────────────────
        episode.status = "generating"
        episode.updated_at = datetime.utcnow()
        project.status = "generating"
        project.updated_at = datetime.utcnow()
        session.add(episode)
        session.add(project)
        session.commit()

        _emit(project_id, episode_number, "status", status="generating",
              message=f"Starting episode {episode_number}: {episode.title}")

        # ── Parallel Web Search: scene accuracy pre-flight ───────────────────
        stmt_scenes = select(Scene).where(
            Scene.project_id == project_id,
            Scene.chapter_number == episode_number,
        ).order_by(Scene.scene_number)
        scenes = session.exec(stmt_scenes).all()

        _emit(project_id, episode_number, "progress", step="parallel_accuracy",
              pct=5, message=f"Running Parallel Web Search accuracy checks on {len(scenes)} scenes…")

        if project.book_analysis_path:
            analysis_json = Path(project.book_analysis_path).read_text()
            analysis_dict = json.loads(analysis_json)
            time_period = analysis_dict.get("time_period", "contemporary")
        else:
            time_period = "contemporary"

        for sc in scenes:
            try:
                result = verify_scene_accuracy(sc.description, time_period)
                existing = json.loads(sc.script_json) if sc.script_json and sc.script_json != "{}" else {}
                existing["accuracy_check"] = {"answer": result.answer[:800], "sources": result.sources[:3]}
                sc.script_json = json.dumps(existing)
                session.add(sc)
                _emit(project_id, episode_number, "scene_accuracy",
                      scene_number=sc.scene_number, slug=sc.slug,
                      accuracy_answer=result.answer[:400])
            except Exception as exc:
                _emit(project_id, episode_number, "warning",
                      message=f"Accuracy check scene {sc.scene_number} failed (non-fatal): {exc}")

        session.commit()
        _emit(project_id, episode_number, "progress", step="parallel_accuracy",
              pct=15, message="Accuracy checks done — starting LangGraph generation")

        # ── Build chapter-scoped LangGraph ───────────────────────────────────
        checkpointer, store = get_memory(postgres_uri=postgres_uri)
        app = build_movie_graph(checkpointer=checkpointer, store=store)

        slug_title = project.title.lower().replace(" ", "_")[:40]
        thread_id = f"{slug_title}_ep{episode_number:03d}"
        output_dir = Path("output") / f"project_{project_id}" / f"ep{episode_number:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load the book analysis for this run
        if not project.book_analysis_path or not Path(project.book_analysis_path).exists():
            _emit(project_id, episode_number, "error", message="book_analysis_path not set; run ingest first")
            _emit(project_id, episode_number, "done")
            return

        initial = {
            "book_source": project.book_path,
            "book_title": project.title,
            "book_author": project.author,
            "genre": project.genre,
            "style": project.style,
            "clip_duration_seconds": 8,
            "output_dir": str(output_dir),
            "max_clip_retries": 2,
            "completed_clips": [],
            "passed_scene_keys": [],
            "style_notes": [],
            "log": [],
            "status": "running",
            # Chapter filter: start iteration at the target chapter
            "current_chapter_index": episode_number - 1,
            "current_scene_index": 0,
            "current_retry_count": 0,
            "current_quality_issues": [],
            # Inject pre-loaded analysis so load_book_node can be skipped
            "_chapter_only": episode_number,
        }

        config = {"configurable": {"thread_id": thread_id}}
        total_scenes = len(scenes)
        done_scenes = 0

        try:
            for event in app.stream(initial, config=config, stream_mode="updates"):
                for node_name, node_state in event.items():
                    log_lines = node_state.get("log", [])
                    for line in log_lines:
                        _emit(project_id, episode_number, "log", node=node_name, message=line)

                    # Sync artefacts to DB
                    _sync_scene_to_db(session, project_id, episode_number, node_state)
                    session.commit()

                    # Compute progress percentage
                    if node_state.get("completed_clips"):
                        done_scenes = len(node_state["completed_clips"])
                        pct = 15 + int(80 * done_scenes / max(total_scenes, 1))
                        _emit(project_id, episode_number, "progress",
                              step="generation", pct=pct,
                              scenes_done=done_scenes, scenes_total=total_scenes,
                              node=node_name)

                    # Emit storyboard + video as they are created
                    if node_state.get("current_storyboard_path"):
                        _emit(project_id, episode_number, "storyboard",
                              scene_number=node_state.get("current_scene", {}).get("scene_number"),
                              path=node_state["current_storyboard_path"])
                    if node_state.get("current_video_path"):
                        _emit(project_id, episode_number, "video_clip",
                              scene_number=node_state.get("current_scene", {}).get("scene_number"),
                              path=node_state["current_video_path"],
                              quality_score=node_state.get("current_quality_score", 0.0))

        except Exception as exc:
            episode.status = "error"
            episode.updated_at = datetime.utcnow()
            session.add(episode)
            session.commit()
            _emit(project_id, episode_number, "error", message=str(exc))
            _emit(project_id, episode_number, "done")
            return

        # ── Mark episode done ────────────────────────────────────────────────
        # Look for assembled episode video
        ep_video = output_dir / "episode.mp4"
        if ep_video.exists():
            episode.episode_video_path = str(ep_video)

        episode.status = "done"
        episode.updated_at = datetime.utcnow()
        session.add(episode)

        # Check if all episodes are done — if so mark project done
        all_episodes = session.exec(
            select(Episode).where(Episode.project_id == project_id)
        ).all()
        if all(ep.status == "done" for ep in all_episodes):
            project.status = "done"
        else:
            project.status = "ready"
        project.updated_at = datetime.utcnow()
        session.add(project)
        session.commit()

        _emit(project_id, episode_number, "status", status="done",
              message=f"Episode {episode_number} complete — {done_scenes} scenes generated")
        _emit(project_id, episode_number, "done")
