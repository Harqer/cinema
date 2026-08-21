"""
api/routers/scenes.py
─────────────────────────────────────────────────────────────────────────────
Scene detail endpoint.

  GET /api/projects/{id}/scenes/{scene_key}

scene_key format: "ch{chapter:03d}_sc{scene:03d}"  e.g. "ch001_sc003"
or fall back to integer scene id.
"""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from api.db import Scene, get_session

router = APIRouter()


class SceneDetail(BaseModel):
    id: int
    episode_id: int
    project_id: int
    chapter_number: int
    scene_number: int
    global_scene_number: int
    slug: str
    location_name: str
    description: str
    emotional_beat: str
    camera_suggestion: str
    dialogue_excerpt: str
    characters: list[str]
    status: str
    script: dict
    storyboard_path: str
    video_path: str
    video_mime: str
    quality_score: float
    retry_count: int
    created_at: datetime
    updated_at: datetime


@router.get("/{project_id}/scenes/{scene_key}", response_model=SceneDetail)
def get_scene(
    project_id: int,
    scene_key: str,
    session: Session = Depends(get_session),
) -> SceneDetail:
    """
    Retrieve full scene detail by scene_key (e.g. "ch001_sc003") or integer id.

    The storyboard_path and video_path are relative to the /output static mount.
    """
    scene = _resolve_scene(project_id, scene_key, session)
    return SceneDetail(
        id=scene.id,
        episode_id=scene.episode_id,
        project_id=scene.project_id,
        chapter_number=scene.chapter_number,
        scene_number=scene.scene_number,
        global_scene_number=scene.global_scene_number,
        slug=scene.slug,
        location_name=scene.location_name,
        description=scene.description,
        emotional_beat=scene.emotional_beat,
        camera_suggestion=scene.camera_suggestion,
        dialogue_excerpt=scene.dialogue_excerpt,
        characters=json.loads(scene.characters_json or "[]"),
        status=scene.status,
        script=json.loads(scene.script_json or "{}"),
        storyboard_path=scene.storyboard_path,
        video_path=scene.video_path,
        video_mime=scene.video_mime,
        quality_score=scene.quality_score,
        retry_count=scene.retry_count,
        created_at=scene.created_at,
        updated_at=scene.updated_at,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_scene(project_id: int, scene_key: str, session: Session) -> Scene:
    """Accept either "ch001_sc003" slug keys or plain integer IDs."""
    # Try slug key format "chXXX_scYYY"
    if scene_key.startswith("ch") and "_sc" in scene_key:
        try:
            parts = scene_key.split("_sc")
            chapter_number = int(parts[0][2:])
            scene_number = int(parts[1])
            scene = session.exec(
                select(Scene).where(
                    Scene.project_id == project_id,
                    Scene.chapter_number == chapter_number,
                    Scene.scene_number == scene_number,
                )
            ).first()
            if scene:
                return scene
        except (ValueError, IndexError):
            pass

    # Fall back to integer id
    try:
        scene_id = int(scene_key)
        scene = session.get(Scene, scene_id)
        if scene and scene.project_id == project_id:
            return scene
    except ValueError:
        pass

    raise HTTPException(status_code=404, detail=f"Scene '{scene_key}' not found")
