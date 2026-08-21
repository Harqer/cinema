"""
api/db.py
─────────────────────────────────────────────────────────────────────────────
Lightweight SQLite database via SQLModel.

Schema:
  Project   — one per book
  Episode   — one per chapter (many-to-one Project)
  Scene     — one per BookScene  (many-to-one Episode)

In production swap SQLite for Postgres by changing DATABASE_URL.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlmodel import Field, Relationship, Session, SQLModel, create_engine

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./cinema.db")
engine = create_engine(DATABASE_URL, echo=False)


# ── Models ────────────────────────────────────────────────────────────────────

class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    author: str = ""
    genre: str = "drama"
    style: str = "cinematic"
    book_path: str = ""          # path to uploaded PDF on disk
    book_analysis_path: str = "" # path to saved BookAnalysis JSON
    status: str = "created"      # created | ingesting | ready | generating | done | error
    total_chapters: int = 0
    total_scenes: int = 0
    movie_path: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    episodes: list["Episode"] = Relationship(back_populates="project")


class Episode(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    chapter_number: int          # 1-based
    title: str
    summary: str = ""
    act: int = 1                 # film act (1/2/3)
    emotional_arc: str = ""
    total_scenes: int = 0
    status: str = "pending"      # pending | generating | done | error
    episode_video_path: str = "" # assembled episode MP4
    thumbnail_path: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    project: Optional[Project] = Relationship(back_populates="episodes")
    scenes: list["Scene"] = Relationship(back_populates="episode")


class Scene(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    episode_id: int = Field(foreign_key="episode.id")
    project_id: int
    chapter_number: int
    scene_number: int            # within chapter
    global_scene_number: int = 0 # across entire book
    slug: str = ""
    location_name: str = ""
    description: str = ""
    emotional_beat: str = ""
    camera_suggestion: str = ""
    dialogue_excerpt: str = ""
    characters_json: str = "[]"  # JSON list of character names
    status: str = "pending"      # pending | scripting | storyboarding | generating | done | error | skipped
    script_json: str = "{}"
    storyboard_path: str = ""
    video_path: str = ""
    video_mime: str = "video/mp4"
    quality_score: float = 0.0
    retry_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    episode: Optional[Episode] = Relationship(back_populates="scenes")


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
