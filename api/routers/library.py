"""
api/routers/library.py
─────────────────────────────────────────────────────────────────────────────
Library endpoints — browse completed movies.

  GET /api/library           — list all projects with status="done"
  GET /api/library/{id}      — movie detail with all episodes
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from api.db import Episode, Project, get_session

router = APIRouter()


class MovieCard(BaseModel):
    id: int
    title: str
    author: str
    genre: str
    style: str
    total_chapters: int
    total_scenes: int
    movie_path: str
    created_at: datetime
    updated_at: datetime


class EpisodeCard(BaseModel):
    chapter_number: int
    title: str
    act: int
    total_scenes: int
    episode_video_path: str
    thumbnail_path: str


class MovieDetail(BaseModel):
    id: int
    title: str
    author: str
    genre: str
    style: str
    total_chapters: int
    total_scenes: int
    movie_path: str
    created_at: datetime
    updated_at: datetime
    episodes: list[EpisodeCard]


@router.get("", response_model=list[MovieCard])
def list_library(session: Session = Depends(get_session)) -> list[MovieCard]:
    """Return all completed movies, newest first."""
    projects = session.exec(
        select(Project)
        .where(Project.status == "done")
        .order_by(Project.updated_at.desc())
    ).all()
    return [
        MovieCard(
            id=p.id,
            title=p.title,
            author=p.author,
            genre=p.genre,
            style=p.style,
            total_chapters=p.total_chapters,
            total_scenes=p.total_scenes,
            movie_path=p.movie_path,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
        for p in projects
    ]


@router.get("/{project_id}", response_model=MovieDetail)
def get_movie(
    project_id: int,
    session: Session = Depends(get_session),
) -> MovieDetail:
    project = session.get(Project, project_id)
    if project is None or project.status != "done":
        raise HTTPException(status_code=404, detail="Movie not found in library")

    episodes = session.exec(
        select(Episode)
        .where(Episode.project_id == project_id)
        .order_by(Episode.chapter_number)
    ).all()

    ep_cards = [
        EpisodeCard(
            chapter_number=ep.chapter_number,
            title=ep.title,
            act=ep.act,
            total_scenes=ep.total_scenes,
            episode_video_path=ep.episode_video_path,
            thumbnail_path=ep.thumbnail_path,
        )
        for ep in episodes
    ]

    return MovieDetail(
        id=project.id,
        title=project.title,
        author=project.author,
        genre=project.genre,
        style=project.style,
        total_chapters=project.total_chapters,
        total_scenes=project.total_scenes,
        movie_path=project.movie_path,
        created_at=project.created_at,
        updated_at=project.updated_at,
        episodes=ep_cards,
    )
