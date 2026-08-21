"""
api/routers/projects.py
─────────────────────────────────────────────────────────────────────────────
CRUD endpoints for the top-level Project resource.

  POST   /api/projects        — create a project
  GET    /api/projects        — list all projects (newest first)
  GET    /api/projects/{id}   — get one project with episode list
  PATCH  /api/projects/{id}   — update title/author/genre/style
  DELETE /api/projects/{id}   — delete project + all child rows
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from api.db import Episode, Project, Scene, get_session

router = APIRouter()


# ── Request / Response bodies ─────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    title: str
    author: str = ""
    genre: str = "drama"
    style: str = "cinematic"


class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    genre: Optional[str] = None
    style: Optional[str] = None


class EpisodeSummary(BaseModel):
    id: int
    chapter_number: int
    title: str
    summary: str
    act: int
    total_scenes: int
    status: str
    episode_video_path: str
    thumbnail_path: str


class ProjectDetail(BaseModel):
    id: int
    title: str
    author: str
    genre: str
    style: str
    status: str
    total_chapters: int
    total_scenes: int
    movie_path: str
    created_at: datetime
    updated_at: datetime
    episodes: list[EpisodeSummary] = []


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("", response_model=ProjectDetail, status_code=201)
def create_project(
    body: ProjectCreate,
    session: Session = Depends(get_session),
) -> ProjectDetail:
    project = Project(**body.model_dump())
    session.add(project)
    session.commit()
    session.refresh(project)
    return _project_detail(project, session)


@router.get("", response_model=list[ProjectDetail])
def list_projects(session: Session = Depends(get_session)) -> list[ProjectDetail]:
    projects = session.exec(select(Project).order_by(Project.created_at.desc())).all()
    return [_project_detail(p, session) for p in projects]


@router.get("/{project_id}", response_model=ProjectDetail)
def get_project(
    project_id: int,
    session: Session = Depends(get_session),
) -> ProjectDetail:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _project_detail(project, session)


@router.patch("/{project_id}", response_model=ProjectDetail)
def update_project(
    project_id: int,
    body: ProjectUpdate,
    session: Session = Depends(get_session),
) -> ProjectDetail:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(project, field, value)
    project.updated_at = datetime.utcnow()
    session.add(project)
    session.commit()
    session.refresh(project)
    return _project_detail(project, session)


@router.delete("/{project_id}", status_code=204)
def delete_project(
    project_id: int,
    session: Session = Depends(get_session),
) -> None:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    # Cascade delete episodes + scenes
    episodes = session.exec(select(Episode).where(Episode.project_id == project_id)).all()
    for ep in episodes:
        scenes = session.exec(select(Scene).where(Scene.episode_id == ep.id)).all()
        for sc in scenes:
            session.delete(sc)
        session.delete(ep)
    session.delete(project)
    session.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _project_detail(project: Project, session: Session) -> ProjectDetail:
    episodes = session.exec(
        select(Episode)
        .where(Episode.project_id == project.id)
        .order_by(Episode.chapter_number)
    ).all()
    ep_summaries = [
        EpisodeSummary(
            id=ep.id,
            chapter_number=ep.chapter_number,
            title=ep.title,
            summary=ep.summary,
            act=ep.act,
            total_scenes=ep.total_scenes,
            status=ep.status,
            episode_video_path=ep.episode_video_path,
            thumbnail_path=ep.thumbnail_path,
        )
        for ep in episodes
    ]
    return ProjectDetail(
        id=project.id,
        title=project.title,
        author=project.author,
        genre=project.genre,
        style=project.style,
        status=project.status,
        total_chapters=project.total_chapters,
        total_scenes=project.total_scenes,
        movie_path=project.movie_path,
        created_at=project.created_at,
        updated_at=project.updated_at,
        episodes=ep_summaries,
    )
