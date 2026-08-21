"""
cinema_studio/workflows/image_to_video.py
─────────────────────────────────────────────────────────────────────────────
LangGraph StateGraph — Image → High-Fidelity Video Workflow
─────────────────────────────────────────────────────────────────────────────

Graph topology (ComfyUI canvas equivalent):

  START
    │
    ▼
  load_image          (LoadImage node)
    │
    ▼
  analyze_image       (AnalyzeImage node — multimodal function calling)
    │
    ▼
  enhance_prompt      (EnhancePrompt node — Parallel research + LLM rewrite)
    │
    ▼
  generate_video ◄────────────────────────────────────────┐
    │                                                      │
    ▼                                                      │
  validate_output ── quality_gate() ── RETRY ─────────────┘
    │
    ▼ (passes OR max_retries hit)
   END

The conditional edge `quality_gate` routes:
  - "retry"  → enhance_prompt  (appends quality issues to the prompt)
  - "end"    → END             (pass or exhausted retries)
  - "failed" → END             (hard error)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from cinema_studio.workflows.state import VideoWorkflowState
from cinema_studio.workflows.nodes import (
    load_image,
    analyze_image,
    enhance_prompt,
    generate_video,
    validate_output,
)


# ── Conditional routing function ──────────────────────────────────────────────

def quality_gate(
    state: VideoWorkflowState,
) -> Literal["retry", "end", "failed"]:
    """
    Decide what happens after ValidateOutput:

    - "retry"  → loop back to enhance_prompt so the prompt can be adjusted
                 based on the quality issues, then regenerate.
    - "end"    → the video passed QA — write the manifest and exit.
    - "failed" → hard error or retries exhausted — exit with failure status.
    """
    status = state.get("status", "running")

    if status == "complete":
        return "end"
    if status == "failed":
        return "failed"

    # status == "running" → still retrying
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if retry_count >= max_retries:
        return "failed"

    return "retry"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(checkpointer: MemorySaver | None = None) -> StateGraph:
    """
    Construct and compile the image-to-video LangGraph StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistence/resume.
                      Defaults to in-memory (MemorySaver).

    Returns:
        Compiled LangGraph app ready for .invoke() / .stream().
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(VideoWorkflowState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("load_image", load_image)
    graph.add_node("analyze_image", analyze_image)
    graph.add_node("enhance_prompt", enhance_prompt)
    graph.add_node("generate_video", generate_video)
    graph.add_node("validate_output", validate_output)

    # ── Deterministic edges (like fixed wires in ComfyUI) ─────────────────────
    graph.add_edge(START, "load_image")
    graph.add_edge("load_image", "analyze_image")
    graph.add_edge("analyze_image", "enhance_prompt")
    graph.add_edge("enhance_prompt", "generate_video")
    graph.add_edge("generate_video", "validate_output")

    # ── Conditional edge: quality gate (the retry loop) ───────────────────────
    graph.add_conditional_edges(
        "validate_output",
        quality_gate,
        {
            "retry": "enhance_prompt",   # loop back: fix prompt → regenerate
            "end": END,
            "failed": END,
        },
    )

    return graph.compile(checkpointer=checkpointer)


# ── High-level runner ─────────────────────────────────────────────────────────

def run_workflow(
    image_path: str | Path,
    output_dir: str | Path = "output/videos",
    user_prompt: str = "",
    style: str = "cinematic",
    duration_seconds: int = 5,
    max_retries: int = 2,
    thread_id: str = "default",
    stream: bool = False,
) -> VideoWorkflowState:
    """
    Run the complete image-to-video LangGraph workflow.

    Args:
        image_path:       Path to the source image file.
        output_dir:       Directory to write the generated video.
        user_prompt:      Optional text direction from the user.
        style:            Visual style — "cinematic", "documentary", "surreal", etc.
        duration_seconds: Target video length in seconds.
        max_retries:      Max regeneration attempts on quality failure.
        thread_id:        LangGraph thread ID for checkpoint isolation.
        stream:           If True, print node updates as they execute.

    Returns:
        Final :class:`VideoWorkflowState` after the graph completes.
    """
    app = build_graph()

    initial_state: VideoWorkflowState = {
        "image_path": str(image_path),
        "output_dir": str(output_dir),
        "user_prompt": user_prompt,
        "style": style,
        "duration_seconds": duration_seconds,
        "max_retries": max_retries,
        "retry_count": 0,
        "quality_issues": [],
        "status": "running",
    }

    config = {"configurable": {"thread_id": thread_id}}

    if stream:
        final_state: VideoWorkflowState = {}
        for event in app.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_state in event.items():
                status = node_state.get("status", "")
                score = node_state.get("quality_score", "")
                score_str = f" (score={score:.2f})" if isinstance(score, float) else ""
                issues = node_state.get("quality_issues", [])
                print(f"  [{node_name}]{score_str} status={status or 'running'}")
                for issue in issues:
                    print(f"    ⚠  {issue}")
                final_state = {**final_state, **node_state}
        return final_state
    else:
        return app.invoke(initial_state, config=config)
