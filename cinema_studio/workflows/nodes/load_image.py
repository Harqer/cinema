"""
cinema_studio/workflows/nodes/load_image.py
─────────────────────────────────────────────────────────────────────────────
NODE: LoadImage
─────────────────────────────────────────────────────────────────────────────
ComfyUI analogy: "Load Image" node — reads the source image from disk and
puts raw bytes + MIME type onto the wire so all downstream nodes can access it
without re-reading the file.

Inputs:  state.image_path
Outputs: state.image_bytes, state.image_mime
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from cinema_studio.workflows.state import VideoWorkflowState


_SUPPORTED_MIMES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def load_image(state: VideoWorkflowState) -> VideoWorkflowState:
    """
    Load the input image from disk.

    Reads state.image_path, detects MIME type, and returns the raw bytes.
    Fails fast with state.error if the file doesn't exist or is unsupported.
    """
    path = Path(state["image_path"])

    if not path.exists():
        return {
            **state,
            "error": f"Image file not found: {path}",
            "status": "failed",
        }

    suffix = path.suffix.lower()
    mime = _SUPPORTED_MIMES.get(suffix)
    if mime is None:
        # Fallback: let mimetypes guess
        guessed, _ = mimetypes.guess_type(str(path))
        mime = guessed or "image/jpeg"

    raw = path.read_bytes()

    return {
        **state,
        "image_bytes": raw,
        "image_mime": mime,
        "status": "running",
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 2),
        "quality_issues": state.get("quality_issues", []),
    }
