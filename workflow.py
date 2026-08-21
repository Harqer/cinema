#!/usr/bin/env python3
"""
workflow.py — Agentic Cinema Image-to-Video Workflow CLI

Usage:
    python workflow.py --image photo.jpg --style cinematic --duration 8

    python workflow.py \\
        --image landscape.png \\
        --prompt "A lone hiker crests a ridge at golden hour" \\
        --style documentary \\
        --duration 10 \\
        --output output/videos \\
        --stream

    # Visualise the graph structure (no API calls):
    python workflow.py --visualize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Cinema — LangGraph Image-to-Video Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", type=Path, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="", help="Optional user direction")
    parser.add_argument(
        "--style",
        type=str,
        default="cinematic",
        choices=["cinematic", "documentary", "surreal", "noir", "horror", "romantic"],
        help="Visual style preset",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Target video duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/videos"),
        help="Output directory for generated video",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max quality-gate retries (default: 2)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print node-by-node progress as the graph executes",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Print the graph structure and exit (no API calls)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.visualize:
        from cinema_studio.workflows.image_to_video import build_graph
        app = build_graph()
        # LangGraph provides a Mermaid diagram via .get_graph().draw_mermaid()
        try:
            print(app.get_graph().draw_mermaid())
        except Exception:
            print("Graph nodes:", list(app.get_graph().nodes.keys()))
        return

    if not args.image:
        print("Error: --image is required. Use --visualize to inspect the graph.", file=sys.stderr)
        sys.exit(1)

    if not args.image.exists():
        print(f"Error: image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    from cinema_studio.workflows.image_to_video import run_workflow

    print(f"\n🎬  Agentic Cinema — Image-to-Video Workflow")
    print(f"   Image:    {args.image}")
    print(f"   Style:    {args.style}")
    print(f"   Duration: {args.duration}s")
    print(f"   Output:   {args.output}")
    print()

    result = run_workflow(
        image_path=args.image,
        output_dir=args.output,
        user_prompt=args.prompt,
        style=args.style,
        duration_seconds=args.duration,
        max_retries=args.max_retries,
        stream=args.stream,
    )

    status = result.get("status", "unknown")
    video_path = result.get("video_path", "")
    score = result.get("quality_score", 0.0)
    issues = result.get("quality_issues", [])

    print()
    if status == "complete":
        print(f"✅  Video generated successfully!")
        print(f"   Path:          {video_path}")
        print(f"   Quality score: {score:.2f}")
    elif status == "failed":
        print(f"❌  Workflow failed.")
        error = result.get("error", "")
        if error:
            print(f"   Error: {error}")
    else:
        print(f"⚠  Workflow ended with status: {status}")
        if video_path:
            print(f"   Last output: {video_path}")

    if issues:
        print("   Quality issues logged:")
        for issue in issues:
            print(f"     • {issue}")


if __name__ == "__main__":
    main()
