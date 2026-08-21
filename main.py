#!/usr/bin/env python3
"""
main.py — Agentic Cinema CLI

Usage:
    python main.py --book path/to/book.pdf --title "Pride and Prejudice" \
                   --author "Jane Austen" --genre "romance, period drama"

    # Or pipe plain text:
    cat book.txt | python main.py --title "My Novel" --author "Author" --genre "thriller"

    # ADK session mode (interactive):
    python main.py --interactive
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from cinema_studio.orchestrator import run_full_pipeline, run_orchestrator_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic Cinema — Book-to-Movie Production Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--book", type=Path, default=None, help="Path to book PDF or text file")
    parser.add_argument("--title", type=str, required=False, help="Book title")
    parser.add_argument("--author", type=str, default="Unknown", help="Book author")
    parser.add_argument("--genre", type=str, default="drama", help="Genre(s), comma-separated")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory for production artefacts",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive ADK session mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if hasattr(args, "command") and args.command == "movie":
        from cinema_studio.workflows.movie_graph import produce_movie
        print(f"\n🎬  Agentic Cinema — Full Movie Production")
        print(f"   Book:   {args.book or '(stdin)'}")
        print(f"   Title:  {args.title}")
        print(f"   Style:  {args.style}")
        print(f"   Output: {args.output}\n")
        source = str(args.book) if args.book and args.book.exists() else sys.stdin.read()
        result = produce_movie(
            book_source=source,
            book_title=args.title,
            book_author=args.author,
            genre=args.genre,
            style=args.style,
            clip_duration_seconds=getattr(args, "clip_duration", 8),
            output_dir=args.output,
            stream=True,
        )
        movie_path = result.get("movie_path", "")
        clips = result.get("completed_clips", [])
        duration = result.get("movie_duration_seconds", 0)
        print(f"\n✅  Movie complete!")
        print(f"   Output:   {movie_path}")
        print(f"   Clips:    {len(clips)}")
        print(f"   Duration: ~{int(duration)//60}m{int(duration)%60}s")
        return

    if args.interactive:
        print("Agentic Cinema — Interactive Studio Mode")
        print("Type your production request (e.g. 'Adapt Dune by Frank Herbert'):")
        message = input("> ").strip()
        if not message:
            print("No message provided. Exiting.")
            sys.exit(1)
        response = asyncio.run(run_orchestrator_session(message))
        print("\n" + response)
        return

    # ── Pipeline mode ──────────────────────────────────────────────────────────
    if not args.title:
        print("Error: --title is required for pipeline mode.", file=sys.stderr)
        sys.exit(1)

    if args.book and args.book.exists():
        source = args.book
        print(f"Loading book from: {source}")
    elif not sys.stdin.isatty():
        source = sys.stdin.read()
        print("Reading book from stdin...")
    else:
        # Demo mode — use a short story synopsis as stand-in text
        source = (
            f"[Demo mode] This is a placeholder for the full text of '{args.title}' "
            f"by {args.author}. In production, provide --book path or pipe text via stdin."
        )
        print("⚠  No book source provided — running in demo mode with placeholder text.")

    bible = run_full_pipeline(
        book_source=source,
        book_title=args.title,
        book_author=args.author,
        genre=args.genre,
        output_dir=args.output,
    )

    print(f"\n📽  Production complete!")
    print(f"   Scenes: {bible['screenplay']['scene_count']}")
    print(f"   Locations: {bible['location_package']['total_unique_locations']}")
    print(f"   Audio assets: {bible['audio_package']['total_assets']}")
    print(f"   Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
