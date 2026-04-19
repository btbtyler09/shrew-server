"""CLI for the Shrew document conversion pipeline."""

import argparse
import logging
import os
import sys

from .models import PipelineConfig
from .pipeline import run_pipeline


def parse_page_range(s: str) -> tuple[int, int]:
    """Parse a page range string like '1-5' or '3'."""
    if "-" in s:
        parts = s.split("-", 1)
        start, end = int(parts[0]), int(parts[1])
    else:
        start = end = int(s)
    if start < 1 or end < start:
        raise ValueError(f"Invalid page range: {s}")
    return start, end


def _run_convert(args):
    """Run the document conversion pipeline."""
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    vlm_url = args.vlm_url
    vlm_model = args.vlm_model
    api_key = args.api_key

    if "openrouter" in vlm_url and not api_key:
        print("Error: API key required for OpenRouter. Set VLM_API_KEY or use --api-key.", file=sys.stderr)
        sys.exit(1)

    if not vlm_model:
        print("Error: No model specified. Use --vlm-model or set VLM_MODEL.", file=sys.stderr)
        sys.exit(1)

    page_range = None
    if args.pages:
        try:
            page_range = parse_page_range(args.pages)
        except ValueError:
            print(f"Invalid page range: {args.pages}. Use format '1-5' or '3'.", file=sys.stderr)
            sys.exit(1)

    config = PipelineConfig(
        vlm_url=vlm_url,
        vlm_model=vlm_model,
        api_key=api_key,
        low_dpi=args.low_dpi,
        high_dpi=args.high_dpi,
        vlm_concurrency=args.vlm_concurrency,
        skip_stage3=args.skip_stage3,
        page_range=page_range,
        shrew_vllm_url=args.shrew_vllm_url,
        shrew_async_stage3=args.async_stage3,
        accurate=not args.shrew_vllm_url,
        section_max_tokens=args.section_max_tokens,
    )

    result = run_pipeline(args.input, args.output_dir, config)

    log = result.processing_log
    print(f"\nClean markdown: {args.output_dir}/clean.md")
    print(f"Processing time: {log['total_time_seconds']:.1f}s")
    print(f"Pages: {log['total_pages']}, Figures: {log['total_figures']}")


def _run_serve(args):
    """Start the Shrew HTTP server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: Server dependencies not installed. Run: pip install shrew[server]",
              file=sys.stderr)
        sys.exit(1)

    if args.host:
        os.environ["SHREW_HOST"] = args.host
    if args.port:
        os.environ["SHREW_PORT"] = str(args.port)
    if args.workers:
        os.environ["SHREW_WORKERS"] = str(args.workers)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    host = args.host or os.environ.get("SHREW_HOST", "0.0.0.0")
    port = args.port or int(os.environ.get("SHREW_PORT", "8080"))
    workers = args.workers or int(os.environ.get("SHREW_WORKERS", "1"))

    uvicorn.run(
        "app.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="debug" if args.verbose else "info",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="shrew",
        description="Shrew — document to clean markdown + structured JSON",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Convert subcommand
    convert_parser = subparsers.add_parser(
        "convert", help="Convert a document to markdown",
    )
    convert_parser.add_argument("input", help="Input file path (PDF, image, or office document)")
    convert_parser.add_argument("-o", "--output-dir", required=True, help="Output directory")
    convert_parser.add_argument(
        "--vlm-url",
        default=os.environ.get("VLM_URL", "http://localhost:8000"),
        help="VLM server URL (env: VLM_URL)",
    )
    convert_parser.add_argument(
        "--vlm-model",
        default=os.environ.get("VLM_MODEL"),
        help="VLM model name (env: VLM_MODEL)",
    )
    convert_parser.add_argument(
        "--api-key",
        default=os.environ.get("VLM_API_KEY"),
        help="API key for VLM endpoint (env: VLM_API_KEY)",
    )
    convert_parser.add_argument("--low-dpi", type=int, default=100, help="DPI for overview images")
    convert_parser.add_argument("--high-dpi", type=int, default=200, help="DPI for VLM transcription images")
    convert_parser.add_argument("--vlm-concurrency", type=int, default=4, help="Concurrent VLM calls")
    convert_parser.add_argument(
        "--shrew-vllm-url",
        default=os.environ.get("SHREW_VLLM_URL"),
        help="Shrew vLLM/llama.cpp URL for structured extraction (env: SHREW_VLLM_URL)",
    )
    convert_parser.add_argument("--async-stage3", action="store_true", help="Run structured extraction tasks concurrently")
    convert_parser.add_argument("--skip-stage3", action="store_true", help="Skip structured extraction")
    convert_parser.add_argument("--section-max-tokens", type=int, default=6000, help="Max tokens per section for chunking (default: 6000)")
    convert_parser.add_argument("--pages", type=str, default=None, help="Page range (e.g., '1-5' or '3')")
    convert_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # Serve subcommand
    serve_parser = subparsers.add_parser(
        "serve", help="Start the HTTP server",
    )
    serve_parser.add_argument("--host", default=None, help="Bind address (env: SHREW_HOST)")
    serve_parser.add_argument("--port", type=int, default=None, help="Bind port (env: SHREW_PORT)")
    serve_parser.add_argument("--workers", type=int, default=None, help="Uvicorn workers (env: SHREW_WORKERS)")
    serve_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # Backward compat: bare file path → convert subcommand
    if len(sys.argv) > 1 and sys.argv[1] not in ("convert", "serve", "-h", "--help"):
        sys.argv.insert(1, "convert")

    args = parser.parse_args()

    if args.command == "serve":
        _run_serve(args)
    elif args.command == "convert":
        _run_convert(args)
    else:
        parser.print_help()
