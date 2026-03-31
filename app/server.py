"""Shrew HTTP server — FastAPI wrapper around the pipeline.

Endpoints:
    POST /v1/convert         — convert a document to markdown + structured JSON
    POST /v1/convert/stream  — same, with SSE progress streaming
    GET  /health             — readiness probe
"""

import asyncio
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time

import requests
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from .cli import parse_page_range
from .models import PipelineConfig, PipelineResult
from .pipeline import CancelledException, run_pipeline
from .rasterizer import IMAGE_EXTENSIONS, OFFICE_EXTENSIONS
from .progress import ProgressReporter

logger = logging.getLogger("shrew.server")

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class ServerConfig:
    """Server configuration loaded from environment variables."""

    vlm_url: str = "http://localhost:8000"
    vlm_model: str = ""
    api_key: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    vlm_concurrency: int = 4
    pipeline_concurrency: int = 3
    shrew_vllm_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            vlm_url=os.environ.get("VLM_URL", "http://localhost:8000"),
            vlm_model=os.environ.get("VLM_MODEL", ""),
            api_key=os.environ.get("VLM_API_KEY"),
            host=os.environ.get("SHREW_HOST", "0.0.0.0"),
            port=int(os.environ.get("SHREW_PORT", "8080")),
            workers=int(os.environ.get("SHREW_WORKERS", "1")),
            vlm_concurrency=int(os.environ.get("VLM_CONCURRENCY", "4")),
            pipeline_concurrency=int(os.environ.get("PIPELINE_CONCURRENCY", "3")),
            shrew_vllm_url=os.environ.get("SHREW_VLLM_URL"),
        )


# ── Global state ─────────────────────────────────────────────────────────────

_config: Optional[ServerConfig] = None
_figure_converter = None
_shrew_lora_map: Optional[dict] = None
_shrew_lora_format: str = "none"
_vlm_pool = None
_pipeline_sem: Optional[asyncio.Semaphore] = None
_pipeline_gate: Optional[threading.Semaphore] = None


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, clean up on shutdown."""
    global _config, _figure_converter, _vlm_pool, _pipeline_sem, _pipeline_gate

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.getLogger("shrew").setLevel(getattr(logging, log_level, logging.INFO))

    start = time.time()
    _config = ServerConfig.from_env()

    from concurrent.futures import ThreadPoolExecutor
    _vlm_pool = ThreadPoolExecutor(max_workers=_config.vlm_concurrency)
    _pipeline_sem = asyncio.Semaphore(_config.pipeline_concurrency)
    _pipeline_gate = threading.Semaphore(_config.pipeline_concurrency)

    # Auto-detect VLM model name if not set
    if not _config.vlm_model:
        try:
            resp = requests.get(f"{_config.vlm_url.rstrip('/')}/v1/models", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                _config.vlm_model = models[0]["id"]
                logger.info(f"Auto-detected VLM model: {_config.vlm_model}")
        except Exception as e:
            logger.warning(f"Failed to auto-detect VLM model: {e}")

    logger.info("=" * 60)
    logger.info("SHREW SERVER STARTING")
    logger.info(f"  VLM: {_config.vlm_url} / {_config.vlm_model}")
    logger.info(f"  VLM concurrency: {_config.vlm_concurrency}")
    logger.info(f"  Pipeline concurrency: {_config.pipeline_concurrency}")
    logger.info("=" * 60)

    # Initialize heron-101 figure detector
    try:
        from .docling_client import create_figure_converter
        figure_device = os.environ.get("DOCLING_DEVICE", "auto")
        _figure_converter = create_figure_converter(device=figure_device)
        if _figure_converter:
            logger.info("Heron-101 figure detector ready")
        else:
            logger.info("torch/transformers not available — figure detection disabled")
    except Exception as e:
        logger.warning(f"Failed to initialize figure detector: {e}")
        _figure_converter = None

    # Discover LoRA adapters from Shrew vLLM server
    if _config.shrew_vllm_url:
        global _shrew_lora_map, _shrew_lora_format
        url = _config.shrew_vllm_url.rstrip("/")

        # Try vLLM first (adapters exposed as model IDs)
        try:
            resp = requests.get(f"{url}/v1/models", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            lora_models = [m for m in models if m.get("parent")]
            if lora_models:
                _shrew_lora_map = {m["id"]: m["id"] for m in lora_models}
                _shrew_lora_format = "openai"
        except Exception:
            pass

        # Fall back to llama.cpp
        if not _shrew_lora_map:
            try:
                resp = requests.get(f"{url}/lora-adapters", timeout=5)
                resp.raise_for_status()
                adapters = resp.json()
                _shrew_lora_map = {}
                for a in adapters:
                    name = os.path.splitext(os.path.basename(a["path"]))[0]
                    _shrew_lora_map[name] = a["id"]
                _shrew_lora_format = "llamacpp"
            except Exception:
                pass

        if _shrew_lora_map:
            logger.info(f"Shrew vLLM: {_config.shrew_vllm_url} ({_shrew_lora_format} format)")
            logger.info(f"  Adapters: {list(_shrew_lora_map.keys())}")
        else:
            logger.warning(f"Shrew vLLM: {_config.shrew_vllm_url} — no adapters found")

    startup_time = time.time() - start
    logger.info(f"Server ready in {startup_time:.1f}s")
    logger.info("=" * 60)

    yield

    if _vlm_pool:
        _vlm_pool.shutdown(wait=True)
    logger.info("Shrew server stopped")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Shrew",
    description="Document to markdown + structured JSON",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


_SUPPORTED_EXTENSIONS = {".pdf"} | IMAGE_EXTENSIONS | OFFICE_EXTENSIONS


def _save_upload(upload: UploadFile) -> str:
    """Save uploaded file to a temp path and return the path."""
    suffix = os.path.splitext(upload.filename or "doc.pdf")[1].lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}",
        )
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        total = 0
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                os.close(fd)
                os.unlink(path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large (max {MAX_FILE_SIZE // 1024 // 1024} MB)",
                )
            os.write(fd, chunk)
    except HTTPException:
        raise
    except Exception:
        os.close(fd)
        os.unlink(path)
        raise
    else:
        os.close(fd)
    return path


def _resolve_model(model_field: Optional[str]) -> tuple[str, str, Optional[str]]:
    """Resolve model from request or use server default."""
    return _config.vlm_url, model_field or _config.vlm_model, _config.api_key


@app.post("/v1/convert")
async def convert(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    pages: Optional[str] = Form(None),
    skip_stage3: bool = Form(False),
    high_dpi: int = Form(200),
):
    """Convert a document to markdown + structured JSON."""
    if _config is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    vlm_url, vlm_model, api_key = _resolve_model(model)

    if not vlm_model:
        raise HTTPException(
            status_code=400,
            detail="No model specified. Set VLM_MODEL or pass model field.",
        )

    page_range = None
    if pages:
        try:
            page_range = parse_page_range(pages)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid page range: {pages}. Use '1-5' or '3'.",
            )

    tmp_path = _save_upload(file)
    try:
        output_dir = tempfile.mkdtemp(prefix="shrew_")
    except Exception:
        os.unlink(tmp_path)
        raise

    async with _pipeline_sem:
        try:
            loop = asyncio.get_running_loop()
            config = PipelineConfig(
                vlm_url=vlm_url,
                vlm_model=vlm_model,
                api_key=api_key,
                high_dpi=high_dpi,
                vlm_concurrency=_config.vlm_concurrency,
                skip_stage3=skip_stage3,
                page_range=page_range,
                accurate=not _config.shrew_vllm_url,
                shrew_vllm_url=_config.shrew_vllm_url,
                shrew_lora_map=_shrew_lora_map,
                shrew_lora_format=_shrew_lora_format,
                shrew_async_stage3=os.environ.get("SHREW_ASYNC_STAGE3", "").lower() in ("1", "true", "yes"),
                section_max_tokens=int(os.environ.get("SECTION_MAX_TOKENS", "9000")),
            )

            def _run():
                return run_pipeline(
                    tmp_path, output_dir, config,
                    figure_converter=_figure_converter,
                    vlm_pool=_vlm_pool,
                )

            result = await loop.run_in_executor(None, _run)
            response = _build_response(result, skip_stage3)
            return JSONResponse(content=response)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )
        finally:
            os.unlink(tmp_path)
            shutil.rmtree(output_dir, ignore_errors=True)


@app.post("/v1/convert/stream")
async def convert_stream(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    pages: Optional[str] = Form(None),
    skip_stage3: bool = Form(False),
    high_dpi: int = Form(200),
):
    """Convert a document with SSE streaming progress."""
    if _config is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    vlm_url, vlm_model, api_key = _resolve_model(model)

    if not vlm_model:
        raise HTTPException(
            status_code=400,
            detail="No model specified. Set VLM_MODEL or pass model field.",
        )

    page_range = None
    if pages:
        try:
            page_range = parse_page_range(pages)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid page range: {pages}. Use '1-5' or '3'.",
            )

    tmp_path = _save_upload(file)
    try:
        output_dir = tempfile.mkdtemp(prefix="shrew_")
    except Exception:
        os.unlink(tmp_path)
        raise

    progress = ProgressReporter()

    def run_in_thread():
        _pipeline_gate.acquire()
        try:
            config = PipelineConfig(
                vlm_url=vlm_url,
                vlm_model=vlm_model,
                api_key=api_key,
                high_dpi=high_dpi,
                vlm_concurrency=_config.vlm_concurrency,
                skip_stage3=skip_stage3,
                page_range=page_range,
                accurate=not _config.shrew_vllm_url,
                shrew_vllm_url=_config.shrew_vllm_url,
                shrew_lora_map=_shrew_lora_map,
                shrew_lora_format=_shrew_lora_format,
                shrew_async_stage3=os.environ.get("SHREW_ASYNC_STAGE3", "").lower() in ("1", "true", "yes"),
                section_max_tokens=int(os.environ.get("SECTION_MAX_TOKENS", "9000")),
            )

            result = run_pipeline(
                tmp_path, output_dir, config,
                figure_converter=_figure_converter,
                progress=progress,
                vlm_pool=_vlm_pool,
            )

            response = _build_response(result, skip_stage3)
            progress.emit_complete(response)

        except CancelledException:
            logger.info("Pipeline cancelled (client disconnected)")
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            progress.emit_error(str(e))
        finally:
            _pipeline_gate.release()
            os.unlink(tmp_path)
            shutil.rmtree(output_dir, ignore_errors=True)
            progress.sentinel()

    def _drain_queue():
        while True:
            try:
                return progress.queue.get(timeout=1.0)
            except queue.Empty:
                if progress.is_cancelled():
                    return None

    async def event_generator():
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

        loop = asyncio.get_running_loop()
        try:
            while True:
                event = await loop.run_in_executor(None, _drain_queue)
                if event is None:
                    break
                evt_name = event.pop("event", "progress")
                yield {"event": evt_name, "data": json.dumps(event)}
        except asyncio.CancelledError:
            progress.cancel()
            thread.join(timeout=5)
            raise

    return EventSourceResponse(event_generator())


def _build_response(result: PipelineResult, skip_stage3: bool) -> dict:
    """Build the spec-format JSON response."""
    response = {
        "markdown": result.clean_markdown,
        "images": result.structured_json.get("images", []),
        "processing_log": {
            "total_pages": result.processing_log.get("total_pages", 0),
            "total_figures": result.processing_log.get("total_figures", 0),
            "total_time_seconds": result.processing_log.get("total_time_seconds", 0),
        },
    }

    if not skip_stage3:
        response["metadata"] = result.structured_json.get("metadata", {})
        response["summary"] = result.structured_json.get("summary", "")
        response["semantic_chunks"] = result.structured_json.get("semantic_chunks", [])

    return response
