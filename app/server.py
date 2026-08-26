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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from sse_starlette.sse import EventSourceResponse

from .cli import parse_page_range
from .models import PipelineConfig, PipelineResult
from .pipeline import CancelledException, run_pipeline
from .rasterizer import (
    CSV_EXTENSION,
    IMAGE_EXTENSIONS,
    OFFICE_EXTENSIONS,
    SPREADSHEET_EXTENSIONS,
    TEXT_EXTENSIONS,
    classify_file,
)
from .progress import ProgressReporter
from .structured_page import (
    EMPTY_200_ALERT_THRESHOLD,
    MAX_MODEL_LEN,
    MAX_TOKENS,
    STREAM_GUARD_CONSECUTIVE,
    STREAM_GUARD_ENABLED,
    STREAM_GUARD_RATIO,
    STREAM_GUARD_WINDOW_CHARS,
    check_output_budget,
    empty_200_streak,
    warn_if_budget_unreachable,
)
from .structured_pipeline import IMAGE_TRANSFORM, run_structured_pipeline
from .ui import UI_HTML
from .vlm_client import VLMClient

logger = logging.getLogger("shrew.server")

# Upload byte cap, env-tunable. Enforced twice: a Content-Length middleware
# rejects before the framework spools the body, and _save_upload counts bytes
# as a backstop for chunked bodies with no declared length.
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "100"))
MAX_FILE_SIZE = int(MAX_UPLOAD_MB * 1024 * 1024)


# ── Adapter constants ───────────────────────────────────────────────────────

UNIFIED_ADAPTER_NAMES = ("doc_processing", "shrew")
STAGE3_TASKS = ("extract_metadata", "summarize_document", "semantic_chunk")


def _normalize_unified_adapter(adapter_map: dict) -> Optional[tuple[dict, str]]:
    """Expand a unified adapter (one entry named `doc_processing` or `shrew`)
    into a per-task map. Returns (expanded_map, detected_name) or None."""
    for name in UNIFIED_ADAPTER_NAMES:
        if name in adapter_map:
            expanded = {task: adapter_map[name] for task in STAGE3_TASKS}
            return expanded, name
    return None


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
# Context length the VLM is actually serving, read from /v1/models at startup.
# None means the probe failed; the configured constant is used as a fallback.
_served_max_model_len: Optional[int] = None


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, clean up on shutdown."""
    global _config, _figure_converter, _vlm_pool, _pipeline_sem, _pipeline_gate

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    shrew_logger = logging.getLogger("shrew")
    shrew_logger.setLevel(getattr(logging, log_level, logging.INFO))
    if not shrew_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s  %(message)s"))
        shrew_logger.addHandler(handler)

    start = time.time()
    _config = ServerConfig.from_env()
    _sweep_stale_tmp()

    # Every throttle in this process (VLM gate, rasterize lock, pipeline
    # semaphores, the /health empty-200 streak) is PER-PROCESS: uvicorn
    # spawns workers rather than forking, so nothing is shared. Effective
    # concurrency multiplies by the worker count (audit issue #14).
    workers = int(os.environ.get("SHREW_WORKERS", "1") or 1)
    if workers > 1:
        logger.warning(
            f"SHREW_WORKERS={workers}: concurrency gates are per-worker — "
            f"effective VLM/pipeline concurrency is {workers}x the configured "
            f"values and the rasterization memory throttle is per-worker. "
            f"Size VLM_CONCURRENCY/PIPELINE_CONCURRENCY accordingly.")

    from concurrent.futures import ThreadPoolExecutor
    _vlm_pool = ThreadPoolExecutor(max_workers=_config.vlm_concurrency)
    _pipeline_sem = asyncio.Semaphore(_config.pipeline_concurrency)
    _pipeline_gate = threading.Semaphore(_config.pipeline_concurrency)

    # Auto-detect the VLM model name and the served context length. The context
    # length is not cosmetic: `max_model_len` bounds prompt + output together, so
    # a server started with too small a value silently truncates large-bucket
    # pages below the configured max_tokens (§3.1 — misdiagnosed as model
    # truncation twice). Check the real server, not our own constant.
    global _served_max_model_len
    try:
        resp = requests.get(f"{_config.vlm_url.rstrip('/')}/v1/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            if not _config.vlm_model:
                _config.vlm_model = models[0]["id"]
                logger.info(f"Auto-detected VLM model: {_config.vlm_model}")
            served = next((m.get("max_model_len") for m in models
                           if m.get("id") == _config.vlm_model), None)
            _served_max_model_len = served or models[0].get("max_model_len")
    except Exception as e:
        logger.warning(f"Failed to query VLM /v1/models: {e}")

    warn_if_budget_unreachable(_served_max_model_len or MAX_MODEL_LEN)

    logger.info("=" * 60)
    logger.info("SHREW SERVER STARTING")
    logger.info(f"  VLM: {_config.vlm_url} / {_config.vlm_model}")
    logger.info(f"  Context: max_model_len={_served_max_model_len or 'unknown'} "
                f"max_tokens={MAX_TOKENS} transform={IMAGE_TRANSFORM}")
    logger.info(f"  VLM concurrency: {_config.vlm_concurrency}")
    logger.info(f"  Pipeline concurrency: {_config.pipeline_concurrency}")
    logger.info("=" * 60)

    # The heron-101 figure detector is a LEGACY-pipeline dependency only —
    # the structured path gets figure bboxes from the OCR model itself. Load
    # it lazily on the first conventional/vlm request instead of paying ~5s
    # and several GB of RAM at startup for a model the default path never uses.
    logger.info("Figure detector: lazy (loads on first legacy-mode request)")

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
            logger.info(f"Doc Processing model: {_config.shrew_vllm_url} ({_shrew_lora_format} format)")
            logger.info(f"  Adapters: {list(_shrew_lora_map.keys())}")
            normalized = _normalize_unified_adapter(_shrew_lora_map)
            if normalized is not None:
                _shrew_lora_map, detected = normalized
                logger.info(f"  Unified adapter detected: {detected} → routing all Stage 3 tasks")
            else:
                logger.warning(
                    f"  No unified adapter ({' or '.join(UNIFIED_ADAPTER_NAMES)}) "
                    f"found among {list(_shrew_lora_map.keys())} — falling back to main VLM"
                )
                _shrew_lora_map = None
        else:
            logger.warning(f"Doc Processing model: {_config.shrew_vllm_url} — no adapters found")

    # Background readiness refresher (audit issue #17): keeps a readiness stamp
    # warm so the per-request gate consults the cache instead of running an
    # inference that would queue behind real OCR under load and false-503.
    # Runs the deep probe (optionally incl. a tiny inference) off the request
    # path, so admission never pays for it.
    from .vlm_client import READINESS_TTL_S

    async def _readiness_refresher():
        interval = max(5.0, READINESS_TTL_S / 2)
        loop = asyncio.get_running_loop()
        while True:
            try:
                probe = VLMClient(base_url=_config.vlm_url, model=_config.vlm_model,
                                  api_key=_config.api_key)
                await loop.run_in_executor(None, probe.probe)
                if _config.shrew_vllm_url:
                    sp = VLMClient(base_url=_config.shrew_vllm_url, model="Qwen3.5-2B")
                    await loop.run_in_executor(None, sp.probe)
            except Exception as e:
                logger.warning(f"Readiness refresher error: {e}")
            await asyncio.sleep(interval)

    _readiness_task = asyncio.ensure_future(_readiness_refresher())

    startup_time = time.time() - start
    logger.info(f"Server ready in {startup_time:.1f}s")
    logger.info("=" * 60)

    yield

    _readiness_task.cancel()
    if _vlm_pool:
        _vlm_pool.shutdown(wait=True)
    logger.info("Shrew server stopped")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Shrew",
    description="Document to markdown + structured JSON",
    version="0.3.4",
    lifespan=lifespan,
)


@app.middleware("http")
async def _reject_oversize_uploads(request, call_next):
    """Reject oversized convert bodies from the Content-Length header, BEFORE
    the framework spools the multipart body to disk (it spools first, so the
    in-handler cap alone still pays the full disk/RAM cost). Chunked bodies
    without a length fall through to _save_upload's byte-count backstop."""
    if request.url.path.startswith("/v1/convert"):
        try:
            declared = int(request.headers.get("content-length", "0"))
        except ValueError:
            declared = 0
        # 1 MiB margin for multipart framing around the file part.
        if declared > MAX_FILE_SIZE + 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={"error": (f"request body {declared // (1024*1024)} MB "
                                   f"exceeds MAX_UPLOAD_MB={MAX_UPLOAD_MB:.0f}")},
            )
    return await call_next(request)


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
async def ui():
    """Built-in conversion viewer: upload a document, watch progress, view
    the result as rendered document / markdown / JSON. Self-contained page,
    no external assets."""
    return HTMLResponse(UI_HTML)


def _fallback_health() -> dict:
    from . import fallback as fb
    if not fb.enabled():
        return {"enabled": False}
    return {"enabled": True, "model": fb.FALLBACK_MODEL or "(auto)",
            "glyph_target": fb.FALLBACK_GLYPH_TARGET,
            "max_long_edge": fb.FALLBACK_MAX_LONG_EDGE}


@app.get("/health")
async def health():
    unavailable = []
    vlm = VLMClient(base_url=_config.vlm_url, model=_config.vlm_model, api_key=_config.api_key)
    if not vlm.is_ready():
        unavailable.append("vlm")
    if _config.shrew_vllm_url:
        shrew_vlm = VLMClient(base_url=_config.shrew_vllm_url, model="Qwen3.5-2B")
        if not shrew_vlm.is_ready():
            unavailable.append("shrew_vlm")
    if unavailable:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "unavailable": unavailable},
        )

    # §5.3 empty-200 watch: consecutive blank completions returned with HTTP
    # 200 are the TP-rank-desync signature. The model server answers health
    # checks normally while producing nothing, so this is the only signal.
    streak = empty_200_streak()
    if streak >= EMPTY_200_ALERT_THRESHOLD:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "empty_200_streak": streak,
                     "detail": "consecutive empty completions — restart the model server"},
        )

    # Serving-contract config. `capacity.ok` false means the served
    # max_model_len cannot reach the configured max_tokens on some tile bucket:
    # those pages will be truncated silently, and the truncation looks exactly
    # like a loop downstream. That is a misconfiguration, not a runtime fault,
    # so it degrades the probe too.
    capacity = check_output_budget(_served_max_model_len or MAX_MODEL_LEN)
    body = {
        "status": "ok",
        "empty_200_streak": streak,
        "image_transform": IMAGE_TRANSFORM,
        "capacity": capacity,
        "loop_guard": {
            "enabled": STREAM_GUARD_ENABLED,
            "window_chars": STREAM_GUARD_WINDOW_CHARS,
            "ratio": STREAM_GUARD_RATIO,
            "consecutive": STREAM_GUARD_CONSECUTIVE,
        },
        "fallback": _fallback_health(),
        "vlm_readiness": vlm.readiness_snapshot(),
    }
    if not capacity["ok"]:
        body["status"] = "degraded"
        return JSONResponse(status_code=503, content=body)
    return body


_SUPPORTED_EXTENSIONS = (
    {".pdf"} | IMAGE_EXTENSIONS | OFFICE_EXTENSIONS
    | SPREADSHEET_EXTENSIONS | TEXT_EXTENSIONS | {CSV_EXTENSION}
)
_SKIP_CHUNKING_CLASSES = {"spreadsheet", "csv"}
# Every supported class now has a modality under the shrew-ocr-preview
# contract: pdf/image/office rasterize to the image arm, text/csv/spreadsheet
# go through their deterministic extractor into the text arm.
_STRUCTURED_ELIGIBLE_CLASSES = {"pdf", "image", "office",
                                "text", "csv", "spreadsheet"}
# "raw" is the structured pipeline rendered as flat text instead of JSON.
_STRUCTURED_MODES = {"structured", "raw"}


_figure_converter_lock = threading.Lock()
_figure_converter_loaded = False


def _get_figure_converter():
    """Load the legacy heron-101 figure detector on first use (thread-safe)."""
    global _figure_converter, _figure_converter_loaded
    if _figure_converter_loaded:
        return _figure_converter
    with _figure_converter_lock:
        if _figure_converter_loaded:
            return _figure_converter
        try:
            from .docling_client import create_figure_converter
            device = os.environ.get("DOCLING_DEVICE", "auto")
            _figure_converter = create_figure_converter(device=device)
            logger.info("Heron-101 figure detector loaded (lazy)"
                        if _figure_converter else
                        "torch/transformers unavailable — figure detection disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize figure detector: {e}")
            _figure_converter = None
        _figure_converter_loaded = True
    return _figure_converter


def _sweep_stale_tmp(max_age_h: float | None = None) -> int:
    """Remove stale shrew_* conversion dirs / uploads / LO profiles in tmp.

    Normal requests clean up in their finally blocks — but an OOM-killed
    worker, a container restart, or a crash mid-conversion leaks the whole
    tree (upload + per-page PNGs, easily GBs per incident) and nothing else
    ever deletes it. Runs once at startup; age threshold keeps live
    conversions of other workers safe.
    """
    if max_age_h is None:
        max_age_h = float(os.environ.get("SHREW_TMP_MAX_AGE_H", "24"))
    cutoff = time.time() - max_age_h * 3600
    tmp = tempfile.gettempdir()
    removed = 0
    try:
        names = os.listdir(tmp)
    except OSError:
        return 0
    for name in names:
        if not name.startswith(("shrew_", "shrew_up_", "shrew_lo_")):
            continue
        path = os.path.join(tmp, name)
        try:
            if os.path.getmtime(path) > cutoff:
                continue
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.unlink(path)
            removed += 1
        except OSError:
            continue
    if removed:
        logger.info(f"Swept {removed} stale temp artifact(s) older than {max_age_h:.0f}h")
    return removed


def _save_upload(upload: UploadFile) -> str:
    """Save uploaded file to a temp path and return the path."""
    suffix = os.path.splitext(upload.filename or "doc.pdf")[1].lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}",
        )
    # shrew_up_ prefix makes stale uploads sweepable at startup (issue #11).
    fd, path = tempfile.mkstemp(prefix="shrew_up_", suffix=suffix)
    total = 0
    try:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=(f"File too large (max {MAX_UPLOAD_MB:.0f} MB; "
                            f"set MAX_UPLOAD_MB to raise)"),
                )
            os.write(fd, chunk)
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
    request: Request,
    file: UploadFile = File(...),
    pipeline_mode: str = Form("structured"),
    model: Optional[str] = Form(None),
    pages: Optional[str] = Form(None),
    format: str = Form("json"),          # "json" | "markdown"
    skip_extraction: bool = Form(False),
    skip_stage3: bool = Form(False),     # deprecated alias of skip_extraction
    high_dpi: int = Form(200),
):
    if format not in ("json", "markdown"):
        raise HTTPException(422, "format must be 'json' or 'markdown'")
    skip_extraction = skip_extraction or skip_stage3  # skip_stage3: deprecated alias
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

    # Pre-flight VLM health check
    vlm = VLMClient(base_url=vlm_url, model=vlm_model, api_key=api_key)
    if not vlm.is_ready():
        return JSONResponse(status_code=503, content={"error": "VLM server unavailable"})
    if _config.shrew_vllm_url:
        shrew_vlm = VLMClient(base_url=_config.shrew_vllm_url, model="Qwen3.5-2B")
        if not shrew_vlm.is_ready():
            return JSONResponse(status_code=503, content={"error": "Shrew VLM server unavailable"})

    tmp_path = _save_upload(file)
    try:
        output_dir = tempfile.mkdtemp(prefix="shrew_")
    except Exception:
        os.unlink(tmp_path)
        raise

    skip_chunking = classify_file(tmp_path) in _SKIP_CHUNKING_CLASSES

    # try encloses the semaphore acquisition: a request cancelled while
    # QUEUED (client gone, server shutdown) must still clean its upload and
    # output dir — with cleanup inside the async-with body, cancel-at-acquire
    # leaked both (audit issue #11).
    # The pipeline is made cancellable via this ProgressReporter (audit issue
    # #16). Two triggers can cancel it:
    #  1. best-effort client-disconnect detection — RELIABLE ONLY behind a
    #     proxy that resets the upstream socket; bare uvicorn does NOT deliver
    #     http.disconnect to a non-streaming handler while it computes the
    #     response. For GUARANTEED mid-request cancellation, use
    #     /v1/convert/stream, whose streamed response lets uvicorn see the
    #     disconnect.
    #  2. an optional hard wall-clock deadline (SHREW_CONVERT_DEADLINE_S, 0 =
    #     off) — the reliable bound on an abandoned or runaway single request.
    #     Left off by default so legitimate large books are never truncated.
    progress = ProgressReporter()
    deadline_s = float(os.environ.get("SHREW_CONVERT_DEADLINE_S", "0") or 0)

    async def _watch_cancel():
        waited = 0.0
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("Client disconnected — cancelling /v1/convert")
                    progress.cancel()
                    return
                if deadline_s and waited >= deadline_s:
                    logger.warning(
                        f"/v1/convert exceeded SHREW_CONVERT_DEADLINE_S={deadline_s:.0f}s "
                        f"— cancelling")
                    progress.cancel()
                    return
                await asyncio.sleep(1.0)
                waited += 1.0
        except asyncio.CancelledError:
            return

    try:
        async with _pipeline_sem:
            loop = asyncio.get_running_loop()
            config = PipelineConfig(
                vlm_url=vlm_url,
                vlm_model=vlm_model,
                api_key=api_key,
                high_dpi=high_dpi,
                vlm_concurrency=_config.vlm_concurrency,
                skip_stage3=skip_extraction,
                skip_chunking=skip_chunking,
                page_range=page_range,
                accurate=not _config.shrew_vllm_url,
                shrew_vllm_url=_config.shrew_vllm_url,
                shrew_lora_map=_shrew_lora_map,
                shrew_lora_format=_shrew_lora_format,
                shrew_async_stage3=os.environ.get("SHREW_ASYNC_STAGE3", "").lower() in ("1", "true", "yes"),
                section_max_tokens=int(os.environ.get("SECTION_MAX_TOKENS", "6000")),
            )

            def _run():
                if pipeline_mode in _STRUCTURED_MODES and classify_file(tmp_path) in _STRUCTURED_ELIGIBLE_CLASSES:
                    return run_structured_pipeline(
                        tmp_path, output_dir, config,
                        raw=(pipeline_mode == "raw"),
                        progress=progress,
                    )
                return run_pipeline(
                    tmp_path, output_dir, config,
                    figure_converter=_get_figure_converter(),
                    vlm_pool=_vlm_pool,
                    progress=progress,
                )

            watcher = asyncio.ensure_future(_watch_cancel())
            try:
                result = await loop.run_in_executor(None, _run)
            finally:
                watcher.cancel()
            response = _build_response(result, skip_extraction)
            if format == "markdown":
                # structured markdown only — same assembly the JSON carries in
                # its "markdown" key, as a text/markdown body
                return PlainTextResponse(response["markdown"],
                                         media_type="text/markdown; charset=utf-8")
            return JSONResponse(content=response)

    except CancelledException:
        # Client went away and the pipeline unwound at a page boundary. The
        # response is moot (nobody is listening); log quietly, don't 500.
        logger.info("Pipeline cancelled before completion (disconnect or deadline)")
        return JSONResponse(status_code=499, content={"error": "conversion cancelled"})
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        shutil.rmtree(output_dir, ignore_errors=True)


@app.post("/v1/convert/stream")
async def convert_stream(
    file: UploadFile = File(...),
    pipeline_mode: str = Form("structured"),
    model: Optional[str] = Form(None),
    pages: Optional[str] = Form(None),
    skip_extraction: bool = Form(False),
    skip_stage3: bool = Form(False),     # deprecated alias of skip_extraction
    high_dpi: int = Form(200),
):
    skip_extraction = skip_extraction or skip_stage3  # skip_stage3: deprecated alias
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

    # Pre-flight VLM health check
    vlm = VLMClient(base_url=vlm_url, model=vlm_model, api_key=api_key)
    if not vlm.is_ready():
        raise HTTPException(status_code=503, detail="VLM server unavailable")
    if _config.shrew_vllm_url:
        shrew_vlm = VLMClient(base_url=_config.shrew_vllm_url, model="Qwen3.5-2B")
        if not shrew_vlm.is_ready():
            raise HTTPException(status_code=503, detail="Shrew VLM server unavailable")

    tmp_path = _save_upload(file)
    try:
        output_dir = tempfile.mkdtemp(prefix="shrew_")
    except Exception:
        os.unlink(tmp_path)
        raise

    skip_chunking = classify_file(tmp_path) in _SKIP_CHUNKING_CLASSES

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
                skip_stage3=skip_extraction,
                skip_chunking=skip_chunking,
                page_range=page_range,
                accurate=not _config.shrew_vllm_url,
                shrew_vllm_url=_config.shrew_vllm_url,
                shrew_lora_map=_shrew_lora_map,
                shrew_lora_format=_shrew_lora_format,
                shrew_async_stage3=os.environ.get("SHREW_ASYNC_STAGE3", "").lower() in ("1", "true", "yes"),
                section_max_tokens=int(os.environ.get("SECTION_MAX_TOKENS", "6000")),
            )

            if pipeline_mode in _STRUCTURED_MODES and classify_file(tmp_path) in _STRUCTURED_ELIGIBLE_CLASSES:
                result = run_structured_pipeline(
                    tmp_path, output_dir, config, progress=progress,
                    raw=(pipeline_mode == "raw"),
                )
            else:
                result = run_pipeline(
                    tmp_path, output_dir, config,
                    figure_converter=_get_figure_converter(),
                    progress=progress,
                    vlm_pool=_vlm_pool,
                )

            response = _build_response(result, skip_extraction)
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
        started = False
        loop = asyncio.get_running_loop()
        try:
            thread.start()
            started = True
            while True:
                event = await loop.run_in_executor(None, _drain_queue)
                if event is None:
                    break
                evt_name = event.pop("event", "progress")
                yield {"event": evt_name, "data": json.dumps(event)}
        except asyncio.CancelledError:
            progress.cancel()
            # NO thread.join here: this except body runs ON the event loop,
            # and the pipeline thread rarely exits within seconds — every
            # disconnect used to freeze ALL requests (SSE, /health, new
            # connections) for the full join timeout. The worker thread owns
            # its cleanup (run_in_thread's finally) and exits at the next
            # page boundary after progress.cancel().
            if not started:
                # Cancelled before the worker existed: nothing else will
                # clean the upload/output dir — do it here (audit issue #11).
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                shutil.rmtree(output_dir, ignore_errors=True)
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

    for key in ("modality", "gates", "failed_pages"):
        if key in result.processing_log:
            response["processing_log"][key] = result.processing_log[key]

    # raw mode returns no structured_json at all: the stage-3 keys are omitted
    # rather than returned empty, so a caller can't mistake "no model ran" for
    # "the model found nothing".
    if not skip_stage3 and result.structured_json:
        response["metadata"] = result.structured_json.get("metadata", {})
        response["summary"] = result.structured_json.get("summary", "")
        response["semantic_chunks"] = result.structured_json.get("semantic_chunks", [])
        response["tables"] = result.structured_json.get("tables", [])

    return response
