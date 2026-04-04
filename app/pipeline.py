"""Pipeline orchestrator — VLM transcription pipeline.

Flow: Prepare pages → per-page (VLM transcription ∥ figure detection) → assemble → structured extraction

Produces: {markdown, images[], metadata, summary, chunks}
"""

import base64
import io
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .generation import get_generation_params
from PIL import Image

from .docling_client import create_figure_converter, detect_figures
from .models import PipelineConfig, PipelineResult
from .prompts import DIRECT_CONVERT_PROMPT, FIGURE_CLASSIFY_PROMPT
from .rasterizer import prepare_pages
from .structured import extract_metadata, generate_summary, semantic_chunk
from .vlm_client import VLMClient, make_image_content, make_text_content

logger = logging.getLogger("shrew.pipeline")

# pypdfium2 uses a C library that is not thread-safe — serialize all rasterization
_rasterize_lock = threading.Lock()

# Padding in pixels added to detected bboxes when cropping figures
FIGURE_CROP_PAD_X = 15
FIGURE_CROP_PAD_Y = 30


class CancelledException(Exception):
    """Raised when client disconnects and processing should stop."""
    pass


class VLMTranscriptionError(Exception):
    """Retryable VLM transcription failure."""
    def __init__(self, message: str, elapsed: float):
        super().__init__(message)
        self.elapsed = elapsed


# ── Adaptive repetition loop detection ─────────────────────────────────────

_METRICS_PATH = Path(__file__).parent / "page_metrics.json"


class PageMetrics:
    """Persistent per-page VLM metrics with adaptive thresholds.

    Persists max_time and max_chars across documents in a JSON file.
    Thresholds linearly tighten from generous defaults to learned values
    over the first 10 documents.
    """

    INITIAL_TIMEOUT = 120.0      # seconds, used for doc 1
    INITIAL_CHAR_LIMIT = 10_000  # chars, used for doc 1
    MARGIN = float(os.environ.get("VLM_TIMEOUT_MARGIN", "1.5"))
    RAMP_DOCS = 10               # docs to fully tighten thresholds

    def __init__(self, vlm_concurrency: int = 4):
        self._lock = threading.Lock()
        self._vlm_concurrency = vlm_concurrency
        self._concurrency_changed = False
        # Persistent state (loaded from / saved to disk)
        self.docs_processed: int = 0
        self.pages_processed: int = 0
        self.max_time_s: float = 0.0
        self.max_chars: int = 0
        # Per-document session state (not persisted)
        self._session_times: list[float] = []
        self._session_sizes: list[int] = []
        self._load()

    def _load(self) -> None:
        try:
            if _METRICS_PATH.exists():
                data = json.loads(_METRICS_PATH.read_text())
                self.docs_processed = data.get("docs_processed", 0)
                self.pages_processed = data.get("pages_processed", 0)
                self.max_time_s = data.get("max_time_s", 0.0)
                self.max_chars = data.get("max_chars", 0)
                stored_concurrency = data.get("vlm_concurrency", 4)
                if stored_concurrency != self._vlm_concurrency and self.max_time_s > 0:
                    scale = self._vlm_concurrency / stored_concurrency
                    logger.info(
                        f"Concurrency changed {stored_concurrency} → {self._vlm_concurrency}, "
                        f"scaling max_time_s {self.max_time_s:.1f}s → {self.max_time_s * scale:.1f}s"
                    )
                    self.max_time_s *= scale
                    self._concurrency_changed = True
                logger.info(
                    f"Loaded page metrics: {self.docs_processed} docs, "
                    f"max_time={self.max_time_s:.1f}s, max_chars={self.max_chars}"
                )
        except Exception as e:
            logger.warning(f"Failed to load page metrics: {e}")

    def _save(self) -> None:
        try:
            data = {
                "docs_processed": self.docs_processed,
                "pages_processed": self.pages_processed,
                "max_time_s": round(self.max_time_s, 2),
                "max_chars": self.max_chars,
                "vlm_concurrency": self._vlm_concurrency,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            _METRICS_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save page metrics: {e}")

    def _lerp(self, initial: float, learned: float) -> float:
        if self.docs_processed >= self.RAMP_DOCS or learned <= 0:
            return learned if learned > 0 else initial
        t = self.docs_processed / self.RAMP_DOCS
        return initial + (learned - initial) * t

    @property
    def timeout(self) -> float:
        with self._lock:
            if self.max_time_s <= 0:
                return self.INITIAL_TIMEOUT
            learned = self.max_time_s * self.MARGIN
            return max(30.0, self._lerp(self.INITIAL_TIMEOUT, learned))

    @property
    def char_threshold(self) -> int:
        with self._lock:
            if self.max_chars <= 0:
                return self.INITIAL_CHAR_LIMIT
            learned = int(self.max_chars * self.MARGIN)
            return max(5000, int(self._lerp(self.INITIAL_CHAR_LIMIT, learned)))

    def record_page(self, elapsed: float, chars: int) -> None:
        with self._lock:
            self._session_times.append(elapsed)
            self._session_sizes.append(chars)

    def is_outlier(self, chars: int) -> bool:
        return chars > self.char_threshold

    def is_time_outlier(self, elapsed: float) -> bool:
        if self.max_time_s <= 0:
            return False
        return elapsed > self.max_time_s * self.MARGIN

    def finish_document(self) -> None:
        with self._lock:
            if not self._session_times:
                return
            session_max_time = max(self._session_times)
            session_max_chars = max(self._session_sizes)
            if self._concurrency_changed:
                # First doc at new concurrency — replace time, don't merge
                self.max_time_s = session_max_time
                self._concurrency_changed = False
            else:
                self.max_time_s = max(self.max_time_s, session_max_time)
            self.max_chars = max(self.max_chars, session_max_chars)
            self.docs_processed += 1
            self.pages_processed += len(self._session_times)
            self._session_times.clear()
            self._session_sizes.clear()
            self._save()
            logger.info(
                f"Updated page metrics: {self.docs_processed} docs, "
                f"max_time={self.max_time_s:.1f}s, max_chars={self.max_chars}"
            )


def _has_repetition_loop(text: str, window: int = 200, min_repeats: int = 3) -> bool:
    """Detect repetitive text loops in VLM output."""
    if len(text) < window * min_repeats:
        return False
    tail = text[-window:]
    return text.count(tail) >= min_repeats


# ── Per-page helpers ────────────────────────────────────────────────────────


def _vlm_transcribe_one_page(
    page_no: int,
    hires_path: Path,
    total_pages: int,
    config: PipelineConfig,
    timeout: int | None = None,
    max_tokens: int = 16384,
) -> tuple[str, float]:
    """Transcribe one page directly from its image via VLM.

    Returns (text, elapsed_seconds).
    """
    logger.info(f"VLM transcribe: starting page {page_no}")
    vlm = VLMClient(
        base_url=config.vlm_url,
        model=config.vlm_model,
        api_key=config.api_key,
    )

    user_content = [
        make_image_content(hires_path, detail="high"),
        make_text_content(f"Page {page_no} of {total_pages}."),
    ]

    params = get_generation_params(config.vlm_model, "transcribe")
    start = time.time()
    try:
        result = vlm.simple_completion(
            system_prompt=DIRECT_CONVERT_PROMPT,
            user_content=user_content,
            max_tokens=max_tokens,
            timeout=timeout,
            **params,
        )
        # Strip markdown fences if the VLM wrapped its output
        text = result.strip()
        if text.startswith("```markdown"):
            text = text[len("```markdown"):].strip()
        if text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        elapsed = time.time() - start
        logger.info(f"VLM transcribe: page {page_no} -> {len(text)} chars in {elapsed:.1f}s")
        return text, elapsed

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"VLM transcribe failed on page {page_no} after {elapsed:.1f}s: {e}")
        raise VLMTranscriptionError(str(e), elapsed) from e


def _classify_figure_crop(
    crop_image: Image.Image,
    config: PipelineConfig,
) -> bool:
    """Classify a cropped region as figure (True) or artifact (False) via VLM."""
    buf = io.BytesIO()
    crop_image.save(buf, format="PNG")
    b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_data}"

    # Use Shrew 2B when available, fall back to main VLM
    if not config.accurate and config.shrew_vllm_url:
        vlm = VLMClient(base_url=config.shrew_vllm_url, model="Qwen3.5-2B")
    else:
        vlm = VLMClient(
            base_url=config.vlm_url,
            model=config.vlm_model,
            api_key=config.api_key,
        )

    user_content = [
        {"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}},
        make_text_content("Classify this cropped region."),
    ]

    params = get_generation_params(vlm.model, "classify_figure")

    # llama.cpp requires LoRA adapters in every request (even at scale 0)
    if config.shrew_lora_format == "llamacpp" and config.shrew_lora_map:
        extra = params.get("extra_params") or {}
        extra["lora"] = [{"id": aid, "scale": 0.0} for aid in config.shrew_lora_map.values()]
        params["extra_params"] = extra
    try:
        result = vlm.simple_completion(
            system_prompt=FIGURE_CLASSIFY_PROMPT,
            user_content=user_content,
            max_tokens=16,
            **params,
        )
        answer = result.strip().lower()
        is_figure = answer.startswith("figure")
        logger.debug(f"Figure classification: {answer} -> {'keep' if is_figure else 'discard'}")
        return is_figure
    except Exception as e:
        logger.warning(f"Figure classification failed: {e} — keeping crop")
        return True


def _crop_and_filter_figures(
    display_path: Path,
    figures: list[dict],
    page_no: int,
    config: PipelineConfig,
    output_dir: str,
) -> list[dict]:
    """Crop detected figure bboxes from the display image, filter via VLM.

    Returns list of image dicts:
        [{"data": "<base64>", "format": "png", "caption": "...", "page": N}, ...]
    """
    if not figures:
        return []

    img = Image.open(display_path)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    results = []
    for i, fig in enumerate(figures):
        bbox = fig["bbox"]
        l = int(bbox["l"])
        t = int(bbox["t"])
        r = int(bbox["r"])
        b = int(bbox["b"])

        # Pad
        l -= FIGURE_CROP_PAD_X
        t -= FIGURE_CROP_PAD_Y
        r += FIGURE_CROP_PAD_X
        b += FIGURE_CROP_PAD_Y

        # Clamp to image bounds
        l = max(0, l)
        t = max(0, t)
        r = min(img.width, r)
        b = min(img.height, b)

        if r <= l or b <= t:
            continue

        crop = img.crop((l, t, r, b))

        if crop.width < 20 or crop.height < 20:
            continue

        is_figure = _classify_figure_crop(crop, config)
        if not is_figure:
            logger.info(f"Page {page_no}: discarded crop {i} as artifact")
            continue

        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

        caption = fig.get("caption", "")

        results.append({
            "data": b64_data,
            "format": "png",
            "caption": caption,
            "page": page_no,
            "bbox": {"l": l, "t": t, "r": r, "b": b},
        })

        crop.save(os.path.join(figures_dir, f"figure_page{page_no}_{i}.png"))

    logger.info(f"Page {page_no}: {len(results)} figures kept from {len(figures)} detected")
    return results


def _process_one_page(
    page_no: int,
    display_path: Path,
    hires_path: Path,
    page_dims: tuple[float, float],
    total_pages: int,
    config: PipelineConfig,
    figure_converter,
    output_dir: str,
    metrics: PageMetrics | None = None,
) -> tuple[int, str, list[dict]]:
    """Process a single page: parallel VLM transcription + figure detection.

    Returns (page_no, markdown, images_list).
    """
    MAX_ERROR_RETRIES = 2
    RETRY_BACKOFF = 2.0  # seconds
    OUTLIER_CONFIRM_RATIO = 0.7  # retry >= 70% of original = confirmed valid

    figures = []
    markdown = ""
    elapsed = 0.0
    error_type = None
    timeout = int(metrics.timeout) if metrics else None

    with ThreadPoolExecutor(max_workers=2) as mini_pool:
        vlm_future = mini_pool.submit(
            _vlm_transcribe_one_page, page_no, hires_path,
            total_pages, config, timeout=timeout,
        )

        if figure_converter is not None:
            fig_future = mini_pool.submit(detect_figures, display_path, figure_converter)
        else:
            fig_future = None

        try:
            markdown, elapsed = vlm_future.result()
        except VLMTranscriptionError as e:
            markdown, elapsed = "", e.elapsed
            error_type = type(e).__name__

        if fig_future is not None:
            try:
                figures = fig_future.result()
            except Exception as e:
                logger.warning(f"Page {page_no}: figure detection failed: {e}")
                figures = []

    # ── Error retry (timeout / bad response / etc) ────────────────────────
    if error_type is not None:
        for attempt in range(1, MAX_ERROR_RETRIES + 1):
            time.sleep(RETRY_BACKOFF * attempt)
            retry_margin = 2.0 * attempt  # attempt 1 → 2x, attempt 2 → 4x
            retry_timeout = int(timeout * retry_margin) if timeout else None
            logger.info(
                f"Page {page_no}: error retry {attempt}/{MAX_ERROR_RETRIES} "
                f"(reason: {error_type})"
            )
            try:
                markdown, elapsed = _vlm_transcribe_one_page(
                    page_no, hires_path, total_pages, config,
                    timeout=retry_timeout,
                )
                error_type = None
                break
            except VLMTranscriptionError as e:
                markdown, elapsed = "", e.elapsed
                error_type = type(e).__name__
                logger.warning(f"Page {page_no}: retry {attempt} failed: {e}")
        if error_type is not None:
            logger.error(
                f"Page {page_no}: all {MAX_ERROR_RETRIES} retries exhausted"
            )

    # ── Repetition loop / outlier detection and retry ─────────────────────
    is_loop = _has_repetition_loop(markdown)
    is_outlier = metrics.is_outlier(len(markdown)) if metrics else False
    is_slow = metrics.is_time_outlier(elapsed) if metrics else False

    if markdown and (is_loop or is_outlier or is_slow):
        reason = (
            "repetition loop" if is_loop
            else f"size outlier ({len(markdown)} > {metrics.char_threshold})" if is_outlier
            else f"time outlier ({elapsed:.1f}s > {metrics.max_time_s * metrics.MARGIN:.1f}s)"
        )
        logger.warning(
            f"Page {page_no}: {reason} detected "
            f"({len(markdown)} chars, {elapsed:.1f}s). Retrying..."
        )
        try:
            retry_text, retry_elapsed = _vlm_transcribe_one_page(
                page_no, hires_path, total_pages, config,
                timeout=timeout, max_tokens=4096,
            )
        except VLMTranscriptionError as e:
            logger.warning(f"Page {page_no}: outlier retry failed: {e}")
            retry_text, retry_elapsed = "", e.elapsed
        if retry_text and len(retry_text) < len(markdown):
            if len(retry_text) >= len(markdown) * OUTLIER_CONFIRM_RATIO:
                # Retry is similar length — page is genuinely dense, not a glitch
                logger.info(
                    f"Page {page_no}: retry similar size "
                    f"({len(retry_text)} vs {len(markdown)} chars), "
                    f"confirmed valid — keeping original"
                )
                if metrics:
                    metrics.record_page(elapsed, len(markdown))
            else:
                # Retry is significantly shorter — original was a glitch
                logger.info(
                    f"Page {page_no}: retry shorter "
                    f"({len(retry_text)} vs {len(markdown)} chars), using retry"
                )
                markdown, elapsed = retry_text, retry_elapsed
        else:
            logger.info(
                f"Page {page_no}: keeping original ({len(markdown)} chars)"
            )
        # Don't record outlier stats unless confirmed valid (handled above)
    elif metrics and error_type is None:
        metrics.record_page(elapsed, len(markdown))

    # ── Figure cropping and image tag insertion ────────────────────────────
    page_images = _crop_and_filter_figures(
        display_path, figures, page_no, config, output_dir,
    )

    if page_images:
        img_lines = []
        for idx, img_info in enumerate(page_images):
            caption = img_info.get("caption", "Figure")
            if not caption:
                caption = "Figure"
            img_lines.append(f"![{caption}](img:{idx})")
        markdown = markdown.rstrip() + "\n\n" + "\n".join(img_lines)

    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    dirty_path = os.path.join(pages_dir, f"page_{page_no:04d}_dirty.md")
    with open(dirty_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    failed = error_type is not None
    return page_no, markdown, page_images, failed


# ── VLM Pipeline ────────────────────────────────────────────────────────────


def run_pipeline(
    input_path: str,
    output_dir: str,
    config: PipelineConfig,
    figure_converter=None,
    progress=None,
    vlm_pool=None,
) -> PipelineResult:
    """Run VLM pipeline: prepare pages → (VLM transcription ∥ figure detection) → assemble.

    Supports PDF, image, and office document inputs.

    Args:
        input_path: Path to input file (PDF, image, or office document).
        output_dir: Directory for all outputs.
        config: Pipeline configuration.
        figure_converter: Optional shared heron-101 converter for figure detection.
        progress: Optional ProgressReporter for SSE streaming.
        vlm_pool: Optional shared ThreadPoolExecutor for VLM concurrency.

    Returns:
        PipelineResult with clean markdown, images, and structured JSON.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.basename(input_path)
    logger.info(f"{'=' * 60}")
    logger.info(f"SHREW PIPELINE: {basename}")
    logger.info(f"{'=' * 60}")

    # ── Step 1: Prepare page images ─────────────────────────────────────────
    if progress:
        progress.emit(0, "Preparing pages...")

    raster_start = time.time()
    with _rasterize_lock:
        page_images, total_pages, page_dims = prepare_pages(
            input_path, output_dir,
            low_dpi=config.low_dpi, high_dpi=config.high_dpi,
            page_range=config.page_range,
        )

    if config.page_range:
        start_p, end_p = config.page_range
    else:
        start_p, end_p = 1, total_pages
    page_numbers = list(range(start_p, min(end_p, total_pages) + 1))
    if not page_numbers:
        raise ValueError(f"Page range {start_p}-{end_p} is outside document ({total_pages} pages)")
    logger.info(f"Processing pages {page_numbers[0]}-{page_numbers[-1]} "
                f"({len(page_numbers)} pages)")

    raster_time = time.time() - raster_start
    logger.info(f"Page preparation complete: {len(page_images)} pages ({raster_time:.1f}s)")

    # ── Step 2: Initialize figure detector (CPU, layout-only) ───────────────
    if figure_converter is not None:
        figure_init_time = 0.0
    else:
        figure_start = time.time()
        figure_converter = create_figure_converter()
        figure_init_time = time.time() - figure_start
    if figure_converter:
        logger.info(f"Figure detector ready ({figure_init_time:.1f}s)")
    else:
        logger.info("Figure detection disabled")

    if progress:
        progress.emit(5, "Rasterization complete")
        if progress.is_cancelled():
            raise CancelledException()

    # ── Step 3: Process pages concurrently ──────────────────────────────────
    _local_pool = None
    try:
        if vlm_pool is None:
            _local_pool = ThreadPoolExecutor(max_workers=config.vlm_concurrency)
            vlm_pool = _local_pool
        process_start = time.time()

        page_results: dict[int, tuple[str, list[dict]]] = {}
        n_pages = len(page_numbers)
        skip_stage3 = config.skip_stage3
        metrics = PageMetrics(vlm_concurrency=config.vlm_concurrency)

        if progress:
            progress.emit(5, f"Transcribing pages (0/{n_pages})...")

        futures = {}
        for pno in page_numbers:
            if pno not in page_images:
                logger.warning(f"No image for page {pno}, skipping")
                continue
            display_path, hires_path = page_images[pno]
            dims = page_dims.get(pno, (612.0, 792.0))
            fut = vlm_pool.submit(
                _process_one_page, pno, display_path, hires_path, dims,
                total_pages, config, figure_converter, output_dir, metrics,
            )
            futures[fut] = pno

        pages_done = 0
        failed_pages = []
        for fut in as_completed(futures):
            pno = futures[fut]
            try:
                _, md, imgs, failed = fut.result()
                page_results[pno] = (md, imgs)
                if failed:
                    failed_pages.append(pno)
            except Exception as e:
                logger.error(f"Page {pno} processing failed: {e}")
                page_results[pno] = ("", [])
                failed_pages.append(pno)
            pages_done += 1
            if progress:
                upper = 85 if skip_stage3 else 60
                pct = 5 + int((upper - 5) * pages_done / n_pages)
                progress.emit(pct, f"Transcribing pages ({pages_done}/{n_pages})...")
                if progress.is_cancelled():
                    raise CancelledException()

        metrics.finish_document()
        process_time = time.time() - process_start
        logger.info(f"Page processing complete: {process_time:.1f}s")

        if failed_pages:
            failed_pages.sort()
            raise RuntimeError(
                f"{len(failed_pages)}/{len(page_numbers)} pages failed VLM transcription "
                f"after all retries (pages: {failed_pages})"
            )

        # ── Step 4: Assemble final document with global image numbering ─────
        all_images: list[dict] = []
        global_img_offset = 0
        parts = []

        for pno in page_numbers:
            md, page_imgs = page_results.get(pno, ("", []))

            if page_imgs:
                for local_idx, img_info in enumerate(page_imgs):
                    global_idx = global_img_offset + local_idx
                    md = md.replace(f"(img:{local_idx})", f"(img:{global_idx})")
                    all_images.append({
                        "index": global_idx,
                        "data": img_info["data"],
                        "format": img_info["format"],
                        "caption": img_info.get("caption", ""),
                        "page": pno,
                    })
                global_img_offset += len(page_imgs)

            parts.append(f"<page {pno}>")
            parts.append(md)
            parts.append(f"</page {pno}>")
            parts.append("")

        clean_markdown = "\n".join(parts)

        from .postprocess import postprocess_markdown
        clean_markdown = postprocess_markdown(clean_markdown)

        # VLM-based heading hierarchy fix
        if not config.skip_stage3:
            from .structured import fix_heading_hierarchy
            if not config.accurate and config.shrew_vllm_url:
                heading_vlm = VLMClient(
                    base_url=config.shrew_vllm_url, model="Qwen3.5-2B",
                )
                heading_fallback = VLMClient(
                    base_url=config.vlm_url, model=config.vlm_model,
                    api_key=config.api_key,
                )
            else:
                heading_vlm = VLMClient(
                    base_url=config.vlm_url, model=config.vlm_model,
                    api_key=config.api_key,
                )
                heading_fallback = None
            clean_markdown = fix_heading_hierarchy(
                clean_markdown,
                vlm_client=heading_vlm,
                fallback_vlm_client=heading_fallback,
                lora_adapters=config.shrew_lora_map,
                lora_format=config.shrew_lora_format,
            )

        clean_path = os.path.join(output_dir, "clean.md")
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(clean_markdown)

        dirty_parts = []
        for pno in page_numbers:
            md, _ = page_results.get(pno, ("", []))
            dirty_parts.append(f"<page {pno}>\n{md}\n</page {pno}>")
        with open(os.path.join(output_dir, "dirty.md"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(dirty_parts))

        pages_dir = os.path.join(output_dir, "pages")
        os.makedirs(pages_dir, exist_ok=True)
        for pno in page_numbers:
            md, _ = page_results.get(pno, ("", []))
            clean_page_path = os.path.join(pages_dir, f"page_{pno:04d}_clean.md")
            with open(clean_page_path, "w", encoding="utf-8") as f:
                f.write(md)

        total_figures = len(all_images)
        logger.info(f"Assembled: {len(page_numbers)} pages, {total_figures} figures")

        if progress:
            pct = 90 if skip_stage3 else 70
            progress.emit(pct, f"Document assembled ({total_figures} figures)")
            if progress.is_cancelled():
                raise CancelledException()

        # ── Step 5: Structured extraction ───────────────────────────────────
        structured_json: dict = {}
        stage3_time = 0.0

        if all_images:
            structured_json["images"] = all_images

        if not config.skip_stage3:
            stage3_start = time.time()

            s3_lora = None
            s3_lora_fmt = "none"

            if not config.accurate and config.shrew_vllm_url:
                logger.info("Structured extraction: fast mode via Shrew vLLM")
                stage3_vlm = VLMClient(
                    base_url=config.shrew_vllm_url, model="Qwen3.5-2B",
                )
                s3_lora = config.shrew_lora_map
                s3_lora_fmt = config.shrew_lora_format
            else:
                logger.info("Structured extraction: main VLM")
                stage3_vlm = VLMClient(
                    base_url=config.vlm_url, model=config.vlm_model, api_key=config.api_key,
                )

            num_pages = len(page_numbers)
            s3_kwargs = dict(lora_adapters=s3_lora, lora_format=s3_lora_fmt)

            if config.shrew_async_stage3:
                if progress:
                    progress.emit(70, "Running structured extraction (async)...")
                    if progress.is_cancelled():
                        raise CancelledException()

                logger.info("Structured extraction: running metadata + summary + chunking in parallel")
                with ThreadPoolExecutor(max_workers=3) as s3_exec:
                    meta_f = s3_exec.submit(extract_metadata, clean_markdown, basename, input_path, stage3_vlm, num_pages, **s3_kwargs)
                    summ_f = s3_exec.submit(generate_summary, clean_markdown, stage3_vlm, **s3_kwargs)
                    chunk_f = s3_exec.submit(semantic_chunk, clean_markdown, stage3_vlm, section_max_tokens=config.section_max_tokens, **s3_kwargs)
                    metadata = meta_f.result()
                    summary = summ_f.result()
                    chunks = chunk_f.result()
            else:
                if progress:
                    progress.emit(70, "Extracting metadata...")
                    if progress.is_cancelled():
                        raise CancelledException()

                metadata = extract_metadata(clean_markdown, basename, input_path, stage3_vlm, num_pages, **s3_kwargs)

                if progress:
                    progress.emit(80, "Generating summary...")
                    if progress.is_cancelled():
                        raise CancelledException()

                summary = generate_summary(clean_markdown, stage3_vlm, **s3_kwargs)

                if progress:
                    progress.emit(90, "Chunking document...")
                    if progress.is_cancelled():
                        raise CancelledException()

                chunks = semantic_chunk(clean_markdown, stage3_vlm, section_max_tokens=config.section_max_tokens, **s3_kwargs)

            metadata["num_chunks"] = len(chunks)
            stage3_time = time.time() - stage3_start
            logger.info(f"Structured extraction complete: {len(chunks)} chunks, {stage3_time:.1f}s")

            structured_json.update({
                "metadata": metadata,
                "summary": summary,
                "semantic_chunks": chunks,
            })

            json_out_path = os.path.join(output_dir, "structured.json")
            with open(json_out_path, "w", encoding="utf-8") as f:
                json.dump(structured_json, f, indent=2)
        else:
            logger.info("Structured extraction: SKIPPED (--skip-stage3)")

        # ── Processing log ──────────────────────────────────────────────────
        total_time = time.time() - start_time

        processing_log = {
            "input_file": input_path,
            "output_dir": output_dir,
            "total_pages": len(page_numbers),
            "total_figures": total_figures,
            "raster_time_seconds": raster_time,
            "figure_init_seconds": figure_init_time,
            "process_time_seconds": process_time,
            "stage3_time_seconds": stage3_time,
            "total_time_seconds": total_time,
            "config": {
                "vlm_url": config.vlm_url,
                "vlm_model": config.vlm_model,
                "high_dpi": config.high_dpi,
                "vlm_concurrency": config.vlm_concurrency,
            },
        }

        log_path = os.path.join(output_dir, "processing_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(processing_log, f, indent=2)

        fig_note = f", {total_figures} figures" if total_figures else ""
        logger.info(f"{'=' * 60}")
        logger.info(f"PIPELINE COMPLETE: {basename}")
        logger.info(f"  Time: {total_time:.1f}s (raster: {raster_time:.1f}s, "
                    f"process: {process_time:.1f}s)")
        logger.info(f"  Pages: {len(page_numbers)}{fig_note}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"{'=' * 60}")

        return PipelineResult(
            clean_markdown=clean_markdown,
            structured_json=structured_json,
            processing_log=processing_log,
        )
    finally:
        if _local_pool is not None:
            _local_pool.shutdown(wait=False)
