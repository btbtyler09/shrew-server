"""Document page rasterization and format conversion.

Supports three input categories that need page images:
- PDF: rasterize via pypdfium2 (two resolutions per page)
- Images: create page structure directly via Pillow (multi-page TIFF supported)
- Office docs: convert to PDF via LibreOffice headless, then rasterize

Text-family inputs (txt/md/rtf/html/csv/eml/msg, plus spreadsheets) bypass
this module entirely — see text_extract.py and spreadsheet_extract.py.
"""

import logging
import multiprocessing
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import pypdfium2
from PIL import Image

logger = logging.getLogger("shrew.rasterizer")

# PDFium (pypdfium2's C core) is not thread-safe, and its Python-side
# finalizers run on whatever thread drops the last reference — a lock around
# our own calls cannot serialize that native teardown. Worse, any PDFium
# fault (bad PDF, lifetime slip, upstream bug) is a segfault that takes the
# whole server down with every in-flight request. So all PDFium work runs in
# a short-lived SPAWNED worker process: a native crash there surfaces as a
# clean RuntimeError for that one document (exit code 139 observed in
# production on an 889-page conversion under overlapped /v1/convert/stream
# requests). The lock survives as a resource throttle — one document
# rasterizes at a time, bounding peak memory/disk pressure. RLock because
# pipeline call sites already hold it when they reach prepare_pages.
RASTERIZE_LOCK = threading.RLock()

_MP_CTX = multiprocessing.get_context("spawn")

# Wall clock for one document's rasterization. PDFium can also HANG in native
# code on pathological PDFs; without a bound that wedges the worker (and the
# throttle lock) forever. 30 min covers 1000+-page books at 200 DPI with
# a wide margin (~2 min measured for 889 pages).
RASTER_TIMEOUT_S_DEFAULT = 1800.0


def _isolated_entry(conn, fn_name: str, args: tuple) -> None:
    """Worker-process entry: run a rasterizer function and pipe back the result.

    Runs in a fresh spawn'd interpreter — this module is re-imported there, so
    the target is resolved by name (closures/monkeypatches don't cross spawn).
    """
    # The child's stderr is inherited from the server process, so configure
    # logging to preserve the per-10-pages progress lines operators rely on.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Test-only fault injection: spawn children inherit os.environ, and a
    # fresh import can't see pytest monkeypatches — env vars are the only
    # practical way to make the worker crash/hang on demand in tests.
    if os.environ.get("SHREW_RASTER_TEST_CRASH"):
        os.kill(os.getpid(), signal.SIGSEGV)
    if os.environ.get("SHREW_RASTER_TEST_HANG"):
        time.sleep(float(os.environ["SHREW_RASTER_TEST_HANG"]))
    try:
        result = globals()[fn_name](*args)
        conn.send(("ok", result))
    except Exception as e:  # noqa: BLE001 — full fidelity back to the parent
        import traceback
        conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    finally:
        conn.close()


def _run_isolated(fn_name: str, *args):
    """Run a module-level rasterizer function in an isolated worker process.

    Returns its result, or raises RuntimeError if the worker segfaults, hangs
    past the timeout, or errors — the server process itself is never at risk.
    SHREW_RASTER_INPROC=1 skips the subprocess (debugging escape hatch; the
    crash containment is gone but the fixed close-ordering still applies).
    """
    with RASTERIZE_LOCK:
        if os.environ.get("SHREW_RASTER_INPROC") == "1":
            return globals()[fn_name](*args)

        timeout = float(os.environ.get("SHREW_RASTER_TIMEOUT_S",
                                       RASTER_TIMEOUT_S_DEFAULT))
        recv, send = _MP_CTX.Pipe(duplex=False)
        proc = _MP_CTX.Process(
            target=_isolated_entry, args=(send, fn_name, args), daemon=True)
        proc.start()
        send.close()  # parent keeps only the read end
        try:
            if not recv.poll(timeout):
                raise RuntimeError(
                    f"rasterization timed out after {timeout:.0f}s "
                    f"(worker killed; document skipped, server unaffected)")
            try:
                status, payload = recv.recv()
            except EOFError:
                # Worker died before sending — native crash (segfault) path.
                proc.join(5)
                raise RuntimeError(
                    f"rasterization worker crashed (exit code {proc.exitcode})"
                    " — likely a malformed PDF or a PDFium fault; the server"
                    " is unaffected") from None
            if status == "err":
                raise RuntimeError(f"rasterization failed in worker:\n{payload}")
            return payload
        finally:
            recv.close()
            if proc.is_alive():
                proc.terminate()
            proc.join(5)
            if proc.is_alive():  # terminate ignored (stuck in native code)
                proc.kill()
                proc.join(5)

# Long-edge ceiling for a rendered page, in pixels. A fixed DPI is blind to the
# page's physical size: a poster-sized page (or one with corrupt MediaBox
# metadata) rendered at 200 DPI produces a 300-700 MP image, which PIL then
# refuses to open (decompression-bomb guard, ~179 MP) — the doc 500s before the
# model transform ever runs. Clamping the long edge loses nothing: the bucket
# transform downsizes to at most 2304x3072 anyway, and the routing rule is
# scale-invariant (effective glyph = native_glyph_px * min(bw/w, bh/h); halving
# the render halves both factors' numerator and denominator alike), so the
# bucket choice is unchanged as long as the clamp stays above the B3 grid.
RENDER_MAX_LONG_EDGE = 6000


def _clamped_scale(page, dpi: float) -> float:
    """Render scale for `dpi`, reduced if the page's long edge would exceed
    RENDER_MAX_LONG_EDGE px. pdfium page sizes are in points (1/72 inch)."""
    scale = dpi / 72
    w, h = page.get_size()
    long_edge = max(w, h) * scale
    if long_edge > RENDER_MAX_LONG_EDGE:
        scale *= RENDER_MAX_LONG_EDGE / long_edge
    return scale

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}
OFFICE_EXTENSIONS = {".docx", ".pptx", ".doc", ".ppt", ".odt", ".odp"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".ods"}
TEXT_EXTENSIONS = {".txt", ".md", ".rtf", ".html", ".htm", ".eml", ".msg"}
CSV_EXTENSION = ".csv"

# Cap image dimensions to prevent OOM
MAX_IMAGE_DIM = 10000


def classify_file(file_path: str) -> str:
    """Classify input file by extension.

    Returns 'pdf', 'image', 'office', 'spreadsheet', 'text', or 'csv'.
    Raises ValueError for unsupported extensions.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in OFFICE_EXTENSIONS:
        return "office"
    if ext in SPREADSHEET_EXTENSIONS:
        return "spreadsheet"
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext == CSV_EXTENSION:
        return "csv"
    raise ValueError(f"Unsupported file type: {ext}")


def convert_office_to_pdf(office_path: str, output_dir: str) -> str:
    """Convert an office document (docx/pptx/odt/odp/doc/ppt) to PDF
    using LibreOffice headless.

    Returns path to the converted PDF file.
    Raises RuntimeError if LibreOffice is not installed or conversion fails.
    """
    basename = os.path.basename(office_path)
    logger.info(f"Converting {basename} to PDF via LibreOffice")

    try:
        result = subprocess.run(
            ["libreoffice", "--headless", "--convert-to", "pdf",
             "--outdir", output_dir, office_path],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "LibreOffice is not installed. "
            "Install it to process office documents: apt install libreoffice"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"LibreOffice conversion timed out after 120s for {basename}"
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice conversion failed for {basename}: {result.stderr}"
        )

    stem = os.path.splitext(basename)[0]
    pdf_path = os.path.join(output_dir, f"{stem}.pdf")
    if not os.path.exists(pdf_path):
        raise RuntimeError(
            f"LibreOffice did not produce expected PDF at {pdf_path}"
        )

    logger.info(f"Converted {basename} → {stem}.pdf")
    return pdf_path


def prepare_image_pages(
    image_path: str,
    output_dir: str,
    low_dpi: int = 100,
    high_dpi: int = 200,
) -> tuple[dict[int, tuple[Path, Path]], int, dict[int, tuple[float, float]]]:
    """Create display+hires page PNGs from an image file.

    Multi-page TIFFs produce one page per frame. GIFs use first frame only.

    Returns:
        (page_images, total_pages, page_dims) matching the rasterize_pages contract.
    """
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)

    img = Image.open(image_path)
    basename = os.path.basename(image_path)
    ext = os.path.splitext(image_path)[1].lower()

    # Determine number of frames
    n_frames = getattr(img, "n_frames", 1)
    is_multipage_tiff = ext in {".tiff", ".tif"} and n_frames > 1
    if ext == ".gif" and n_frames > 1:
        logger.warning(f"Animated GIF ({n_frames} frames) — using first frame only")
        n_frames = 1

    if is_multipage_tiff:
        logger.info(f"Multi-page TIFF: {n_frames} pages from {basename}")
    else:
        logger.info(f"Image input: {basename}")

    # Get native DPI (default 150 if not embedded)
    dpi_info = img.info.get("dpi")
    native_dpi = dpi_info[0] if dpi_info and dpi_info[0] > 0 else 150

    display_scale = low_dpi / high_dpi
    page_images = {}
    page_dims = {}

    for frame_idx in range(n_frames):
        if n_frames > 1:
            img.seek(frame_idx)

        page_no = frame_idx + 1
        frame = img.convert("RGB")

        # Cap very large images
        w, h = frame.size
        if max(w, h) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(w, h)
            frame = frame.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h = frame.size
            logger.info(f"  Page {page_no}: resized to {w}x{h} (capped at {MAX_IMAGE_DIM}px)")

        # Save hires version (original size or capped)
        hires_path = Path(pages_dir) / f"page_{page_no:04d}_hires.png"
        frame.save(str(hires_path))

        # Save display version (scaled down)
        dw, dh = int(w * display_scale), int(h * display_scale)
        display = frame.resize((dw, dh), Image.LANCZOS)
        display_path = Path(pages_dir) / f"page_{page_no:04d}_display.png"
        display.save(str(display_path))

        page_images[page_no] = (display_path, hires_path)
        # Dimensions in PDF points (1/72 inch)
        page_dims[page_no] = (w * 72 / native_dpi, h * 72 / native_dpi)

        if page_no % 10 == 0 or page_no == n_frames:
            logger.info(f"  Prepared page {page_no}/{n_frames}")

    img.close()
    total_pages = n_frames
    logger.info(f"Prepared {total_pages} page(s) from {basename}")
    return page_images, total_pages, page_dims


def prepare_pages(
    file_path: str,
    output_dir: str,
    low_dpi: int = 100,
    high_dpi: int = 200,
    page_range: tuple[int, int] | None = None,
) -> tuple[dict[int, tuple[Path, Path]], int, dict[int, tuple[float, float]]]:
    """Prepare page images from any supported file format.

    Unified entry point that handles PDFs, images, and office documents.

    Returns:
        (page_images, total_pages, page_dims) where:
        - page_images: {page_no: (display_path, hires_path)}
        - total_pages: int
        - page_dims: {page_no: (width_pts, height_pts)}
    """
    file_type = classify_file(file_path)

    if file_type == "image":
        if page_range:
            logger.warning("page_range ignored for image input")
        return prepare_image_pages(file_path, output_dir, low_dpi, high_dpi)

    if file_type == "office":
        file_path = convert_office_to_pdf(file_path, output_dir)
    elif file_type in {"text", "csv", "spreadsheet"}:
        raise ValueError(
            f"Text/CSV/spreadsheet input should bypass prepare_pages "
            f"(file_type={file_type})"
        )

    # PDF path (original or converted from office/spreadsheet).
    # One isolated worker call does count + dims + rendering in a single
    # document open — see _run_isolated for the crash-containment contract.
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    logger.info(f"Rasterizing {os.path.basename(file_path)} in isolated worker")
    raw_images, total_pages, page_dims = _run_isolated(
        "_raster_worker", file_path, pages_dir, low_dpi, high_dpi, page_range)
    page_images = {n: (Path(d), Path(h)) for n, (d, h) in raw_images.items()}
    logger.info(f"Rasterized {len(page_images)} pages to {pages_dir}")
    return page_images, total_pages, page_dims


def _raster_worker(
    pdf_path: str,
    pages_dir: str,
    low_dpi: int,
    high_dpi: int,
    page_range: tuple[int, int] | None,
) -> tuple[dict[int, tuple[str, str]], int, dict[int, tuple[float, float]]]:
    """All PDFium work for one document: count, dims, and two-DPI renders.

    Runs inside the isolated worker (see _run_isolated). Native-object
    lifetime is strict here — every page and bitmap is closed before its
    parent, and nothing outlives doc.close(). The previous implementation
    left the final loop iteration's page/bitmap locals alive across
    doc.close(); their finalizers then touched a freed document — the
    production exit-139 segfault on large documents.

    Returns str paths (picklable across the process boundary).
    """
    doc = pypdfium2.PdfDocument(pdf_path)
    try:
        total_pages = len(doc)
        logger.info(
            f"Rasterizing {total_pages} pages from {os.path.basename(pdf_path)}")
        start = (page_range[0] - 1) if page_range else 0
        end = min(page_range[1] if page_range else total_pages, total_pages)

        result: dict[int, tuple[str, str]] = {}
        dims: dict[int, tuple[float, float]] = {}
        for page_idx in range(total_pages):
            page_no = page_idx + 1  # 1-indexed
            page = doc[page_idx]
            try:
                w, h = page.get_size()
                dims[page_no] = (w, h)
                if not (start <= page_idx < end):
                    continue

                display_path = os.path.join(
                    pages_dir, f"page_{page_no:04d}_display.png")
                hires_path = os.path.join(
                    pages_dir, f"page_{page_no:04d}_hires.png")
                for out_path, dpi in ((display_path, low_dpi),
                                      (hires_path, high_dpi)):
                    bitmap = page.render(scale=_clamped_scale(page, dpi))
                    try:
                        # to_pil() may return a VIEW over the PDFium-owned
                        # buffer (PIL.frombuffer doesn't copy RGB) — the PIL
                        # image must be fully consumed and closed before the
                        # bitmap it borrows from.
                        pil_img = bitmap.to_pil()
                        try:
                            pil_img.save(out_path)
                        finally:
                            pil_img.close()
                    finally:
                        bitmap.close()
                result[page_no] = (display_path, hires_path)

                if page_no % 10 == 0 or page_no == total_pages:
                    logger.info(f"  Rasterized page {page_no}/{total_pages}")
            finally:
                page.close()
        return result, total_pages, dims
    finally:
        doc.close()


def _dims_worker(pdf_path: str) -> tuple[int, dict[int, tuple[float, float]]]:
    """Count + dims only (no rendering). Same lifetime rules as _raster_worker."""
    doc = pypdfium2.PdfDocument(pdf_path)
    try:
        dims = {}
        for i in range(len(doc)):
            page = doc[i]
            try:
                w, h = page.get_size()
            finally:
                page.close()
            dims[i + 1] = (w, h)
        return len(doc), dims
    finally:
        doc.close()


def rasterize_pages(
    pdf_path: str,
    output_dir: str,
    low_dpi: int = 100,
    high_dpi: int = 200,
    page_range: tuple[int, int] | None = None,
) -> dict[int, tuple[Path, Path]]:
    """Rasterize PDF pages at two resolutions (crash-isolated).

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save page images.
        low_dpi: DPI for overview images.
        high_dpi: DPI for element crop source images.
        page_range: Optional (start, end) 1-indexed inclusive. None = all pages.

    Returns:
        Dict mapping page_no (1-indexed) to (display_image_path, hires_image_path).
    """
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    raw_images, _total, _dims = _run_isolated(
        "_raster_worker", pdf_path, pages_dir, low_dpi, high_dpi, page_range)
    logger.info(f"Rasterized {len(raw_images)} pages to {pages_dir}")
    return {n: (Path(d), Path(h)) for n, (d, h) in raw_images.items()}


def get_page_count_and_dims(pdf_path: str) -> tuple[int, dict[int, tuple[float, float]]]:
    """Get page count and dimensions from a PDF without rasterizing
    (crash-isolated).

    Returns:
        (total_pages, {page_no: (width_pts, height_pts)}) where page_no is 1-indexed
        and dimensions are in PDF points (1/72 inch).
    """
    return _run_isolated("_dims_worker", pdf_path)
