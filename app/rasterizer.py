"""Document page rasterization and format conversion.

Supports three input categories that need page images:
- PDF: rasterize via pypdfium2 (two resolutions per page)
- Images: create page structure directly via Pillow (multi-page TIFF supported)
- Office docs: convert to PDF via LibreOffice headless, then rasterize

Text-family inputs (txt/md/rtf/html/csv/eml/msg, plus spreadsheets) bypass
this module entirely — see text_extract.py and spreadsheet_extract.py.
"""

import logging
import os
import subprocess
from pathlib import Path

import pypdfium2
from PIL import Image

logger = logging.getLogger("shrew.rasterizer")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}
OFFICE_EXTENSIONS = {".docx", ".pptx", ".doc", ".ppt", ".odt", ".odp"}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".ods"}
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

    # PDF path (original or converted from office/spreadsheet)
    total_pages, page_dims = get_page_count_and_dims(file_path)
    page_images = rasterize_pages(file_path, output_dir, low_dpi, high_dpi, page_range)
    return page_images, total_pages, page_dims


def rasterize_pages(
    pdf_path: str,
    output_dir: str,
    low_dpi: int = 100,
    high_dpi: int = 200,
    page_range: tuple[int, int] | None = None,
) -> dict[int, tuple[Path, Path]]:
    """Rasterize PDF pages at two resolutions.

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

    doc = pypdfium2.PdfDocument(pdf_path)
    try:
        total_pages = len(doc)
        logger.info(f"Rasterizing {total_pages} pages from {os.path.basename(pdf_path)}")

        start = (page_range[0] - 1) if page_range else 0
        end = page_range[1] if page_range else total_pages

        result = {}
        for page_idx in range(start, min(end, total_pages)):
            page_no = page_idx + 1  # 1-indexed
            page = doc[page_idx]

            # Display resolution
            display_path = Path(pages_dir) / f"page_{page_no:04d}_display.png"
            bitmap = page.render(scale=low_dpi / 72)
            bitmap.to_pil().save(str(display_path))

            # Extraction resolution
            hires_path = Path(pages_dir) / f"page_{page_no:04d}_hires.png"
            bitmap = page.render(scale=high_dpi / 72)
            bitmap.to_pil().save(str(hires_path))

            result[page_no] = (display_path, hires_path)

            if page_no % 10 == 0 or page_no == total_pages:
                logger.info(f"  Rasterized page {page_no}/{total_pages}")
    finally:
        doc.close()
    logger.info(f"Rasterized {len(result)} pages to {pages_dir}")
    return result


def get_page_count_and_dims(pdf_path: str) -> tuple[int, dict[int, tuple[float, float]]]:
    """Get page count and dimensions from a PDF without rasterizing.

    Returns:
        (total_pages, {page_no: (width_pts, height_pts)}) where page_no is 1-indexed
        and dimensions are in PDF points (1/72 inch).
    """
    doc = pypdfium2.PdfDocument(pdf_path)
    try:
        dims = {}
        for i in range(len(doc)):
            w, h = doc[i].get_size()
            dims[i + 1] = (w, h)
        total = len(doc)
    finally:
        doc.close()
    return total, dims
