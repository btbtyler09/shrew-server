# Vendored from shrew_ocr/train/preprocess.py — canonical source; keep byte-identical logic. Parity test: tests/test_preprocess_parity.py (internal).
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def enhance_v2(gray_u8: np.ndarray) -> np.ndarray:
    """100-DPI grayscale uint8 -> CLAHE local contrast + unsharp mask (the v2 winner).

    CLAHE(clip=2.0, 8x8) recovers faint table lines after the downscale; the unsharp
    mask (amount=0.8, radius=1.2) re-sharpens text edges. Do NOT add line-thickening.
    """
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_u8)
    blur = cv2.GaussianBlur(g, (0, 0), 1.2)
    return cv2.addWeighted(g, 1.8, blur, -0.8, 0)  # amount=0.8, radius=1.2


def prepare_image(img: Image.Image, target_dpi: int = 100, src_dpi: int = 200) -> Image.Image:
    """Stored page image (rendered at `src_dpi`) -> luminance, downscaled to `target_dpi`
    via INTER_AREA, v2-enhanced, returned as RGB (the Qwen processor expects RGB).

    Downscale uses a pure ratio (target/src), so it is page-dimension independent — every
    page was rendered at 200 DPI, halving yields 100 DPI regardless of page size. INTER_AREA
    matches the eval/bake-off downscale path exactly.
    """
    g = np.asarray(img.convert("L"))
    if target_dpi != src_dpi:
        scale = target_dpi / src_dpi
        h, w = g.shape[:2]
        g = cv2.resize(
            g, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return Image.fromarray(enhance_v2(g)).convert("RGB")


# ── Glyph-routed tile buckets (SHREW_OCR_PREVIEW.md §2.1, e4) ─────────────────────────────────────
# SUPERSEDES the fixed DPI-ratio downscale above. The vision tower is SigLIP at a fixed 384x384 tile
# and tile COUNT is chosen by `select_best_resolution` from image_grid_pinpoints, so INPUT RESOLUTION
# is the only control over how finely text is sampled. A fixed ratio downscale put body glyphs below
# what the encoder can represent -- the model was asked to transcribe text that was not in its input,
# and its failure mode at temperature 0 is a repetition loop.
#
# Routing is deterministic (no page-type classifier): effective glyph height = native glyph height x
# grid scale; pick the SMALLEST bucket that reaches GLYPH_TARGET. Buckets are portrait, so a 3:4 page
# is never letterboxed into a square (the stock 1152x1152 ceiling spent ~25% of the horizontal
# resolution on padding).
#
# Measured on 40 held-out looping pages (base model, precision-on-page): 0.76 at the old 658x850
# downscale (24% clean body reads) vs 0.89 native + pinpoints (74%). Corpus controls sit at 0.88.
GLYPH_TARGET = 10.0
BUCKETS = [                      # (name, [w, h], tiles, approx image tokens)
    ("B1", [1152, 1536], 12, 1846),
    ("B2", [1536, 2304], 24, 3182),
    ("B3", [2304, 3072], 48, 6300),
]
SQUARE_BUCKET = ("B0", [1152, 1152], 9, 1188)   # square-ish inputs only (table crops, where the
                                                # image IS the table and 6px native is fine)
BUCKET_PINPOINTS = [[h, w] for _, (w, h), _, _ in BUCKETS] + [[1152, 1152]]
BUCKET_BY_NAME = {n: (wh, ti, tok) for n, wh, ti, tok in BUCKETS + [SQUARE_BUCKET]}


def glyph_height(img: Image.Image, max_side: int = 2600) -> float | None:
    """Median connected-component height in NATIVE pixels -- the routing signal. Components are
    filtered to glyph-shaped blobs so rules, borders, halftone speckle and figures do not skew it."""
    import cv2 as _cv
    import numpy as _np
    W, H = img.size
    s = min(1.0, max_side / max(W, H))
    im = img.convert("L")
    if s < 1.0:
        im = im.resize((int(W * s), int(H * s)), Image.BOX)
    g = _cv.adaptiveThreshold(_np.asarray(im), 255, _cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              _cv.THRESH_BINARY_INV, 31, 10)
    n, _, stats, _ = _cv.connectedComponentsWithStats(g, connectivity=8)
    hs = [stats[i][3] for i in range(1, n)
          if 2 <= stats[i][3] <= 60 and 1 <= stats[i][2] <= 60 and stats[i][4] >= 4
          and 0.08 <= stats[i][2] / max(stats[i][3], 1) <= 6.0]
    if len(hs) < 50:
        return None
    import statistics as _st
    return _st.median(hs) / max(s, 1e-6)


def pick_bucket(size, glyph_px, target: float = GLYPH_TARGET):
    """(w, h) + native glyph px -> (bucket name, [w, h] grid). Square-ish inputs take B0."""
    w, h = size
    if h and 0.9 <= w / h <= 1.15:
        return SQUARE_BUCKET[0], SQUARE_BUCKET[1]
    if not glyph_px:                      # unmeasurable (blank/graphical page): middle bucket
        return "B2", BUCKET_BY_NAME["B2"][0]
    # 5% tolerance: a 3:4 corpus page reaches 9.8px in B1, which is 10px within the measurement's own
    # noise, and B1 is the floor for portrait pages. Without the tolerance every corpus page would
    # jump to B2 and double the token cost of the whole corpus for 0.2px.
    for name, (bw, bh), _, _ in BUCKETS:
        if glyph_px * min(bw / w, bh / h) >= target * 0.95:
            return name, [bw, bh]
    return BUCKETS[-1][0], BUCKETS[-1][1]


def fit_to_bucket(img: Image.Image, glyph_px=None, target: float = GLYPH_TARGET):
    """Resize so the chosen bucket is what select_best_resolution will pick, preserving aspect."""
    name, (bw, bh) = pick_bucket(img.size, glyph_px, target)
    w, h = img.size
    s = min(bw / w, bh / h)
    if abs(s - 1.0) < 0.02:
        return img, name
    return img.resize((max(1, round(w * s)), max(1, round(h * s))),
                      Image.LANCZOS if s > 1 else Image.BOX), name


def prepare_image_bucketed(img: Image.Image, glyph_px: float | None = None):
    """§2.1 image transform: route to a tile bucket, THEN enhance. Returns (RGB image, bucket).

    Order is load-bearing and matches training (`make_batch_transform` tile_mode="bucket"): fit
    first, enhance second. `glyph_px` is the page's cached native glyph height -- pass None to
    measure it here, but prefer caching, since the measurement is the only CPU-bound step and it is
    a fixed property of the image.
    """
    if glyph_px is None:
        glyph_px = glyph_height(img)
    fitted, bucket = fit_to_bucket(img, glyph_px)
    return Image.fromarray(enhance_v2(np.asarray(fitted.convert("L")))).convert("RGB"), bucket
