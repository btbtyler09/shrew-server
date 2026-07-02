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
