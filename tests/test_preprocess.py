import numpy as np
from PIL import Image
from app.preprocess import prepare_image, enhance_v2


def test_prepare_image_downscales_and_returns_rgb():
    img = Image.new("RGB", (1700, 2200), "white")  # 8.5x11in at 200 DPI
    out = prepare_image(img, target_dpi=100, src_dpi=200)
    assert out.mode == "RGB"
    assert out.size == (850, 1100)


def test_enhance_v2_preserves_shape_and_dtype():
    g = (np.random.default_rng(0).random((200, 300)) * 255).astype(np.uint8)
    out = enhance_v2(g)
    assert out.shape == (200, 300) and out.dtype == np.uint8
