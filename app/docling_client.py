"""Figure detection via heron-101 layout model (RT-DETR v2).

Uses the docling-layout-heron-101 model directly via HuggingFace
Transformers to detect picture bounding boxes in page images.
"""

import logging
from pathlib import Path

import PIL.Image

logger = logging.getLogger("shrew.figures")

HERON_MODEL = "ds4sd/docling-layout-heron-101"
PICTURE_LABEL = 6  # "Picture" class in heron-101's 17-class label map
CONFIDENCE_THRESHOLD = 0.5
MAX_PAGE_COVERAGE = 0.9  # Ignore detections covering >90% of the page


def create_figure_converter(device: str = "auto"):
    """Load the heron-101 layout model for figure detection.

    Args:
        device: Accelerator device — "auto", "cpu", or "cuda".

    Returns (model, processor) tuple, or None if transformers/torch not installed.
    """
    try:
        import torch
        from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
    except ImportError:
        logger.info("torch/transformers not installed — figure detection disabled")
        return None

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = RTDetrImageProcessor.from_pretrained(HERON_MODEL)
    model = RTDetrV2ForObjectDetection.from_pretrained(HERON_MODEL)
    model.to(device)
    model.eval()

    logger.info(f"Heron-101 figure detector loaded (device={device})")
    return model, processor


def detect_figures(image_path: str | Path, converter) -> list[dict]:
    """Detect picture bounding boxes in a page image.

    Args:
        image_path: Path to a rasterized page image (PNG).
        converter: A (model, processor) tuple from create_figure_converter().

    Returns:
        List of dicts: [{"bbox": {"l", "t", "r", "b"}, "caption": ""}, ...]
        Coordinates are in pixels (image coordinate space, top-left origin).
    """
    if converter is None:
        return []

    import torch

    model, processor = converter
    image_path = Path(image_path)

    try:
        image = PIL.Image.open(image_path).convert("RGB")
        w, h = image.size

        inputs = processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[h, w]], device=model.device),
            threshold=CONFIDENCE_THRESHOLD,
        )[0]

        page_area = w * h
        figures = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() != PICTURE_LABEL:
                continue

            x1, y1, x2, y2 = box.tolist()
            box_area = (x2 - x1) * (y2 - y1)
            if box_area / page_area > MAX_PAGE_COVERAGE:
                continue

            figures.append({
                "bbox": {"l": x1, "t": y1, "r": x2, "b": y2},
                "caption": "",
            })

    except Exception as e:
        logger.warning(f"Figure detection failed on {image_path.name}: {e}")
        return []

    logger.info(f"Detected {len(figures)} figure(s) in {image_path.name}")
    return figures
