"""Data types for the Shrew pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BBox:
    """Bounding box coordinates."""
    l: float
    t: float
    r: float
    b: float
    coord_origin: str = "BOTTOMLEFT"

    def to_top_left(self, page_height: float) -> "BBox":
        """Convert BOTTOMLEFT origin to TOPLEFT origin."""
        if self.coord_origin == "TOPLEFT":
            return BBox(self.l, self.t, self.r, self.b, "TOPLEFT")
        return BBox(
            l=self.l,
            t=page_height - self.t,
            r=self.r,
            b=page_height - self.b,
            coord_origin="TOPLEFT",
        )

    def to_pixels(self, dpi: int) -> tuple[int, int, int, int]:
        """Convert from PDF points (72 DPI) to pixel coordinates.
        Returns (left, top, right, bottom) in pixels.
        Must be in TOPLEFT origin first.
        """
        scale = dpi / 72.0
        return (
            int(self.l * scale),
            int(self.t * scale),
            int(self.r * scale),
            int(self.b * scale),
        )


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""
    vlm_url: str = "http://localhost:8000"
    vlm_model: str = ""
    low_dpi: int = 100
    high_dpi: int = 200
    accurate: bool = True
    shrew_vllm_url: Optional[str] = None
    shrew_lora_map: Optional[dict] = None
    shrew_lora_format: str = "none"
    shrew_async_stage3: bool = False
    skip_stage3: bool = False
    skip_chunking: bool = False
    section_max_tokens: int = 6000
    vlm_concurrency: int = 4
    page_range: Optional[tuple[int, int]] = None
    api_key: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of the pipeline."""
    clean_markdown: str
    structured_json: dict
    processing_log: dict
