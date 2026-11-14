"""compose_markdown(v05_json) -> str: one page of v0.5 model output -> the page
markdown shrew-server v0.5 ships.

A pure function (no I/O, no model call) built on the SAME code the server runs
(assembly.assemble_document + structured_pipeline.synthesize_markdown), so an
external scorer (a benchmark harness) composes exactly what /v1/convert
returns for that page. Spec: compose.md (beside this file).

    from app.compose import compose_markdown
    md = compose_markdown(json.loads(raw_model_output))
"""
from __future__ import annotations

import copy

from .assembly import assemble_document
from .contract import envelope, normalize_v05

_LISTS = ("semantic_chunks", "figures", "tables")
# assembly.crop_bbox refuses boxes under this share of the page (a crop that
# small is not shipped, so the figure renders as a text placeholder instead).
_MIN_CROP_FRAC = 0.015


def _croppable(bbox) -> bool:
    e = envelope(bbox)
    if e is None:
        return False
    x0, y0, x1, y1 = (max(0, min(1000, c)) for c in e)
    return x1 > x0 and y1 > y0 and (x1 - x0) * (y1 - y0) / 1e6 >= _MIN_CROP_FRAC


def compose_markdown(v05_json: dict, *, page: int = 1, page_tags: bool = False,
                     image_page: bool = True) -> str:
    """Page markdown for one parsed v0.5 page output.

    page_tags   keep the server's ``<page N> ... </page N>`` wrapper (default off).
    image_page  True (an uploaded page image / PDF page): figures with a usable
                box become ``![caption](img:N)`` image refs, as the server ships
                them with their crops. False (text input): no crops exist, so
                every figure is a ``[Figure: caption]`` line.
    The input is not modified.
    """
    if not isinstance(v05_json, dict):
        raise TypeError("compose_markdown takes the parsed page JSON object")
    data = copy.deepcopy(v05_json)
    for k in _LISTS:
        if not isinstance(data.get(k), list):
            data[k] = []
    data["semantic_chunks"] = [c for c in data["semantic_chunks"] if isinstance(c, dict)]
    data["figures"] = [f for f in data["figures"] if isinstance(f, dict)]
    data["tables"] = [t for t in data["tables"] if isinstance(t, dict)]
    if not isinstance(data.get("metadata"), dict):
        data["metadata"] = {}
    data.setdefault("summary", None)
    normalize_v05(data)                     # the server's tolerant repairs

    # Imported here: structured_pipeline pulls in the whole pipeline module.
    from .structured_pipeline import synthesize_markdown

    doc = assemble_document("compose", None, "image" if image_page else "text",
                            [{"page": page, "ok": True, "data": data}], stitch=False)
    if image_page:
        for f in doc["figures"]:
            if _croppable(f.get("bbox")):
                f["crop_path"] = "<crop>"   # stands in for the server's crop file
    md = synthesize_markdown(doc)
    if page_tags:
        return md
    head, tail = f"<page {page}>\n", f"\n</page {page}>"
    if md.startswith(head) and md.endswith(tail):
        md = md[len(head):-len(tail)]
    return md.strip()
