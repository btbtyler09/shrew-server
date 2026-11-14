"""Exact prompt token counts for the v0.5 contract.

The v0.5 budget is ``max_tokens = 30,000 - prompt tokens``, so the prompt has to
be counted, not estimated: chars / 3.5 under-counts CJK and number-heavy text
by more than 2x (training text rows go down to 1.58 chars per token), and an
under-count lets prompt + output pass the 30,000 tokens the model was trained
under.

The tokenizer is the one the served v0.5 checkpoint is actually tokenized with
(the released v0.5 checkpoint), vendored in ``app/data`` and
loaded once with the Rust ``tokenizers`` library.

It is NOT the checkpoint's ``tokenizer.json`` read as-is. transformers 5 (the
trainer, vLLM 0.28 on ROCm, vLLM 0.34 on CUDA) builds Granite's tokenizer as a
``GPT2Tokenizer`` with a plain ByteLevel pre-tokenizer and ignores the Split
regex that ``tokenizer.json`` carries, so the raw file counts differently
(``_figure`` is one token there, two when served; the raw file is up to ~8%
low on mixed text). The vendored file is ``backend_tokenizer.to_str()`` of that
effective tokenizer; it is byte-identical under transformers 5.11.0, 5.16.1
and 5.22.2 and for the training base. If a serving stack ever loads the raw
``tokenizer.json`` instead, counts here run slightly high (safe). The prompt
is rendered exactly as the checkpoint's chat template renders the server's two
message shapes (verified against the template: tests/test_v05_budget.py):

    <|start_of_role|>system<|end_of_role|>{command}<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>{text  |  <image>\\n}<|end_of_text|>
    <|start_of_role|>assistant<|end_of_role|>

An image prompt is that count with the single ``<image>`` placeholder replaced
by the processor's image tokens (``contract.v05_image_tokens``).

Without a tokenizer (file missing, library missing, hash mismatch) the count
falls back to the UTF-8 byte length of the text, a true upper bound for a
byte-level BPE; ``status()`` reports which is active and ``/health`` shows it.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

VENDORED = Path(__file__).parent / "data" / "v05_tokenizer.json"
# sha256 of the effective tokenizer (see above), and of the checkpoint's
# tokenizer.json it was built from (the released v0.5 checkpoint).
VENDORED_SHA256 = "d041ce99998f408c77bdad53fce53ddc185d1b10cdbf983d6e9ebeccaba487ca"
CHECKPOINT_TOKENIZER_JSON_SHA256 = "ab67c81a103c0ade199853ccffee38f88374190663662a81cc31393633c42f38"

_PREFIX = "<|start_of_role|>system<|end_of_role|>{command}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>"
_SUFFIX = "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
IMAGE_PLACEHOLDER = "<image>\n"
# Template tokens that are not the command or the content, used by the fallback only.
_FALLBACK_SCAFFOLD = 16

# The checkpoint's chat template REPLACES a user message that contains one of
# six task tags (``<tables_otsl>`` ...) by a canned prompt of up to ~330 tokens
# (its ``expand_tags`` macro). A text page that merely mentions a tag is
# therefore served as that prompt; ``render`` reproduces it so the count is the
# served one. The prompts are copied from chat_template.jinja (sha256 in the file).
TEMPLATE_TAGS = Path(__file__).parent / "data" / "v05_template_tags.json"
_tags: dict = {}


def _template_tags() -> dict:
    if not _tags:
        import json
        _tags.update(json.loads(TEMPLATE_TAGS.read_text(encoding="utf-8")))
    return _tags


def template_tag(content: str) -> str | None:
    """The task tag the chat template would expand in this user content, or None."""
    if "<" not in content:
        return None
    return next((k for k in _template_tags()["order"] if f"<{k}>" in content), None)


def expand_tags(content: str) -> str:
    """User content as the chat template's ``expand_tags`` macro leaves it."""
    tag = template_tag(content)
    if tag is None:
        return content
    prompt = _template_tags()["prompts"][tag]
    if "<image>" not in content:
        return prompt
    before = content.index("<image>") < content.index(f"<{tag}>")
    return ("<image>\n" + prompt) if before else (prompt + "<image>\n")


_lock = threading.Lock()
_state: dict = {"loaded": False, "tokenizer": None, "source": None, "sha256": None, "error": None,
                "fallback_counts": 0}


def _path() -> Path:
    return Path(os.environ.get("SHREW_V05_TOKENIZER") or VENDORED)


def load():
    """The tokenizer, or None (see ``status()['error']``). Loaded once."""
    if _state["loaded"]:
        return _state["tokenizer"]
    with _lock:
        if _state["loaded"]:
            return _state["tokenizer"]
        path = _path()
        try:
            from tokenizers import Tokenizer
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            _state.update(source=str(path), sha256=sha)
            if path == VENDORED and sha != VENDORED_SHA256:
                raise ValueError(f"vendored tokenizer hash {sha[:12]} != {VENDORED_SHA256[:12]}")
            _state["tokenizer"] = Tokenizer.from_str(data.decode("utf-8"))
        except Exception as e:                      # noqa: BLE001 — reported, never fatal
            _state.update(tokenizer=None, error=f"{type(e).__name__}: {e}", source=str(path))
            logger.error("v0.5 tokenizer unavailable (%s): prompt counts fall back to UTF-8 bytes "
                         "(over-counts, safe, less output room)", _state["error"])
        _state["loaded"] = True
        return _state["tokenizer"]


def reset():
    """Forget the loaded tokenizer (tests, env changes)."""
    with _lock:
        _state.update(loaded=False, tokenizer=None, source=None, sha256=None, error=None,
                      fallback_counts=0)


def status() -> dict:
    load()
    return {"exact": _state["tokenizer"] is not None, "source": _state["source"],
            "sha256": _state["sha256"], "matches_served_checkpoint": _state["sha256"] == VENDORED_SHA256,
            "error": _state["error"], "fallback_counts": _state["fallback_counts"]}


# vLLM in its "string" content format (what it picks for this template) hands a
# text message to the template as a plain string, and the template's string
# branch writes eight spaces before it; in "openai" format (a list of parts, as
# in training) there are none. Measured on a live serve 2026-11-14: a text page
# counted 777 was served as 778. Text prompts are counted both ways and charged
# the larger. Image prompts matched the live serve exactly.
STRING_FORMAT_INDENT = " " * 8


def render(command: str, content: str, string_format: bool = False) -> str:
    """The prompt string the chat template produces for system=command, user=content
    (``string_format``: the user content reached the template as a plain string)."""
    body = expand_tags(content)
    if string_format and body is content:
        body = STRING_FORMAT_INDENT + content
    return _PREFIX.format(command=command) + body + _SUFFIX


def count(rendered: str) -> int:
    """Tokens of a rendered prompt (special tokens are in the string)."""
    tok = load()
    if tok is None:
        _state["fallback_counts"] += 1
        return len(rendered.encode("utf-8"))
    return len(tok.encode(rendered, add_special_tokens=False).ids)


def text_tokens(text: str) -> int:
    """Tokens of ``text`` alone (no template): the size of a text page."""
    return count(text)


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Cut ``text`` into consecutive pieces of at most ``max_tokens`` tokens,
    each cut on a token boundary (never inside a character). The pieces join
    back to ``text`` exactly. Last resort for a run with no sentence end and no
    whitespace to split on."""
    if not text:
        return []
    tok = load()
    if tok is None:                                   # bytes bound: every piece <= max_tokens bytes
        pieces, cur, size = [], [], 0
        for ch in text:
            n = len(ch.encode("utf-8"))
            if cur and size + n > max_tokens:
                pieces.append("".join(cur))
                cur, size = [], 0
            cur.append(ch)
            size += n
        return pieces + ["".join(cur)]
    ends = [end for _, end in tok.encode(text, add_special_tokens=False).offsets]
    pieces, start, i = [], 0, 0
    step = max(1, max_tokens)
    while start < len(text):
        j = min(i + step, len(ends))
        cut = ends[j - 1] if j else len(text)
        while cut <= start and j < len(ends):         # tokens that share one character
            j += 1
            cut = ends[j - 1]
        if j >= len(ends):
            cut = len(text)
        piece = text[start:cut]
        # A piece re-tokenizes on its own; if that lands over the bound, give a little back.
        while count(piece) > max_tokens and j - i > 1:
            j -= max(1, (j - i) // 20)
            cut = max(ends[j - 1], start + 1)
            piece = text[start:cut]
        pieces.append(piece)
        start, i = cut, j
    return pieces


def text_prompt_tokens(command: str, text: str) -> int:
    """Prompt tokens of a text call: the larger of the two renderings a server
    can produce (see STRING_FORMAT_INDENT); exact for one, at most 2 over for the other."""
    return max(count(render(command, text)), count(render(command, text, string_format=True)))


def image_prompt_tokens(command: str, image_tokens: int) -> int:
    """Exact prompt tokens of an image call whose image costs ``image_tokens``."""
    if load() is None:
        _state["fallback_counts"] += 1
        return _FALLBACK_SCAFFOLD + len(command.encode("utf-8")) + 2 + image_tokens
    return count(render(command, IMAGE_PLACEHOLDER)) - 1 + image_tokens
