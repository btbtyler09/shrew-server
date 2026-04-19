"""Structured extraction via VLM.

Produces metadata, summary, and semantic chunks from clean markdown.
Uses the same pipeline prompts designed for Shrew dataset generation.
Eventually Shrew fine-tuned models replace the VLM ("fast" mode).
"""

import hashlib
import json
import logging
import os
import re
from typing import Optional

from .generation import get_generation_params
from .vlm_client import VLMClient

logger = logging.getLogger("shrew.structured")

# ─── Pipeline prompt loading ────────────────────────────────────────────────

_PIPELINES_DIR = os.path.join(os.path.dirname(__file__), "pipelines")


def _load_pipeline(name: str) -> dict:
    """Load a pipeline config from the pipelines directory."""
    path = os.path.join(_PIPELINES_DIR, f"{name}.json")
    with open(path) as f:
        return json.load(f)


# ─── Document sectioning ────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    """Approximate token count (roughly 4 chars per token for English)."""
    return len(text) // 4


def _section_document(
    content: str,
    max_tokens: int = 9000,
    overlap_tokens: int = 1000,
) -> list[dict]:
    """Split document into sections at paragraph boundaries.

    Uses fixed target sizes. Finds good break points near the target.
    """
    total_tokens = _approx_tokens(content)

    if total_tokens <= max_tokens:
        return [{
            "content": content,
            "section_index": 0,
            "total_sections": 1,
            "token_count": total_tokens,
        }]

    paragraphs = content.split("\n\n")
    sections = []
    current_paras = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _approx_tokens(para)

        if current_tokens + para_tokens > max_tokens and current_paras:
            # Save current section
            sections.append("\n\n".join(current_paras))

            # Build overlap from end of current section
            overlap_paras = []
            overlap_tok = 0
            for p in reversed(current_paras):
                pt = _approx_tokens(p)
                if overlap_tok + pt > overlap_tokens:
                    break
                overlap_paras.insert(0, p)
                overlap_tok += pt

            current_paras = overlap_paras
            current_tokens = overlap_tok

        current_paras.append(para)
        current_tokens += para_tokens

    if current_paras:
        sections.append("\n\n".join(current_paras))

    return [
        {
            "content": section,
            "section_index": i,
            "total_sections": len(sections),
            "token_count": _approx_tokens(section),
        }
        for i, section in enumerate(sections)
    ]


def _parse_json_response(text: str) -> dict:
    """Parse JSON from VLM response, handling markdown fences and preamble."""
    # Strip markdown code fences
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    # Find the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Repair attempt: fix common model output issues
    repaired = text
    # Fix unescaped backslashes (LaTeX in JSON strings: \alpha, \epsilon, etc.)
    # Escape any \ not followed by a valid JSON escape char: " \ / b f n r t u
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', repaired)
    # Fix \u not followed by 4 hex digits (e.g., \underset → \\underset)
    repaired = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', repaired)
    # Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    # Close unclosed structures
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    if open_braces > 0 or open_brackets > 0:
        repaired += "]" * open_brackets + "}" * open_braces
        # Clean trailing commas created by closing
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    try:
        result = json.loads(repaired)
        logger.warning("Repaired malformed JSON from model output")
        return result
    except json.JSONDecodeError:
        # Return original text for the caller's error handler
        return json.loads(text)  # will raise


# ─── Extract metadata ───────────────────────────────────────────────────────

def _apply_lora(vlm_client, params, task, lora_adapters, lora_format):
    """Apply LoRA adapter selection to VLMClient params.

    Returns (vlm_client, params, system_prompt_override).
    system_prompt_override is the task name when using LoRA (the adapters were
    trained with the task name as the system message), or None for main VLM.
    """
    if not lora_adapters or task not in lora_adapters:
        return vlm_client, params, None
    if lora_format == "openai":
        vlm_client = VLMClient(base_url=vlm_client.base_url, model=lora_adapters[task])
    elif lora_format == "llamacpp":
        extra = params.get("extra_params") or {}
        # llama.cpp requires all adapters in every request; active at 1.0, others at 0.0
        active_id = lora_adapters[task]
        extra["lora"] = [
            {"id": aid, "scale": 1.0 if aid == active_id else 0.0}
            for aid in lora_adapters.values()
        ]
        params["extra_params"] = extra
    return vlm_client, params, task


def _call_metadata_vlm(
    client: VLMClient,
    base_system_prompt: str,
    user_msg: str,
    lora_adapters: Optional[dict],
    lora_format: str,
) -> dict:
    """One metadata VLM call + parse. Raises JSONDecodeError on bad output."""
    params = get_generation_params(client.model, "structured")
    client, params, sys_override = _apply_lora(
        client, params, "extract_metadata", lora_adapters, lora_format,
    )
    effective_system = sys_override or base_system_prompt
    response = client.simple_completion(
        system_prompt=effective_system,
        user_content=user_msg,
        max_tokens=4096,
        **params,
    )
    return _parse_json_response(response)


def extract_metadata(
    clean_markdown: str,
    filename: str,
    pdf_path: str,
    vlm_client: VLMClient,
    num_pages: int = 0,
    lora_adapters: Optional[dict] = None,
    lora_format: str = "none",
    fallback_vlm_client: Optional[VLMClient] = None,
    fallback_lora_adapters: Optional[dict] = None,
    fallback_lora_format: str = "none",
) -> dict:
    """Extract document metadata via VLM.

    Order of attempts: primary → primary retry → fallback (if provided).
    """
    pipeline = _load_pipeline("extract_metadata")

    # Use first section only (metadata is in the front matter)
    sections = _section_document(clean_markdown, max_tokens=8000)
    first_section = sections[0]["content"]

    user_msg = pipeline["generation_user_template"].replace(
        "{filename}", filename
    ).replace(
        "{first_section_content}", first_section
    ).replace(
        "{section_content}", first_section
    )
    base_system_prompt = pipeline["generation_system_prompt"]

    logger.info("Extracting metadata via VLM")
    metadata: Optional[dict] = None
    try:
        metadata = _call_metadata_vlm(
            vlm_client, base_system_prompt, user_msg, lora_adapters, lora_format,
        )
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse metadata JSON: {e}")
        logger.info("Retrying metadata (primary)...")
        try:
            metadata = _call_metadata_vlm(
                vlm_client, base_system_prompt, user_msg, lora_adapters, lora_format,
            )
        except (json.JSONDecodeError, Exception) as e2:
            logger.warning(f"Metadata primary retry failed: {e2}")
            if fallback_vlm_client is not None:
                logger.warning(
                    f"Falling back to main VLM ({fallback_vlm_client.model}) for metadata"
                )
                try:
                    metadata = _call_metadata_vlm(
                        fallback_vlm_client, base_system_prompt, user_msg,
                        fallback_lora_adapters, fallback_lora_format,
                    )
                except (json.JSONDecodeError, Exception) as e3:
                    logger.error(f"Metadata fallback also failed: {e3}")
    except Exception as e:
        logger.warning(f"Metadata call failed: {e}")
        if fallback_vlm_client is not None:
            logger.warning(
                f"Falling back to main VLM ({fallback_vlm_client.model}) for metadata"
            )
            try:
                metadata = _call_metadata_vlm(
                    fallback_vlm_client, base_system_prompt, user_msg,
                    fallback_lora_adapters, fallback_lora_format,
                )
            except Exception as e3:
                logger.error(f"Metadata fallback also failed: {e3}")

    if metadata is None:
        metadata = {
            "title": None, "authors": [], "organization": None,
            "year": None, "type": None, "keywords": [],
        }

    # Add computed fields
    sha = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    metadata["id"] = sha.hexdigest()[:16]
    metadata["source_pages"] = num_pages
    metadata["num_chunks"] = 0  # filled after chunking

    logger.info(f"Metadata: title={metadata.get('title', '?')}, "
                f"authors={len(metadata.get('authors', []))}, "
                f"type={metadata.get('type', '?')}")
    return metadata


# ─── Generate summary ────────────────────────────────────────────────────────

def _call_summary_vlm(
    client: VLMClient,
    base_system_prompt: str,
    user_msg: str,
    lora_adapters: Optional[dict],
    lora_format: str,
) -> str:
    """One summary VLM call. Raises on error; no JSON parse."""
    params = get_generation_params(client.model, "structured")
    client, params, sys_override = _apply_lora(
        client, params, "summarize_document", lora_adapters, lora_format,
    )
    effective_system = sys_override or base_system_prompt
    return client.simple_completion(
        system_prompt=effective_system,
        user_content=user_msg,
        max_tokens=4096,
        **params,
    )


def generate_summary(
    clean_markdown: str,
    vlm_client: VLMClient,
    lora_adapters: Optional[dict] = None,
    lora_format: str = "none",
    fallback_vlm_client: Optional[VLMClient] = None,
    fallback_lora_adapters: Optional[dict] = None,
    fallback_lora_format: str = "none",
) -> str:
    """Generate a concise document summary.

    Order of attempts: primary → primary retry → fallback (if provided).
    Summary output is free-form text — fallback only triggers if the call
    itself raises (network, parse, timeout, etc.).
    """
    pipeline = _load_pipeline("summarize_document")

    sections = _section_document(clean_markdown, max_tokens=8000)
    first_section = sections[0]["content"]

    user_msg = pipeline["generation_user_template"].replace(
        "{section_content}", first_section
    )
    base_system_prompt = pipeline["generation_system_prompt"]

    logger.info("Generating summary via VLM")
    summary: Optional[str] = None
    try:
        summary = _call_summary_vlm(
            vlm_client, base_system_prompt, user_msg, lora_adapters, lora_format,
        )
    except Exception as e:
        logger.warning(f"Summary primary failed: {e}")
        logger.info("Retrying summary (primary)...")
        try:
            summary = _call_summary_vlm(
                vlm_client, base_system_prompt, user_msg, lora_adapters, lora_format,
            )
        except Exception as e2:
            logger.warning(f"Summary primary retry failed: {e2}")
            if fallback_vlm_client is not None:
                logger.warning(
                    f"Falling back to main VLM ({fallback_vlm_client.model}) for summary"
                )
                try:
                    summary = _call_summary_vlm(
                        fallback_vlm_client, base_system_prompt, user_msg,
                        fallback_lora_adapters, fallback_lora_format,
                    )
                except Exception as e3:
                    logger.error(f"Summary fallback also failed: {e3}")

    if not summary:
        summary = ""

    # Clean up — remove any JSON wrapper or markdown formatting
    summary = summary.strip().strip('"')
    logger.info(f"Summary: {len(summary)} chars")
    return summary


# ─── Semantic chunking ───────────────────────────────────────────────────────

# Max chars of previous-chunk context passed to a continuation request.
# Approx 1k tokens at 4 chars/token; bounds chunk-input growth so the model
# never exceeds its context window when sections happen to produce one big
# chunk that would otherwise be re-fed verbatim as continuation context.
_MAX_PREV_CHUNK_CHARS = 4000


def load_chunk_pipeline() -> dict:
    """Load the semantic_chunk pipeline config (exposed for streaming caller)."""
    return _load_pipeline("semantic_chunk")


def _truncate_prev_chunk(text: Optional[str]) -> str:
    """Trim previous-chunk context to a safe size for continuation prompts."""
    if not text:
        return ""
    if len(text) <= _MAX_PREV_CHUNK_CHARS:
        return text
    # Keep the tail — that's what semantically connects to the next section
    return "…" + text[-(_MAX_PREV_CHUNK_CHARS - 1):]


def _compute_chunk_max_tokens(section: dict, content: str, pipeline: dict) -> int:
    """Cap chunk output at ~2.5x input so runaway generation ends fast.

    A well-behaved chunker produces ≈1x the input (content preserved + JSON
    overhead). 2.5x leaves room for reformatting; the configured pipeline
    ceiling (e.g., 20000) is an absolute upper bound; 2048 is a floor so
    small sections don't get starved.
    """
    section_tokens = section.get("token_count") or _approx_tokens(content)
    dynamic_cap = int(section_tokens * 2.5)
    configured_cap = pipeline.get("max_output_tokens", 16384)
    return max(2048, min(dynamic_cap, configured_cap))


def _call_chunk_vlm(
    client: VLMClient,
    system_prompt: str,
    user_msg: str,
    max_tokens: int,
    lora_adapters: Optional[dict],
    lora_format: str,
) -> list[dict]:
    """One chunking VLM call + parse. Raises JSONDecodeError on bad output."""
    params = get_generation_params(client.model, "structured")
    client, params, sys_override = _apply_lora(
        client, params, "semantic_chunk", lora_adapters, lora_format,
    )
    effective_system = sys_override or system_prompt
    response = client.simple_completion(
        system_prompt=effective_system,
        user_content=user_msg,
        max_tokens=max_tokens,
        timeout=600,
        **params,
    )
    result = _parse_json_response(response)
    return result.get("semantic_chunks", [])


def chunk_one_section(
    section: dict,
    is_first: bool,
    last_chunk_content: Optional[str],
    next_chunk_id: int,
    vlm_client: VLMClient,
    lora_adapters: Optional[dict] = None,
    lora_format: str = "none",
    pipeline: Optional[dict] = None,
    section_label: Optional[str] = None,
    fallback_vlm_client: Optional[VLMClient] = None,
    fallback_lora_adapters: Optional[dict] = None,
    fallback_lora_format: str = "none",
) -> tuple[list[dict], int, Optional[str]]:
    """Chunk one section via VLM. Returns (chunks, new_next_chunk_id, last_chunk_content).

    Order of attempts:
      1. Primary (shrew) call
      2. Retry primary once on JSON parse failure
      3. If fallback_vlm_client is set, one call against the main VLM

    If all attempts fail, returns ([], next_chunk_id, last_chunk_content).
    """
    if pipeline is None:
        pipeline = _load_pipeline("semantic_chunk")
    content = section["content"]
    label = section_label or f"section {section.get('section_index', '?')}"

    if is_first:
        system_prompt = pipeline["generation_system_prompt"]
        user_msg = pipeline["generation_user_template"].replace(
            "{section_content}", content
        )
    else:
        system_prompt = pipeline["continuation_system_prompt"]
        prev_for_prompt = _truncate_prev_chunk(last_chunk_content)
        user_msg = pipeline["continuation_user_template"].replace(
            "{previous_chunk}", prev_for_prompt
        ).replace(
            "{next_chunk_id}", str(next_chunk_id)
        ).replace(
            "{section_content}", content
        )

    max_tokens = _compute_chunk_max_tokens(section, content, pipeline)
    logger.info(
        f"Chunking {label} (~{section.get('token_count', '?')} tokens, "
        f"max_out={max_tokens}, next_id={next_chunk_id})"
    )

    chunks: list[dict] = []
    try:
        chunks = _call_chunk_vlm(
            vlm_client, system_prompt, user_msg, max_tokens,
            lora_adapters, lora_format,
        )
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse chunks JSON for {label}: {e}")
        logger.info(f"Retrying {label} (primary)...")
        try:
            chunks = _call_chunk_vlm(
                vlm_client, system_prompt, user_msg, max_tokens,
                lora_adapters, lora_format,
            )
        except (json.JSONDecodeError, Exception) as e2:
            logger.warning(f"Primary retry failed for {label}: {e2}")
            if fallback_vlm_client is not None:
                logger.warning(
                    f"Falling back to main VLM ({fallback_vlm_client.model}) for {label}"
                )
                try:
                    chunks = _call_chunk_vlm(
                        fallback_vlm_client, system_prompt, user_msg, max_tokens,
                        fallback_lora_adapters, fallback_lora_format,
                    )
                except (json.JSONDecodeError, Exception) as e3:
                    logger.error(f"Fallback also failed for {label}: {e3}")
                    chunks = []
    except Exception as e:
        logger.warning(f"Chunk call failed for {label}: {e}")
        if fallback_vlm_client is not None:
            logger.warning(
                f"Falling back to main VLM ({fallback_vlm_client.model}) for {label}"
            )
            try:
                chunks = _call_chunk_vlm(
                    fallback_vlm_client, system_prompt, user_msg, max_tokens,
                    fallback_lora_adapters, fallback_lora_format,
                )
            except Exception as e3:
                logger.error(f"Fallback also failed for {label}: {e3}")
                chunks = []

    if chunks:
        for chunk in chunks:
            chunk["chunk_id"] = str(next_chunk_id)
            next_chunk_id += 1
        last_chunk_content = chunks[-1].get("content", "") or last_chunk_content
        logger.info(f"  Got {len(chunks)} chunks from {label}")
    else:
        logger.warning(f"  No chunks returned for {label}")

    return chunks, next_chunk_id, last_chunk_content


def semantic_chunk(
    clean_markdown: str,
    vlm_client: VLMClient,
    lora_adapters: Optional[dict] = None,
    lora_format: str = "none",
    section_max_tokens: int = 6000,
    fallback_vlm_client: Optional[VLMClient] = None,
    fallback_lora_adapters: Optional[dict] = None,
    fallback_lora_format: str = "none",
) -> list[dict]:
    """Split document into semantic chunks.

    For long documents, sections the text and processes each section
    sequentially with continuation context. If a fallback VLM client is
    provided, per-section chunking falls back to it when the primary
    model fails to produce valid JSON.
    """
    pipeline = _load_pipeline("semantic_chunk")

    sections = _section_document(
        clean_markdown, max_tokens=section_max_tokens, overlap_tokens=1000,
    )
    all_chunks = []
    last_chunk_content = None
    next_chunk_id = 1

    logger.info(f"Semantic chunking: {len(sections)} sections")

    for i, section in enumerate(sections):
        chunks, next_chunk_id, last_chunk_content = chunk_one_section(
            section,
            is_first=(i == 0),
            last_chunk_content=last_chunk_content,
            next_chunk_id=next_chunk_id,
            vlm_client=vlm_client,
            lora_adapters=lora_adapters,
            lora_format=lora_format,
            pipeline=pipeline,
            section_label=f"section {i + 1}/{len(sections)}",
            fallback_vlm_client=fallback_vlm_client,
            fallback_lora_adapters=fallback_lora_adapters,
            fallback_lora_format=fallback_lora_format,
        )
        all_chunks.extend(chunks)

    all_chunks = _dedup_consecutive_chunks(all_chunks)
    logger.info(f"Semantic chunking complete: {len(all_chunks)} total chunks")
    return all_chunks


# ─── Chunk deduplication ───────────────────────────────────────────────────

def _dedup_consecutive_chunks(
    chunks: list[dict], threshold: float = 0.6,
) -> list[dict]:
    """Remove near-duplicate consecutive chunks from section boundary overlap."""
    if len(chunks) < 2:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_words = set(result[-1].get("content", "").split())
        curr_words = set(chunks[i].get("content", "").split())
        if not prev_words or not curr_words:
            result.append(chunks[i])
            continue
        jaccard = len(prev_words & curr_words) / len(prev_words | curr_words)
        if jaccard > threshold:
            # Keep the longer chunk
            if len(curr_words) > len(prev_words):
                result[-1] = chunks[i]
            logger.info(f"Chunk dedup: dropped chunk (jaccard={jaccard:.2f})")
        else:
            result.append(chunks[i])

    # Renumber sequentially
    for i, chunk in enumerate(result):
        chunk["chunk_id"] = str(i + 1)

    if len(result) < len(chunks):
        logger.info(f"Chunk dedup: {len(chunks)} -> {len(result)} chunks")
    return result


# ─── Heading hierarchy fix ─────────────────────────────────────────────────

def fix_heading_hierarchy(
    clean_markdown: str,
    vlm_client: VLMClient,
    fallback_vlm_client: Optional[VLMClient] = None,
    lora_adapters: Optional[dict] = None,
    lora_format: str = "none",
    return_changes: bool = False,
):
    """Fix heading levels using VLM understanding of document structure.

    Extracts headings, sends them to the model as plain markdown,
    parses the corrected headings, and reinserts them.
    Falls back to the fallback client if the primary fails.

    By default returns the corrected markdown string. If return_changes=True,
    returns (markdown, changes_map) where changes_map maps stripped heading
    text → new_level. Streaming callers use this map to patch chunk content
    after the fact.
    """
    pipeline = _load_pipeline("fix_headings")

    # Extract headings with line positions
    lines = clean_markdown.split("\n")
    headings = []
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            headings.append({"line_idx": i, "level": len(m.group(1)), "text": m.group(2)})

    def _ret(md: str, changes: dict) -> object:
        return (md, changes) if return_changes else md

    if len(headings) < 3:
        return _ret(clean_markdown, {})

    # Format as plain markdown headings
    heading_md = "\n".join("#" * h["level"] + " " + h["text"] for h in headings)
    user_msg = pipeline["generation_user_template"].replace("{heading_list}", heading_md)
    system_prompt = pipeline["generation_system_prompt"]

    # Try primary client, then fallback
    for client in [vlm_client, fallback_vlm_client]:
        if client is None:
            continue
        params = get_generation_params(client.model, "structured")
        # Apply LoRA scale 0 for llamacpp (no adapter for this task)
        if lora_format == "llamacpp" and lora_adapters:
            extra = params.get("extra_params") or {}
            extra["lora"] = [{"id": aid, "scale": 0.0} for aid in lora_adapters.values()]
            params["extra_params"] = extra

        try:
            response = client.simple_completion(
                system_prompt=system_prompt,
                user_content=user_msg,
                max_tokens=4096,
                **params,
            )

            # Parse response: extract heading lines
            fixed = []
            for rline in response.strip().split("\n"):
                rline = rline.strip()
                m = re.match(r"^(?:\d+\.\s*)?(#{1,6})\s+(.+)$", rline)
                if m:
                    fixed.append({"level": len(m.group(1)), "text": m.group(2)})

            changes_map: dict[str, int] = {}

            # Match back to input by text — handles hallucinated extras
            if len(fixed) == len(headings):
                # Exact count match — apply in order
                changes = 0
                for orig, fix in zip(headings, fixed):
                    new_level = fix["level"]
                    if 1 <= new_level <= 6 and new_level != orig["level"]:
                        lines[orig["line_idx"]] = "#" * new_level + " " + orig["text"]
                        changes_map[orig["text"].strip()] = new_level
                        changes += 1
                logger.info(f"Heading fix: {changes} changes via {client.model}")
                return _ret("\n".join(lines), changes_map)

            # Count mismatch — try matching by text content
            fix_map = {}
            for f in fixed:
                fix_map.setdefault(f["text"].strip(), f["level"])

            changes = 0
            for orig in headings:
                new_level = fix_map.get(orig["text"].strip())
                if new_level and 1 <= new_level <= 6 and new_level != orig["level"]:
                    lines[orig["line_idx"]] = "#" * new_level + " " + orig["text"]
                    changes_map[orig["text"].strip()] = new_level
                    changes += 1

            if changes > 0:
                logger.info(f"Heading fix: {changes} changes via {client.model} (text-matched)")
                return _ret("\n".join(lines), changes_map)

            logger.warning(f"Heading fix: no usable changes from {client.model}")

        except Exception as e:
            logger.warning(f"Heading fix failed with {client.model}: {e}")
            continue

    logger.warning("Heading hierarchy fix failed on all models, returning original")
    return _ret(clean_markdown, {})


def patch_chunk_headings(chunks: list[dict], changes_map: dict) -> int:
    """Apply heading-level changes to chunk content. Returns number of edits."""
    if not changes_map or not chunks:
        return 0
    edits = 0
    pat = re.compile(r"^(#{1,6})\s+(.+)$")
    for chunk in chunks:
        content = chunk.get("content", "")
        if not content or "#" not in content:
            continue
        new_lines = []
        changed = False
        for line in content.split("\n"):
            m = pat.match(line)
            if m:
                text = m.group(2).strip()
                new_level = changes_map.get(text)
                if new_level and 1 <= new_level <= 6 and new_level != len(m.group(1)):
                    new_lines.append("#" * new_level + " " + m.group(2))
                    changed = True
                    edits += 1
                    continue
            new_lines.append(line)
        if changed:
            chunk["content"] = "\n".join(new_lines)
    return edits
