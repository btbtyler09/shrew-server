"""Centralized generation parameters for VLM calls.

Prompts and parameters are tuned for modern Qwen 3.x instruct models
(3.5, 3.6, and later). When such a model is detected, model-specific
flags (reasoning/thinking disable) are included automatically. Other
VLMs get the same sampling params but without Qwen-specific flags.
"""

import os
import re


def is_qwen3x(model_name: str) -> bool:
    """Check if a model name refers to a modern Qwen 3.x variant.

    Matches: "qwen3.5-35b", "qwen3.6-35b-8bit", "qwen/qwen3.5-35b-a3b",
             "Qwen/Qwen3.6-397B-A17B", "qwen-3.7-...", etc.
    Does not match bare Qwen3 (e.g., "qwen3-8b") — those predate the
    thinking-tokens API and may not accept the flags.
    """
    return bool(re.search(r"qwen[\s._-]*3\.?[5-9]", model_name, re.IGNORECASE))


def _qwen3x_flags() -> dict:
    """Qwen 3.x specific flags to disable reasoning/thinking.

    chat_template_kwargs works on all backends (OpenRouter, vLLM, llama.cpp).
    reasoning is OpenRouter-specific but harmless elsewhere.
    """
    return {
        "reasoning": {"enabled": False},
        "chat_template_kwargs": {"enable_thinking": False},
    }


_STAGE_PARAMS = {
    "transcribe": {
        "temperature": 0.7,
        "extra": {"top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
    },
    "classify_figure": {
        "temperature": 0.7,
        "extra": {"top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
    },
    "structured": {
        # Match Qwen 3.x instruct-general-tasks recommendation. Running the
        # 2B model outside this distribution (low temp, no top_p/k) was a
        # major source of malformed JSON from the chunking adapter.
        "temperature": 0.7,
        "extra": {"top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
    },
    # shrew-ocr-preview §3, revised 2026-08-16: first pass is temperature 0
    # plus presence_penalty 0.3 and NOTHING else — no top_p/top_k/min_p, no
    # frequency/repetition penalties, no structured-output enforcement, no
    # template kwargs. The penalty was adopted from a 900-run decode matrix on
    # the production stack (shrew-server-private#4): one-shot 0.887 vs 0.880
    # penalty-free, table-page one-shot 1.000 vs 0.975, content/table fidelity
    # and chunk order flat — and it breaks the deterministic-loop rut that
    # greedy re-enters on retry. First-pass ENFORCEMENT was measured and
    # REJECTED there: grammar-constrained table transcription grinds past the
    # token cap (table one-shot 0.225) with worse table fidelity; enforcement
    # stays retry-only. presence_penalty is a flat per-distinct-token tax, so
    # repeated JSON scaffolding is safe (frequency_penalty would not be).
    # `no_model_flags` keeps the Qwen merge below from injecting anything if
    # the operator points VLM_MODEL at a Qwen-named endpoint.
    "structured_page": {
        "temperature": 0,
        "extra": ({"presence_penalty": float(os.environ.get("SHREW_TRY1_PP", "0.3"))}
                  if float(os.environ.get("SHREW_TRY1_PP", "0.3")) else {}),
        "no_model_flags": True,
    },
}


def get_generation_params(model_name: str, stage: str) -> dict:
    """Get generation parameters for a VLM call.

    Args:
        model_name: The model being called (used for Qwen 3.5 detection).
        stage: One of "transcribe", "classify_figure", "structured",
               "structured_page".

    Returns:
        Dict with "temperature" and "extra_params" keys.
    """
    cfg = _STAGE_PARAMS[stage]
    extra = dict(cfg["extra"])

    if is_qwen3x(model_name) and not cfg.get("no_model_flags"):
        extra.update(_qwen3x_flags())

    return {
        "temperature": cfg["temperature"],
        "extra_params": extra or None,
    }
