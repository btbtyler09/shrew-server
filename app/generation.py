"""Centralized generation parameters for VLM calls.

Prompts and parameters are tuned for modern Qwen 3.x instruct models
(3.5, 3.6, and later). When such a model is detected, model-specific
flags (reasoning/thinking disable) are included automatically. Other
VLMs get the same sampling params but without Qwen-specific flags.
"""

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
        # shrew 2B outside this distribution (low temp, no top_p/k) was a
        # major source of malformed JSON from the chunking adapter.
        "temperature": 0.7,
        "extra": {"top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
    },
}


def get_generation_params(model_name: str, stage: str) -> dict:
    """Get generation parameters for a VLM call.

    Args:
        model_name: The model being called (used for Qwen 3.5 detection).
        stage: One of "transcribe", "classify_figure", "structured".

    Returns:
        Dict with "temperature" and "extra_params" keys.
    """
    cfg = _STAGE_PARAMS[stage]
    extra = dict(cfg["extra"])

    if is_qwen3x(model_name):
        extra.update(_qwen3x_flags())

    return {
        "temperature": cfg["temperature"],
        "extra_params": extra or None,
    }
