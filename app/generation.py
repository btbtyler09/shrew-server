"""Centralized generation parameters for VLM calls.

Prompts and parameters are tuned for Qwen 3.5 models. When a Qwen 3.5
model is detected, model-specific flags (reasoning/thinking disable)
are included automatically. Other VLMs get the same sampling params
but without Qwen-specific flags.
"""

import re


def is_qwen35(model_name: str) -> bool:
    """Check if a model name refers to a Qwen 3.5 variant.

    Matches: "qwen3.5-35b", "qwen/qwen3.5-35b-a3b",
             "Qwen/Qwen3.5-397B-A17B", etc.
    """
    return bool(re.search(r"qwen[\s._-]*3\.?5", model_name, re.IGNORECASE))


def _qwen35_flags() -> dict:
    """Qwen 3.5 specific flags to disable reasoning/thinking.

    chat_template_kwargs works on all backends (OpenRouter, vLLM, llama.cpp).
    reasoning is OpenRouter-specific but harmless elsewhere.
    """
    return {
        "reasoning": {"enabled": False},
        "chat_template_kwargs": {"enable_thinking": False},
    }


_STAGE_PARAMS = {
    "transcribe": {
        "temperature": 0.3,
        "extra": {"top_p": 0.8, "top_k": 20, "repetition_penalty": 1.1},
    },
    "classify_figure": {
        "temperature": 0.7,
        "extra": {"top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
    },
    "structured": {
        "temperature": 0.2,
        "extra": {},
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

    if is_qwen35(model_name):
        extra.update(_qwen35_flags())

    return {
        "temperature": cfg["temperature"],
        "extra_params": extra or None,
    }
