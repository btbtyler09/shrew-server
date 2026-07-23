from app.generation import get_generation_params


def test_structured_page_stage_is_bare_greedy():
    """SHREW_OCR_PREVIEW.md §3: first-pass sampling is EXACTLY temperature 0
    with no other params. Gate F measured penalty-free unconstrained greedy —
    any decode knob added here voids every published eval number, and
    smoke_test_preview will report DRIFT."""
    p = get_generation_params("shrew-ocr-preview", "structured_page")
    assert p["temperature"] == 0
    assert not p["extra_params"], f"expected no extra params, got {p['extra_params']}"


def test_structured_page_rejects_every_forbidden_knob():
    ex = get_generation_params("shrew-ocr-preview", "structured_page")["extra_params"] or {}
    for knob in ("top_p", "top_k", "min_p", "repetition_penalty",
                 "frequency_penalty", "presence_penalty", "guided_json",
                 "structured_outputs", "chat_template_kwargs"):
        assert knob not in ex, f"{knob} must not be sent on the first pass"


def test_structured_page_ignores_qwen_flag_merge():
    """The stage opts out of model-specific flags: whatever the operator names
    the endpoint, the contract admits no extra params."""
    p = get_generation_params("qwen3.5-35b", "structured_page")
    assert p["temperature"] == 0
    assert not p["extra_params"]


def test_other_stages_still_get_their_qwen_flags():
    """The opt-out is scoped to structured_page and must not regress the
    legacy pipeline stages."""
    ex = get_generation_params("qwen3.5-35b", "structured")["extra_params"]
    assert ex["chat_template_kwargs"] == {"enable_thinking": False}
