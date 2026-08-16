from app.generation import get_generation_params


def test_structured_page_stage_is_greedy_plus_mild_presence_penalty():
    """§3 revised 2026-08-16 (900-run decode matrix on the production stack,
    shrew-server-private#4): first pass is EXACTLY temperature 0 +
    presence_penalty 0.3 and nothing else. The penalty measured one-shot
    0.887 vs 0.880 bare, table-page one-shot 1.000 vs 0.975, with content and
    table fidelity flat — and it breaks the deterministic loop rut greedy
    re-enters on retry. presence is a flat per-distinct-token tax, safe for
    repeated JSON scaffolding (frequency_penalty is NOT and stays banned)."""
    p = get_generation_params("shrew-ocr-preview", "structured_page")
    assert p["temperature"] == 0
    assert p["extra_params"] == {"presence_penalty": 0.3}


def test_structured_page_rejects_every_forbidden_knob():
    ex = get_generation_params("shrew-ocr-preview", "structured_page")["extra_params"] or {}
    for knob in ("top_p", "top_k", "min_p", "repetition_penalty",
                 "frequency_penalty", "guided_json",
                 "structured_outputs", "chat_template_kwargs"):
        assert knob not in ex, f"{knob} must not be sent on the first pass"
    assert set(ex) <= {"presence_penalty"}   # the ONE admitted knob


def test_structured_page_ignores_qwen_flag_merge():
    """The stage opts out of model-specific flags: whatever the operator names
    the endpoint, the contract admits only the §3 presence penalty."""
    p = get_generation_params("qwen3.5-35b", "structured_page")
    assert p["temperature"] == 0
    assert p["extra_params"] == {"presence_penalty": 0.3}


def test_other_stages_still_get_their_qwen_flags():
    """The opt-out is scoped to structured_page and must not regress the
    legacy pipeline stages."""
    ex = get_generation_params("qwen3.5-35b", "structured")["extra_params"]
    assert ex["chat_template_kwargs"] == {"enable_thinking": False}
