from app.generation import get_generation_params


def test_structured_page_stage_params():
    p = get_generation_params("shrew-9b", "structured_page")
    assert p["temperature"] == 0.3
    ex = p["extra_params"]
    assert ex["top_p"] == 0.8 and ex["top_k"] == 20 and ex["min_p"] == 0.0
    assert "repetition_penalty" not in ex
    assert ex["chat_template_kwargs"] == {"enable_thinking": False}
