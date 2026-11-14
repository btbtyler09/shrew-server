"""latex_wrap.wrap_bare_latex — pure, idempotent, delimiter-only edits."""
import pytest
from app.latex_wrap import wrap_bare_latex as w, wrap_record


def test_bare_aligned_line_becomes_display():
    s = r"\begin{aligned} a &= b \\ c &= d \end{aligned}"
    assert w(s) == "$$" + s + "$$"


def test_multiline_env_stays_one_display():
    s = "Intro.\n\n\\begin{align*}\na &= 1 \\\\\nb &= 2\n\\end{align*}\n\nOutro."
    out = w(s)
    assert "$$\\begin{align*}" in out and "\\end{align*}$$" in out
    assert out.startswith("Intro.\n\n") and out.endswith("\n\nOutro.")


def test_lhs_and_glue_words_make_pure_line_display():
    s = r"A = \begin{bmatrix} 3 & 5 \\ 1 & 2 \end{bmatrix} and b = \begin{bmatrix} 7 \\ 1 \end{bmatrix}."
    assert w(s) == "$$" + s + "$$"


def test_env_inside_prose_is_wrapped_inline_by_default_and_display_on_request():
    s = r"The product is \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} as the identity matrix of the space."
    assert w(s) == s.replace(r"\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}", r"$\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$")
    assert "$$\\begin{pmatrix}" in w(s, mixed="display")


def test_pure_latex_line_wrapped():
    assert w(r"\frac{f'(x)}{f(x)} = -\frac{4x - 1}{2x - 1}") == r"$$\frac{f'(x)}{f(x)} = -\frac{4x - 1}{2x - 1}$$"
    assert w(r"= 3.14 \times 16 \times 2") == r"$$= 3.14 \times 16 \times 2$$"
    assert w(r"∴ OE=\sqrt{3},") == r"$$∴ OE=\sqrt{3},$$"


def test_text_inside_math_is_not_prose():
    s = r"\vdots \quad a_n = n^2 \cdot 2^n \quad \text{是公比为 2 的等比数列}"
    assert w(s) == "$$" + s + "$$"


@pytest.mark.parametrize("s", [
    r"$$\begin{aligned} a &= b \end{aligned}$$",
    r"$x = \frac{1}{2}$",
    r"Let \(a_n = \sum k\) be given.",
    "```\n\\begin{aligned} a \\end{aligned}\n```",
    r"# Title \frac{a}{b}",
    r"| \frac{a}{b} | c |",
    r"- item \frac{a}{b} = c",
    r"1. Simplify \frac{a}{b} and express your answer",
    r"~~\frac{4-3a}{b}=5~~",
    r"\begin{enumerate} \item one \end{enumerate}",
    r"Area A = \int_{2}^{4} 3x^{2} \, dx where the region lies under the curve",
    "plain text with no latex",
    r"x = 5",
    "",
])
def test_left_alone(s):
    assert w(s) == s


def test_idempotent_and_only_inserts_dollars():
    s = ("Text.\n\n\\begin{aligned} a &= b \\\\ c &= d \\end{aligned}\n\n"
         "\\frac{a}{b} = c\n\nSee \\begin{pmatrix} 1 \\end{pmatrix} for details of the thing.\n")
    o = w(s)
    assert w(o) == o
    assert o.replace("$", "") == s.replace("$", "")


def test_nested_same_env_and_unclosed():
    s = r"\begin{cases} x & \begin{cases} 1 \\ 2 \end{cases} \\ y \end{cases}"
    assert w(s) == "$$" + s + "$$"
    u = r"\begin{aligned} a &= b"
    assert w(u) == u


def test_non_str_passthrough():
    assert w(None) is None and w(5) == 5


def test_bad_mode():
    with pytest.raises(ValueError):
        w(r"\frac{a}{b}", mixed="nope")


def test_wrap_record_touches_only_chunks_and_markdown():
    rec = {"markdown": "x\n\\frac{a}{b} = c\n", "images": [], "metadata": {"k": 1},
           "semantic_chunks": [{"chunk_id": "p1_c1", "content": "\\frac{a}{b} = c", "title": "t"},
                               {"chunk_id": "p1_c2", "content": "plain"}]}
    new, st = wrap_record(rec)
    assert new["semantic_chunks"][0]["content"] == "$$\\frac{a}{b} = c$$"
    assert new["semantic_chunks"][1] == rec["semantic_chunks"][1]
    assert new["markdown"] == "x\n$$\\frac{a}{b} = c$$\n"
    assert new["metadata"] is rec["metadata"] and rec["markdown"].startswith("x\n\\frac")
    assert st == {"chunks_changed": 1, "markdown_changed": 1}


def test_env_settings_fail_fast(monkeypatch):
    import pytest
    from app.latex_wrap import env_settings
    monkeypatch.delenv("SHREW_LATEX_WRAP", raising=False)
    monkeypatch.delenv("SHREW_LATEX_WRAP_MIXED", raising=False)
    assert env_settings() == (True, "inline")
    monkeypatch.setenv("SHREW_LATEX_WRAP", "off")
    monkeypatch.setenv("SHREW_LATEX_WRAP_MIXED", " Display ")
    assert env_settings() == (False, "display")
    for name, bad in (("SHREW_LATEX_WRAP", "maybe"), ("SHREW_LATEX_WRAP_MIXED", "both")):
        monkeypatch.setenv("SHREW_LATEX_WRAP", "1"); monkeypatch.setenv("SHREW_LATEX_WRAP_MIXED", "inline")
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError):
            env_settings()
