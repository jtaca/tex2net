# tests/test_rewriting.py

from tex2net import rewriting
from tex2net.rewriting import summarize_t5


def test_summarize_t5():
    text = "Alice meets Bob. Bob loves Alice."
    summary = summarize_t5(text)
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_summarize_t5_falls_back_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(
        rewriting,
        "_load_t5_components",
        lambda model_name: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    summary = summarize_t5("Alice meets Bob. Bob thanks Alice.")

    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "Alice" in summary or "Bob" in summary
