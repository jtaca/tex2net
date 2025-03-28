# tests/test_rewriting.py

import pytest
from tex2net.rewriting import summarize_t5

def test_summarize_t5():
    text = "Alice meets Bob. Bob loves Alice."
    summary = summarize_t5(text)
    assert isinstance(summary, str)
    assert len(summary) > 0
