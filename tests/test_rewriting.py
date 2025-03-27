# tests/test_rewriting.py

import pytest
from tex2net.rewriting import rewrite_with_t5_base, summarize_t5

def test_rewrite_with_t5_base():
    text = "Alice loves Bob."
    rewritten = rewrite_with_t5_base(text)
    assert isinstance(rewritten, str)
    assert len(rewritten) > 0

def test_summarize_t5():
    text = "Alice meets Bob. Bob loves Alice."
    summary = summarize_t5(text)
    assert isinstance(summary, str)
    assert len(summary) > 0
