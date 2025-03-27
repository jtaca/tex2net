# tests/test_analysis.py

import pytest
import networkx as nx
from tex2net.analysis import (
    calculate_degree_centrality,
    detect_communities,
    analyze_graph_characteristics
)

def test_calculate_degree_centrality():
    G = nx.DiGraph()
    G.add_edge("Alice", "Bob", actions=["meets"], sentence_ids=[1])
    centrality = calculate_degree_centrality(G)
    assert isinstance(centrality, dict)
    assert "Alice" in centrality and "Bob" in centrality

def test_detect_communities():
    G = nx.DiGraph()
    G.add_edge("Alice", "Bob", actions=["meets"], sentence_ids=[1])
    G.add_edge("Bob", "Charlie", actions=["greets"], sentence_ids=[2])
    communities = detect_communities(G)
    assert isinstance(communities, dict)

def test_analyze_graph_characteristics(capfd):
    G = nx.DiGraph()
    G.add_edge("Alice", "Bob", actions=["meets"], sentence_ids=[1])
    from tex2net.analysis import analyze_graph_characteristics
    analyze_graph_characteristics(G)
    out, err = capfd.readouterr()
    assert "Number of Nodes:" in out
