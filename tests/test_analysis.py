import networkx as nx
from tex2net.analysis import (
    analyze_network_resilience,
    calculate_degree_centrality,
    calculate_k_core_numbers,
    detect_communities,
    identify_bridge_characters,
    summarize_graph_metrics,
    analyze_graph_characteristics,
    compare_avg_shortest_path,
)


def build_sample_graph():
    graph = nx.DiGraph()
    graph.add_edge("Alice", "Bob", actions=["meets"], sentence_ids=[1])
    graph.add_edge("Bob", "Charlie", actions=["greets"], sentence_ids=[2])
    graph.add_edge("Charlie", "Dana", actions=["supports"], sentence_ids=[3])
    graph.add_edge("Bob", "Dana", actions=["helps"], sentence_ids=[4])
    return graph


def test_calculate_degree_centrality():
    graph = build_sample_graph()
    centrality = calculate_degree_centrality(graph)
    assert isinstance(centrality, dict)
    assert "Alice" in centrality and "Bob" in centrality


def test_detect_communities():
    communities = detect_communities(build_sample_graph())
    assert isinstance(communities, dict)


def test_analyze_graph_characteristics(capfd):
    metrics = analyze_graph_characteristics(build_sample_graph())
    out, err = capfd.readouterr()
    assert "Number of Nodes:" in out
    assert metrics["number_of_nodes"] == 4


def test_identify_bridge_characters_flags_connector():
    graph = build_sample_graph()
    bridges = identify_bridge_characters(graph)
    assert bridges["Bob"]["articulation_point"] is True


def test_summarize_graph_metrics_exposes_network_science_features():
    summary = summarize_graph_metrics(build_sample_graph())
    assert summary["number_of_nodes"] == 4
    assert summary["top_pagerank_character"] in {"Bob", "Dana", "Charlie"}
    assert isinstance(summary["bridge_characters"], list)


def test_analyze_network_resilience_removes_central_nodes():
    resilience = analyze_network_resilience(build_sample_graph(), removal_fraction=0.25)
    assert resilience["removed_nodes"]
    assert 0 <= resilience["largest_component_ratio"] <= 1


def test_calculate_k_core_numbers_returns_all_nodes():
    core_numbers = calculate_k_core_numbers(build_sample_graph())
    assert set(core_numbers) == {"Alice", "Bob", "Charlie", "Dana"}


def test_compare_avg_shortest_path_handles_directed_graphs():
    graph_a = build_sample_graph()
    graph_b = build_sample_graph()
    graph_b.add_edge("Dana", "Alice", actions=["returns"], sentence_ids=[5])
    difference = compare_avg_shortest_path(graph_a, graph_b)
    assert difference >= 0
