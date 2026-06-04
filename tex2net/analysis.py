from __future__ import annotations

import math

import networkx as nx


def _to_undirected(graph):
    return graph.to_undirected() if graph.is_directed() else graph.copy()


def _largest_connected_component(graph):
    undirected = _to_undirected(graph)
    if undirected.number_of_nodes() == 0:
        return undirected
    if nx.is_connected(undirected):
        return undirected
    component_nodes = max(nx.connected_components(undirected), key=len)
    return undirected.subgraph(component_nodes).copy()


def _safe_correlation(series_a, series_b):
    if len(series_a) < 2 or len(series_b) < 2:
        return 1.0

    mean_a = sum(series_a) / len(series_a)
    mean_b = sum(series_b) / len(series_b)
    centered_a = [value - mean_a for value in series_a]
    centered_b = [value - mean_b for value in series_b]
    std_a = math.sqrt(sum(value * value for value in centered_a))
    std_b = math.sqrt(sum(value * value for value in centered_b))
    if std_a == 0 and std_b == 0:
        return 1.0
    if std_a == 0 or std_b == 0:
        return 0.0

    numerator = sum(value_a * value_b for value_a, value_b in zip(centered_a, centered_b))
    correlation = numerator / (std_a * std_b)
    return float(correlation)


def _safe_degree_assortativity(graph):
    undirected = _to_undirected(graph)
    if undirected.number_of_edges() < 2:
        return 0.0

    source_degrees = []
    target_degrees = []
    for source, target in undirected.edges():
        source_degrees.append(undirected.degree(source))
        target_degrees.append(undirected.degree(target))
    return _safe_correlation(source_degrees, target_degrees)


def plot_character_centralities(graph, output_path="character_centralities.png", show=True):
    """Plot degree centrality and return the figure for reuse in notebooks or tests."""
    import matplotlib.pyplot as plt

    degree_centralities = nx.degree_centrality(graph)
    if not degree_centralities:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Character Centrality and Influence")
        return fig, ax

    sorted_centralities = sorted(degree_centralities.items(), key=lambda item: item[1], reverse=True)
    characters, centralities = zip(*sorted_centralities)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(characters, centralities)
    ax.set_xlabel("Characters")
    ax.set_ylabel("Degree Centrality")
    ax.set_title("Character Centrality and Influence")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    return fig, ax


def detect_communities_and_plot(graph, output_file="character_network_communities.html"):
    """Detect communities and render an interactive community-colored network."""
    from pyvis.network import Network

    partition = detect_communities(graph, as_partition=True)
    net = Network(notebook=False, width="100%", height="100%", bgcolor="#222222", font_color="white")
    communities = set(partition.values())
    community_colors = [f"#{hex(x)[2:]:0>6}" for x in range(256, 256 + max(1, len(communities)))]
    for node, community_index in partition.items():
        net.add_node(node, title=node, group=community_index, color=community_colors[community_index % len(community_colors)])
    for source, target in graph.edges():
        net.add_edge(source, target)
    net.show(output_file)
    return partition


def plot_community_interactions(graph, partition, output_path="community_interactions.png", show=True):
    """Plot interaction counts between communities and return the underlying matrix."""
    import matplotlib.pyplot as plt
    import pandas as pd

    community_ids = sorted(set(partition.values()))
    interaction_counts = pd.DataFrame(0, index=community_ids, columns=community_ids)
    for source, target in graph.edges():
        source_community = partition[source]
        target_community = partition[target]
        interaction_counts.at[source_community, target_community] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(interaction_counts, cmap="hot", interpolation="nearest")
    fig.colorbar(image, label="Interaction Counts")
    ax.set_title("Community Interactions")
    ax.set_xlabel("Target Community")
    ax.set_ylabel("Source Community")
    if output_path:
        fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    return fig, ax, interaction_counts


def describe_degree(graph, show=True):
    """Plot the degree distribution and return the figure."""
    import matplotlib.pyplot as plt

    degrees = [degree for _, degree in graph.degree()]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(degrees, bins=min(20, max(1, len(degrees))), color="lightblue")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title("Degree Distribution")
    if show:
        plt.show()
    return fig, ax


def analyze_temporal_relationships(graph, verbose=True):
    """Return temporal interaction metrics while preserving the existing printed report."""
    temporal_distances = []
    sentence_ids = []
    for _, _, data in graph.edges(data=True):
        ordered_sentence_ids = sorted(data.get("sentence_ids", []))
        sentence_ids.extend(ordered_sentence_ids)
        if len(ordered_sentence_ids) > 1:
            temporal_distances.extend(
                ordered_sentence_ids[index + 1] - ordered_sentence_ids[index]
                for index in range(len(ordered_sentence_ids) - 1)
            )

    metrics = {
        "average_temporal_distance": (sum(temporal_distances) / len(temporal_distances)) if temporal_distances else None,
        "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
        "reciprocity": nx.reciprocity(graph) if graph.is_directed() and graph.number_of_edges() else 0.0,
        "interaction_span": (max(sentence_ids) - min(sentence_ids)) if sentence_ids else 0,
        "interaction_events": len(sentence_ids),
    }

    if verbose:
        print("Temporal Relationship Analysis")
        print("-----------------------------")
        print(f"Average Temporal Distance: {metrics['average_temporal_distance']}")
        print(f"Density of Temporal Interactions: {metrics['density']}")
        print(f"Reciprocity of the Graph: {metrics['reciprocity']}")
        print(f"Interaction Span: {metrics['interaction_span']}")
        print(f"Interaction Events: {metrics['interaction_events']}")
        print()

    return metrics


def analyze_graph_characteristics(graph, verbose=True):
    """Return structural graph characteristics and optionally print them."""
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    metrics = {
        "number_of_nodes": num_nodes,
        "number_of_edges": num_edges,
        "average_degree": (sum(dict(graph.degree()).values()) / num_nodes) if num_nodes > 0 else 0.0,
        "average_clustering_coefficient": nx.average_clustering(_to_undirected(graph)) if num_nodes > 1 else 0.0,
        "density": nx.density(graph) if num_nodes > 1 else 0.0,
        "degree_assortativity": _safe_degree_assortativity(graph),
    }

    if verbose:
        print("Graph Characteristics:")
        print(f"Number of Nodes: {metrics['number_of_nodes']}")
        print(f"Number of Edges: {metrics['number_of_edges']}")
        print(f"Average Degree: {metrics['average_degree']}")
        print(f"Average Clustering Coefficient: {metrics['average_clustering_coefficient']}")
        print(f"Density: {metrics['density']}")
        print(f"Degree Assortativity: {metrics['degree_assortativity']}")

    return metrics


def detect_communities(graph, as_partition=False):
    """Detect communities using Louvain and return either groups or a node partition."""
    from networkx.algorithms import community

    undirected_graph = _to_undirected(graph)
    if undirected_graph.number_of_nodes() == 0:
        return {} if as_partition else {}
    if undirected_graph.number_of_edges() == 0:
        partition = {node: index for index, node in enumerate(undirected_graph.nodes())}
    else:
        partition = {}
        for community_index, community_nodes in enumerate(community.greedy_modularity_communities(undirected_graph)):
            for node in community_nodes:
                partition[node] = community_index
    if as_partition:
        return partition

    communities = {}
    for character, community_id in partition.items():
        communities.setdefault(community_id, []).append(character)
    return communities


def calculate_degree_centrality(graph):
    """Return degree centrality."""
    return nx.degree_centrality(graph)


def calculate_pagerank(graph):
    """Return PageRank scores as a narrative prominence signal."""
    if graph.number_of_nodes() == 0:
        return {}

    nodes = list(graph.nodes())
    node_count = len(nodes)
    damping = 0.85
    base_score = (1.0 - damping) / node_count
    scores = {node: 1.0 / node_count for node in nodes}

    for _ in range(100):
        updated_scores = {node: base_score for node in nodes}
        for node in nodes:
            outgoing = list(graph.successors(node)) if graph.is_directed() else list(graph.neighbors(node))
            if outgoing:
                share = scores[node] / len(outgoing)
                for neighbor in outgoing:
                    updated_scores[neighbor] += damping * share
            else:
                share = scores[node] / node_count
                for neighbor in nodes:
                    updated_scores[neighbor] += damping * share

        delta = sum(abs(updated_scores[node] - scores[node]) for node in nodes)
        scores = updated_scores
        if delta < 1e-9:
            break

    return scores


def calculate_k_core_numbers(graph):
    """Return k-core numbers for character cohesion analysis."""
    undirected_graph = _to_undirected(graph)
    if undirected_graph.number_of_nodes() == 0:
        return {}
    if undirected_graph.number_of_edges() == 0:
        return {node: 0 for node in undirected_graph.nodes()}
    return nx.core_number(undirected_graph)


def identify_bridge_characters(graph):
    """Identify structurally important bridge characters."""
    undirected_graph = _largest_connected_component(graph)
    if undirected_graph.number_of_nodes() == 0:
        return {}

    articulation_points = set(nx.articulation_points(undirected_graph)) if undirected_graph.number_of_edges() else set()
    betweenness = nx.betweenness_centrality(undirected_graph, endpoints=True)
    return {
        node: {
            "betweenness_centrality": float(betweenness.get(node, 0.0)),
            "articulation_point": node in articulation_points,
        }
        for node in undirected_graph.nodes()
    }


def summarize_graph_metrics(graph):
    """Return a compact network-science summary of the story graph."""
    pagerank = calculate_pagerank(graph)
    k_core = calculate_k_core_numbers(graph)
    bridges = identify_bridge_characters(graph)
    largest_component = _largest_connected_component(graph)

    summary = {
        "number_of_nodes": graph.number_of_nodes(),
        "number_of_edges": graph.number_of_edges(),
        "density": nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
        "reciprocity": nx.reciprocity(graph) if graph.is_directed() and graph.number_of_edges() else 0.0,
        "average_clustering": nx.average_clustering(_to_undirected(graph)) if graph.number_of_nodes() > 1 else 0.0,
        "average_shortest_path": nx.average_shortest_path_length(largest_component) if largest_component.number_of_nodes() > 1 else 0.0,
        "degree_assortativity": _safe_degree_assortativity(graph),
        "top_pagerank_character": max(pagerank, key=pagerank.get) if pagerank else None,
        "max_k_core": max(k_core.values()) if k_core else 0,
        "bridge_characters": [node for node, metrics in bridges.items() if metrics["articulation_point"]],
    }
    return summary


def analyze_network_resilience(graph, removal_fraction=0.1):
    """Measure how much the largest component shrinks after removing central nodes."""
    undirected_graph = _to_undirected(graph)
    if undirected_graph.number_of_nodes() == 0:
        return {"removed_nodes": [], "largest_component_ratio": 0.0}

    node_count = undirected_graph.number_of_nodes()
    removal_count = max(1, math.ceil(node_count * removal_fraction)) if node_count else 0
    ranked_nodes = sorted(nx.degree_centrality(undirected_graph).items(), key=lambda item: item[1], reverse=True)
    removed_nodes = [node for node, _ in ranked_nodes[:removal_count]]

    reduced_graph = undirected_graph.copy()
    reduced_graph.remove_nodes_from(removed_nodes)
    if reduced_graph.number_of_nodes() == 0:
        largest_component_ratio = 0.0
    elif reduced_graph.number_of_edges() == 0:
        largest_component_ratio = 1 / node_count
    else:
        largest_component = max(nx.connected_components(reduced_graph), key=len)
        largest_component_ratio = len(largest_component) / node_count

    return {
        "removed_nodes": removed_nodes,
        "largest_component_ratio": largest_component_ratio,
    }


def jaccard_similarity(set1, set2):
    """Compute the Jaccard similarity between two sets."""
    union = set1.union(set2)
    if not union:
        return 1.0
    return len(set1.intersection(set2)) / len(union)


def compare_node_similarity(g1, g2):
    """Compare node overlap using Jaccard similarity."""
    return jaccard_similarity(set(g1.nodes()), set(g2.nodes()))


def compare_edge_similarity(g1, g2, directed=False):
    """Compare edge overlap using Jaccard similarity."""
    if directed:
        edges1 = set(g1.edges())
        edges2 = set(g2.edges())
    else:
        edges1 = {frozenset(edge) for edge in g1.edges()}
        edges2 = {frozenset(edge) for edge in g2.edges()}
    return jaccard_similarity(edges1, edges2)


def compare_degree_distributions(g1, g2):
    """Compare degree distributions through Pearson correlation."""
    union_nodes = sorted(set(g1.nodes()).union(set(g2.nodes())))
    degrees1 = [g1.degree(node) if node in g1 else 0 for node in union_nodes]
    degrees2 = [g2.degree(node) if node in g2 else 0 for node in union_nodes]
    return _safe_correlation(degrees1, degrees2)


def compare_average_clustering(g1, g2):
    """Return the absolute difference in average clustering."""
    clustering1 = nx.average_clustering(_to_undirected(g1)) if g1.number_of_nodes() > 1 else 0.0
    clustering2 = nx.average_clustering(_to_undirected(g2)) if g2.number_of_nodes() > 1 else 0.0
    return abs(clustering1 - clustering2)


def compute_graph_edit_distance(g1, g2):
    """Compute graph edit distance, returning `None` if NetworkX cannot compute it."""
    try:
        return nx.graph_edit_distance(_to_undirected(g1), _to_undirected(g2))
    except Exception:
        return None


def compare_graph_density(g1, g2):
    """Return the absolute difference in graph density."""
    density1 = nx.density(g1) if g1.number_of_nodes() > 1 else 0.0
    density2 = nx.density(g2) if g2.number_of_nodes() > 1 else 0.0
    return abs(density1 - density2)


def compare_avg_shortest_path(g1, g2):
    """Compare average shortest path lengths on the largest connected components."""
    component1 = _largest_connected_component(g1)
    component2 = _largest_connected_component(g2)
    shortest_path_1 = nx.average_shortest_path_length(component1) if component1.number_of_nodes() > 1 else 0.0
    shortest_path_2 = nx.average_shortest_path_length(component2) if component2.number_of_nodes() > 1 else 0.0
    return abs(shortest_path_1 - shortest_path_2)


def compare_modularity(g1, g2):
    """Compare modularity differences between two graphs."""
    from networkx.algorithms import community

    def modularity(graph):
        undirected = _to_undirected(graph)
        if undirected.number_of_edges() == 0:
            return 0.0
        communities = community.greedy_modularity_communities(undirected)
        return community.modularity(undirected, communities)

    return abs(modularity(g1) - modularity(g2))


def compare_betweenness_correlations(g1, g2):
    """Compare betweenness centrality distributions using Pearson correlation."""
    union_nodes = sorted(set(g1.nodes()).union(set(g2.nodes())))
    betweenness_g1 = nx.betweenness_centrality(_to_undirected(g1), endpoints=True) if g1.number_of_nodes() else {}
    betweenness_g2 = nx.betweenness_centrality(_to_undirected(g2), endpoints=True) if g2.number_of_nodes() else {}
    betweenness1 = [betweenness_g1.get(node, 0.0) for node in union_nodes]
    betweenness2 = [betweenness_g2.get(node, 0.0) for node in union_nodes]
    return _safe_correlation(betweenness1, betweenness2)


def compare_graphs(g1, g2, directed_edges=False):
    """Compare two character networks using multiple topology measures."""
    pagerank1 = calculate_pagerank(g1)
    pagerank2 = calculate_pagerank(g2)
    union_nodes = sorted(set(g1.nodes()).union(set(g2.nodes())))
    pagerank_corr = _safe_correlation(
        [pagerank1.get(node, 0.0) for node in union_nodes],
        [pagerank2.get(node, 0.0) for node in union_nodes],
    )

    return {
        "node_jaccard": compare_node_similarity(g1, g2),
        "edge_jaccard": compare_edge_similarity(g1, g2, directed=directed_edges),
        "degree_corr": compare_degree_distributions(g1, g2),
        "avg_clustering_diff": compare_average_clustering(g1, g2),
        "graph_edit_distance": compute_graph_edit_distance(g1, g2),
        "density_diff": compare_graph_density(g1, g2),
        "avg_shortest_path_diff": compare_avg_shortest_path(g1, g2),
        "modularity_diff": compare_modularity(g1, g2),
        "betweenness_corr": compare_betweenness_correlations(g1, g2),
        "pagerank_corr": pagerank_corr,
    }


