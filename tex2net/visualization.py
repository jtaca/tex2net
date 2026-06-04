from __future__ import annotations

from collections import Counter

import networkx as nx


def _prepare_figure(figsize=(10, 10)):
    import matplotlib.pyplot as plt

    plt.rcParams["axes.facecolor"] = "white"
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    return fig, ax


def _finish_plot(fig, ax, title, show=True, output_path=None, axis_off=True):
    import matplotlib.pyplot as plt

    if axis_off:
        ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def visualize_graph(graph, title="Character Relationships", show=True, output_path=None):
    """Render a relationship graph with edge labels and optional file export."""
    layout = nx.spring_layout(graph, seed=42, k=4)
    fig, ax = _prepare_figure(figsize=(10, 10))
    nx.draw_networkx_nodes(graph, pos=layout, node_color="lightblue", node_size=7000, ax=ax)
    nx.draw_networkx_labels(graph, pos=layout, font_size=12, font_color="black", font_weight="bold", ax=ax)

    edge_labels = {}
    for source, target, data in graph.edges(data=True):
        actions = data.get("actions", [])
        sentence_ids = data.get("sentence_ids", [])
        weight = max(1, len(actions))
        bidirectional = data.get("bidirectional", True)
        edge_color = "lightgray" if bidirectional else "gray"
        arrow_style = "-|>" if bidirectional else "->"
        evidence_count = len(data.get("evidence_sentences", []))
        edge_labels[(source, target)] = f"{', '.join(actions)} (IDs: {', '.join(map(str, sentence_ids))}; ev: {evidence_count})"
        nx.draw_networkx_edges(
            graph,
            pos=layout,
            edgelist=[(source, target)],
            edge_color=edge_color,
            arrowstyle=arrow_style,
            connectionstyle="arc3,rad=0.1",
            width=weight * 1.5,
            ax=ax,
        )

    nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=edge_labels, font_size=8, font_color="gray", ax=ax)
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    return _finish_plot(fig, ax, title, show=show, output_path=output_path)


def visualize_pyvis_graph(graph, output_file="character_relationships.html", notebook=False):
    """Render an interactive Pyvis graph and return the output file path."""
    from pyvis.network import Network

    try:
        net = Network(
            notebook=notebook,
            width="100%",
            height="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=True,
            cdn_resources="remote",
        )
    except TypeError:
        net = Network(notebook=notebook, width="100%", height="100%", bgcolor="#ffffff", font_color="black", directed=True)
    for node, data in graph.nodes(data=True):
        aliases = data.get("aliases", [])
        title = node if not aliases else f"{node}<br>Aliases: {', '.join(aliases)}"
        net.add_node(node, size=30, title=title, color="lightblue", font={"color": "black", "size": 12, "face": "arial"})

    for source, target, data in graph.edges(data=True):
        actions = data.get("actions", [])
        sentence_ids = data.get("sentence_ids", [])
        weight = max(1, len(actions))
        bidirectional = data.get("bidirectional", True)
        edge_color = "#c4c4c4" if bidirectional else "#8a8a8a"
        edge_label = f"{', '.join(actions)}<br>Sentence IDs: {', '.join(map(str, sentence_ids))}"
        net.add_edge(source, target, width=weight * 1.5, color=edge_color, arrowStrikethrough=bidirectional, title=edge_label)

    net.barnes_hut(gravity=-8000, spring_length=110, central_gravity=0.1, damping=0.6)
    net.write_html(output_file, open_browser=False, notebook=notebook)
    return output_file


def visualize_directed_graph(graph, title="Character Relationships Directed Graph", show=True, output_path=None):
    """Render a directed graph with multiline edge labels."""
    pos = nx.spring_layout(graph, seed=42, k=3)
    fig, ax = _prepare_figure(figsize=(10, 10))
    nx.draw_networkx_nodes(graph, pos=pos, node_color="lightblue", node_size=3000, ax=ax)
    nx.draw_networkx_labels(graph, pos=pos, font_size=10, font_color="black", font_weight="bold", ax=ax)

    edge_labels = {}
    for source, target, data in graph.edges(data=True):
        actions = data.get("actions", [])
        sentence_ids = data.get("sentence_ids", [])
        edge_labels[(source, target)] = "\n".join(f"{action} (sent: {sentence_id})" for action, sentence_id in zip(actions, sentence_ids))

    nx.draw_networkx_edges(graph, pos=pos, arrows=True, arrowstyle="->", connectionstyle="arc3,rad=0.1", width=1.5, ax=ax)
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, font_size=8, font_color="gray", label_pos=0.5, ax=ax)
    return _finish_plot(fig, ax, title, show=show, output_path=output_path)


def visualize_directed_graph_styled(graph, title="Character Relationships Directed Graph", show=True, output_path=None):
    """Render a styled directed graph with larger nodes and weighted edges."""
    layout = nx.spring_layout(graph, seed=42, k=4)
    fig, ax = _prepare_figure(figsize=(10, 10))
    nx.draw_networkx_nodes(graph, pos=layout, node_color="lightblue", node_size=7000, ax=ax)
    nx.draw_networkx_labels(graph, pos=layout, font_size=12, font_color="black", font_weight="bold", ax=ax)

    edge_labels = {}
    for source, target, data in graph.edges(data=True):
        actions = data.get("actions", [])
        sentence_ids = data.get("sentence_ids", [])
        weight = max(1, len(actions))
        bidirectional = data.get("bidirectional", True)
        edge_color = "lightgray" if bidirectional else "gray"
        arrow_style = "fancy" if bidirectional else "->"
        edge_labels[(source, target)] = "\n".join(f"{action} (sent: {sentence_id})" for action, sentence_id in zip(actions, sentence_ids))
        nx.draw_networkx_edges(
            graph,
            pos=layout,
            edgelist=[(source, target)],
            edge_color=edge_color,
            arrowstyle=arrow_style,
            arrows=True,
            arrowsize=20,
            connectionstyle="arc3,rad=0.1",
            width=weight * 1.5,
            ax=ax,
            node_size=7000,
        )

    nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=edge_labels, font_size=8, font_color="gray", ax=ax)
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    return _finish_plot(fig, ax, title, show=show, output_path=output_path)


def visualize_directed_graph_styled_communities(
    graph,
    title="Character Relationships Directed Graph",
    edge_annotation="none",
    show=True,
    output_path=None,
):
    """Render the largest connected component using community coloring and centrality sizing."""
    import matplotlib.cm as cm
    from networkx.algorithms import community

    undirected = graph.to_undirected()
    fig, ax = _prepare_figure(figsize=(20, 15))
    if undirected.number_of_nodes() == 0:
        return _finish_plot(fig, ax, title, show=show, output_path=output_path)

    if nx.is_connected(undirected):
        component = undirected
    else:
        component = undirected.subgraph(max(nx.connected_components(undirected), key=len)).copy()

    try:
        centrality = nx.betweenness_centrality(component, k=min(10, max(1, component.number_of_nodes() - 1)), endpoints=True)
    except ValueError:
        centrality = nx.degree_centrality(component)

    communities = list(community.greedy_modularity_communities(component)) if component.number_of_edges() else [set(component.nodes())]
    community_index = {node: index for index, nodes in enumerate(communities) for node in nodes}
    cmap = cm.get_cmap("Pastel1", max(1, len(communities)))
    node_colors = [cmap(community_index[node]) for node in component.nodes()]
    node_sizes = [max(3000, centrality.get(node, 0.0) * 20000) for node in component.nodes()]
    pos = nx.spring_layout(component, k=0.15, seed=4572321)

    nx.draw_networkx_nodes(component, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_labels(component, pos, font_size=12, font_color="black", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(component, pos, edge_color="gainsboro", alpha=0.4, ax=ax)

    if edge_annotation.lower() != "none":
        edge_labels = {}
        for source, target, data in component.edges(data=True):
            actions = data.get("actions", [])
            sentence_ids = data.get("sentence_ids", [])
            if edge_annotation.lower() == "number":
                label = f"{len(actions)}"
            elif edge_annotation.lower() == "full":
                label = "\n".join(f"{action} (sent: {sentence_id})" for action, sentence_id in zip(actions, sentence_ids))
            else:
                label = ""
            edge_labels[(source, target)] = label
        nx.draw_networkx_edge_labels(component, pos, edge_labels=edge_labels, font_size=10, font_color="gray", ax=ax)

    ax.margins(0.1, 0.05)
    return _finish_plot(fig, ax, title, show=show, output_path=output_path)


def visualize_interaction_temporality(graph, mode="histogram", show=True, output_path=None):
    """Visualize temporal interaction patterns without requiring NumPy."""
    import matplotlib.pyplot as plt

    sentence_ids = []
    for _, _, data in graph.edges(data=True):
        sentence_ids.extend(data.get("sentence_ids", []))

    fig, ax = _prepare_figure(figsize=(10, 6))
    ax.axis("on")
    if not sentence_ids:
        ax.text(0.5, 0.5, "No temporal data available", ha="center", va="center")
        return _finish_plot(fig, ax, "Interaction Temporality", show=show, output_path=output_path, axis_off=False)

    ordered_sentence_ids = sorted(sentence_ids)
    if mode.lower() == "histogram":
        counts = Counter(ordered_sentence_ids)
        ax.bar(list(counts.keys()), list(counts.values()), color="skyblue", edgecolor="black")
        ax.set_xlabel("Sentence Number")
        ax.set_ylabel("Number of Interactions")
        title = "Histogram of Interactions Over Time"
    elif mode.lower() == "timeline":
        events = [data.get("sentence_ids", []) for _, _, data in graph.edges(data=True) if data.get("sentence_ids")]
        ax.eventplot(events, orientation="horizontal", colors="skyblue")
        ax.set_xlabel("Sentence Number")
        title = "Timeline of Interactions"
    elif mode.lower() == "cumulative":
        counts = Counter(ordered_sentence_ids)
        cumulative_x = []
        cumulative_y = []
        running_total = 0
        for sentence_id in sorted(counts):
            running_total += counts[sentence_id]
            cumulative_x.append(sentence_id)
            cumulative_y.append(running_total)
        ax.plot(cumulative_x, cumulative_y, marker="o", color="skyblue")
        ax.set_xlabel("Sentence Number")
        ax.set_ylabel("Cumulative Interactions")
        title = "Cumulative Interactions Over Time"
    else:
        raise ValueError("mode must be one of: histogram, timeline, cumulative")

    ax.set_title(title)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


