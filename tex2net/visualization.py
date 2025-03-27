# character_interaction_graph/visualization.py

import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

def visualize_graph(graph, title="Character Relationships"):
    """
    Visualiza o grafo usando matplotlib.
    """
    layout = nx.spring_layout(graph, seed=42, k=4)
    plt.rcParams["axes.facecolor"] = "white"
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    nx.draw_networkx_nodes(graph, pos=layout, node_color="lightblue", node_size=7000, ax=ax)
    nx.draw_networkx_labels(graph, pos=layout, font_size=12, font_color="black", font_weight="bold", ax=ax)
    edge_labels = {}
    for source, target, data in graph.edges(data=True):
        actions = data["actions"]
        sentence_ids = data["sentence_ids"]
        weight = len(actions)
        bidirectional = data.get("bidirectional", True)
        edge_color = "lightgray" if bidirectional else "gray"
        arrow_style = "-|>" if bidirectional else "->"
        edge_labels[(source, target)] = f"{', '.join(actions)} (IDs: {', '.join(map(str, sentence_ids))})"
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
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

def visualize_pyvis_graph(graph, output_file="character_relationships.html"):
    """
    Visualiza o grafo de forma interativa utilizando Pyvis.
    """
    layout = nx.spring_layout(graph, seed=42, k=1.5)
    net = Network(notebook=True, width="100%", height="100%", bgcolor="#ffffff", font_color="black", directed=True)
    for node in graph.nodes:
        net.add_node(node, size=30, title=node, color="lightblue", font={"color": "black", "size": 12, "face": "bold"})
    for source, target, data in graph.edges(data=True):
        actions = data["actions"]
        sentence_ids = data["sentence_ids"]
        weight = len(actions)
        bidirectional = data.get("bidirectional", True)
        edge_color = "lightgray" if bidirectional else "gray"
        arrow_style = "-|>" if bidirectional else "->"
        edge_label = f"{', '.join(actions)} (IDs: {', '.join(map(str, sentence_ids))})"
        net.add_edge(source, target, width=weight * 1.5, color=edge_color, arrowStrikethrough=bidirectional, title=edge_label)
    net.barnes_hut(gravity=-8000, spring_length=100, central_gravity=0.1, damping=0.6)
    net.show(output_file)
