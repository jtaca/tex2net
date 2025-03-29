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


def visualize_directed_graph(graph, title="Character Relationships Directed Graph"):
    """
    Parameters:
        graph (networkx.DiGraph): The directed graph to be visualized.
        title (str): The title for the visualization.
    Visualizes a directed graph using Matplotlib and labels edges with actions,
    each on a new line.
    """
    import matplotlib.pyplot as plt

    pos = nx.spring_layout(graph, seed=42, k=3)
    plt.figure(figsize=(10, 10), facecolor="white")
    nx.draw_networkx_nodes(graph, pos=pos, node_color="lightblue", node_size=3000)
    nx.draw_networkx_labels(graph, pos=pos, font_size=10, font_color="black", font_weight="bold")

    # Build edge labels (multi-line)
    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        actions = data.get("actions", [])
        s_ids = data.get("sentence_ids", [])
        combined_label = []
        for a, sid in zip(actions, s_ids):
            # Example: "discussing ideas (sent: 1)"
            combined_label.append(f"{a} (sent: {sid})")
        # Put each label on a new line
        edge_labels[(u, v)] = "\n".join(combined_label)

    nx.draw_networkx_edges(
        graph,
        pos=pos,
        arrows=True,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.1",
        width=1.5
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos=pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="gray",
        label_pos=0.5
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.show()



def visualize_directed_graph_styled(graph, title="Character Relationships Directed Graph"):
    """
    Visualizes a directed graph with increased spacing using Matplotlib.
    
    Parameters:
        graph (networkx.DiGraph): The directed graph to be visualized.
        title (str): The title for the visualization.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    # Create a directed graph layout with increased spacing
    layout = nx.spring_layout(graph, seed=42, k=4)

    # Set the background color to white
    plt.rcParams["axes.facecolor"] = "white"

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")

    # Draw nodes and labels
    nx.draw_networkx_nodes(graph, pos=layout, node_color="lightblue", node_size=7000, ax=ax)
    nx.draw_networkx_labels(graph, pos=layout, font_size=12, font_color="black", font_weight="bold", ax=ax)

    # Draw edges with annotations
    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        actions = data.get("actions", [])
        s_ids = data.get("sentence_ids", [])
        weight = len(actions)
        bidirectional = data.get("bidirectional", True)
        edge_color = "lightgray" if bidirectional else "gray"
        arrow_style = "fancy" if bidirectional else "->"
        combined_label = []
        
        for a, sid in zip(actions, s_ids):
            # Example: "discussing ideas (sent: 1)"
            combined_label.append(f"{a} (sent: {sid})")
        # Put each label on a new line
        edge_labels[(u, v)] = "\n".join(combined_label)

        nx.draw_networkx_edges(
            graph,
            pos=layout,
            edgelist=[(u, v)],
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

    # Set axis limits and remove them
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.axis("off")

    # Set title
    ax.set_title(title, fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community
import matplotlib.cm as cm

def visualize_directed_graph_styled_communities(graph, title="Character Relationships Directed Graph", edge_annotation="none"):
    """
    Visualizes a directed graph using a community-based layout.
    The graph is first converted to its largest connected undirected component,
    then nodes are colored by community (using a pastel colormap for light colors)
    and sized by a centrality measure.
    
    If betweenness centrality calculation fails (due to a small population),
    degree centrality is used instead and a message is printed.
    
    Parameters:
        graph (networkx.DiGraph): The directed graph to be visualized.
        title (str): Title for the visualization.
        edge_annotation (str): How to annotate edges. Options:
            - "none": no annotation,
            - "number": annotate with number of interactions,
            - "full": annotate with full details (each action and sentence id).
    """

    from networkx.algorithms import community
    # Convert to undirected and select the largest connected component.
    H = graph.to_undirected()
    components = nx.connected_components(H)
    largest_component = max(components, key=len)
    H = H.subgraph(largest_component).copy()
    
    # Compute centrality for node sizing.
    try:
        centrality = nx.betweenness_centrality(H, k=10, endpoints=True)
    except ValueError as e:
        print("Betweenness centrality calculation failed (likely due to small population). Using degree centrality instead.")
        centrality = nx.degree_centrality(H)
    
    # Compute community structure using the greedy modularity algorithm.
    lpc = community.greedy_modularity_communities(H)
    community_index = {node: i for i, com in enumerate(lpc) for node in com}
    
    # Use a pastel colormap to get very light colors.
    num_communities = max(community_index.values()) + 1
    cmap = cm.get_cmap("Pastel1", num_communities)
    node_colors = [cmap(community_index[node]) for node in H.nodes()]
    
    # Set up the layout with increased spacing.
    pos = nx.spring_layout(H, k=0.15, seed=4572321)
    
    # Prepare node sizes.
    node_size = [centrality[node] * 20000 for node in H.nodes()]
    
    # Set up the figure.
    fig, ax = plt.subplots(figsize=(20, 15), facecolor="white")
    
    # Draw nodes and labels.
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=node_size, ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=12, font_color="black", font_weight="bold", ax=ax)
    
    # Draw edges.
    nx.draw_networkx_edges(H, pos, edge_color="gainsboro", alpha=0.4, ax=ax)
    
    # Optionally, annotate edges.
    if edge_annotation.lower() != "none":
        edge_labels = {}
        for u, v, data in H.edges(data=True):
            actions = data.get("actions", [])
            sentence_ids = data.get("sentence_ids", [])
            if edge_annotation.lower() == "number":
                label = f"{len(actions)}"
            elif edge_annotation.lower() == "full":
                combined_label = []
                for a, sid in zip(actions, sentence_ids):
                    combined_label.append(f"{a} (sent: {sid})")
                label = "\n".join(combined_label)
            else:
                label = ""
            edge_labels[(u, v)] = label
        
        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=10, font_color="gray", ax=ax)
    
    # Title.
    font = {"color": "k", "fontweight": "bold", "fontsize": 20}
    ax.set_title(title, fontdict=font)
    
    # Instead of drawing the red legend text on the figure,
    # print the legend info to the console.
    print("Legend: node color = community structure")
    print("Legend: node size = centrality measure")
    
    # Remove axis and adjust margins.
    ax.axis("off")
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    
    plt.show()




import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def visualize_interaction_temporality(graph, mode="histogram"):
    """
    Visualizes the temporality of interactions in a character network.
    
    This function expects that each edge in the graph has an attribute 'sentence_ids',
    which is a list of sentence numbers (or time markers) where the interaction occurs.
    
    Parameters:
        graph (networkx.DiGraph or networkx.Graph): The character network.
        mode (str): The mode of visualization:
            - "histogram": A histogram of interactions per sentence.
            - "timeline": An event plot showing interactions along a horizontal timeline.
            - "cumulative": A plot showing the cumulative count of interactions over time.
    
    Returns:
        None. Displays the plot using matplotlib.
        
    Robust error handling is applied to catch and report issues during processing.
    """
    try:
        # Extract all sentence_ids from graph edges.
        sentence_ids = []
        for u, v, data in graph.edges(data=True):
            s_ids = data.get("sentence_ids", [])
            if s_ids:
                sentence_ids.extend(s_ids)
        
        if not sentence_ids:
            print("No temporal data ('sentence_ids') found in graph edges.")
            return
        
        # Ensure sentence_ids is a numpy array of numbers.
        sentence_ids = np.array(sentence_ids, dtype=float)
        
        if mode.lower() == "histogram":
            try:
                plt.figure(figsize=(10, 6))
                # Use bins covering the full range of sentence numbers.
                bins = np.arange(np.min(sentence_ids), np.max(sentence_ids) + 2) - 0.5
                plt.hist(sentence_ids, bins=bins, color="skyblue", edgecolor="black")
                plt.xlabel("Sentence Number")
                plt.ylabel("Number of Interactions")
                plt.title("Histogram of Interactions Over Time (by Sentence Number)")
                plt.show()
            except Exception as e:
                print(f"Failed to create histogram: {e}")
        
        elif mode.lower() == "timeline":
            try:
                # Create an event plot: each edge's events are one line of markers.
                events = []
                for u, v, data in graph.edges(data=True):
                    s_ids = data.get("sentence_ids", [])
                    if s_ids:
                        events.append(s_ids)
                plt.figure(figsize=(10, 6))
                plt.eventplot(events, orientation='horizontal', colors='skyblue')
                plt.xlabel("Sentence Number")
                plt.title("Timeline of Interactions")
                plt.show()
            except Exception as e:
                print(f"Failed to create timeline plot: {e}")
        
        elif mode.lower() == "cumulative":
            try:
                min_sent = int(np.min(sentence_ids))
                max_sent = int(np.max(sentence_ids))
                # Create bins for each sentence number.
                bins = np.arange(min_sent, max_sent + 1)
                hist, bin_edges = np.histogram(sentence_ids, bins=bins)
                cumulative = np.cumsum(hist)
                plt.figure(figsize=(10, 6))
                plt.plot(bin_edges[:-1], cumulative, marker='o', color='skyblue')
                plt.xlabel("Sentence Number")
                plt.ylabel("Cumulative Interactions")
                plt.title("Cumulative Interactions Over Time")
                plt.show()
            except Exception as e:
                print(f"Failed to create cumulative plot: {e}")
        
        else:
            print(f"Mode '{mode}' not recognized. Please choose 'histogram', 'timeline', or 'cumulative'.")
    
    except Exception as e:
        print(f"An error occurred while visualizing temporality: {e}")


