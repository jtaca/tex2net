# character_interaction_graph/analysis.py

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pyvis.network import Network
import community as community_louvain

def plot_character_centralities(graph):
    """
    Calcula e plota a centralidade de grau dos personagens no grafo.
    """
    degree_centralities = nx.degree_centrality(graph)
    sorted_centralities = sorted(degree_centralities.items(), key=lambda item: item[1], reverse=True)
    characters, centralities = zip(*sorted_centralities)
    plt.figure(figsize=(10, 6))
    plt.bar(characters, centralities)
    plt.xlabel('Characters')
    plt.ylabel('Degree Centrality')
    plt.title('Character Centrality and Influence')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('character_centralities.png', dpi=300)
    plt.show()

def detect_communities_and_plot(graph):
    """
    Detecta comunidades e visualiza a rede com informações comunitárias.
    """
    partition = community_louvain.best_partition(graph.to_undirected())
    net = Network(notebook=True, width="100%", height="100%", bgcolor="#222222", font_color="white")
    communities = set(partition.values())
    community_colors = [f"#{hex(x)[2:]:0>6}" for x in range(256, 256 + len(communities))]
    for node, community_index in partition.items():
        net.add_node(node, title=node, group=community_index, color=community_colors[community_index % len(community_colors)])
    for edge in graph.edges():
        net.add_edge(edge[0], edge[1])
    net.show("character_network_communities.html")
    return partition

def plot_community_interactions(graph, partition):
    """
    Analisa e plota interações entre comunidades.
    """
    community_members = {}
    for node, community in partition.items():
        community_members.setdefault(community, []).append(node)
    interaction_counts = pd.DataFrame(0, index=range(len(community_members)), columns=range(len(community_members)))
    for edge in graph.edges():
        source_community = partition[edge[0]]
        target_community = partition[edge[1]]
        interaction_counts.at[source_community, target_community] += 1
    plt.figure(figsize=(8, 6))
    plt.imshow(interaction_counts, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Interaction Counts')
    plt.title('Community Interactions')
    plt.xlabel('Target Community')
    plt.ylabel('Source Community')
    plt.savefig('community_interactions.png', dpi=300)
    plt.show()

def describe_degree(graph):
    """
    Plota a distribuição de grau do grafo.
    """
    degrees = [degree for _, degree in graph.degree()]
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=20, color='lightblue')
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.title('Degree Distribution')
    plt.show()

def analyze_temporal_relationships(graph):
    """
    Analisa métricas temporais das interações do grafo.
    """
    temporal_distances = []
    for _, _, data in graph.edges(data=True):
        sentence_ids = data["sentence_ids"]
        if len(sentence_ids) > 1:
            temporal_distances.extend([sentence_ids[i+1] - sentence_ids[i] for i in range(len(sentence_ids)-1)])
    average_temporal_distance = sum(temporal_distances) / len(temporal_distances) if temporal_distances else None
    density = nx.density(graph)
    reciprocity = nx.reciprocity(graph)
    print("Temporal Relationship Analysis")
    print("-----------------------------")
    print(f"Average Temporal Distance: {average_temporal_distance}")
    print(f"Density of Temporal Interactions: {density}")
    print(f"Reciprocity of the Graph: {reciprocity}")
    print()

def analyze_graph_characteristics(graph):
    """
    Calcula e imprime características estruturais do grafo.
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    average_degree = sum(dict(graph.degree()).values()) / num_nodes if num_nodes > 0 else 0
    average_clustering = nx.average_clustering(graph)
    print("Graph Characteristics:")
    print(f"Number of Nodes: {num_nodes}")
    print(f"Number of Edges: {num_edges}")
    print(f"Average Degree: {average_degree}")
    print(f"Average Clustering Coefficient: {average_clustering}")

def detect_communities(graph):
    """
    Detecta comunidades utilizando o algoritmo de Louvain.
    """
    undirected_graph = graph.to_undirected()
    partition = community_louvain.best_partition(undirected_graph)
    communities = {}
    for character, community_id in partition.items():
        communities.setdefault(community_id, []).append(character)
    return communities

def calculate_degree_centrality(graph):
    """
    Calcula e retorna a centralidade de grau do grafo.
    """
    return nx.degree_centrality(graph)
