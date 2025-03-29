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


import networkx as nx
import numpy as np

def jaccard_similarity(set1, set2):
    """
    Compute the Jaccard similarity between two sets.
    
    In a character network, if set1 and set2 are sets of characters or edges,
    this returns the fraction of common elements relative to the total unique elements.
    """
    try:
        union = set1.union(set2)
        if not union:
            return 1.0
        return len(set1.intersection(set2)) / len(union)
    except Exception as e:
        print(f"Error in jaccard_similarity: {e}")
        return 0.0

def compare_node_similarity(g1, g2):
    """
    Compare the node sets (characters) of two graphs using Jaccard similarity.
    
    Intuition: This measure tells us what fraction of characters is shared between two networks.
    """
    try:
        nodes1 = set(g1.nodes())
        nodes2 = set(g2.nodes())
        return jaccard_similarity(nodes1, nodes2)
    except Exception as e:
        print(f"Error in compare_node_similarity: {e}")
        return 0.0

def compare_edge_similarity(g1, g2, directed=False):
    """
    Compare the edge sets (interactions) of two graphs using Jaccard similarity.
    
    Parameters:
        directed (bool): If False, edges are treated as undirected.
    
    Intuition: This measure tells us what fraction of interactions is common between the two networks.
    """
    try:
        if directed:
            edges1 = set(g1.edges())
            edges2 = set(g2.edges())
        else:
            # For undirected comparison, treat each edge as a frozenset of nodes
            edges1 = set(frozenset(e) for e in g1.edges())
            edges2 = set(frozenset(e) for e in g2.edges())
        return jaccard_similarity(edges1, edges2)
    except Exception as e:
        print(f"Error in compare_edge_similarity: {e}")
        return 0.0

def compare_degree_distributions(g1, g2):
    """
    Compare the degree distributions of two graphs.
    
    This function computes the Pearson correlation coefficient between the
    degree vectors (number of interactions per character) over the union of nodes.
    
    Intuition: Similar degree distributions imply that characters in both networks 
    have similar levels of connectivity.
    """
    try:
        union_nodes = set(g1.nodes()).union(set(g2.nodes()))
        degrees1 = [g1.degree(node) if node in g1 else 0 for node in union_nodes]
        degrees2 = [g2.degree(node) if node in g2 else 0 for node in union_nodes]
        if len(degrees1) < 2:
            return 1.0
        corr = np.corrcoef(degrees1, degrees2)[0, 1]
        return corr
    except Exception as e:
        print(f"Error in compare_degree_distributions: {e}")
        return 0.0

def compare_average_clustering(g1, g2):
    """
    Compare the average clustering coefficients of two graphs.
    
    The average clustering coefficient indicates the tendency of characters to form 
    tightly-knit groups. The function returns the absolute difference.
    
    Intuition: A small difference suggests that both networks have similar clique structures.
    """
    try:
        clustering1 = nx.average_clustering(g1)
        clustering2 = nx.average_clustering(g2)
        return abs(clustering1 - clustering2)
    except Exception as e:
        print(f"Error in compare_average_clustering: {e}")
        return None

def compute_graph_edit_distance(g1, g2):
    """
    Compute the graph edit distance between two graphs.
    
    Graph edit distance is the minimum number of modifications (node/edge insertions or deletions)
    needed to transform one graph into the other.
    
    Intuition: A lower graph edit distance means that the overall structure of the two networks is very similar.
    
    Note: This calculation can be computationally expensive for larger graphs.
    """
    try:
        ged = nx.graph_edit_distance(g1, g2)
        return ged
    except Exception as e:
        print(f"Graph edit distance computation failed: {e}")
        return None

def compare_graph_density(g1, g2):
    """
    Compare the densities of two graphs.
    
    Density is defined as the ratio of the number of edges to the number of possible edges.
    
    Intuition: In character networks, density reflects how interconnected the characters are.
    A small difference indicates similar levels of overall interaction.
    """
    try:
        def density(g):
            n = g.number_of_nodes()
            if n <= 1:
                return 0
            # For undirected graphs: 2*E/(n*(n-1))
            return 2 * g.number_of_edges() / (n * (n - 1))
        return abs(density(g1) - density(g2))
    except Exception as e:
        print(f"Error in compare_graph_density: {e}")
        return None

def compare_avg_shortest_path(g1, g2):
    """
    Compare the average shortest path lengths of two graphs.
    
    The average shortest path length gives an idea of how 'close' characters are in the network.
    This function returns the absolute difference between the average shortest path lengths
    computed on the largest connected components.
    
    Intuition: Similar average distances imply that characters in both networks have similar reachability.
    """
    try:
        def avg_shortest_path(g):
            if nx.is_connected(g):
                return nx.average_shortest_path_length(g)
            else:
                # Compute on the largest connected component.
                comp = max(nx.connected_components(g), key=len)
                subg = g.subgraph(comp)
                return nx.average_shortest_path_length(subg)
        return abs(avg_shortest_path(g1) - avg_shortest_path(g2))
    except Exception as e:
        print(f"Error in compare_avg_shortest_path: {e}")
        return None

def compare_modularity(g1, g2):
    """
    Compare the modularity of the community structures in two graphs.
    
    Modularity measures the strength of division of a network into communities.
    We use the greedy modularity algorithm to detect communities and compute their modularity.
    
    Intuition: Similar modularity scores indicate that both networks have a comparable quality of community division.
    """
    try:
        from networkx.algorithms import community
        def modularity(g):
            communities = community.greedy_modularity_communities(g)
            return community.modularity(g, communities)
        return abs(modularity(g1) - modularity(g2))
    except Exception as e:
        print(f"Error in compare_modularity: {e}")
        return None

def compare_betweenness_correlations(g1, g2):
    """
    Compare the betweenness centrality distributions of two graphs using Pearson correlation.
    
    Betweenness centrality indicates how often a character acts as a bridge between other characters.
    
    Intuition: A high correlation means that key connector roles are similarly distributed in both networks.
    """
    try:
        union_nodes = set(g1.nodes()).union(set(g2.nodes()))
        betweenness_g1 = nx.betweenness_centrality(g1, endpoints=True)
        betweenness_g2 = nx.betweenness_centrality(g2, endpoints=True)
        betweenness1 = [betweenness_g1.get(node, 0) for node in union_nodes]
        betweenness2 = [betweenness_g2.get(node, 0) for node in union_nodes]
        if len(betweenness1) < 2:
            return 1.0
        corr = np.corrcoef(betweenness1, betweenness2)[0, 1]
        return corr
    except Exception as e:
        print(f"Error in compare_betweenness_correlations: {e}")
        return None

def compare_graphs(g1, g2, directed_edges=False):
    """
    Compare two character networks using multiple topology measures.
    
    In the context of character networks, these measures capture:
      - How many characters and interactions are shared (node and edge Jaccard similarities).
      - Whether the overall connectivity patterns (degree distributions) are similar.
      - If the tendency to form cliques (clustering coefficients) is similar.
      - How many modifications are needed to transform one network into the other (graph edit distance).
      - Differences in density, average distances between characters, the quality of community structure (modularity),
        and the roles of key connector characters (betweenness centrality).
    
    Parameters:
        g1, g2: The character graphs to compare.
        directed_edges (bool): Whether to treat edges as directed when comparing edge sets.
        
    Returns:
        dict: A dictionary containing the following keys:
            - node_jaccard: Jaccard similarity of node sets.
            - edge_jaccard: Jaccard similarity of edge sets.
            - degree_corr: Pearson correlation between degree distributions.
            - avg_clustering_diff: Absolute difference in average clustering coefficients.
            - graph_edit_distance: The computed graph edit distance.
            - density_diff: Absolute difference in graph densities.
            - avg_shortest_path_diff: Absolute difference in average shortest path lengths.
            - modularity_diff: Absolute difference in modularity values.
            - betweenness_corr: Pearson correlation between betweenness centrality distributions.
    """
    comparisons = {}
    try:
        comparisons['node_jaccard'] = compare_node_similarity(g1, g2)
    except Exception as e:
        print(f"Error comparing node similarity: {e}")
        comparisons['node_jaccard'] = None
        
    try:
        comparisons['edge_jaccard'] = compare_edge_similarity(g1, g2, directed=directed_edges)
    except Exception as e:
        print(f"Error comparing edge similarity: {e}")
        comparisons['edge_jaccard'] = None
        
    try:
        comparisons['degree_corr'] = compare_degree_distributions(g1, g2)
    except Exception as e:
        print(f"Error comparing degree distributions: {e}")
        comparisons['degree_corr'] = None
        
    try:
        comparisons['avg_clustering_diff'] = compare_average_clustering(g1, g2)
    except Exception as e:
        print(f"Error comparing average clustering: {e}")
        comparisons['avg_clustering_diff'] = None
        
    try:
        comparisons['graph_edit_distance'] = compute_graph_edit_distance(g1, g2)
    except Exception as e:
        print(f"Error computing graph edit distance: {e}")
        comparisons['graph_edit_distance'] = None
        
    try:
        comparisons['density_diff'] = compare_graph_density(g1, g2)
    except Exception as e:
        print(f"Error comparing graph density: {e}")
        comparisons['density_diff'] = None
        
    try:
        comparisons['avg_shortest_path_diff'] = compare_avg_shortest_path(g1, g2)
    except Exception as e:
        print(f"Error comparing average shortest path: {e}")
        comparisons['avg_shortest_path_diff'] = None
        
    try:
        comparisons['modularity_diff'] = compare_modularity(g1, g2)
    except Exception as e:
        print(f"Error comparing modularity: {e}")
        comparisons['modularity_diff'] = None
        
    try:
        comparisons['betweenness_corr'] = compare_betweenness_correlations(g1, g2)
    except Exception as e:
        print(f"Error comparing betweenness correlations: {e}")
        comparisons['betweenness_corr'] = None
        
    return comparisons


