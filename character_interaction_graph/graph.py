# character_interaction_graph/graph.py

import spacy
import networkx as nx
from itertools import combinations

def create_character_graph(text):
    # Carregar o modelo spaCy grande para inglês
    nlp = spacy.load('en_core_web_lg')
    doc = nlp(text)

    characters = []
    relationships = []

    # Extração de personagens a partir das entidades nomeadas
    for entity in doc.ents:
        if entity.label_ == "PERSON":
            character_name = entity.text
            if character_name not in characters:
                characters.append(character_name)

    # Criação do grafo direcionado
    graph = nx.DiGraph()
    graph.add_nodes_from(characters)

    sentence_id = 0
    for sentence in doc.sents:
        sentence_id += 1
        mentioned_characters = []
        for entity in sentence.ents:
            if entity.label_ == "PERSON":
                character_name = entity.text
                if character_name not in characters:
                    characters.append(character_name)
                mentioned_characters.append(character_name)
        if len(mentioned_characters) > 1:
            relationships.append(mentioned_characters)
            # Extrair ação baseada na estrutura da frase
            action_tokens = []
            for token in sentence:
                if token.dep_ == "ROOT":
                    action_tokens.append(token.text)
                elif token.dep_ == "xcomp" and token.head.text in ["be", "are", "is"]:
                    action_tokens.append(token.text)
                elif token.dep_ == "attr" and token.head.dep_ == "ROOT":
                    action_tokens.append(token.text)
                elif token.dep_ == "prep" and token.head.dep_ == "ROOT" and token.head.head.text in ["be", "are", "is"]:
                    action_tokens.append(token.text)
            action = " ".join(action_tokens) if action_tokens else ""

            # Adicionar arestas entre personagens com a ação extraída
            for i in range(len(mentioned_characters) - 1):
                for j in range(i + 1, len(mentioned_characters)):
                    source = mentioned_characters[i]
                    target = mentioned_characters[j]
                    if action and source in sentence.text and target in sentence.text:
                        if graph.has_edge(source, target):
                            graph[source][target]["actions"].append(action)
                            graph[source][target]["sentence_ids"].append(sentence_id)
                        else:
                            graph.add_edge(source, target, actions=[action], sentence_ids=[sentence_id], bidirectional=True)
                    else:
                        if graph.has_edge(source, target):
                            graph[source][target]["actions"].append(action)
                            graph[source][target]["sentence_ids"].append(sentence_id)
                        else:
                            graph.add_edge(source, target, actions=[action], sentence_ids=[sentence_id], bidirectional=False)
    return graph, characters, relationships

def join_similar_nodes(graph, characters):
    # Mesclar nós com nomes semelhantes
    character_combinations = combinations(characters, 2)
    for character1, character2 in character_combinations:
        if character1.lower() in character2.lower() or character2.lower() in character1.lower():
            if character1 in graph.nodes and character2 in graph.nodes:
                try:
                    if graph.has_edge(character1, character2):
                        graph[character1][character2]["actions"] += graph[character2][character1]["actions"]
                        graph[character1][character2]["sentence_ids"] += graph[character2][character1]["sentence_ids"]
                        graph.remove_edge(character2, character1)
                    elif graph.has_edge(character2, character1):
                        graph[character2][character1]["actions"] += graph[character1][character2]["actions"]
                        graph[character2][character1]["sentence_ids"] += graph[character1][character2]["sentence_ids"]
                        graph.remove_edge(character1, character2)
                except Exception:
                    pass
                graph.remove_node(character2)
    return graph
