# tests/test_graph.py

import pytest
from tex2net.graph import create_character_graph, join_similar_nodes

def test_create_character_graph():
    text = "Alice meets Bob. Bob greets Alice."
    graph, characters, relationships = create_character_graph(text)
    assert "Alice" in characters
    assert "Bob" in characters
    assert graph.number_of_nodes() >= 2

def test_join_similar_nodes():
    text = "Alice meets Alicia. Alicia greets Alice."
    graph, characters, _ = create_character_graph(text)
    new_graph = join_similar_nodes(graph, characters)
    # Após mesclar nós com nomes semelhantes, o número de nós deve ser menor que o original
    assert new_graph.number_of_nodes() < len(characters)
