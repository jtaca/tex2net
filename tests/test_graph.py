import networkx as nx

from tex2net.graph import (
    count_cooccurrences,
    create_character_graph,
    create_character_graph_llm_chunked,
    create_character_graph_llm_long_text,
    extract_person_entities,
    find_character_alias_groups,
    join_similar_nodes,
)


def test_create_character_graph(fake_nlp):
    text = "Alice meets Bob. Bob greets Alice."
    graph, characters, relationships = create_character_graph(text, nlp=fake_nlp)

    assert "Alice" in characters
    assert "Bob" in characters
    assert relationships == [["Alice", "Bob"], ["Bob", "Alice"]]
    assert count_cooccurrences(graph, "Alice", "Bob") == 2

    edge_payload = graph["Alice"]["Bob"]
    assert edge_payload["actions"][0] == "meets"
    assert edge_payload["sentence_ids"] == [1]
    assert edge_payload["evidence_sentences"] == ["Alice meets Bob."]


def test_extract_person_entities_uses_fallback(fake_nlp):
    people = extract_person_entities("Alice meets Bob in Paris.", nlp=fake_nlp)
    assert people == ["Alice", "Bob", "Paris"]


def test_join_similar_nodes_merges_aliases_but_not_short_false_positives():
    graph = nx.DiGraph()
    graph.add_edge("Alice", "Bob", actions=["meets"], sentence_ids=[1], evidence_sentences=["Alice meets Bob"], bidirectional=True)
    graph.add_edge("Alicea", "Bob", actions=["greets"], sentence_ids=[2], evidence_sentences=["Alicea greets Bob"], bidirectional=True)
    graph.add_edge("Ann", "Bob", actions=["mentions"], sentence_ids=[3], evidence_sentences=["Ann mentions Bob"], bidirectional=True)
    graph.add_edge("Anna", "Bob", actions=["mentions"], sentence_ids=[4], evidence_sentences=["Anna mentions Bob"], bidirectional=True)

    merged_graph = join_similar_nodes(graph, ["Alice", "Alicea", "Ann", "Anna"], similarity_threshold=0.9)

    assert "Alice" in merged_graph.nodes
    assert "Alicea" not in merged_graph.nodes
    assert "Ann" in merged_graph.nodes
    assert "Anna" in merged_graph.nodes
    assert merged_graph["Alice"]["Bob"]["sentence_ids"] == [1, 2]


def test_find_character_alias_groups_is_stricter_than_substring_matching():
    groups = find_character_alias_groups(["Alice", "Alicea", "Ann", "Anna"], similarity_threshold=0.9)
    assert ["Alice", "Alicea"] in groups
    assert ["Ann"] in groups
    assert ["Anna"] in groups


def test_create_character_graph_llm_chunked_preserves_global_sentence_ids(fake_nlp):
    text = "Alice meets Bob. Bob greets Alice. Charlie admires Dana. Dana supports Charlie. Alice helps Dana. Bob thanks Charlie."
    graph, characters, relationships = create_character_graph_llm_chunked(text, chunk_size=2, nlp=fake_nlp, qa_pipeline=False)

    all_sentence_ids = []
    for _, _, data in graph.edges(data=True):
        all_sentence_ids.extend(data.get("sentence_ids", []))

    assert set(characters) >= {"Alice", "Bob", "Charlie", "Dana"}
    assert len(relationships) == 6
    assert min(all_sentence_ids) == 1
    assert max(all_sentence_ids) == 6


def test_create_character_graph_llm_long_text_preserves_offsets_across_chunks(fake_nlp):
    text = "Alice meets Bob. Bob greets Alice. Charlie admires Dana. Dana supports Charlie. Alice helps Dana. Bob thanks Charlie."
    graph, _, _ = create_character_graph_llm_long_text(
        text,
        max_chunk_chars=45,
        chunk_size=2,
        nlp=fake_nlp,
        qa_pipeline=False,
    )

    all_sentence_ids = []
    for _, _, data in graph.edges(data=True):
        all_sentence_ids.extend(data.get("sentence_ids", []))

    assert min(all_sentence_ids) == 1
    assert max(all_sentence_ids) == 6
