from tex2net.graph import create_character_graph_llm_long_text


def test_long_corpus_stress_preserves_global_sentence_ids(fake_nlp):
    sentences = []
    names = [("Alice", "Bob"), ("Charlie", "Dana"), ("Alice", "Dana"), ("Bob", "Charlie")]
    for index in range(40):
        source, target = names[index % len(names)]
        sentences.append(f"{source} meets {target} in chapter {index + 1}.")

    graph, characters, relationships = create_character_graph_llm_long_text(
        " ".join(sentences),
        max_chunk_chars=120,
        chunk_size=3,
        nlp=fake_nlp,
        qa_pipeline=False,
    )

    all_sentence_ids = []
    for _, _, data in graph.edges(data=True):
        all_sentence_ids.extend(data.get("sentence_ids", []))

    assert set(characters) >= {"Alice", "Bob", "Charlie", "Dana"}
    assert len(relationships) == 40
    assert min(all_sentence_ids) == 1
    assert max(all_sentence_ids) == 40