from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
from itertools import combinations
import re
import warnings

import networkx as nx


DEFAULT_SPACY_MODELS = ("en_core_web_lg", "en_core_web_sm")
DEFAULT_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "for", "from", "had", "has", "have",
    "he", "her", "hers", "him", "his", "i", "in", "is", "it", "its", "of", "on",
    "or", "our", "she", "that", "the", "their", "them", "they", "to", "was", "we",
    "were", "with", "you", "your",
}


@lru_cache(maxsize=4)
def get_nlp_pipeline(model_candidates=DEFAULT_SPACY_MODELS):
    """Load a spaCy pipeline once and fall back to a blank English pipeline."""
    import spacy

    candidates = tuple(model_candidates or DEFAULT_SPACY_MODELS)
    for model_name in candidates:
        try:
            return spacy.load(model_name)
        except OSError:
            continue

    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


@lru_cache(maxsize=1)
def get_question_answering_pipeline():
    """Load the optional QA pipeline lazily so core extraction stays lightweight."""
    from transformers import pipeline

    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


def _resolve_nlp(nlp=None, model_candidates=None):
    return nlp or get_nlp_pipeline(tuple(model_candidates or DEFAULT_SPACY_MODELS))


def _deduplicate_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _normalize_character_name(name):
    return re.sub(r"[^a-z0-9\s]", "", name.lower()).strip()


def _fallback_character_candidates(text_string):
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text_string)
    return _deduplicate_preserve_order(candidates)


def _extract_sentence_spans(text, nlp_pipeline):
    doc = nlp_pipeline(text)
    sentences = [sentence for sentence in doc.sents if sentence.text.strip()]
    if not sentences:
        sentences = [doc[:]]
    return doc, sentences


def clean_action_label(action_label, char_list):
    """Normalize an action label by removing repeated character names and trimming text."""
    for char in char_list:
        pattern = r"(?i)\b" + re.escape(char) + r"\b"
        action_label = re.sub(pattern, "", action_label)

    action_label = " ".join(action_label.split())
    if len(action_label) > 50:
        action_label = action_label[:50] + "..."
    return action_label.strip()


def extract_person_entities(text_string, nlp=None, model_candidates=None):
    """Extract PERSON entities from text using spaCy with a regex fallback."""
    nlp_pipeline = _resolve_nlp(nlp=nlp, model_candidates=model_candidates)
    doc = nlp_pipeline(text_string)
    people = [ent.text for ent in getattr(doc, "ents", []) if ent.label_ == "PERSON"]
    if people:
        return _deduplicate_preserve_order(people)
    return _fallback_character_candidates(text_string)


def _extract_sentence_characters(sentence, nlp_pipeline):
    people = [ent.text for ent in getattr(sentence, "ents", []) if ent.label_ == "PERSON"]
    if people:
        return _deduplicate_preserve_order(people)
    return extract_person_entities(sentence.text, nlp=nlp_pipeline)


def _extract_action_label(sentence, characters):
    action_tokens = []
    character_tokens = {
        token
        for character in characters
        for token in _normalize_character_name(character).split()
        if token
    }

    for token in sentence:
        dependency = getattr(token, "dep_", "")
        part_of_speech = getattr(token, "pos_", "")
        if dependency == "ROOT" and token.text.strip():
            action_tokens.append(token.lemma_ if getattr(token, "lemma_", "") else token.text)
        elif dependency == "xcomp" and getattr(token.head, "text", "") in {"be", "are", "is"}:
            action_tokens.append(token.text)
        elif dependency == "attr" and getattr(token.head, "dep_", "") == "ROOT":
            action_tokens.append(token.text)
        elif part_of_speech == "VERB" and not action_tokens:
            action_tokens.append(token.lemma_ if getattr(token, "lemma_", "") else token.text)

    if not action_tokens:
        for token in sentence:
            cleaned = token.text.strip(" ,.;:!?\"'()[]{}").lower()
            if cleaned and cleaned.isalpha() and cleaned not in DEFAULT_STOPWORDS and cleaned not in character_tokens:
                action_tokens.append(cleaned)
                break

    action = clean_action_label(" ".join(action_tokens), characters)
    return action or "interacts"


def _merge_edge_payload(existing_payload, new_payload):
    for field in ("actions", "sentence_ids", "evidence_sentences"):
        existing_payload.setdefault(field, [])
        existing_payload[field].extend(new_payload.get(field, []))
    existing_payload["bidirectional"] = existing_payload.get("bidirectional", False) or new_payload.get("bidirectional", False)


def _add_interaction(graph, source, target, action, sentence_id, sentence_text, bidirectional=True):
    payload = {
        "actions": [action],
        "sentence_ids": [sentence_id],
        "evidence_sentences": [sentence_text],
        "bidirectional": bidirectional,
    }
    if graph.has_edge(source, target):
        _merge_edge_payload(graph[source][target], payload)
    else:
        graph.add_edge(source, target, **payload)


def create_character_graph(text, nlp=None, model_candidates=None, merge_similar=False, similarity_threshold=0.9):
    """Create a directed character graph from narrative text."""
    if not text or not text.strip():
        return nx.DiGraph(), [], []

    nlp_pipeline = _resolve_nlp(nlp=nlp, model_candidates=model_candidates)
    _, sentences = _extract_sentence_spans(text, nlp_pipeline)
    characters = extract_person_entities(text, nlp=nlp_pipeline)
    graph = nx.DiGraph()
    graph.add_nodes_from(characters)
    relationships = []

    for sentence_id, sentence in enumerate(sentences, start=1):
        sentence_characters = _extract_sentence_characters(sentence, nlp_pipeline)
        if not sentence_characters:
            continue

        characters = _deduplicate_preserve_order(characters + sentence_characters)
        for character in sentence_characters:
            graph.add_node(character)

        if len(sentence_characters) <= 1:
            continue

        relationships.append(sentence_characters)
        action = _extract_action_label(sentence, sentence_characters)
        for source, target in combinations(sentence_characters, 2):
            _add_interaction(graph, source, target, action, sentence_id, sentence.text.strip(), bidirectional=True)

    if merge_similar:
        graph = join_similar_nodes(graph, list(graph.nodes), similarity_threshold=similarity_threshold)

    return graph, list(graph.nodes), relationships


def count_cooccurrences(graph, char1, char2):
    """Count distinct sentence IDs mentioning both characters across both directions."""
    sentence_ids = set()
    if graph.has_edge(char1, char2):
        sentence_ids.update(graph[char1][char2].get("sentence_ids", []))
    if graph.has_edge(char2, char1):
        sentence_ids.update(graph[char2][char1].get("sentence_ids", []))
    return len(sentence_ids)


def create_character_graph_llm_chunked(
    text,
    chunk_size=5,
    nlp=None,
    qa_pipeline=None,
    sentence_id_offset=0,
    similarity_threshold=0.9,
):
    """Process long text in chunks and keep global sentence IDs stable."""
    if not text or not text.strip():
        return nx.DiGraph(), [], []

    nlp_pipeline = _resolve_nlp(nlp=nlp)
    _, sentences = _extract_sentence_spans(text, nlp_pipeline)

    if qa_pipeline is None:
        try:
            qa_pipeline = get_question_answering_pipeline()
        except Exception as exc:
            warnings.warn(
                f"Falling back to heuristic chunked extraction because the QA pipeline could not be loaded: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            qa_pipeline = False

    graph = nx.DiGraph()
    all_characters = []
    relationships = []
    conversation_so_far = ""
    indexed_sentences = list(enumerate(sentences, start=sentence_id_offset + 1))

    for chunk_start in range(0, len(indexed_sentences), chunk_size):
        chunk = indexed_sentences[chunk_start:chunk_start + chunk_size]
        for sent_id, sentence in chunk:
            text_sent = sentence.text.strip()
            sentence_characters = extract_person_entities(text_sent, nlp=nlp_pipeline)
            action_label = _extract_action_label(sentence, sentence_characters or _fallback_character_candidates(text_sent))

            if qa_pipeline:
                question_for_chars = (
                    "Which characters (PERSONs) are interacting in this sentence:\n\n"
                    f"'{text_sent}'\n\n"
                    f"Conversation so far:\n\n'{conversation_so_far}'"
                )
                question_for_action = (
                    "What is the main action or interaction described in this sentence:\n\n"
                    f"'{text_sent}'"
                )

                try:
                    result_chars = qa_pipeline(question=question_for_chars, context=conversation_so_far + "\n" + text_sent)
                    sentence_characters.extend(extract_person_entities(result_chars.get("answer", ""), nlp=nlp_pipeline))
                except Exception as exc:
                    warnings.warn(f"QA character extraction failed for sentence {sent_id}: {exc}", RuntimeWarning, stacklevel=2)

                try:
                    result_action = qa_pipeline(question=question_for_action, context=text_sent)
                    candidate_action = clean_action_label(result_action.get("answer", "").strip(), sentence_characters)
                    if candidate_action:
                        action_label = candidate_action
                except Exception as exc:
                    warnings.warn(f"QA action extraction failed for sentence {sent_id}: {exc}", RuntimeWarning, stacklevel=2)

            sentence_characters = _deduplicate_preserve_order(sentence_characters)
            all_characters = _deduplicate_preserve_order(all_characters + sentence_characters)

            for character in sentence_characters:
                graph.add_node(character)

            if len(sentence_characters) > 1:
                relationships.append(sentence_characters)
                for source, target in combinations(sentence_characters, 2):
                    _add_interaction(graph, source, target, action_label or "interacts", sent_id, text_sent, bidirectional=True)

            conversation_so_far += text_sent + " "

    graph = join_similar_nodes(graph, list(graph.nodes), similarity_threshold=similarity_threshold)
    return graph, list(graph.nodes), relationships


def _is_similar_character(name_a, name_b, similarity_threshold=0.9):
    normalized_a = _normalize_character_name(name_a)
    normalized_b = _normalize_character_name(name_b)
    if not normalized_a or not normalized_b:
        return False
    if normalized_a == normalized_b:
        return True

    ratio = SequenceMatcher(None, normalized_a, normalized_b).ratio()
    tokens_a = set(normalized_a.split())
    tokens_b = set(normalized_b.split())
    if tokens_a and tokens_b and (tokens_a.issubset(tokens_b) or tokens_b.issubset(tokens_a)) and ratio >= max(0.84, similarity_threshold - 0.08):
        return True
    if min(len(normalized_a), len(normalized_b)) >= 5 and ratio >= similarity_threshold:
        return True
    return False


def find_character_alias_groups(characters, similarity_threshold=0.9):
    """Group highly similar character names while preserving first-seen order."""
    groups = []
    for character in _deduplicate_preserve_order(characters):
        for group in groups:
            if any(_is_similar_character(character, member, similarity_threshold=similarity_threshold) for member in group):
                group.append(character)
                break
        else:
            groups.append([character])
    return groups


def _merge_nodes(graph, canonical_name, alias_name):
    if canonical_name == alias_name or canonical_name not in graph or alias_name not in graph:
        return

    graph.nodes[canonical_name].setdefault("aliases", [])
    graph.nodes[canonical_name]["aliases"] = _deduplicate_preserve_order(graph.nodes[canonical_name]["aliases"] + [alias_name])

    for predecessor, _, data in list(graph.in_edges(alias_name, data=True)):
        if predecessor == canonical_name:
            continue
        payload = {
            "actions": list(data.get("actions", [])),
            "sentence_ids": list(data.get("sentence_ids", [])),
            "evidence_sentences": list(data.get("evidence_sentences", [])),
            "bidirectional": data.get("bidirectional", False),
        }
        if graph.has_edge(predecessor, canonical_name):
            _merge_edge_payload(graph[predecessor][canonical_name], payload)
        else:
            graph.add_edge(predecessor, canonical_name, **payload)

    for _, successor, data in list(graph.out_edges(alias_name, data=True)):
        if successor == canonical_name:
            continue
        payload = {
            "actions": list(data.get("actions", [])),
            "sentence_ids": list(data.get("sentence_ids", [])),
            "evidence_sentences": list(data.get("evidence_sentences", [])),
            "bidirectional": data.get("bidirectional", False),
        }
        if graph.has_edge(canonical_name, successor):
            _merge_edge_payload(graph[canonical_name][successor], payload)
        else:
            graph.add_edge(canonical_name, successor, **payload)

    graph.remove_node(alias_name)


def join_similar_nodes(graph, characters, similarity_threshold=0.9):
    """Merge nodes with highly similar names while retaining edge evidence."""
    for group in find_character_alias_groups(characters, similarity_threshold=similarity_threshold):
        if len(group) < 2:
            continue
        canonical_name = group[0]
        for alias_name in group[1:]:
            _merge_nodes(graph, canonical_name, alias_name)
    return graph


def create_character_graph_llm_long_text(
    text,
    max_chunk_chars=1000,
    chunk_size=5,
    nlp=None,
    qa_pipeline=None,
    similarity_threshold=0.9,
):
    """Create a graph from arbitrarily long text while preserving global sentence IDs."""
    if not text or not text.strip():
        return nx.DiGraph(), [], []

    nlp_pipeline = _resolve_nlp(nlp=nlp)
    _, sentence_spans = _extract_sentence_spans(text, nlp_pipeline)
    sentences = [sentence.text.strip() for sentence in sentence_spans if sentence.text.strip()]

    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_chunk and current_length + sentence_length > max_chunk_chars:
            chunks.append(current_chunk)
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        chunks.append(current_chunk)

    global_graph = nx.DiGraph()
    global_relationships = []
    sentence_offset = 0

    for chunk in chunks:
        chunk_text = " ".join(chunk)
        chunk_graph, _, chunk_relationships = create_character_graph_llm_chunked(
            chunk_text,
            chunk_size=chunk_size,
            nlp=nlp_pipeline,
            qa_pipeline=qa_pipeline,
            sentence_id_offset=sentence_offset,
            similarity_threshold=similarity_threshold,
        )

        for node, data in chunk_graph.nodes(data=True):
            if node not in global_graph:
                global_graph.add_node(node, **data)
            elif data.get("aliases"):
                global_graph.nodes[node].setdefault("aliases", [])
                global_graph.nodes[node]["aliases"] = _deduplicate_preserve_order(global_graph.nodes[node]["aliases"] + data["aliases"])

        for source, target, data in chunk_graph.edges(data=True):
            payload = {
                "actions": list(data.get("actions", [])),
                "sentence_ids": list(data.get("sentence_ids", [])),
                "evidence_sentences": list(data.get("evidence_sentences", [])),
                "bidirectional": data.get("bidirectional", False),
            }
            if global_graph.has_edge(source, target):
                _merge_edge_payload(global_graph[source][target], payload)
            else:
                global_graph.add_edge(source, target, **payload)

        global_relationships.extend(chunk_relationships)
        sentence_offset += len(chunk)

    global_graph = join_similar_nodes(global_graph, list(global_graph.nodes), similarity_threshold=similarity_threshold)
    return global_graph, list(global_graph.nodes), global_relationships

