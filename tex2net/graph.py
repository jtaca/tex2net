# character_interaction_graph/graph.py

import spacy
import networkx as nx
from itertools import combinations
from transformers import pipeline



def create_character_graph(text):
    """
    Creates a directed character graph from the input text using spaCy for NER.
    """
    # Load the 'en_core_web_lg' model (you can switch to 'en_core_web_sm' if needed)
    nlp = spacy.load('en_core_web_lg')

    # Process the text
    doc = nlp(text)

    # Initialize character and relationship lists
    characters = []
    relationships = []

    # Identify person entities
    for entity in doc.ents:
        if entity.label_ == "PERSON":
            character_name = entity.text
            if character_name not in characters:
                characters.append(character_name)

    # Create a NetworkX directed graph
    graph = nx.DiGraph()
    graph.add_nodes_from(characters)

    # Initialize sentence ID counter
    sentence_id = 0

    # Process each sentence to extract relationships
    for sentence in doc.sents:
        sentence_id += 1
        mentioned_characters = []

        # Identify characters in the sentence
        for entity in sentence.ents:
            if entity.label_ == "PERSON":
                character_name = entity.text
                if character_name not in characters:
                    characters.append(character_name)
                mentioned_characters.append(character_name)

        # If more than one character is mentioned, treat it as a relationship
        if len(mentioned_characters) > 1:
            relationships.append(mentioned_characters)

            # Determine the main action (relationship) based on sentence structure
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

            # Add edges between all mentioned characters
            for i in range(len(mentioned_characters) - 1):
                for j in range(i + 1, len(mentioned_characters)):
                    source = mentioned_characters[i]
                    target = mentioned_characters[j]

                    # Decide direction of edge based on who performed the action
                    if action and source in sentence.text and target in sentence.text:
                        if graph.has_edge(source, target):
                            graph[source][target]["actions"].append(action)
                            graph[source][target]["sentence_ids"].append(sentence_id)
                        else:
                            graph.add_edge(
                                source,
                                target,
                                actions=[action],
                                sentence_ids=[sentence_id],
                                bidirectional=True
                            )
                    else:
                        if graph.has_edge(source, target):
                            graph[source][target]["actions"].append(action)
                            graph[source][target]["sentence_ids"].append(sentence_id)
                        else:
                            graph.add_edge(
                                source,
                                target,
                                actions=[action],
                                sentence_ids=[sentence_id],
                                bidirectional=False
                            )

    return graph, characters, relationships


def clean_action_label(action_label, char_list):
    """
    Remove repeated character names from the action label (case-insensitive),
    trim extra whitespace, and truncate overly long text.
    """
    for char in char_list:
        # Regex word boundary to avoid partial matches
        pattern = r"(?i)\b" + re.escape(char) + r"\b"
        action_label = re.sub(pattern, "", action_label)

    # Remove extra spaces
    action_label = " ".join(action_label.split())

    # Truncate if too long
    max_length = 50
    if len(action_label) > max_length:
        action_label = action_label[:max_length] + "..."

    return action_label


def extract_person_entities(text_string):
    """Extract PERSON entities from a string using spaCy."""
    temp_doc = nlp(text_string)
    return [ent.text for ent in temp_doc.ents if ent.label_ == "PERSON"]


def count_cooccurrences(graph, char1, char2):
    """
    Count how many distinct sentence IDs mention both char1 and char2
    (considering edges in both directions).
    """
    sentence_ids = set()
    if graph.has_edge(char1, char2):
        sentence_ids.update(graph[char1][char2].get("sentence_ids", []))
    if graph.has_edge(char2, char1):
        sentence_ids.update(graph[char2][char1].get("sentence_ids", []))
    return len(sentence_ids)

def create_character_graph_llm_chunked(text, chunk_size=5):


    """
    Creates a directed character graph from very long text by processing it
    in 'chunks' of sentences. Each chunk uses an LLM (QA pipeline) to infer
    which characters are interacting and what the main action is.

    :param text: The full text to analyze.
    :param chunk_size: Number of sentences to process at once before continuing.
    :return: (graph, characters, relationships)
        graph: A NetworkX directed graph of interactions
        characters: A list of unique characters discovered
        relationships: A list of character groupings per chunk-sentence
    """

    nlp = spacy.load("en_core_web_sm")

    # QA pipeline for extracting interactions from text
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    # Split the text into spaCy sentences
    doc = nlp(text)
    sentences = list(doc.sents)

    # A global graph to accumulate results
    graph = nx.DiGraph()
    all_characters = set()
    relationships = []

    conversation_so_far = ""  # We'll build context incrementally
    current_chunk = []

    def process_chunk(chunk_sentences, conversation_context):
        """
        Process a list of sentences as one 'chunk':
         - For each sentence, do QA-based detection of characters + action.
         - Update the graph, relationships, and global character set.
        Returns the updated conversation context (for the next chunk).
        """
        nonlocal graph, all_characters, relationships

        for sent_id, sentence in enumerate(chunk_sentences, start=1):
            text_sent = sentence.text.strip()
            # 1) Direct mentions via spaCy
            direct_mentions = extract_person_entities(text_sent)

            # 2) QA-based detection
            question_for_chars = (
                f"Which characters (PERSONs) are interacting in this sentence:\n\n"
                f"'{text_sent}'\n\n"
                f"Conversation so far:\n\n'{conversation_context}'"
            )
            question_for_action = (
                f"What is the main action or interaction described in this sentence:\n\n"
                f"'{text_sent}'"
            )

            try:
                result_chars = qa_pipeline(
                    question=question_for_chars, 
                    context=conversation_context + "\n" + text_sent
                )
                found_chars_str = result_chars.get("answer", "")
            except Exception as e:
                found_chars_str = ""
                print(f"QA for characters failed: {e}")

            try:
                result_action = qa_pipeline(
                    question=question_for_action, 
                    context=text_sent
                )
                action_label = result_action.get("answer", "").strip()
            except Exception as e:
                action_label = ""
                print(f"QA for action failed: {e}")

            # Combine direct + QA-based mentions
            qa_chars = extract_person_entities(found_chars_str)
            sentence_chars = set(direct_mentions + qa_chars)

            # Clean the action label
            action_label = clean_action_label(action_label, sentence_chars)

            # If more than one character => treat as a relationship
            if len(sentence_chars) > 1:
                relationships.append(list(sentence_chars))

                # Add to global set
                all_characters.update(sentence_chars)

                # Create edges among them
                char_list = list(sentence_chars)
                for i in range(len(char_list) - 1):
                    for j in range(i + 1, len(char_list)):
                        source = char_list[i]
                        target = char_list[j]

                        if graph.has_edge(source, target):
                            graph[source][target]["actions"].append(action_label)
                            graph[source][target]["sentence_ids"].append(sent_id)
                        else:
                            graph.add_edge(
                                source,
                                target,
                                actions=[action_label],
                                sentence_ids=[sent_id],
                                bidirectional=True
                            )
            else:
                # Still add single characters to global set, even if no edges
                all_characters.update(sentence_chars)

            # Update conversation context
            conversation_context += text_sent + " "

        return conversation_context

    # Process the text in chunks of chunk_size sentences
    for i, sentence in enumerate(sentences, start=1):
        current_chunk.append(sentence)
        if i % chunk_size == 0:
            # Process this chunk
            conversation_so_far = process_chunk(current_chunk, conversation_so_far)
            current_chunk = []

    # Process any leftover sentences
    if current_chunk:
        conversation_so_far = process_chunk(current_chunk, conversation_so_far)

    # Ensure all discovered characters are in the graph
    for char in all_characters:
        if char not in graph.nodes:
            graph.add_node(char)

    return graph, list(all_characters), relationships


def join_similar_nodes(graph, characters):
    """
    Merges nodes with very similar names (e.g., 'Winston' and 'Winstons')
    by transferring edges from the duplicate node to the main node.
    """
    from itertools import combinations
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




