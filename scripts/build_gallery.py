from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tex2net import (
    analyze_graph_characteristics,
    analyze_temporal_relationships,
    calculate_degree_centrality,
    calculate_pagerank,
    create_character_graph,
    create_character_graph_llm_chunked,
    create_character_graph_llm_long_text,
    detect_communities,
    identify_bridge_characters,
    join_similar_nodes,
    plot_character_centralities,
    plot_community_interactions,
    summarize_graph_metrics,
    visualize_directed_graph_styled,
    visualize_directed_graph_styled_communities,
    visualize_interaction_temporality,
    visualize_pyvis_graph,
)

DOCS_DIR = ROOT / "docs"
GENERATED_DIR = DOCS_DIR / "generated"


class _FakeEntity:
    def __init__(self, text, label_="PERSON"):
        self.text = text
        self.label_ = label_


class _FakeHead:
    def __init__(self, text="", dep_=""):
        self.text = text
        self.dep_ = dep_


class _FakeToken:
    def __init__(self, text, dep_="", pos_="", lemma_=None, head=None):
        self.text = text
        self.dep_ = dep_
        self.pos_ = pos_
        self.lemma_ = lemma_ if lemma_ is not None else text.lower()
        self.head = head or _FakeHead()


class _FakeSentence:
    def __init__(self, text):
        self.text = text.strip()
        names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", self.text)
        self.ents = [_FakeEntity(name) for name in dict.fromkeys(names)]

        raw_tokens = re.findall(r"[A-Za-z']+", self.text)
        root_index = None
        for index, token in enumerate(raw_tokens):
            if token.isalpha() and token[:1].islower():
                root_index = index
                break

        self._tokens = []
        for index, token in enumerate(raw_tokens):
            is_root = index == root_index
            dep = "ROOT" if is_root else ""
            pos = "VERB" if is_root else ("PROPN" if token[:1].isupper() else "")
            self._tokens.append(_FakeToken(token, dep_=dep, pos_=pos, lemma_=token.lower()))

        root_token = next((token for token in self._tokens if token.dep_ == "ROOT"), _FakeToken("interacts", dep_="ROOT", pos_="VERB"))
        for token in self._tokens:
            token.head = root_token

    def __iter__(self):
        return iter(self._tokens)


class _FakeDoc:
    def __init__(self, text):
        sentence_texts = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text.strip()) if segment.strip()]
        self.sents = [_FakeSentence(sentence_text) for sentence_text in sentence_texts] or [_FakeSentence(text)]
        self.ents = []
        for sentence in self.sents:
            self.ents.extend(sentence.ents)

    def __getitem__(self, item):
        return self


class _FakeNLP:
    pipe_names = ("sentencizer",)

    def __call__(self, text):
        return _FakeDoc(text)


FAKE_NLP = _FakeNLP()


def _long_chronicle_text() -> str:
    chapters = [
        "Aria briefed Captain Hale beside the harbor gate.",
        "Captain Hale introduced Aria to Scholar Neri in the watchtower.",
        "Scholar Neri warned Mayor Sol about hidden ledgers.",
        "Mayor Sol summoned Guard Petra to the council chamber.",
        "Guard Petra escorted Aria through the archive hall.",
        "Aria questioned Broker Venn under the lantern bridge.",
        "Broker Venn bargained with Captain Hale near the grain market.",
        "Captain Hale reassured Mayor Sol during the storm alarm.",
        "Scholar Neri compared notes with Guard Petra before dawn.",
        "Mayor Sol confronted Broker Venn about missing seals.",
        "Aria and Scholar Neri mapped tunnels beneath the citadel.",
        "Captain Hale and Guard Petra sealed the eastern gate.",
        "Broker Venn met Oracle Ilex in a shuttered chapel.",
        "Oracle Ilex challenged Aria with a riddle about the harbor bells.",
        "Oracle Ilex advised Mayor Sol to trust Scholar Neri.",
        "Guard Petra reported to Captain Hale after the market riot.",
        "Aria persuaded Mayor Sol to open the hidden vault.",
        "Scholar Neri thanked Oracle Ilex after reading the final ledger.",
        "Broker Venn apologized to Guard Petra before leaving the city.",
        "Captain Hale promised Aria that the harbor would remain open.",
    ]
    return " ".join(chapters)


EXAMPLES = [
    {
        "slug": "salon-of-rivals",
        "title": "Salon of Rivals",
        "description": "A compact court-intrigue scene that highlights interaction extraction, community structure, and bridge characters.",
        "builder": "basic",
        "text": (
            "Alice debated policy with Bob in the marble salon. "
            "Bob reassured Clara after the vote. "
            "Clara challenged Diana during the reception. "
            "Diana thanked Alice before meeting Edgar. "
            "Edgar praised Bob in front of Clara."
        ),
        "merge_similar": False,
    },
    {
        "slug": "alias-resolution",
        "title": "Alias Resolution",
        "description": "A short investigation where near-duplicate names collapse into cleaner character identities.",
        "builder": "basic",
        "text": (
            "Mira Stone met Captain Ivo at sunrise. "
            "Mira Stonne briefed Captain Ivo beside the river. "
            "Captain Ivo introduced Mira Stone to Doctor Vale. "
            "Doctor Vahle warned Mira Stonne about Auditor Sen. "
            "Auditor Sen confronted Captain Ivo in the square."
        ),
        "merge_similar": True,
        "similarity_threshold": 0.84,
    },
    {
        "slug": "council-in-motion",
        "title": "Council in Motion",
        "description": "Chunked extraction over a denser political sequence, emphasizing temporal patterns and evolving alliances.",
        "builder": "chunked",
        "text": (
            "Helena convened Marcus and Octavia in the war room. "
            "Marcus consulted Priya over the city maps. "
            "Octavia questioned Helena near the signal fire. "
            "Priya introduced Jonas to Marcus before dusk. "
            "Jonas warned Helena about Octavia. "
            "Helena negotiated with Priya after the alarm. "
            "Marcus and Jonas secured the western gate. "
            "Octavia apologized to Priya during the retreat."
        ),
    },
    {
        "slug": "long-corpus-chronicle",
        "title": "Long-Corpus Chronicle",
        "description": "A longer synthetic chronicle designed to exercise stable sentence IDs, chunking, and long-range temporal structure.",
        "builder": "long",
        "text": _long_chronicle_text(),
    },
]
def _top_items(values, limit=5, places=4):
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=True)
    return [{"name": name, "value": round(float(value), places)} for name, value in ordered[:limit]]


def _relative(path: Path) -> str:
    return path.relative_to(DOCS_DIR).as_posix()


def _build_graph(example):
    if example["builder"] == "basic":
        graph, characters, relationships = create_character_graph(example["text"], nlp=FAKE_NLP)
        if example.get("merge_similar"):
            graph = join_similar_nodes(
                graph,
                characters,
                similarity_threshold=example.get("similarity_threshold", 0.9),
            )
            characters = list(graph.nodes())
        return graph, characters, relationships

    if example["builder"] == "chunked":
        return create_character_graph_llm_chunked(example["text"], qa_pipeline=False, chunk_size=3, nlp=FAKE_NLP)

    if example["builder"] == "long":
        return create_character_graph_llm_long_text(
            example["text"],
            qa_pipeline=False,
            chunk_size=4,
            max_chunk_chars=220,
            nlp=FAKE_NLP,
        )

    raise ValueError(f"Unknown builder: {example['builder']}")


def _generate_assets(example, graph):
    example_dir = GENERATED_DIR / example["slug"]
    example_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "interactive_graph": example_dir / "interactive_graph.html",
        "styled_graph": example_dir / "styled_graph.png",
        "community_graph": example_dir / "community_graph.png",
        "temporal_histogram": example_dir / "temporal_histogram.png",
        "temporal_cumulative": example_dir / "temporal_cumulative.png",
        "centrality_chart": example_dir / "centrality_chart.png",
        "community_heatmap": example_dir / "community_heatmap.png",
        "metrics": example_dir / "metrics.json",
    }

    visualize_pyvis_graph(graph, output_file=str(files["interactive_graph"]))
    visualize_directed_graph_styled(graph, title=example["title"], show=False, output_path=str(files["styled_graph"]))
    visualize_directed_graph_styled_communities(
        graph,
        title=f"{example['title']} Communities",
        edge_annotation="number",
        show=False,
        output_path=str(files["community_graph"]),
    )
    visualize_interaction_temporality(graph, mode="histogram", show=False, output_path=str(files["temporal_histogram"]))
    visualize_interaction_temporality(graph, mode="cumulative", show=False, output_path=str(files["temporal_cumulative"]))

    plot_character_centralities(graph, output_path=str(files["centrality_chart"]), show=False)

    partition = detect_communities(graph, as_partition=True)
    if partition:
        plot_community_interactions(graph, partition, output_path=str(files["community_heatmap"]), show=False)

    return files, partition


def _example_record(example):
    graph, characters, relationships = _build_graph(example)
    files, partition = _generate_assets(example, graph)

    summary_metrics = summarize_graph_metrics(graph)
    graph_characteristics = analyze_graph_characteristics(graph, verbose=False)
    temporal_metrics = analyze_temporal_relationships(graph, verbose=False)
    degree = calculate_degree_centrality(graph)
    pagerank = calculate_pagerank(graph)
    bridges = identify_bridge_characters(graph)

    record = {
        "slug": example["slug"],
        "title": example["title"],
        "description": example["description"],
        "builder": example["builder"],
        "text_excerpt": textwrap.shorten(example["text"], width=220, placeholder="…"),
        "character_count": len(characters),
        "relationship_groups": len(relationships),
        "community_count": len(set(partition.values())) if partition else 0,
        "metrics": summary_metrics,
        "graph_characteristics": graph_characteristics,
        "temporal_metrics": temporal_metrics,
        "top_degree": _top_items(degree),
        "top_pagerank": _top_items(pagerank),
        "bridge_characters": [
            {
                "name": name,
                "betweenness_centrality": round(values["betweenness_centrality"], 4),
                "articulation_point": values["articulation_point"],
            }
            for name, values in sorted(bridges.items(), key=lambda item: item[1]["betweenness_centrality"], reverse=True)
            if values["articulation_point"] or values["betweenness_centrality"] > 0
        ],
        "assets": {name: _relative(path) for name, path in files.items()},
    }

    files["metrics"].write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    records = [_example_record(example) for example in EXAMPLES]
    overview = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "example_count": len(records),
        "total_nodes": sum(example["metrics"]["number_of_nodes"] for example in records),
        "total_edges": sum(example["metrics"]["number_of_edges"] for example in records),
        "max_span": max(example["temporal_metrics"].get("interaction_span", 0) for example in records),
        "examples": records,
    }
    (GENERATED_DIR / "examples.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
