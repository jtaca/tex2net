async function loadGallery() {
  const response = await fetch("generated/examples.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to load gallery data.");
  }
  return response.json();
}

function setOverview(data) {
  document.getElementById("example-count").textContent = data.example_count;
  document.getElementById("node-count").textContent = data.total_nodes;
  document.getElementById("edge-count").textContent = data.total_edges;
  document.getElementById("max-span").textContent = `${data.max_span} sentences`;
}

function formatBuilderLabel(builder) {
  const labels = {
    basic: "Direct extraction",
    chunked: "Chunked extraction",
    long: "Long-corpus extraction",
  };
  return labels[builder] || builder;
}

function addMetricTiles(container, example) {
  const tiles = [
    ["Nodes", example.metrics.number_of_nodes],
    ["Edges", example.metrics.number_of_edges],
    ["Density", Number(example.metrics.density || 0).toFixed(3)],
    ["Top PageRank", example.metrics.top_pagerank_character || "—"],
  ];

  tiles.forEach(([label, value]) => {
    const tile = document.createElement("div");
    tile.className = "metric-tile";
    tile.innerHTML = `<dt class="metric-label">${label}</dt><dd>${value}</dd>`;
    container.appendChild(tile);
  });
}

function buildAnnotations(example) {
  const notes = [
    `Average clustering: ${Number(example.metrics.average_clustering || 0).toFixed(3)}`,
    `Average shortest path: ${Number(example.metrics.average_shortest_path || 0).toFixed(3)}`,
    `Interaction span: ${example.temporal_metrics.interaction_span} sentences`,
    `Relationship groups observed: ${example.relationship_groups}`,
  ];
  return notes;
}

function fillRankingList(listElement, items, formatter) {
  listElement.textContent = "";
  if (!items || items.length === 0) {
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent = "No prominent entries for this example.";
    listElement.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("li");
    row.innerHTML = `<span>${item.name}</span><span>${formatter(item)}</span>`;
    listElement.appendChild(row);
  });
}

function createExampleCard(example) {
  const template = document.getElementById("example-template");
  const fragment = template.content.cloneNode(true);

  fragment.querySelector(".example-builder").textContent = formatBuilderLabel(example.builder);
  fragment.querySelector(".example-title").textContent = example.title;
  fragment.querySelector(".example-subtitle").textContent = `Top PageRank character: ${example.metrics.top_pagerank_character || "—"}. Max k-core: ${example.metrics.max_k_core}.`;
  fragment.querySelector(".example-characters").textContent = `${example.character_count} characters`;
  fragment.querySelector(".example-communities").textContent = `${example.community_count} communities`;
  fragment.querySelector(".example-description").textContent = example.description;
  fragment.querySelector(".example-excerpt").textContent = example.text_excerpt;

  addMetricTiles(fragment.querySelector(".mini-metrics"), example);

  const frame = fragment.querySelector(".interactive-frame");
  const link = fragment.querySelector(".interactive-link");
  frame.src = example.assets.interactive_graph;
  link.href = example.assets.interactive_graph;

  fragment.querySelector(".community-image").src = example.assets.community_graph;
  fragment.querySelector(".temporal-image").src = example.assets.temporal_cumulative;
  fragment.querySelector(".centrality-image").src = example.assets.centrality_chart;

  const annotations = fragment.querySelector(".annotation-list");
  buildAnnotations(example).forEach((note) => {
    const item = document.createElement("li");
    item.textContent = note;
    annotations.appendChild(item);
  });

  fillRankingList(
    fragment.querySelector(".pagerank-list"),
    example.top_pagerank,
    (item) => item.value.toFixed(4),
  );

  fillRankingList(
    fragment.querySelector(".bridge-list"),
    example.bridge_characters,
    (item) => `${item.betweenness_centrality.toFixed(4)}${item.articulation_point ? " · articulation" : ""}`,
  );

  return fragment;
}

async function main() {
  const gallery = document.getElementById("gallery");
  try {
    const data = await loadGallery();
    setOverview(data);
    data.examples.forEach((example) => {
      gallery.appendChild(createExampleCard(example));
    });
  } catch (error) {
    const message = document.createElement("div");
    message.className = "empty-state";
    message.textContent = error.message;
    gallery.appendChild(message);
  }
}

main();
