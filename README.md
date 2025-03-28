# tex2net

[![Build Status](https://img.shields.io/github/actions/workflow/status/jtaca/tex2net/python-package.yml?branch=main)](https://github.com/jtaca/tex2net/actions)
[![PyPI version](https://img.shields.io/pypi/v/tex2net)](https://pypi.org/project/tex2net)
[![License](https://img.shields.io/github/license/jtaca/tex2net)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/tex2net)](https://pypi.org/project/tex2net)

## Overview

**tex2net** is a Python library designed to convert narrative texts into interactive network graphs that visualize character interactions. The library leverages Natural Language Processing (NLP) with spaCy, text rewriting and summarization using Transformer models (T5), network analysis with NetworkX, and interactive visualization with Pyvis and Matplotlib. This tool is perfect for researchers, data scientists, and digital humanities scholars looking to explore complex narrative structures.

## Features

- **Text Preprocessing & Entity Extraction:** Extract characters from text using spaCy's Named Entity Recognition.
- **Graph Construction:** Build directed graphs representing character interactions, annotated with actions and sentence IDs.
- **Node Merging:** Combine nodes with similar character names for consistency.
- **Text Rewriting & Summarization:** Use T5 models to rephrase or summarize text, emphasizing clear character-verb-character structures.
- **Network Analysis:** Compute centrality measures, detect communities, analyze temporal relationships, and more.
- **Visualization:** Create both static (Matplotlib) and interactive (Pyvis) visualizations for in-depth narrative analysis.

# Usage Example

Below is an example of how to use tex2net to create a character graph from sample text:

```python
# Import functions from the tex2net package
from tex2net.graph import create_character_graph, join_similar_nodes

# Sample text data (this could be text from a novel, article, or LaTeX file)
text_data = """
Alice was discussing ideas with Bob. Later, Bob met Charlie and they collaborated on a project.
Alice was also involved in the conversation.
"""

# Create a character graph from the text data
graph = create_character_graph(text_data)

# Optionally, join similar nodes to merge duplicate character entries
joined_graph = join_similar_nodes(graph)

# Visualize the graph by generating an HTML file (using pyvis or a similar tool)
joined_graph.show("character_graph.html")

print("Graph generated and saved as character_graph.html")

```

More functions are instantiated in example_usage.ipynb.

## Installation

Install the package via pip:

```bash
pip install tex2net

```

Make sure you have all the required dependencies. You can also install them using the provided requirements.txt if needed.

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues to improve the functionality of this library. Please follow the code style and add tests for any new features.

## License

This project is licensed under the GPL-3.0 License.

---

Happy storytelling and network exploring!

Citation

If you find tex2net helpful in your academic or professional work, please cite the following paper:

graphql
Copy
@inproceedings{aparicio2023tex2net,
  title={tex2net: A Package for Storytelling Using Network Models},
  author={Aparicio, Joao Tiago and Arsenio, Elisabete and Henriques, Rui},
  booktitle={Proceedings of the 41st ACM International Conference on Design of Communication},
  pages={119--125},
  year={2023}
}

or 

Aparicio, J. T., Arsenio, E., & Henriques, R. (2023, October). tex2net: A Package for Storytelling Using Network Models. In Proceedings of the 41st ACM International Conference on Design of Communication (pp. 119-125).

You can find the paper at ACM Digital Library: https://dl.acm.org/doi/pdf/10.1145/3615335.3623022.



