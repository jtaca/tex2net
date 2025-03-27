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

## Installation

Install the package via pip:

```bash
pip install tex2net

```

Make sure you have all the required dependencies. You can also install them using the provided requirements.txt if needed.

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues to improve the functionality of this library. Please follow the code style and add tests for any new features.

## License

This project is licensed under the MIT License.

---

Happy storytelling and network exploring!

