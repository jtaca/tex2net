from setuptools import setup, find_packages

setup(
    name='character_interaction_graph',
    version='1.22',
    description='Biblioteca para criação e análise de grafos de interações de personagens a partir de textos narrativos utilizando NLP e análise de grafos',
    author='jtaca',
    author_email='joaotiagoaparicio@gmail.com',
    packages=find_packages(),
    install_requires=[
        'spacy>=3.0',
        'networkx>=2.0',
        'matplotlib>=3.0',
        'transformers>=4.0',
        'torch',
        'pyvis',
        'python-louvain',
        'pandas',
        'sentencepiece',
        'en-core-web-lg>=3.0.0',
        'numpy<2'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GPL-3 License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)