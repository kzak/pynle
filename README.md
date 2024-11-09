# pynle

`pynle` is a Python package for Neural Linear Embedding, based on [Mochihashi 2021].

[Mochihashi 2021]: "Researcher2Vec: ニューラル線形モデルによる自然言語処理研究者の可視化と推薦", NLP, 2021.

## Installation

```py
pip install git+https://github.com/kzak/pynle
```

## Usage

```py
from pynle import NeuralLinearEmbedding

# It is assumed that a corpus is given.
# The corpus should be of type list[list[str]] as shown below.
# corpus = [["w1", "w2", ...], ["w3", "w1", ...], ...]

nle = NeuralLinearEmbedding(k=2)
doc_embeddings, _, word_embeddings = nle.fit_transform(corpus)

assert len(corpus) == len(doc_embeddings)
```

Please see `notebooks/pynle_example.py` for a more detailed example.
