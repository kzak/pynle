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

n_latent_dim = 2
nle = NeuralLinearEmbedding(k=n_latent_dim)
doc_embeddings, word_embeddings = nle.fit_transform(corpus)

assert len(corpus) == doc_embeddings.shape[0]
assert n_latent_dim == doc_embeddings.shape[1]
```

Please see `notebooks/pynle_example.py` for a more detailed example.
