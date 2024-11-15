from pathlib import Path

from pynle import NeuralLinearEmbedding


def load_corpus():
    print(Path().resolve())

    with open("./tests/fixtures/text.txt") as f:
        lines = f.readlines()

    corpus = []
    for line in lines:
        line = line.strip()
        if line == "":
            continue

        corpus.append(line.split())
    return corpus


def test_pynle():
    corpus = load_corpus()

    N = len(corpus)
    K = 2
    V = len(set(sum(corpus, [])))

    nle = NeuralLinearEmbedding(k=2)
    nle.fit(corpus)
    doc_embeddings, word_embeddings = nle.transform()

    assert (N, K) == doc_embeddings.shape
    assert (V, K) == word_embeddings.shape
