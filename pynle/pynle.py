from collections import defaultdict
from logging import INFO, Formatter, StreamHandler, getLogger

import numpy as np
import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def build_logger():
    logger = getLogger(__name__)
    logger.setLevel(INFO)

    fmt = "[%(filename)s:%(lineno)d][%(levelname)s][%(asctime)s] %(message)s"
    stream_hander = StreamHandler()
    stream_hander.setFormatter(Formatter(fmt=fmt))
    logger.addHandler(stream_hander)
    return logger


logger = build_logger()


def build_word_freq(words: list[str]) -> dict[str, int]:
    word_freq: dict[str, int] = defaultdict(lambda: 0)
    for word in words:
        word_freq[word] += 1
    return word_freq


def build_word_prob(words: list[str]) -> dict[str, float]:
    word_freq = build_word_freq(words)
    N = sum(word_freq.values())
    word_prob = {}
    for w, n in word_freq.items():
        word_prob[w] = n / N
    return word_prob


class NeuralLinearEmbedding:
    def __init__(self, k: int):
        self.k = k

    def fit(self, corpus: list[list[str]]):
        words_all = [word for doc in corpus for word in doc]  # flatten corpus
        word_prob = build_word_prob(words_all)  # p(w)

        cond_prob: dict[int, dict[str, float]] = {}  # p(w|d)
        for d, words in enumerate(corpus):
            cond_prob[d] = build_word_prob(words)

        N = len(corpus)  # The number of documents
        V = len(word_prob.keys())  # The number of vocabularies
        word_to_id = dict(zip(word_prob.keys(), range(V)))

        logger.info(f"Building PMI matrix ({N}, {V})")
        PMI = np.zeros(shape=(N, V))  # PMI matrix
        for d, pw_d in tqdm.tqdm(cond_prob.items()):
            for w, pw_d in pw_d.items():
                pw = word_prob[w]
                pmi = np.log(pw_d / pw)  # log p(d, w) / p(d) p(w) = log p(w|d) / p(d)
                v = word_to_id[w]
                PMI[d, v] = pmi

        self.word_prob = word_prob
        self.cond_prob = cond_prob
        self.PMI = PMI
        self.word_to_id = word_to_id
        self.id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))

    def transform(self):
        logger.info("Executing SVD. Please wait.")
        PMI_sparse = csr_matrix(self.PMI)
        U, _, Vt = svds(PMI_sparse, k=self.k)
        return U, Vt.T

    def fit_transform(self, corpus: list[list[str]]):
        self.fit(corpus)
        return self.transform()
