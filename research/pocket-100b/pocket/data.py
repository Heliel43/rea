"""A structured synthetic corpus: enough language for a toy model to learn
conditioning, topic routing and non-uniform FFN usage.

Real pretrained weights would be better experimental subjects, but downloading a
70B checkpoint is exactly the thing this lab is trying to make unnecessary, and a
3 GiB machine cannot hold one anyway.  So the subject is a corpus the experiments
can afford, with the three structures the study needs:

  * *conditional next-token structure* -- so loss is a meaningful quality signal;
  * *topics* -- so a gate has something to route on;
  * *heavy-tailed transitions* -- so activation and routing statistics are not
    uniform by construction.

Documents look like ``<bos> <topic> w w w ... <eos>``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

BOS, EOS = 0, 1
N_TOPICS = 4
TOPIC_BASE = 2
FIRST_WORD = TOPIC_BASE + N_TOPICS


class Corpus:
    def __init__(self, vocab: int = 40, n_topics: int = N_TOPICS, seed: int = 0, doc_len: int = 24):
        if vocab < FIRST_WORD + 8:
            raise ValueError("vocabulary too small for the topic markers")
        self.vocab = vocab
        self.n_topics = n_topics
        self.doc_len = doc_len
        self.seed = seed
        rng = random.Random(seed)
        words = list(range(FIRST_WORD, vocab))
        # Each topic owns a vocabulary region that overlaps only partly with the
        # others: content words are shared, but preferences differ sharply.
        self.weights: List[Dict[int, List[Tuple[int, float]]]] = []
        for topic in range(n_topics):
            table: Dict[int, List[Tuple[int, float]]] = {}
            for w in words:
                k = rng.choice([2, 3, 4])
                succ = rng.sample(words, k)
                # Zipf-ish preference over successors, fixed per topic+word.
                probs = [1.0 / (i + 1) ** 1.3 for i in range(k)]
                s = sum(probs)
                table[w] = [(t, p / s) for t, p in zip(succ, probs)]
            self.weights.append(table)

    # -- sampling ---------------------------------------------------------- #
    def document(self, rng: random.Random, topic: int) -> List[int]:
        out = [BOS, TOPIC_BASE + topic]
        w = rng.choice(list(self.weights[topic]))
        for _ in range(self.doc_len - 3):
            out.append(w)
            choices = self.weights[topic][w]
            r = rng.random()
            acc = 0.0
            nxt = choices[-1][0]
            for t, p in choices:
                acc += p
                if r <= acc:
                    nxt = t
                    break
            w = nxt
        out.append(EOS)
        return out

    def sequences(self, n_docs: int, seed: int = 1234) -> List[List[int]]:
        rng = random.Random(seed)
        return [self.document(rng, rng.randrange(self.n_topics)) for _ in range(n_docs)]

    def batched(self, n_docs: int, seed: int = 1234) -> List[Tuple[List[int], List[int]]]:
        "Documents split into (input, target) pairs of ``doc_len - 1`` positions."

        out = []
        for doc in self.sequences(n_docs, seed):
            out.append((doc[:-1], doc[1:]))
        return out


def corpus_fingerprint(corpus: Corpus) -> str:
    "Cheap, stable identity for the corpus a checkpoint was trained on."

    import hashlib

    h = hashlib.sha256()
    h.update(f"{corpus.vocab}/{corpus.n_topics}/{corpus.doc_len}/{corpus.seed}".encode())
    for table in corpus.weights:
        for w in sorted(table):
            for t, p in table[w]:
                h.update(f"{w}>{t}:{p:.6f};".encode())
    return h.hexdigest()[:16]
