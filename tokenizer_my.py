"""
Byte-level BPE tokenizer. Loads merge rules produced by train_tokenizer.py
and implements load() / encode() / decode().

Interface expected by train.py / evaluate.py:
    load() -> tokenizer object with .encode(str), .decode(list[int]), .vocab_size
    Called with NO arguments — merges.txt is located relative to this file,
"""

import os


class BPETokenizer:
    def __init__(self):
        self.merges = []
        self.merge_rank = {}
        self.vocab = {}     
        self._init_base_vocab()

    def _init_base_vocab(self):
        # tokens 0..255 are always the raw bytes
        self.vocab = {i: bytes([i]) for i in range(256)}

    def load_merges(self, merges_path):
        self._init_base_vocab()
        self.merges = []
        self.merge_rank = {}

        with open(merges_path, "r", encoding="utf-8") as f:
            for rank, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                a_str, b_str = line.split()
                a, b = int(a_str), int(b_str)

                new_id = 256 + rank
                self.vocab[new_id] = self.vocab[a] + self.vocab[b]
                self.merges.append((a, b))
                self.merge_rank[(a, b)] = rank

        return self

    def encode(self, text):
        ids = list(text.encode("utf-8"))

        if not self.merge_rank:
            return ids  # no merges loaded -> plain byte tokenizer

        while len(ids) >= 2:
            best_pair = None
            best_rank = None
            for a, b in zip(ids, ids[1:]):
                rank = self.merge_rank.get((a, b))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = (a, b)

            if best_pair is None:
                break  # no more merges apply to what's left

            new_id = 256 + best_rank
            ids = self._merge(ids, best_pair, new_id)

        return ids

    def decode(self, ids):
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _merge(ids, pair, new_id):
        merged = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    @property
    def vocab_size(self):
        return len(self.vocab)


def load():
    tok = BPETokenizer()
    merges_path = "merges.txt"
    tok.load_merges(merges_path)
    return tok