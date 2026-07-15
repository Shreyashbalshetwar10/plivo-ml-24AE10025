"""
Trains a simple byte-level BPE tokenizer on a text corpus and saves
the learned merge rules to merges.txt.

Usage:
    python train_tokenizer.py train_corpus.txt merges.txt --num_merges 1000 --min_freq 5

merges.txt is the ONLY artifact needed at inference time — tokenizer.py
reconstructs the full vocabulary (token id -> bytes) from it, since:
  - tokens 0..255 are always the raw bytes
  - merge i (0-indexed, in file order) always produces token id 256+i
"""

import argparse
import time
from collections import Counter


def get_pair_counts(ids):
    counts = Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def merge_ids(ids, pair, new_id):
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


def train_bpe(text_bytes, num_merges, min_freq, verbose=True):
    ids = list(text_bytes)  # start as raw bytes: 0..255
    merges = []

    for i in range(num_merges):
        pair_counts = get_pair_counts(ids)
        if not pair_counts:
            if verbose:
                print("Stopping: no more adjacent pairs (sequence collapsed to 1 token).")
            break

        best_pair, best_count = max(pair_counts.items(), key=lambda kv: kv[1])

        if best_count < min_freq:
            if verbose:
                print(f"Stopping early at merge {i}: "
                      f"best pair frequency {best_count} < min_freq {min_freq}")
            break

        new_id = 256 + i
        ids = merge_ids(ids, best_pair, new_id)
        merges.append(best_pair)

        if verbose and (i % 50 == 0 or i == num_merges - 1):
            print(f"merge {i:4d}: {best_pair} -> {new_id}  "
                  f"(count={best_count}, seq_len={len(ids)})")

    return merges


def save_merges(merges, path):
    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")


def main():
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("corpus_path", help="Path to training corpus (text file).", default ="data/train_corpus.txt")
    parser.add_argument("output_path", help="Where to save merges.txt", default="data/merges.txt")
    parser.add_argument("--num_merges", type=int, default=1000,
                         help="Max number of merge operations (default: 1000).")
    parser.add_argument("--min_freq", type=int, default=5,
                         help="Stop merging once best pair frequency drops below this (default: 5).")
    args = parser.parse_args()

    with open(args.corpus_path, "rb") as f:
        text_bytes = f.read()

    print(f"Loaded corpus: {len(text_bytes):,} bytes")

    t0 = time.time()
    merges = train_bpe(text_bytes, args.num_merges, args.min_freq)
    t1 = time.time()

    save_merges(merges, args.output_path)

    final_vocab_size = 256 + len(merges)
    print(f"\nDone in {t1 - t0:.1f}s")
    print(f"Learned {len(merges)} merges -> vocab size = {final_vocab_size}")
    print(f"Saved merges to {args.output_path}")


if __name__ == "__main__":
    main()