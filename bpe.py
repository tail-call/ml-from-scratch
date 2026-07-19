from collections import defaultdict
from collections.abc import Mapping
from typing import TypeAlias

Token: TypeAlias = int
TokenPair: TypeAlias = tuple[int, int]


def adjacent_pairs_frequencies(tokens: list[Token]) -> Mapping[TokenPair, int]:
    """Count frequencies of adjacent pairs."""

    counts: defaultdict[TokenPair, Token] = defaultdict(int)

    for i in range(len(tokens) - 1):
        counts[(tokens[i], tokens[i + 1])] = (
            counts.get((tokens[i], tokens[i + 1]), 0) + 1
        )

    return counts


def replace_pair_with_token(
    tokens: list[Token], pair: TokenPair, replacement_token: Token
) -> list[Token]:
    """Replace all occurrences of a pair with a new token ID."""
    new_ids = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_ids.append(replacement_token)
            i += 2
        else:
            new_ids.append(tokens[i])
            i += 1
    return new_ids


class SimpleBPE:
    def __init__(self, vocabulary_size: int):
        self.vocabulary_size: int = vocabulary_size

        self.merges: dict[TokenPair, Token] = {}
        "Map of (char1, char2) -> merged_char"

        self.vocabulary: dict[Token, bytes] = {}

    def train(self, text: str):
        """Train BPE on raw text until vocab_size is reached."""
        # Start with standard byte/character tokens (0-255)
        # For simplicity, we convert characters to their UTF-8 byte representations
        tokens: list[Token] = list(text.encode("utf-8"))
        num_merges = self.vocabulary_size - 256

        for i in range(num_merges):
            print(f"Merge {i}/{num_merges}...")
            frequencies = adjacent_pairs_frequencies(tokens)
            if not frequencies:
                # No more pairs to merge
                break

            # Find the most frequent pair
            best_pair = max(frequencies, key=lambda k: frequencies[k])
            new_id = 256 + i

            # Record the merge
            self.merges[best_pair] = new_id
            tokens = replace_pair_with_token(tokens, best_pair, new_id)

        # Build vocabulary mapping (token_id -> bytes)
        self.vocabulary = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocabulary[idx] = self.vocabulary[p0] + self.vocabulary[p1]

    def encode(self, text):
        """Encode new text into BPE token IDs."""
        tokens = list(text.encode("utf-8"))
        if len(tokens) <= 1:
            return tokens

        while True:
            # Find the pair that occurred earliest in our training merges
            frequencies = adjacent_pairs_frequencies(tokens)
            # Find which of the available pairs exists in our merge rules
            pair = min(frequencies.keys(), key=lambda p: self.merges.get(p, float("inf")))

            # If the best pair isn't in our merges, we are done
            if pair not in self.merges:
                break

            # Otherwise, merge it
            tokens = replace_pair_with_token(tokens, pair, self.merges[pair])

        return tokens

    def decode(self, ids):
        """Decode token IDs back into a readable string."""
        text_bytes = b"".join(self.vocabulary[idx] for idx in ids)
        # errors='replace' handles any malformed UTF-8 fragments gracefully
        return text_bytes.decode("utf-8", errors="replace")


bpe = SimpleBPE(4000)

with open(
    "/Users/scales/Documents/Books/txt/Bible.txt"
) as file:
    text = file.read()
    bpe.train(text)

raw = "В начале сотворил Господь"

[bpe.vocabulary[x].decode("utf-8", errors="replace") for x in bpe.encode(raw)]
for k in bpe.vocabulary:
    print(bpe.vocabulary[k].decode("utf-8", errors="replace"))
import random

bpe.decode([random.randint(520, 1023) for x in range(200)])
