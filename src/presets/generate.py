"""Generative data augmentation room — n-gram based synthetic state generator."""

import json
import random
from collections import Counter, defaultdict
from room_base import RoomBase


class GenerateRoom(RoomBase):
    """Learns the distribution of room states and generates realistic synthetic ones."""

    def __init__(self, name: str = "generate", n: int = 2):
        super().__init__(name)
        self.n = n
        self._buffer: list[str] = []
        self._ngram_counts: dict[str, Counter] = defaultdict(Counter)
        self._trained = False

    def feed(self, *states: str) -> None:
        """Accept real states (strings) for learning."""
        for s in states:
            if isinstance(s, str) and s:
                self._buffer.append(s)

    def train_step(self) -> dict:
        """Build n-gram frequency model from accumulated states."""
        for state in self._buffer:
            tokens = state.split()
            for i in range(len(tokens) - self.n + 1):
                prefix = " ".join(tokens[i : i + self.n - 1])
                next_tok = tokens[i + self.n - 1]
                self._ngram_counts[prefix][next_tok] += 1
        self._trained = True
        ngrams = sum(len(v) for v in self._ngram_counts)
        return {"states_consumed": len(self._buffer), "ngrams_learned": ngrams}

    def predict(self, max_tokens: int = 30, seed: str | None = None) -> str:
        """Generate a synthetic state by sampling from the n-gram distribution."""
        if not self._trained or not self._ngram_counts:
            return ""
        if seed is not None:
            tokens = seed.split()
        else:
            prefix = random.choice(list(self._ngram_counts.keys()))
            tokens = prefix.split()
        for _ in range(max_tokens - len(tokens)):
            prefix = " ".join(tokens[-(self.n - 1) :]) if self.n > 1 else ""
            if prefix not in self._ngram_counts:
                break
            candidates = list(self._ngram_counts[prefix].elements())
            tokens.append(random.choice(candidates))
        return " ".join(tokens)

    def export_model(self) -> str:
        """Return JSON of n-gram frequencies."""
        data = {k: dict(v) for k, v in self._ngram_counts.items()}
        return json.dumps({"n": self.n, "ngrams": data}, indent=2)
