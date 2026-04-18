"""SupervisedRoom — statistical supervised learning from labeled pairs."""

import json
import hashlib
from collections import Counter
from room_base import RoomBase


class SupervisedRoom(RoomBase):
    """Learns input→label mappings via frequency counting."""

    def __init__(self, room_id: str = "supervised", **kwargs):
        super().__init__(room_id, **kwargs)
        self._buffer: list[dict] = []
        self.label_map: dict[str, dict[str, int]] = {}  # hash → {label: count}

    @staticmethod
    def _hash_input(value) -> str:
        """Deterministic hash of any serializable input."""
        raw = json.dumps(value, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def feed(self, data: dict) -> None:
        """Accept a dict with 'input' and 'label' keys."""
        if "input" not in data or "label" not in data:
            raise ValueError("feed() requires {input, label} dict")
        self._buffer.append({"input": data["input"], "label": str(data["label"])})

    def train_step(self) -> dict:
        """Ingest buffered pairs into the frequency lookup table."""
        added = 0
        for item in self._buffer:
            h = self._hash_input(item["input"])
            bucket = self.label_map.setdefault(h, {})
            bucket[item["label"]] = bucket.get(item["label"], 0) + 1
            added += 1
        self._buffer.clear()
        return {"pairs_processed": added, "unique_inputs": len(self.label_map)}

    def predict(self, input_value) -> dict:
        """Return the most common label for the given input."""
        h = self._hash_input(input_value)
        bucket = self.label_map.get(h)
        if not bucket:
            return {"label": None, "confidence": 0.0}
        total = sum(bucket.values())
        best_label, best_count = Counter(bucket).most_common(1)[0]
        return {"label": best_label, "confidence": best_count / total}

    def export_model(self) -> str:
        """Export the label map as JSON."""
        return json.dumps(self.label_map, indent=2)
