"""
Active Learning preset — model chooses what data to learn from.
"""
import hashlib, json, random
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from room_base import RoomBase
except ImportError:
    from ..room_base import RoomBase


class ActiveRoom(RoomBase):
    """Active learning room. Queries for labels on the most uncertain examples."""
    
    def __init__(self, room_id: str, **kwargs):
        super().__init__(room_id, preset="active", **kwargs)
        self.budget = kwargs.get("budget", 100)
        self.query_size = kwargs.get("query_size", 10)
        self._label_counts = defaultdict(lambda: defaultdict(int))
        self._queried = set()
    
    def feed(self, data: Any, **kwargs) -> Dict:
        if isinstance(data, dict) and "input" in data and "label" in data:
            h = self._hash(data["input"])
            self._label_counts[h][data["label"]] += 1
            self._queried.add(h)
            return {"labeled": True, "hash": h}
        return {"status": "need_input_and_label"}
    
    def train_step(self, batch: List[Dict]) -> Dict:
        for tile in batch:
            h = tile.get("state_hash", "")
            action = tile.get("action", "")
            reward = tile.get("reward", 0)
            if reward > 0:
                self._label_counts[h][action] += 1
        return {"status": "trained", "labeled_states": len(self._label_counts)}
    
    def predict(self, input: Any) -> Dict:
        h = self._hash(str(input))
        counts = self._label_counts.get(h, {})
        total = sum(counts.values())
        if total == 0:
            return {"label": None, "uncertainty": 1.0, "confidence": 0.0}
        best = max(counts, key=counts.get)
        confidence = counts[best] / total
        return {"label": best, "uncertainty": 1 - confidence, "confidence": round(confidence, 3)}
    
    def query_uncertain(self, candidates: List[str] = None, top_k: int = None) -> List[str]:
        """Return the most uncertain inputs that need labeling."""
        if top_k is None: top_k = self.query_size
        if candidates:
            scored = [(c, self.predict(c)["uncertainty"]) for c in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [c for c, _ in scored[:top_k]]
        return []
    
    def export_model(self, format: str = "json") -> Optional[bytes]:
        model = {"room_id": self.room_id, "preset": "active",
                 "labels": {k: dict(v) for k, v in self._label_counts.items()},
                 "queried": len(self._queried), "budget": self.budget}
        return json.dumps(model, indent=2).encode()
    
    def _hash(self, s: str) -> str:
        return hashlib.md5(s.encode()).hexdigest()[:8]
