"""ContrastiveRoom – learns similarity by contrasting anchor/positive/negative triples."""

import json
import math
from collections import defaultdict
try:
    from room_base import RoomBase
except ImportError:
    from .room_base import RoomBase


class ContrastiveRoom(RoomBase):
    """Statistical contrastive learner. No torch – just vector math on dicts of floats."""

    def __init__(self, room_id="contrastive", **kw):
        super().__init__(room_id, **kw)
        self.vectors: dict[str, dict[str, float]] = {}
        self.sim_matrix: dict[str, dict[str, float]] = defaultdict(dict)
        self._triplets: list[dict] = []

    # ------------------------------------------------------------------
    def feed(self, data=None):
        """Accept {anchor, positive, negative} dicts. Each value is a feature dict."""
        if data is None:
            data = {"anchor": {"id": "a", "features": {"x": 1.0}},
                    "positive": {"id": "b", "features": {"x": 0.9}},
                    "negative": {"id": "c", "features": {"x": 0.1}}}
        if isinstance(data, str):
            data = {"anchor": {"id": data, "features": {"x": 1.0}},
                    "positive": {"id": data+"p", "features": {"x": 0.9}},
                    "negative": {"id": data+"n", "features": {"x": 0.1}}}
        self._triplets.append(data)
        for role in ("anchor", "positive", "negative"):
            item = data.get(role)
            if item and "id" in item:
                self.vectors[item["id"]] = item.get("features", {})

    # ------------------------------------------------------------------
    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
        na = math.sqrt(sum(v * v for v in a.values())) or 1e-12
        nb = math.sqrt(sum(v * v for v in b.values())) or 1e-12
        return dot / (na * nb)

    # ------------------------------------------------------------------
    def train_step(self) -> dict:
        """Build pairwise cosine similarities and reinforce positive pairs."""
        ids = list(self.vectors)
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                sim = self._cosine(self.vectors[id_a], self.vectors[id_b])
                self.sim_matrix[id_a][id_b] = sim
                self.sim_matrix[id_b][id_a] = sim

        margin_hits = 0
        for t in self._triplets:
            aid = t.get("anchor", {}).get("id")
            pid = t.get("positive", {}).get("id")
            nid = t.get("negative", {}).get("id")
            if aid and pid and nid:
                sp = self.sim_matrix.get(aid, {}).get(pid, 0.0)
                sn = self.sim_matrix.get(aid, {}).get(nid, 0.0)
                if sp > sn:
                    margin_hits += 1
        total = max(len(self._triplets), 1)
        return {"accuracy": margin_hits / total, "pairs": len(ids) * (len(ids) - 1) // 2}

    # ------------------------------------------------------------------
    def predict(self, query_id=None, top_k: int = 5) -> list[dict]:
        """Return nearest neighbors with similarity scores."""
        if query_id is None:
            ids = list(self.sim_matrix)
            query_id = ids[0] if ids else ''
        if isinstance(query_id, dict):
            query_id = str(query_id)
        row = self.sim_matrix.get(query_id, {})
        ranked = sorted(row.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": nid, "score": round(s, 6)} for nid, s in ranked]

    # ------------------------------------------------------------------
    def export_model(self) -> str:
        """JSON similarity graph."""
        graph = {k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in self.sim_matrix.items()}
        return json.dumps({"vectors": len(self.vectors), "graph": graph}, indent=2)
