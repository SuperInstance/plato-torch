"""
RefractionRoom — Ideas bending through different agents create new spectra.

Inspired by Forgemaster's 20 refractive synergies. When two repos
(or concepts) hit a prism (agent), they refract into something
neither could produce alone.

Not similarity search — ORTHOGONAL intersection.

API: feed(concept), train_step(), predict(input), export_model()
"""
from typing import Any, Dict, List, Optional
import hashlib


class RefractionRoom:
    """Find refractions — orthogonal intersections that create new ideas."""

    def __init__(self, room_id: str = "refraction", prism_count: int = 3, **kwargs):
        self.room_id = room_id
        self.prism_count = prism_count
        self.concepts: List[Dict] = []
        self.refractions: List[Dict] = []
        self.state = {"total_feeds": 0, "total_refractions": 0, "avg_novelty": 0.0}

    def feed(self, data: Any = None, **kwargs) -> Dict:
        if data is None:
            return {"concepts": len(self.concepts)}
        concept = self._parse_concept(data)
        if concept:
            self.concepts.append(concept)
            self.state["total_feeds"] += 1
        return {"concepts": len(self.concepts), "ready_for_refraction": len(self.concepts) >= 2}

    def train_step(self, batch: Any = None, **kwargs) -> Dict:
        new_refractions = 0
        for i in range(len(self.concepts)):
            for j in range(i + 1, len(self.concepts)):
                a, b = self.concepts[i], self.concepts[j]
                pair_key = f"{a.get('id','')}:{b.get('id','')}"
                if any(r.get("pair_key") == pair_key for r in self.refractions):
                    continue
                orthogonality = self._orthogonality(a, b)
                if orthogonality > 0.3:
                    self.refractions.append({
                        "pair_key": pair_key,
                        "concept_a": a.get("domain", "unknown"),
                        "concept_b": b.get("domain", "unknown"),
                        "orthogonality": round(orthogonality, 4),
                        "novelty": round(orthogonality * (a.get("quality", 0.5) + b.get("quality", 0.5)) / 2, 4),
                        "synthesis": self._synthesize(a, b),
                    })
                    new_refractions += 1
        self.state["total_refractions"] = len(self.refractions)
        if self.refractions:
            self.state["avg_novelty"] = round(
                sum(r["novelty"] for r in self.refractions) / len(self.refractions), 4)
        return {"new_refractions": new_refractions, "total_refractions": len(self.refractions),
                "avg_novelty": self.state["avg_novelty"]}

    def predict(self, input: Any = None, **kwargs) -> Dict:
        if input is None or not self.concepts:
            return {"best_refraction": None, "total_refractions": len(self.refractions)}
        query = self._parse_concept(input)
        if query is None:
            return {"best_refraction": None, "total_refractions": len(self.refractions)}
        best, best_ortho = None, 0.0
        for concept in self.concepts:
            o = self._orthogonality(query, concept)
            if o > best_ortho:
                best_ortho, best = o, concept
        if best and best_ortho > 0.3:
            return {"best_refraction": best.get("domain", "unknown"),
                    "orthogonality": round(best_ortho, 4),
                    "synthesis": self._synthesize(query, best),
                    "total_refractions": len(self.refractions)}
        return {"best_refraction": None, "orthogonality": round(best_ortho, 4),
                "total_refractions": len(self.refractions)}

    def export_model(self, **kwargs) -> Dict:
        return {"room_id": self.room_id, "type": "refraction",
                "concepts": len(self.concepts), "refractions": len(self.refractions),
                "avg_novelty": self.state["avg_novelty"],
                "top_refractions": sorted(self.refractions, key=lambda r: r["novelty"], reverse=True)[:5],
                "doctrine": "Not similarity — REFRACTION. The same idea bent through a different medium."}

    def _parse_concept(self, data: Any) -> Optional[Dict]:
        if isinstance(data, dict):
            data["id"] = data.get("id", hashlib.md5(str(data).encode()).hexdigest()[:8])
            data.setdefault("domain", "general")
            data.setdefault("quality", 0.5)
            data.setdefault("tags", [])
            return data
        elif isinstance(data, str):
            return {"id": hashlib.md5(data.encode()).hexdigest()[:8], "domain": "general",
                    "quality": 0.5, "tags": data.lower().split()[:5], "content": data[:100]}
        return None

    def _orthogonality(self, a: Dict, b: Dict) -> float:
        domain_match = 0.0 if a.get("domain") != b.get("domain") else 0.5
        tags_a, tags_b = set(a.get("tags", [])), set(b.get("tags", []))
        tag_ortho = (1.0 - len(tags_a & tags_b) / max(len(tags_a | tags_b), 1)) if tags_a and tags_b else 0.5
        return (domain_match + tag_ortho) / 2

    def _synthesize(self, a: Dict, b: Dict) -> str:
        return f"{a.get('domain','?')} × {b.get('domain','?')} → new spectrum"
