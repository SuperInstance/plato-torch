"""
FractalRoom — The Generative Rule applied recursively.

The Fractal Doctrine: 'Find existing structure. Inhabit it.
Leave it better than you found it. The structure remembers.'

This room applies one rule at every scale:
tile -> room -> fleet -> city -> civilization.

API: feed(data), train_step(), predict(input), export_model()
"""
from typing import Any, Dict, List, Optional


class FractalRoom:
    """Apply one generative rule recursively at every scale."""

    def __init__(self, room_id: str = "fractal", rule: str = "inhabit_improve_remember",
                 max_depth: int = 5, **kwargs):
        self.room_id = room_id
        self.rule = rule
        self.max_depth = max_depth
        self.structures: List[Dict] = []
        self.scales = {"tile": [], "room": [], "fleet": [], "city": [], "civilization": []}
        self.state = {"total_feeds": 0, "depth_reached": 0,
                      "structures_inhabited": 0, "improvements_made": 0}

    def feed(self, data: Any = None, **kwargs) -> Dict:
        if data is None:
            return {"status": "looking", "structures": len(self.structures)}
        structure = self._parse_structure(data)
        if structure:
            self.structures.append(structure)
            scale = structure.get("scale", "tile")
            if scale in self.scales:
                self.scales[scale].append(structure)
            self.state["total_feeds"] += 1
            self.state["structures_inhabited"] += 1
        active_scales = [s for s, items in self.scales.items() if items]
        self.state["depth_reached"] = len(active_scales)
        return {"status": "inhabited", "depth": self.state["depth_reached"],
                "structures": len(self.structures)}

    def train_step(self, batch: Any = None, **kwargs) -> Dict:
        improvements = 0
        for struct in self.structures:
            if not struct.get("improved", False):
                struct["improved"] = True
                struct["quality"] = min(struct.get("quality", 0.5) + 0.1, 1.0)
                improvements += 1
        self.state["improvements_made"] += improvements
        consistency = 0.0
        active = [s for s, items in self.scales.items() if items]
        if len(active) >= 2:
            counts = [len(self.scales[s]) for s in active]
            consistency = min(counts) / max(counts)
        return {"improvements": improvements, "total_improvements": self.state["improvements_made"],
                "depth": self.state["depth_reached"], "fractal_consistency": round(consistency, 4),
                "structures": len(self.structures)}

    def predict(self, input: Any = None, **kwargs) -> Dict:
        if input is None:
            return {"fractal_depth": self.state["depth_reached"], "max_depth": self.max_depth,
                    "rule": self.rule, "structures": len(self.structures)}
        struct = self._parse_structure(input)
        if struct is None:
            return {"fractal_depth": 0, "rule": self.rule}
        resonances = 0
        for scale, items in self.scales.items():
            for item in items:
                if item.get("content") == struct.get("content"):
                    resonances += 1
                    break
        return {"fractal_depth": resonances, "max_depth": self.max_depth,
                "rule": self.rule, "resonates_with": resonances,
                "consistency": round(resonances / max(self.state["depth_reached"], 1), 4)}

    def export_model(self, **kwargs) -> Dict:
        return {"room_id": self.room_id, "type": "fractal", "rule": self.rule,
                "depth": self.state["depth_reached"], "structures": len(self.structures),
                "improvements": self.state["improvements_made"],
                "doctrine": "Find existing structure. Inhabit it. Leave it better. The structure remembers.",
                "scales": {s: len(items) for s, items in self.scales.items()}}

    def _parse_structure(self, data: Any) -> Optional[Dict]:
        if isinstance(data, dict):
            return {"scale": data.get("scale", "tile"), "quality": float(data.get("quality", 0.5)),
                    "content": data.get("content", str(data)[:100]), "improved": data.get("improved", False)}
        elif isinstance(data, str):
            return {"scale": "tile", "quality": 0.5, "content": data[:100], "improved": False}
        elif isinstance(data, (list, tuple)):
            return {"scale": "room", "quality": 0.5,
                    "content": f"collection of {len(data)} items", "improved": False}
        return None
