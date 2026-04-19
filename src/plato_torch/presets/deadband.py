"""
DeadbandRoom — Navigation by where the rocks are not.

Inspired by Casey's first captain: "I know where they are NOT.
And I have myself a path of safe."

FM's plato-tile-spec TileDomain::NegativeSpace validates this:
tiles about what NOT to do are worth 10x positive tiles.

The room doesn't catalog failures. It maps safe channels.
Agents navigate by attraction to safety, not avoidance of danger.

API: feed(safe_path), train_step(), predict(state), export_model()
"""
from typing import Any, Dict, List, Optional
import math


class DeadbandRoom:
    """Navigate by the deadband — the channel between the rocks."""

    def __init__(self, room_id: str = "deadband", channel_radius: float = 0.3,
                 decay_rate: float = 0.05, negative_space_weight: float = 10.0,
                 **kwargs):
        self.room_id = room_id
        self.channel_radius = channel_radius
        self.decay_rate = decay_rate
        self.negative_space_weight = negative_space_weight
        self.safe_paths: List[Dict] = []
        self.negative_spaces: List[Dict] = []  # Where NOT to go (10x value)
        self.landmarks: List[Dict] = []
        self.state = {
            "total_feeds": 0,
            "total_predictions": 0,
            "channels_discovered": 0,
            "negative_spaces_discovered": 0,
            "avg_safety": 0.0,
        }

    def feed(self, data: Any = None, **kwargs) -> Dict:
        """Record a safe path OR a negative space (where rocks are)."""
        if data is None:
            return {"status": "fed_nothing", "channels": len(self.safe_paths),
                    "negative_spaces": len(self.negative_spaces)}

        # Detect negative space tiles (FM's TileDomain::NegativeSpace)
        if isinstance(data, dict):
            domain = data.get("domain", "").lower()
            if domain in ("negativespace", "negative_space", "negative"):
                return self._feed_negative(data)

        path = self._extract_path(data)
        if path:
            self.safe_paths.append(path)
            self.state["total_feeds"] += 1
            self.state["channels_discovered"] = len(self.safe_paths)
            for point in path.get("points", []):
                self.landmarks.append({
                    "position": point,
                    "weight": 1.0,
                    "source": path.get("source", "unknown"),
                    "feed_num": self.state["total_feeds"],
                })
        return {"status": "channel_recorded", "channels": len(self.safe_paths),
                "landmarks": len(self.landmarks)}

    def _feed_negative(self, data: Dict) -> Dict:
        """Record a negative space — where the rocks ARE. Worth 10x."""
        ns = {
            "position": data.get("position") or data.get("points"),
            "reason": data.get("answer") or data.get("reason") or data.get("question", "unknown"),
            "weight": self.negative_space_weight,
        }
        self.negative_spaces.append(ns)
        self.state["negative_spaces_discovered"] = len(self.negative_spaces)
        return {"status": "negative_space_recorded", "negative_spaces": len(self.negative_spaces),
                "weight": self.negative_space_weight}

    def train_step(self, batch: Any = None, **kwargs) -> Dict:
        """Consolidate. Decay unused landmarks. Strengthen negatives."""
        for lm in self.landmarks:
            lm["weight"] *= (1.0 - self.decay_rate)
            if lm["weight"] < 0.01:
                lm["weight"] = 0.0
        self.landmarks = [lm for lm in self.landmarks if lm["weight"] > 0.0]
        if self.landmarks:
            self.state["avg_safety"] = sum(lm["weight"] for lm in self.landmarks) / len(self.landmarks)
        return {"active_landmarks": len(self.landmarks), "negative_spaces": len(self.negative_spaces),
                "avg_safety": round(self.state["avg_safety"], 4), "channels": len(self.safe_paths)}

    def predict(self, input: Any = None, **kwargs) -> Dict:
        """Predict distance to safe water AND proximity to rocks."""
        self.state["total_predictions"] += 1
        if input is None or (not self.landmarks and not self.negative_spaces):
            return {"distance_to_safe_water": None, "distance_to_rocks": None,
                    "in_channel": False, "confidence": 0.0}

        pos = self._extract_position(input)
        if pos is None:
            return {"distance_to_safe_water": None, "distance_to_rocks": None,
                    "in_channel": False, "confidence": 0.0}

        # Distance to nearest safe water
        min_safe = float('inf')
        nearest_safe = None
        for lm in self.landmarks:
            d = self._distance(pos, lm["position"])
            if d < min_safe:
                min_safe = d
                nearest_safe = lm
        if nearest_safe and nearest_safe["weight"] > 0:
            nearest_safe["weight"] = min(nearest_safe["weight"] + 0.05, 1.0)

        # Distance to nearest rock (negative space)
        min_rock = float('inf')
        nearest_rock = None
        for ns in self.negative_spaces:
            if ns["position"] is not None:
                d = self._distance(pos, ns["position"])
                if d < min_rock:
                    min_rock = d
                    nearest_rock = ns

        in_channel = min_safe <= self.channel_radius if min_safe != float('inf') else False
        near_rocks = min_rock < self.channel_radius if min_rock != float('inf') else False

        return {
            "distance_to_safe_water": round(min_safe, 4) if min_safe != float('inf') else None,
            "distance_to_rocks": round(min_rock, 4) if min_rock != float('inf') else None,
            "in_channel": in_channel and not near_rocks,
            "nearest_safe": nearest_safe["source"] if nearest_safe else None,
            "nearest_rock": nearest_rock["reason"] if nearest_rock else None,
            "confidence": round(nearest_safe["weight"], 4) if nearest_safe else 0.0,
        }

    def export_model(self, **kwargs) -> Dict:
        return {
            "room_id": self.room_id, "type": "deadband",
            "channels": len(self.safe_paths), "landmarks": len(self.landmarks),
            "negative_spaces": len(self.negative_spaces),
            "negative_space_weight": self.negative_space_weight,
            "channel_radius": self.channel_radius, "state": self.state,
            "doctrine": "I know where the rocks are NOT. I have myself a path of safe.",
            "fm_integration": "TileDomain::NegativeSpace from plato-tile-spec",
        }

    def _extract_path(self, data: Any) -> Optional[Dict]:
        if isinstance(data, dict):
            points = data.get("points") or data.get("path") or data.get("safe_path")
            if points:
                return {"points": points if isinstance(points, list) else [points],
                        "source": data.get("source", "feed"), "weight": data.get("weight", 1.0)}
        elif isinstance(data, (list, tuple)):
            return {"points": list(data), "source": "feed", "weight": 1.0}
        elif isinstance(data, (int, float)):
            return {"points": [data], "source": "feed", "weight": 1.0}
        return None

    def _extract_position(self, input: Any):
        if isinstance(input, dict):
            return input.get("position") or input.get("pos") or input.get("state")
        elif isinstance(input, (list, tuple)):
            return list(input)
        elif isinstance(input, (int, float)):
            return [input]
        return None

    def _distance(self, a, b) -> float:
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return math.sqrt(sum((float(x) - float(y))**2 for x, y in zip(a, b)))
        try:
            return abs(float(a) - float(b))
        except (TypeError, ValueError):
            return 1.0
