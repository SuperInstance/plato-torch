"""
DeadbandRoom — Navigation by where the rocks are not.

Inspired by Casey's first captain: "I know where they are NOT.
And I have myself a path of safe."

The room doesn't catalog failures. It maps safe channels.
Agents navigate by attraction to safety, not avoidance of danger.

API: feed(safe_path), train_step(), predict(state) -> distance_to_safe_water
"""
from typing import Any, Dict, List, Optional
import math


class DeadbandRoom:
    """Navigate by the deadband — the channel between the rocks."""

    def __init__(self, room_id: str = "deadband", channel_radius: float = 0.3,
                 decay_rate: float = 0.05, **kwargs):
        self.room_id = room_id
        self.channel_radius = channel_radius
        self.decay_rate = decay_rate
        self.safe_paths: List[Dict] = []
        self.landmarks: List[Dict] = []
        self.state = {
            "total_feeds": 0,
            "total_predictions": 0,
            "channels_discovered": 0,
            "avg_safety": 0.0,
        }

    def feed(self, data: Any = None, **kwargs) -> Dict:
        """Record a safe path through the space. Not rocks — CHANNELS."""
        if data is None:
            return {"status": "fed_nothing", "channels": len(self.safe_paths)}
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
        return {
            "status": "channel_recorded",
            "channels": len(self.safe_paths),
            "landmarks": len(self.landmarks),
        }

    def train_step(self, batch: Any = None, **kwargs) -> Dict:
        """Consolidate safe channels. Decay unused landmarks."""
        active = 0
        for lm in self.landmarks:
            lm["weight"] *= (1.0 - self.decay_rate)
            if lm["weight"] < 0.01:
                lm["weight"] = 0.0
            else:
                active += 1
        self.landmarks = [lm for lm in self.landmarks if lm["weight"] > 0.0]
        if self.landmarks:
            self.state["avg_safety"] = sum(lm["weight"] for lm in self.landmarks) / len(self.landmarks)
        return {
            "active_landmarks": len(self.landmarks),
            "avg_safety": round(self.state["avg_safety"], 4),
            "channels": len(self.safe_paths),
        }

    def predict(self, input: Any = None, **kwargs) -> Dict:
        """Given a state, predict distance to nearest safe water."""
        self.state["total_predictions"] += 1
        if input is None or not self.landmarks:
            return {"distance_to_safe_water": None, "in_channel": False, "confidence": 0.0}
        pos = self._extract_position(input)
        if pos is None:
            return {"distance_to_safe_water": None, "in_channel": False, "confidence": 0.0}
        min_dist = float('inf')
        nearest = None
        for lm in self.landmarks:
            d = self._distance(pos, lm["position"])
            if d < min_dist:
                min_dist = d
                nearest = lm
        if nearest and nearest["weight"] > 0:
            nearest["weight"] = min(nearest["weight"] + 0.05, 1.0)
        in_channel = min_dist <= self.channel_radius if min_dist != float('inf') else False
        return {
            "distance_to_safe_water": round(min_dist, 4) if min_dist != float('inf') else None,
            "in_channel": in_channel,
            "nearest_landmark": nearest["source"] if nearest else None,
            "confidence": round(nearest["weight"], 4) if nearest else 0.0,
        }

    def export_model(self, **kwargs) -> Dict:
        """Export the deadband map — the channels, not the rocks."""
        return {
            "room_id": self.room_id,
            "type": "deadband",
            "channels": len(self.safe_paths),
            "landmarks": len(self.landmarks),
            "channel_radius": self.channel_radius,
            "state": self.state,
            "doctrine": "I know where the rocks are NOT. I have myself a path of safe.",
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
