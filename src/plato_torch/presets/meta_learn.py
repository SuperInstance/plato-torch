"""MetaLearnRoom — learns fast-adaptation profiles per task. Pure Python, no numpy."""

import json, math, hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from room_base import RoomBase
except ImportError:
    from .room_base import RoomBase


class MetaLearnRoom(RoomBase):
    """Meta-learning: adapt to new tasks in 1-3 examples via nearest-task lookup."""

    def __init__(self, room_id: str = "meta_learn", **kwargs):
        super().__init__(room_id, preset="meta_learn", **kwargs)
        self._task_centroids = {}   # task → {feature_hash → count}
        self._task_actions = {}     # task → {state_hash → best_action}
        self._buffer = defaultdict(list)

    def feed(self, data=None, **kwargs) -> Dict:
        if data is None: data = {}
        if isinstance(data, str): data = {"data": data}
        if isinstance(data, dict):
            task = data.get("task", "default")
            return self.observe(data.get("state",""), data.get("action",""), data.get("outcome",""),
                              context={"task": task})
        return {"status": "invalid"}

    def add_task(self, task: str, examples: List[Dict]):
        """Add examples for a task."""
        for ex in examples:
            self._buffer[task].append(ex)

    def train_step(self, batch=None) -> Dict:
        if batch is None:
            return {"status": "ok", "message": "no batch", "preset": "meta_learn"}
        for tile in batch:
            task = tile.get("context", {}).get("task", "default")
            sh = tile.get("state_hash", "")
            action = tile.get("action", "")
            reward = tile.get("reward", 0)

            # Update centroid
            if task not in self._task_centroids:
                self._task_centroids[task] = defaultdict(int)
            self._task_centroids[task][sh] += 1

            # Update best action
            if task not in self._task_actions:
                self._task_actions[task] = defaultdict(lambda: defaultdict(list))
            self._task_actions[task][sh][action].append(reward)

        return {"status": "trained", "tasks": len(self._task_centroids)}

    def _distance(self, query_features: Dict, task: str) -> float:
        """Cosine-like distance between query and task centroid."""
        centroid = self._task_centroids.get(task, {})
        if not centroid:
            return float('inf')
        # Jaccard-like overlap
        shared = set(query_features.keys()) & set(centroid.keys())
        if not shared:
            return float('inf')
        return 1.0 / (len(shared) + 1)  # more overlap = smaller distance

    def predict(self, input=None) -> Dict:
        state = str(input)
        sh = hashlib.md5(state.encode()).hexdigest()[:8]
        query_features = {sh: 1}

        # Find nearest task
        if not self._task_centroids:
            return {"action": None, "task": None, "adapted_in": 0}

        best_task = min(self._task_centroids.keys(),
                       key=lambda t: self._distance(query_features, t))

        # Use that task's knowledge
        task_data = self._task_actions.get(best_task, {}).get(sh, {})
        if task_data:
            best_action = max(task_data, key=lambda a: sum(task_data[a])/len(task_data[a]))
            return {"action": best_action, "task": best_task, "adapted_in": 1}

        return {"action": None, "task": best_task, "adapted_in": 0}

    def export_model(self, format: str = "json") -> Optional[bytes]:
        model = {"room_id": self.room_id, "preset": "meta_learn",
                 "tasks": list(self._task_centroids.keys()),
                 "centroid_sizes": {t: len(c) for t, c in self._task_centroids.items()}}
        return json.dumps(model, indent=2).encode()
