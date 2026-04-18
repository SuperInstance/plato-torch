"""MetaLearnRoom — learns fast-adaptation profiles per task."""

import json
import numpy as np
from collections import defaultdict
from room_base import RoomBase


class MetaLearnRoom(RoomBase):
    """Meta-learning room that adapts to new tasks via nearest-task lookup."""

    def __init__(self, input_dim=64, lr=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.lr = lr
        self.task_profiles = {}  # task_id -> {centroid, weights}
        self._buffer = defaultdict(list)  # task_id -> [(x, y), ...]

    def feed(self, task_id, examples):
        """Accept task_id + list of (input, label) example pairs."""
        for x, y in examples:
            x = np.asarray(x, dtype=np.float32)
            self._buffer[task_id].append((x, float(y)))

    def _build_profile(self, task_id):
        """Build a task adaptation profile from buffered examples."""
        data = self._buffer[task_id]
        xs = np.array([d[0] for d in data])
        ys = np.array([d[1] for d in data])
        centroid = xs.mean(axis=0)
        # Simple linear weights via normal equation (pseudo-inverse)
        X = np.column_stack([xs, np.ones(len(xs))])
        weights = np.linalg.lstsq(X, ys, rcond=None)[0]
        self.task_profiles[task_id] = {
            "centroid": centroid,
            "weights": weights,
            "count": len(data),
        }

    def train_step(self):
        """Build/update fast-adaptation tables for all buffered tasks."""
        for task_id in list(self._buffer):
            if len(self._buffer[task_id]) >= 2:
                self._build_profile(task_id)
        # Clear buffered data after building
        self._buffer.clear()

    def predict(self, examples):
        """Adapt to a new task using 1-3 examples via nearest-task lookup.

        Args:
            examples: list of (input,) or (input, label) pairs.

        Returns:
            Predicted outputs for the input examples.
        """
        if not self.task_profiles:
            return [0.0] * len(examples)

        xs = [np.asarray(e[0], dtype=np.float32) for e in examples]
        # Compute query centroid
        query_centroid = np.mean(xs, axis=0)

        # Find nearest task by centroid distance
        best_task = min(
            self.task_profiles,
            key=lambda tid: np.linalg.norm(
                self.task_profiles[tid]["centroid"] - query_centroid
            ),
        )
        weights = self.task_profiles[best_task]["weights"]

        # Predict using nearest task's weights
        results = []
        for x in xs:
            x_aug = np.append(x, 1.0)
            results.append(float(x_aug @ weights))
        return results

    def export_model(self):
        """Export task adaptation profiles as JSON-serializable dict."""
        out = {}
        for tid, prof in self.task_profiles.items():
            out[tid] = {
                "centroid": prof["centroid"].tolist(),
                "weights": prof["weights"].tolist(),
                "count": prof["count"],
            }
        return json.dumps({"task_profiles": out, "input_dim": self.input_dim}, indent=2)
