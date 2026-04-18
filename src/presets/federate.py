"""FederateRoom — federated averaging across multiple agents."""

import json
import numpy as np
from collections import defaultdict
from room_base import RoomBase


class FederateRoom(RoomBase):
    """Aggregates learning across agents via federated averaging."""

    def __init__(self, model_dim=64):
        super().__init__()
        self.model_dim = model_dim
        self.aggregated = np.zeros(model_dim, dtype=np.float32)
        self.agent_updates = defaultdict(list)  # agent_id -> [update, ...]
        self.round_num = 0

    def feed(self, data):
        """Accept dict with agent_id and gradients_or_updates (array-like)."""
        agent_id = data["agent_id"]
        updates = np.asarray(data["gradients_or_updates"], dtype=np.float32)
        self.agent_updates[agent_id].append(updates)

    def train_step(self):
        """Federated averaging: mean of per-agent averaged updates."""
        if not self.agent_updates:
            return

        per_agent_means = []
        for agent_id, updates_list in self.agent_updates.items():
            mean_update = np.mean(updates_list, axis=0)
            per_agent_means.append(mean_update)

        # Federated average across agents
        fed_avg = np.mean(per_agent_means, axis=0)
        self.aggregated = self.aggregated + fed_avg
        self.agent_updates.clear()
        self.round_num += 1

    def predict(self, x=None):
        """Return consensus prediction (aggregated model applied to x, or raw model)."""
        if x is not None:
            x = np.asarray(x, dtype=np.float32)
            return float(x @ self.aggregated[: len(x)])
        return self.aggregated.tolist()

    def export_model(self):
        """Export aggregated model as JSON."""
        return json.dumps({
            "aggregated": self.aggregated.tolist(),
            "model_dim": self.model_dim,
            "round_num": self.round_num,
        }, indent=2)
