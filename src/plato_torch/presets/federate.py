"""FederateRoom — federated averaging across multiple agents. Pure Python, no numpy."""

import json, hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from ..room_base import RoomBase


class FederateRoom(RoomBase):
    """Aggregates learning across agents via federated averaging."""

    def __init__(self, room_id: str, **kwargs):
        super().__init__(room_id, preset="federate", **kwargs)
        self._agent_models = {}        # agent_id → {state_hash → {action → score}}
        self._consensus = defaultdict(lambda: defaultdict(float))  # merged model
        self.round_num = 0

    def feed(self, data: Any, **kwargs) -> Dict:
        if isinstance(data, dict):
            agent_id = data.get("agent_id", "unknown")
            updates = data.get("updates", {})
            if agent_id not in self._agent_models:
                self._agent_models[agent_id] = {}
            self._agent_models[agent_id].update(updates)
            return {"agent": agent_id, "updates_received": len(updates)}
        return {"status": "invalid"}

    def local_update(self, agent_id: str, gradients: Dict) -> Dict:
        """Accept gradient-like updates from an agent."""
        if agent_id not in self._agent_models:
            self._agent_models[agent_id] = {}
        for key, value in gradients.items():
            self._agent_models[agent_id][key] = self._agent_models[agent_id].get(key, 0) + value
        return {"agent": agent_id, "accepted": True}

    def train_step(self, batch: List[Dict]) -> Dict:
        """Federated averaging: merge all agent models into consensus."""
        # Also process any batch tiles
        for tile in batch:
            sh = tile.get("state_hash", "")
            action = tile.get("action", "")
            reward = tile.get("reward", 0)
            self._consensus[f"{sh}:{action}"] = self._consensus[f"{sh}:{action}"] * 0.9 + reward * 0.1

        # Average agent models
        if self._agent_models:
            all_keys = set()
            for model in self._agent_models.values():
                all_keys.update(model.keys())
            
            for key in all_keys:
                values = [m.get(key, 0) for m in self._agent_models.values()]
                avg = sum(values) / len(values)
                self._consensus[key] = avg

        self.round_num += 1
        return {"round": self.round_num, "agents": len(self._agent_models),
                "consensus_keys": len(self._consensus)}

    def predict(self, input: Any) -> Dict:
        """Consensus prediction."""
        key = str(input)
        value = self._consensus.get(key, 0)
        # Find nearby keys
        nearby = {k: round(v, 3) for k, v in self._consensus.items() if key in k}
        return {"consensus_value": round(value, 3), "nearby": nearby,
                "agents": len(self._agent_models), "round": self.round_num}

    def export_model(self, format: str = "json") -> Optional[bytes]:
        model = {"room_id": self.room_id, "preset": "federate",
                 "round": self.round_num, "agents": list(self._agent_models.keys()),
                 "consensus_size": len(self._consensus)}
        return json.dumps(model, indent=2).encode()
