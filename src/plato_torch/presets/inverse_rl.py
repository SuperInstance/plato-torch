"""Inverse RL preset — observe expert, infer reward function."""
import json, hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from room_base import RoomBase
except ImportError:
    from .room_base import RoomBase


class InverseRLRoom(RoomBase):
    """Watch an expert perform, infer what reward function they're optimizing."""
    
    def __init__(self, room_id: str = "inverse_rl", **kwargs):
        super().__init__(room_id, preset="inverse_rl", **kwargs)
        self._expert_states = defaultdict(list)  # state_hash → [actions]
        self._inferred_rewards = defaultdict(lambda: defaultdict(float))  # state → action → reward
    
    def feed(self, data=None, **kwargs) -> Dict:
        if data is None: data = {}
        if isinstance(data, str): data = {"data": data}
        if isinstance(data, dict):
            return self.observe(data.get("state",""), data.get("action",""), data.get("outcome",""))
        return {"status": "invalid"}
    
    def observe_expert(self, demonstrations: List[Dict]) -> Dict:
        """Bulk-load expert demonstrations."""
        for demo in demonstrations:
            self.observe(demo.get("state",""), demo.get("action",""), demo.get("outcome",""),
                        agent_id="expert")
        return {"demos_loaded": len(demonstrations)}
    
    def train_step(self, batch=None) -> Dict:
        if batch is None:
            return {"status": "ok", "message": "no batch", "preset": "inverse_rl"}
        """Infer reward function from expert behavior using inverse RL heuristic.
        
        Expert chose action A in state S → A gets higher reward than alternatives.
        Frequency of choice + outcome quality = reward signal.
        """
        # Group by state
        state_actions = defaultdict(lambda: defaultdict(list))
        for tile in batch:
            sh = tile.get("state_hash", "")
            action = tile.get("action", "")
            reward = tile.get("reward", 0)
            state_actions[sh][action].append(reward)
        
        # Infer: actions the expert chose frequently with good outcomes → high reward
        for sh, actions in state_actions.items():
            # Count frequency of each action (expert preference)
            total_choices = sum(len(rs) for rs in actions.values())
            for action, rewards in actions.items():
                frequency = len(rewards) / max(total_choices, 1)
                avg_outcome = sum(rewards) / len(rewards) if rewards else 0
                # Reward = preference (frequency) * outcome quality
                self._inferred_rewards[sh][action] = frequency * (1 + avg_outcome)
        
        return {"status": "inferred", "states_analyzed": len(state_actions)}
    
    def predict(self, input=None) -> Dict:
        h = hashlib.md5(str(input).encode()).hexdigest()[:8]
        rewards = dict(self._inferred_rewards.get(h, {}))
        if rewards:
            best = max(rewards, key=rewards.get)
            return {"inferred_rewards": rewards, "expert_would_choose": best}
        return {"inferred_rewards": {}, "expert_would_choose": None}
    
    def export_model(self, format: str = "json") -> Optional[bytes]:
        model = {"room_id": self.room_id, "preset": "inverse_rl",
                 "inferred_rewards": {k: dict(v) for k, v in self._inferred_rewards.items()}}
        return json.dumps(model, indent=2).encode()
