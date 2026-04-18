"""
Imitation Learning preset — clone expert behavior.
"""
import json, hashlib, random
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from room_base import RoomBase
except ImportError:
    from ..room_base import RoomBase


class ImitateRoom(RoomBase):
    """Clone expert behavior from demonstrations."""
    
    def __init__(self, room_id: str, **kwargs):
        super().__init__(room_id, preset="imitate", **kwargs)
        self._expert_actions = defaultdict(list)  # state_hash → list of (action, reward)
        self._clone_accuracy = defaultdict(float)
    
    def feed(self, data: Any, **kwargs) -> Dict:
        if isinstance(data, dict):
            expert = data.get("expert", "unknown")
            return self.observe(data.get("state",""), data.get("action",""),
                              data.get("outcome",""), agent_id=expert)
        return {"status": "invalid"}
    
    def watch_expert(self, expert_id: str, episodes: int = 100) -> Dict:
        """Shortcut: simulate watching an expert for N episodes."""
        for i in range(episodes):
            state = f"expert-demo-{i}"
            action = random.choice(["expert_move_a", "expert_move_b", "expert_move_c"])
            self.observe(state, action, "success", expert_id)
        return {"watched": episodes, "expert": expert_id}
    
    def train_step(self, batch: List[Dict]) -> Dict:
        for tile in batch:
            sh = tile.get("state_hash", "")
            action = tile.get("action", "")
            reward = tile.get("reward", 0)
            self._expert_actions[sh].append((action, reward))
        
        # Build clone model: most common expert action per state
        return {"status": "cloned", "states_learned": len(self._expert_actions)}
    
    def predict(self, input: Any) -> Dict:
        h = hashlib.md5(str(input).encode()).hexdigest()[:8]
        actions = self._expert_actions.get(h, [])
        if not actions:
            return {"action": None, "confidence": 0.0, "source": "no_data"}
        
        # Most common action with highest avg reward
        action_scores = defaultdict(list)
        for a, r in actions:
            action_scores[a].append(r)
        
        scored = {a: sum(rs)/len(rs) for a, rs in action_scores.items()}
        best = max(scored, key=scored.get)
        count = len(action_scores[best])
        
        return {"action": best, "confidence": min(1.0, count/10),
                "source": "expert_clone", "examples": count}
    
    def export_model(self, format: str = "json") -> Optional[bytes]:
        model = {"room_id": self.room_id, "preset": "imitate",
                 "cloned_states": len(self._expert_actions),
                 "actions": {k: list(set(a for a, _ in v)) for k, v in self._expert_actions.items()}}
        return json.dumps(model, indent=2).encode()
