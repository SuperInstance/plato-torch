"""
Continual Learning preset — learn forever without forgetting.
"""
import json, hashlib, math
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from room_base import RoomBase
except ImportError:
    from .room_base import RoomBase


class ContinualRoom(RoomBase):
    """Lifelong learning. EWC-inspired: protects important weights from catastrophic forgetting."""
    
    def __init__(self, room_id: str, **kwargs):
        super().__init__(room_id, preset="continual", **kwargs)
        self.ewc_lambda = kwargs.get("ewc_lambda", 0.4)
        self.memory_size = kwargs.get("memory_size", 200)
        self._knowledge = defaultdict(lambda: defaultdict(list))  # task → state → actions
        self._importance = defaultdict(float)  # state_hash → importance (Fisher-like)
        self._memory_buffer = []  # replay buffer
        self._current_task = "default"
    
    def set_task(self, task: str):
        self._current_task = task
    
    def feed(self, data: Any, **kwargs) -> Dict:
        if isinstance(data, dict):
            task = data.get("task", self._current_task)
            self._current_task = task
            return self.observe(data.get("state",""), data.get("action",""), data.get("outcome",""))
        return {"status": "invalid"}
    
    def train_step(self, batch: List[Dict]) -> Dict:
        # 1. Compute importance (how much each state matters)
        for tile in batch:
            sh = tile.get("state_hash", "")
            reward = abs(tile.get("reward", 0))
            # Importance grows with reward magnitude
            self._importance[sh] = self._importance[sh] * 0.9 + reward * 0.1
        
        # 2. Add to memory buffer (reservoir sampling)
        for tile in batch[-10:]:
            if len(self._memory_buffer) < self.memory_size:
                self._memory_buffer.append(tile)
            else:
                idx = hash(tile.get("state_hash", "")) % len(self._memory_buffer)
                if self._importance.get(tile.get("state_hash",""), 0) > self._importance.get(
                        self._memory_buffer[idx].get("state_hash",""), 0):
                    self._memory_buffer[idx] = tile
        
        # 3. Train on new data + replay buffer
        all_data = batch + random.sample(self._memory_buffer, min(len(self._memory_buffer), 20))
        
        for tile in all_data:
            sh = tile.get("state_hash", "")
            action = tile.get("action", "")
            reward = tile.get("reward", 0)
            self._knowledge[self._current_task][sh].append({"action": action, "reward": reward})
        
        return {"task": self._current_task, "tasks_learned": len(self._knowledge),
                "memory_size": len(self._memory_buffer)}
    
    def predict(self, input: Any) -> Dict:
        h = hashlib.md5(str(input).encode()).hexdigest()[:8]
        # Search across all tasks
        for task, states in self._knowledge.items():
            if h in states:
                actions = states[h]
                best = max(actions, key=lambda a: a["reward"])
                return {"action": best["action"], "task": task, "importance": round(self._importance.get(h, 0), 3)}
        return {"action": None, "task": None, "importance": 0}
    
    def evaluate_task(self, task: str) -> float:
        """Check if old task knowledge is retained."""
        states = self._knowledge.get(task, {})
        if not states: return 0.0
        retained = sum(1 for acts in states.values() if any(a["reward"] > 0 for a in acts))
        return retained / max(len(states), 1)
    
    def export_model(self, format: str = "json") -> Optional[bytes]:
        model = {"room_id": self.room_id, "preset": "continual",
                 "tasks": list(self._knowledge.keys()),
                 "importance_scores": {k: round(v, 3) for k, v in list(self._importance.items())[:50]},
                 "memory_size": len(self._memory_buffer)}
        return json.dumps(model, indent=2).encode()

import random
