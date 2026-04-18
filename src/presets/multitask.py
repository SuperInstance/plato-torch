"""Multi-Task Learning preset — one model, multiple related tasks."""
import json, hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from room_base import RoomBase
except ImportError:
    from ..room_base import RoomBase


class MultitaskRoom(RoomBase):
    """Train a shared backbone with task-specific heads."""
    
    def __init__(self, room_id: str, **kwargs):
        super().__init__(room_id, preset="multitask", **kwargs)
        self._shared = defaultdict(lambda: defaultdict(list))   # shared state patterns
        self._task_heads = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # task → state → action → score
        self._loss_weights = defaultdict(lambda: 1.0)
    
    def add_task(self, task: str, loss_weight: float = 1.0):
        self._loss_weights[task] = loss_weight
    
    def feed(self, data: Any, **kwargs) -> Dict:
        if isinstance(data, dict):
            task = data.get("task", "default")
            return self.observe(data.get("state",""), data.get("action",""), data.get("outcome",""),
                              context={"task": task})
        return {"status": "invalid"}
    
    def train_step(self, batch: List[Dict]) -> Dict:
        for tile in batch:
            task = tile.get("context", {}).get("task", "default")
            sh = tile.get("state_hash", "")
            action = tile.get("action", "")
            reward = tile.get("reward", 0)
            
            # Shared backbone learns cross-task patterns
            self._shared[sh][action].append(reward)
            # Task-specific head
            self._task_heads[task][sh][action] = self._task_heads[task][sh].get(action, 0) + reward
        
        return {"tasks": len(self._task_heads), "shared_states": len(self._shared)}
    
    def predict(self, input: Any, task: str = None) -> Dict:
        h = hashlib.md5(str(input).encode()).hexdigest()[:8]
        
        # Shared prediction (cross-task)
        shared_actions = self._shared.get(h, {})
        shared_best = max(shared_actions, key=lambda a: sum(shared_actions[a])/len(shared_actions[a])) if shared_actions else None
        
        # Task-specific prediction
        task_result = None
        if task and task in self._task_heads:
            task_actions = self._task_heads[task].get(h, {})
            if task_actions:
                task_result = max(task_actions, key=task_actions.get)
        
        return {
            "shared_best": shared_best,
            "task_specific": task_result,
            "task": task,
            "final": task_result or shared_best,
        }
    
    def export_model(self, format: str = "json") -> Optional[bytes]:
        model = {"room_id": self.room_id, "preset": "multitask",
                 "tasks": list(self._task_heads.keys()),
                 "loss_weights": dict(self._loss_weights),
                 "shared_states": len(self._shared)}
        return json.dumps(model, indent=2).encode()
