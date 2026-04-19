"""
Curriculum Learning preset — easy first, then harder.
"""
import json, hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from room_base import RoomBase
except ImportError:
    from .room_base import RoomBase


class CurriculumRoom(RoomBase):
    """Curriculum learning room. Trains on easy examples first, progressively harder."""
    
    def __init__(self, room_id: str, **kwargs):
        super().__init__(room_id, preset="curriculum", **kwargs)
        self.stages = kwargs.get("stages", [
            {"difficulty": "easy", "threshold": 0.9},
            {"difficulty": "medium", "threshold": 0.8},
            {"difficulty": "hard", "threshold": 0.7},
        ])
        self.current_stage = 0
        self._stage_data = defaultdict(list)  # difficulty → tiles
        self._stage_accuracy = defaultdict(float)
        self._knowledge = defaultdict(lambda: defaultdict(list))
    
    def _classify_difficulty(self, tile: Dict) -> str:
        """Classify a tile's difficulty based on reward variance."""
        r = abs(tile.get("reward", 0))
        if r > 0.8: return "easy"
        elif r > 0.3: return "medium"
        else: return "hard"
    
    def feed(self, data=None, **kwargs) -> Dict:
        if data is None: data = {}
        if isinstance(data, str): data = {"data": data}
        if isinstance(data, dict):
            return self.observe(data.get("state",""), data.get("action",""), data.get("outcome",""))
        return {"status": "invalid"}
    
    def train_step(self, batch=None) -> Dict:
        if batch is None:
            return {"status": "ok", "message": "no batch", "preset": "curriculum"}
        # Sort tiles by difficulty
        for tile in batch:
            diff = self._classify_difficulty(tile)
            self._stage_data[diff].append(tile)
            sh = tile.get("state_hash", "")
            action = tile.get("action", "")
            reward = tile.get("reward", 0)
            self._knowledge[diff][sh].append({"action": action, "reward": reward})
        
        # Check if current stage threshold met
        if self.current_stage < len(self.stages):
            stage = self.stages[self.current_stage]
            diff = stage["difficulty"]
            tiles = self._knowledge.get(diff, {})
            if tiles:
                correct = sum(1 for t_list in tiles.values() for t in t_list if t["reward"] > 0)
                total = sum(len(t_list) for t_list in tiles.values())
                accuracy = correct / max(total, 1)
                self._stage_accuracy[diff] = accuracy
                if accuracy >= stage["threshold"] and self.current_stage < len(self.stages) - 1:
                    self.current_stage += 1
        
        return {"stage": self.current_stage, "stages_total": len(self.stages),
                "stage_name": self.stages[self.current_stage]["difficulty"] if self.current_stage < len(self.stages) else "mastered"}
    
    def predict(self, input: Any) -> Dict:
        h = hashlib.md5(str(input).encode()).hexdigest()[:8]
        # Search current stage first, then easier stages
        for i in range(self.current_stage, -1, -1):
            diff = self.stages[i]["difficulty"] if i < len(self.stages) else "easy"
            if h in self._knowledge.get(diff, {}):
                actions = self._knowledge[diff][h]
                best = max(actions, key=lambda a: a["reward"])
                return {"action": best["action"], "stage": diff}
        return {"action": None, "stage": "unknown"}
    
    def export_model(self, format: str = "json") -> Optional[bytes]:
        model = {"room_id": self.room_id, "preset": "curriculum",
                 "current_stage": self.current_stage, "stages": self.stages,
                 "accuracy": dict(self._stage_accuracy),
                 "knowledge_size": {k: len(v) for k, v in self._knowledge.items()}}
        return json.dumps(model, indent=2).encode()
