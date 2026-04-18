"""
Distill preset — knowledge distillation room.

Compress a big teacher model into a tiny student model.
Teacher generates soft labels → student learns from them.
Export as GGUF for CPU-only agents (greenhorns).
"""

import json
import random
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from room_base import RoomBase
except ImportError:
    from ..room_base import RoomBase


class DistillRoom(RoomBase):
    """Knowledge distillation training room.
    
    Teacher (big model or accumulated wisdom) generates soft labels.
    Student (tiny model) learns to mimic the teacher's behavior.
    
    Works without PyTorch (statistical distillation) and with PyTorch
    (neural teacher-student training).
    """
    
    def __init__(self, room_id: str, **kwargs):
        super().__init__(room_id, preset="distill", **kwargs)
        
        self.temperature = kwargs.get("temperature", 4.0)
        self.alpha = kwargs.get("alpha", 0.7)  # weight for distillation loss vs hard labels
        
        # Teacher knowledge: state_hash → {action → probability}
        self._teacher_logits = defaultdict(lambda: defaultdict(float))
        # Student knowledge: state_hash → {action → learned probability}
        self._student_logits = defaultdict(lambda: defaultdict(float))
        # Hard labels from real outcomes
        self._hard_labels = defaultdict(lambda: defaultdict(int))
        
        # Teacher model reference (API or local)
        self._teacher_model = kwargs.get("teacher_model", None)
        self._student_model = kwargs.get("student_model", "tiny-ensign")
    
    def feed(self, data: Any, **kwargs) -> Dict:
        """Feed teacher's knowledge or student's predictions."""
        if isinstance(data, dict):
            if data.get("type") == "teacher":
                return self._feed_teacher(data)
            elif data.get("type") == "student":
                return self._feed_student(data)
            else:
                # Default: treat as interaction
                return self.observe(
                    data.get("state", ""),
                    data.get("action", ""),
                    data.get("outcome", ""),
                    reward=data.get("reward")
                )
        return {"status": "invalid_data"}
    
    def _feed_teacher(self, data: Dict) -> Dict:
        """Record teacher's probability distribution for a state."""
        state = data.get("state", "")
        sh = hashlib.md5(state.encode()).hexdigest()[:8]
        logits = data.get("logits", {})
        
        for action, prob in logits.items():
            self._teacher_logits[sh][action] = prob
        
        return {"state_hash": sh, "teacher_actions": len(logits)}
    
    def _feed_student(self, data: Dict) -> Dict:
        """Record student's prediction for a state."""
        state = data.get("state", "")
        sh = hashlib.md5(state.encode()).hexdigest()[:8]
        logits = data.get("logits", {})
        
        for action, prob in logits.items():
            self._student_logits[sh][action] = prob
        
        return {"state_hash": sh, "student_actions": len(logits)}
    
    def train_step(self, batch: List[Dict]) -> Dict:
        """Distill: align student logits with teacher logits + hard labels."""
        updates = 0
        distill_loss = 0
        
        for tile in batch:
            sh = tile.get("state_hash", "")
            action = tile.get("action", "")
            reward = tile.get("reward", 0)
            
            # Build teacher logits from accumulated data
            if sh not in self._teacher_logits:
                # Infer teacher distribution from reward signals
                self._infer_teacher_logits(sh, batch)
            
            # Build hard labels
            if reward > 0:
                self._hard_labels[sh][action] += 1
            elif reward < 0:
                self._hard_labels[sh][action] -= 1
            
            # Distill: student learns from teacher's soft distribution
            teacher_probs = self._teacher_logits.get(sh, {})
            if teacher_probs:
                # KL divergence direction: move student toward teacher
                for act, t_prob in teacher_probs.items():
                    s_prob = self._student_logits[sh].get(act, 0.5)
                    # Gradient: student should match teacher
                    new_prob = s_prob + 0.01 * (t_prob - s_prob)
                    self._student_logits[sh][act] = max(0, min(1, new_prob))
                
                updates += 1
                # Simplified distillation loss
                for act, t_prob in teacher_probs.items():
                    s_prob = self._student_logits[sh].get(act, 0.5)
                    if s_prob > 0:
                        distill_loss += t_prob * (math.log(t_prob + 1e-8) - math.log(s_prob + 1e-8))
        
        self._save_model()
        
        avg_loss = distill_loss / max(updates, 1)
        
        return {
            "status": "distilled",
            "updates": updates,
            "avg_distill_loss": round(avg_loss, 4),
            "tiles": len(batch),
        }
    
    def predict(self, input: Any) -> Dict:
        """Predict using student model (the distilled version)."""
        state = str(input)
        sh = hashlib.md5(state.encode()).hexdigest()[:8]
        
        student_probs = dict(self._student_logits.get(sh, {}))
        teacher_probs = dict(self._teacher_logits.get(sh, {}))
        
        # Softmax over student logits
        if student_probs:
            total = sum(student_probs.values())
            if total > 0:
                student_probs = {a: round(p/total, 3) for a, p in student_probs.items()}
        
        best_action = None
        if student_probs:
            best_action = max(student_probs, key=student_probs.get)
        
        return {
            "state_hash": sh,
            "student_probs": student_probs,
            "teacher_probs": teacher_probs,
            "best_action": best_action,
            "confidence": min(1.0, len(student_probs) / 5),
        }
    
    def export_model(self, format: str = "json") -> Optional[bytes]:
        """Export student model (the distilled tiny model)."""
        model = {
            "room_id": self.room_id,
            "preset": "distill",
            "student_model": self._student_model,
            "temperature": self.temperature,
            "student_logits": {k: dict(v) for k, v in self._student_logits.items()},
            "teacher_logits": {k: dict(v) for k, v in self._teacher_logits.items()},
            "hard_labels": {k: dict(v) for k, v in self._hard_labels.items()},
            "metadata": {
                "distillable_states": len(self._student_logits),
                "teacher_states": len(self._teacher_logits),
            }
        }
        
        if format == "gguf":
            # GGUF export would go here — for now, return JSON
            model["export_note"] = "GGUF conversion requires llama.py — export as json for now"
        
        return json.dumps(model, indent=2).encode()
    
    def _infer_teacher_logits(self, state_hash: str, batch: List[Dict]):
        """Infer teacher distribution from reward signals."""
        action_rewards = defaultdict(list)
        for tile in batch:
            if tile.get("state_hash") == state_hash:
                action_rewards[tile["action"]].append(tile["reward"])
        
        # Temperature-scaled soft distribution
        logits = {}
        for action, rewards in action_rewards.items():
            avg_r = sum(rewards) / len(rewards) if rewards else 0
            logits[action] = self._softmax_value(avg_r, self.temperature)
        
        # Normalize
        total = sum(logits.values())
        if total > 0:
            logits = {a: v/total for a, v in logits.items()}
        
        self._teacher_logits[state_hash] = defaultdict(float, logits)
    
    def _softmax_value(self, value: float, temperature: float) -> float:
        """Temperature-scaled softmax for a single value."""
        import math
        return math.exp(value / max(temperature, 0.1))
    
    def _save_model(self):
        path = self.ensign_dir / "distill_model.json"
        data = self.export_model("json")
        if data:
            path.write_bytes(data)


# Need math import at top
import math
