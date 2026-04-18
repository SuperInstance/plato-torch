"""
Few-Shot Learning preset — adapt from 1-5 examples.
"""
import json, hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from ..room_base import RoomBase


class FewshotRoom(RoomBase):
    """Few-shot/zero-shot: adapt to new tasks from minimal examples using nearest-prototype."""
    
    def __init__(self, room_id: str, **kwargs):
        super().__init__(room_id, preset="fewshot", **kwargs)
        self.n_shots = kwargs.get("n_shots", 5)
        self._prototypes = defaultdict(lambda: defaultdict(list))  # task → label → [features]
        self._all_tasks = set()
    
    def feed(self, data: Any, **kwargs) -> Dict:
        if isinstance(data, dict):
            task = data.get("task", "default")
            self._all_tasks.add(task)
            return self.observe(data.get("state",""), data.get("action",""), data.get("outcome",""), context={"task": task})
        return {"status": "invalid"}
    
    def adapt_from_examples(self, examples: List[Dict], task: str = "new") -> Dict:
        """Few-shot adaptation: learn from N examples."""
        self._all_tasks.add(task)
        for ex in examples:
            state = ex.get("state", ex.get("input", ""))
            label = ex.get("action", ex.get("label", ""))
            sh = hashlib.md5(state.encode()).hexdigest()[:8]
            self._prototypes[task][label].append(sh)
        return {"task": task, "examples": len(examples), "labels": list(self._prototypes[task].keys())}
    
    def train_step(self, batch: List[Dict]) -> Dict:
        for tile in batch:
            task = tile.get("context", {}).get("task", "default")
            sh = tile.get("state_hash", "")
            label = tile.get("action", "")
            self._prototypes[task][label].append(sh)
        return {"status": "trained", "tasks": len(self._prototypes)}
    
    def predict(self, input: Any) -> Dict:
        h = hashlib.md5(str(input).encode()).hexdigest()[:8]
        # Find closest prototype across all tasks
        best_task = None
        best_label = None
        best_score = -1
        
        for task, labels in self._prototypes.items():
            for label, prototypes in labels.items():
                # Score: how many prototype hashes match (exact) or are nearby
                if h in prototypes:
                    score = prototypes.count(h) / len(prototypes)
                    if score > best_score:
                        best_score = score
                        best_task = task
                        best_label = label
        
        if best_label:
            return {"label": best_label, "task": best_task, "confidence": round(best_score, 3),
                    "shots": len(self._prototypes.get(best_task, {}).get(best_label, []))}
        
        # Zero-shot: no exact match, return most common label across all tasks
        all_labels = defaultdict(int)
        for task_labels in self._prototypes.values():
            for label, protos in task_labels.items():
                all_labels[label] += len(protos)
        if all_labels:
            guess = max(all_labels, key=all_labels.get)
            return {"label": guess, "task": "zero_shot", "confidence": 0.0}
        
        return {"label": None, "task": None, "confidence": 0.0}
    
    def export_model(self, format: str = "json") -> Optional[bytes]:
        model = {"room_id": self.room_id, "preset": "fewshot",
                 "tasks": list(self._prototypes.keys()),
                 "prototypes": {t: {l: len(p) for l, p in ls.items()} for t, ls in self._prototypes.items()}}
        return json.dumps(model, indent=2).encode()
