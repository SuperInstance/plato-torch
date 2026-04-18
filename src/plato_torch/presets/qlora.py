"""QLoRA preset — 4-bit quantized LoRA fine-tuning simulation."""
import json, hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from ..room_base import RoomBase


class QLoRARoom(RoomBase):
    """Simulates QLoRA: 4-bit quantized base + LoRA adapters. Statistical fallback."""
    
    def __init__(self, room_id: str, **kwargs):
        super().__init__(room_id, preset="qlora", **kwargs)
        self.quantization = kwargs.get("quantization", "4bit")
        self.rank = kwargs.get("rank", 16)
        # Quantized base: compressed state→action mapping
        self._quant_base = defaultdict(lambda: defaultdict(int))
        # LoRA deltas: small adaptations on top of quant base
        self._lora_deltas = defaultdict(lambda: defaultdict(float))
        self._precision_loss = 0.0
    
    def _quantize(self, value: float, bits: int = 4) -> float:
        """Simulate quantization: reduce precision of value."""
        levels = 2 ** bits
        return round(value * levels) / levels
    
    def feed(self, data: Any, **kwargs) -> Dict:
        if isinstance(data, dict):
            return self.observe(data.get("state",""), data.get("action",""), data.get("outcome",""))
        return {"status": "invalid"}
    
    def train_step(self, batch: List[Dict]) -> Dict:
        for tile in batch:
            sh = tile.get("state_hash", "")
            action = tile.get("action", "")
            reward = tile.get("reward", 0)
            
            # Quantized base update
            self._quant_base[sh][action] += 1
            
            # LoRA delta update (small, low-rank adjustment)
            current_delta = self._lora_deltas[sh].get(action, 0.0)
            self._lora_deltas[sh][action] = current_delta + 0.01 * (reward - current_delta)
            self._precision_loss += abs(reward - self._quantize(reward))
        
        return {
            "status": "trained",
            "quant_states": len(self._quant_base),
            "lora_states": len(self._lora_deltas),
            "avg_precision_loss": round(self._precision_loss / max(len(batch), 1), 4),
        }
    
    def predict(self, input: Any) -> Dict:
        h = hashlib.md5(str(input).encode()).hexdigest()[:8]
        
        # Base prediction (quantized frequency)
        base_actions = self._quant_base.get(h, {})
        base_total = sum(base_actions.values())
        
        # LoRA-enhanced prediction
        lora_deltas = self._lora_deltas.get(h, {})
        
        results = {}
        for action in set(list(base_actions.keys()) + list(lora_deltas.keys())):
            base_freq = base_actions.get(action, 0) / max(base_total, 1)
            delta = lora_deltas.get(action, 0.0)
            # Combine: quantized base + LoRA delta
            combined = self._quantize(base_freq + delta)
            results[action] = round(combined, 4)
        
        best = max(results, key=results.get) if results else None
        return {"action": best, "combined_scores": results,
                "quantization": self.quantization, "rank": self.rank}
    
    def export_model(self, format: str = "json") -> Optional[bytes]:
        model = {
            "room_id": self.room_id, "preset": "qlora",
            "quantization": self.quantization, "rank": self.rank,
            "base_states": len(self._quant_base),
            "delta_states": len(self._lora_deltas),
            "precision_loss": round(self._precision_loss, 4),
        }
        return json.dumps(model, indent=2).encode()
