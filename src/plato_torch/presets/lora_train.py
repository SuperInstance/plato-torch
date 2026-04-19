"""LoRARoom – parameter-efficient fine-tuning simulation with statistical fallback."""

import json, math, hashlib
from collections import defaultdict
try:
    from room_base import RoomBase
except ImportError:
    from .room_base import RoomBase


class LoRARoom(RoomBase):
    """Simulates LoRA-style adapter training over a base knowledge table."""

    def __init__(self, room_id: str = "lora", rank: int = 8, alpha: float = 1.0, **kwargs):
        super().__init__(room_id=room_id)
        self.rank = rank
        self.alpha = alpha
        self.base_knowledge: dict[str, list[float]] = {}
        self.delta_table: dict[str, dict[str, float]] = {}   # key -> {param: value}
        self.examples: list[tuple[str, str]] = []
        self._step = 0

    # ── data ingestion ────────────────────────────────────────────────

    def feed(self, data=None, instruction: str = None, response: str = None):
        if data is not None and instruction is None:
            if isinstance(data, dict):
                instruction = data.get('instruction', data.get('input', 'test'))
                response = data.get('response', data.get('output', 'test'))
            else:
                instruction = str(data)
                response = str(data)
        if instruction is None: instruction = 'test'
        if response is None: response = 'test'
        """Accept an instruction-response pair for adapter training."""
        instruction = str(instruction)
        response = str(response)
        self.examples.append((instruction, response))
        # tokenise via simple hash windows → pseudo-embedding
        instruction = str(instruction) if not isinstance(instruction, str) else instruction
        key = self._hash(instruction)
        vec = self._embed(response)
        self.base_knowledge[key] = vec

    # ── training step ─────────────────────────────────────────────────

    def train_step(self) -> dict:
        """Build delta table (low-rank adaptations on top of base knowledge)."""
        self._step += 1
        lr = self.alpha / math.sqrt(self._step)
        for key, base_vec in self.base_knowledge.items():
            # compute a low-rank delta: top-rank dimensions scaled by lr
            ranked = sorted(enumerate(base_vec), key=lambda iv: -abs(iv[1]))[: self.rank]
            if key not in self.delta_table:
                self.delta_table[key] = defaultdict(float)
            for idx, val in ranked:
                dim_name = f"d{idx}"
                self.delta_table[key][dim_name] += lr * val
        return {"step": self._step, "keys": len(self.delta_table), "rank": self.rank}

    # ── inference ─────────────────────────────────────────────────────

    def predict(self, instruction=None) -> list[float]:
        if instruction is None: instruction = "test"
        if not isinstance(instruction, str): instruction = str(instruction)
        """Combine base knowledge with learned deltas for a prediction."""
        key = self._hash(instruction)
        base = self.base_knowledge.get(key, self._embed(instruction))
        merged = list(base)
        deltas = self.delta_table.get(key, {})
        for dim_name, delta_val in deltas.items():
            idx = int(dim_name[1:])
            if idx < len(merged):
                merged[idx] += delta_val
        return merged

    # ── export ────────────────────────────────────────────────────────

    def export_model(self) -> str:
        """Return JSON string of the adapter weights (delta table)."""
        serialisable = {k: dict(v) for k, v in self.delta_table.items()}
        payload = {
            "adapter_name": self.room_id,
            "rank": self.rank,
            "alpha": self.alpha,
            "train_step": self._step,
            "examples_seen": len(self.examples),
            "weights": serialisable,
        }
        return json.dumps(payload, indent=2)

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:12]

    @staticmethod
    def _embed(text: str, dim: int = 64) -> list[float]:
        """Cheap statistical embedding from character frequencies."""
        vec = [0.0] * dim
        for i, ch in enumerate(text):
            vec[i % dim] += ord(ch) / 128.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
