"""
RoomBase — abstract base class for all training room presets.

Every training method (supervised, reinforce, evolve, etc.) inherits from this.
Same interface, different training paradigm underneath.
"""

import json
import os
import time
import hashlib
import random
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict


class RoomBase(ABC):
    """Abstract base for self-training PLATO rooms.
    
    Subclasses override:
    - feed(): how data comes in
    - train_step(): one training iteration
    - predict(): inference
    - export_model(): serialize trained artifact
    """
    
    def __init__(self, room_id: str, preset: str = "general",
                 train_threshold: int = 500,
                 ensign_dir: str = "ensigns",
                 buffer_dir: str = "tile_buffers",
                 **kwargs):
        self.room_id = room_id
        self.preset = preset
        self.train_threshold = train_threshold
        
        self.ensign_dir = Path(ensign_dir) / room_id
        self.ensign_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_dir = Path(buffer_dir) / room_id
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        
        # Sentiment (lazy import to avoid circular deps)
        self._sentiment = None
        self._biased_random = None
        
        # Stats
        self._episodes_seen = 0
        self._last_train_time = None
        self._train_count = 0
        
        self._load_state()
    
    # ── Abstract interface (every preset must implement) ──
    
    @abstractmethod
    def feed(self, data: Any, **kwargs) -> Dict:
        """Feed data into the room. Format depends on preset."""
        pass
    
    @abstractmethod
    def train_step(self, batch: List[Dict]) -> Dict:
        """One training step on a batch of data."""
        pass
    
    @abstractmethod
    def predict(self, input: Any) -> Any:
        """Inference using the trained model."""
        pass
    
    @abstractmethod
    def export_model(self, format: str = "json") -> Optional[bytes]:
        """Export the trained model in the specified format."""
        pass
    
    # ── Common interface (shared across all presets) ──
    
    def observe(self, state: str, action: str, outcome: str,
                agent_id: str = "unknown", context: Dict = None,
                reward: float = None) -> Dict:
        """Record an interaction. Works for all presets."""
        if reward is None:
            reward = self._infer_reward(state, action, outcome)
        
        tile = {
            "room_id": self.room_id,
            "preset": self.preset,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "state": state,
            "action": action,
            "outcome": outcome,
            "agent_id": agent_id,
            "context": context or {},
            "reward": reward,
            "state_hash": hashlib.md5(state.encode()).hexdigest()[:8],
        }
        
        self._write_tile(tile)
        self._episodes_seen += 1
        self.maybe_train()
        return tile
    
    def maybe_train(self):
        """Auto-train when threshold hit."""
        if self._count_tiles() >= self.train_threshold and self._should_train():
            self.train()
    
    def train(self) -> Dict:
        """Full training run on accumulated data."""
        tiles = self._load_tiles()
        if not tiles:
            return {"status": "no_data"}
        
        result = self.train_step(tiles)
        
        self._last_train_time = time.time()
        self._train_count += 1
        self._save_state()
        
        result["train_count"] = self._train_count
        return result
    
    def simulate(self, episodes: int = 100) -> Dict:
        """Run self-play simulations (override in subclasses for domain-specific)."""
        tiles_generated = 0
        for ep in range(episodes):
            state = self._generate_synthetic_state(ep)
            action = self._pick_simulated_action(state)
            outcome, reward = self._simulate_outcome(state, action)
            self.observe(state, action, outcome, f"sim-{self.preset}", reward=reward)
            tiles_generated += 1
        return {"episodes": episodes, "tiles_generated": tiles_generated}
    
    def wisdom(self) -> Dict:
        """Room's accumulated knowledge and stats."""
        return {
            "room_id": self.room_id,
            "preset": self.preset,
            "episodes_seen": self._episodes_seen,
            "train_count": self._train_count,
            "tile_count": self._count_tiles(),
            "last_trained": self._last_train_time,
        }
    
    @property
    def sentiment(self):
        """Lazy-load sentiment tracker."""
        if self._sentiment is None:
            from room_sentiment import RoomSentiment, BiasedRandomness
            self._sentiment = RoomSentiment()
            self._biased_random = BiasedRandomness(self._sentiment)
        return self._sentiment
    
    # ── Helpers ──
    
    def _infer_reward(self, state, action, outcome) -> float:
        o = outcome.lower()
        for w in ["won", "success", "good", "pass", "approved", "saved", "correct"]:
            if w in o: return 1.0
        for w in ["lost", "fail", "bad", "error", "rejected", "wrong"]:
            if w in o: return -1.0
        return 0.0
    
    def _should_train(self) -> bool:
        if self._last_train_time is None: return True
        return (time.time() - self._last_train_time) > 300
    
    def _write_tile(self, tile: Dict):
        ts = int(time.time() * 1000)
        path = self.buffer_dir / f"tile_{ts}_{random.randint(0,9999):04d}.json"
        with open(path, "w") as f:
            json.dump(tile, f, indent=2)
    
    def _count_tiles(self) -> int:
        return len(list(self.buffer_dir.glob("tile_*.json")))
    
    def _load_tiles(self) -> List[Dict]:
        tiles = []
        for f in sorted(self.buffer_dir.glob("tile_*.json")):
            with open(f) as fh:
                tiles.append(json.load(fh))
        return tiles
    
    def _generate_synthetic_state(self, ep: int) -> str:
        return f"sim-state-{ep}"
    
    def _pick_simulated_action(self, state: str) -> str:
        return random.choice(["act_a", "act_b", "act_c"])
    
    def _simulate_outcome(self, state: str, action: str) -> Tuple[str, float]:
        return ("neutral", random.uniform(-1, 1))
    
    def _state_file(self) -> Path:
        return self.buffer_dir / "_room_state.json"
    
    def _save_state(self):
        with open(self._state_file(), "w") as f:
            json.dump({
                "episodes_seen": self._episodes_seen,
                "train_count": self._train_count,
                "last_train_time": self._last_train_time,
            }, f)
    
    def _load_state(self):
        sf = self._state_file()
        if sf.exists():
            with open(sf) as f:
                s = json.load(f)
                self._episodes_seen = max(s.get("episodes_seen", 0), self._count_tiles())
                self._train_count = s.get("train_count", 0)
                self._last_train_time = s.get("last_train_time")
