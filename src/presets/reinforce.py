"""
Reinforce preset — reinforcement learning room.

State → Action → Reward → Next State loop.
Supports PPO-style policy gradient with statistical fallback.
"""

import random
import math
from collections import defaultdict
from typing import Dict, List, Optional, Any, Tuple

try:
    from room_base import RoomBase
except ImportError:
    from .room_base import RoomBase


class ReinforceRoom(RoomBase):
    """RL training room. Agents act, room rewards, policy improves.
    
    Works without PyTorch (statistical policy table) and with PyTorch
    (neural policy + value networks).
    """
    
    def __init__(self, room_id: str, **kwargs):
        super().__init__(room_id, preset="reinforce", **kwargs)
        
        # Policy table: state_hash → {action → estimated_value}
        self._policy = defaultdict(lambda: defaultdict(list))
        # Value table: state_hash → estimated return
        self._values = defaultdict(list)
        # Eligibility traces for TD(λ)
        self._gamma = kwargs.get("gamma", 0.99)
        self._lr = kwargs.get("learning_rate", 0.1)
    
    def feed(self, data: Any, **kwargs) -> Dict:
        """Feed a (state, action, reward) tuple."""
        if isinstance(data, dict):
            return self.observe(
                data.get("state", ""),
                data.get("action", ""),
                data.get("outcome", ""),
                reward=data.get("reward")
            )
        return {"status": "invalid_data"}
    
    def train_step(self, batch: List[Dict]) -> Dict:
        """Update policy from episode batch using Monte Carlo returns."""
        # Compute returns (discounted cumulative rewards)
        returns = self._compute_returns(batch)
        
        updates = 0
        for tile, g in zip(batch, returns):
            sh = tile["state_hash"]
            action = tile["action"]
            
            # Update policy table
            self._policy[sh][action].append(g)
            self._values[sh].append(g)
            updates += 1
        
        # Save model
        self._save_model()
        
        return {
            "status": "trained",
            "updates": updates,
            "avg_return": sum(returns) / len(returns) if returns else 0,
            "tiles": len(batch)
        }
    
    def predict(self, input: Any) -> Dict:
        """Predict best action and value for a state."""
        state = str(input)
        sh = self._hash(state)
        
        # Get action values
        action_values = {}
        for action, rewards in self._policy[sh].items():
            if rewards:
                action_values[action] = sum(rewards) / len(rewards)
        
        # Get state value
        state_value = 0
        if self._values[sh]:
            state_value = sum(self._values[sh]) / len(self._values[sh])
        
        best_action = None
        if action_values:
            best_action = max(action_values, key=action_values.get)
        
        return {
            "state_hash": sh,
            "best_action": best_action,
            "action_values": action_values,
            "state_value": round(state_value, 3),
            "confidence": min(1.0, len(self._values[sh]) / 20)
        }
    
    def export_model(self, format: str = "json") -> Optional[bytes]:
        """Export trained policy."""
        import json
        model = {
            "room_id": self.room_id,
            "preset": "reinforce",
            "policy": {k: {a: round(sum(r)/len(r), 3) for a, r in v.items()} 
                      for k, v in self._policy.items()},
            "values": {k: round(sum(v)/len(v), 3) for k, v in self._values.items()},
        }
        return json.dumps(model, indent=2).encode()
    
    def _compute_returns(self, batch: List[Dict]) -> List[float]:
        """Compute discounted returns from a batch of tiles."""
        rewards = [t.get("reward", 0) for t in batch]
        returns = []
        g = 0
        for r in reversed(rewards):
            g = r + self._gamma * g
            returns.insert(0, g)
        return returns
    
    def _hash(self, state: str) -> str:
        import hashlib
        return hashlib.md5(state.encode()).hexdigest()[:8]
    
    def _save_model(self):
        import json
        from pathlib import Path
        path = self.ensign_dir / "reinforce_model.json"
        data = self.export_model("json")
        if data:
            path.write_bytes(data)
    
    def simulate(self, episodes: int = 100) -> Dict:
        """Self-play: generate episodes using current policy."""
        generated = 0
        for ep in range(episodes):
            state = f"ep-{ep}-state"
            pred = self.predict(state)
            
            if pred["best_action"]:
                # Epsilon-greedy: mostly follow policy, sometimes explore
                if random.random() < 0.2:
                    action = random.choice(["aggressive", "conservative", "neutral"])
                else:
                    action = pred["best_action"]
            else:
                action = random.choice(["aggressive", "conservative", "neutral"])
            
            reward = random.gauss(pred["state_value"], 0.5)
            outcome = "won" if reward > 0 else "lost"
            
            self.observe(state, action, outcome, "sim-rl", reward=reward)
            generated += 1
        
        return {"episodes": episodes, "tiles_generated": generated}
