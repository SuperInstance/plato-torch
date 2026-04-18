"""
Neurosymbolic preset — neural instinct + symbolic rules.
"""
import json, hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from room_base import RoomBase
except ImportError:
    from ..room_base import RoomBase


class NeurosymbolicRoom(RoomBase):
    """Hybrid: neural pattern recognition + hard symbolic rules."""
    
    def __init__(self, room_id: str, **kwargs):
        super().__init__(room_id, preset="neurosymbolic", **kwargs)
        self.neural_weight = kwargs.get("neural_weight", 0.6)
        self.rule_weight = kwargs.get("rule_weight", 0.4)
        self.rules = kwargs.get("rules", [])
        self._neural = defaultdict(lambda: defaultdict(list))  # state → action → rewards
    
    def add_rule(self, rule: str):
        """Add a symbolic rule (e.g. 'IF x > 0.7 THEN raise')."""
        self.rules.append(rule)
    
    def feed(self, data: Any, **kwargs) -> Dict:
        if isinstance(data, dict):
            return self.observe(data.get("state",""), data.get("action",""), data.get("outcome",""))
        return {"status": "invalid"}
    
    def train_step(self, batch: List[Dict]) -> Dict:
        for tile in batch:
            sh = tile.get("state_hash", "")
            action = tile.get("action", "")
            reward = tile.get("reward", 0)
            self._neural[sh][action].append(reward)
        return {"status": "trained", "neural_states": len(self._neural), "rules": len(self.rules)}
    
    def _evaluate_rules(self, state: str) -> Optional[str]:
        """Evaluate symbolic rules against state. Return forced action or None."""
        for rule in self.rules:
            # Simple rule evaluation: "IF condition THEN action"
            if "THEN" in rule.upper():
                parts = rule.upper().split("THEN")
                condition = parts[0].replace("IF", "").strip()
                action = parts[1].strip().lower() if len(parts) > 1 else None
                # Simplistic: if any keyword from condition appears in state
                keywords = [w for w in condition.split() if len(w) > 2]
                if any(kw.lower() in state.lower() for kw in keywords):
                    return action
        return None
    
    def _evaluate_neural(self, state_hash: str) -> Optional[str]:
        """Neural instinct: best action from accumulated pattern data."""
        actions = self._neural.get(state_hash, {})
        if not actions: return None
        scored = {a: sum(rs)/len(rs) for a, rs in actions.items()}
        return max(scored, key=scored.get)
    
    def predict(self, input: Any) -> Dict:
        state = str(input)
        sh = hashlib.md5(state.encode()).hexdigest()[:8]
        
        neural_action = self._evaluate_neural(sh)
        rule_action = self._evaluate_rules(state)
        
        # Blend: rules override if mandatory, otherwise weighted blend
        if rule_action and neural_action:
            final = rule_action if self.rule_weight > self.neural_weight else neural_action
        elif rule_action:
            final = rule_action
        elif neural_action:
            final = neural_action
        else:
            final = None
        
        return {
            "neural_vote": neural_action,
            "rule_vote": rule_action,
            "final": final,
            "neural_weight": self.neural_weight,
            "rule_weight": self.rule_weight,
        }
    
    def export_model(self, format: str = "json") -> Optional[bytes]:
        model = {"room_id": self.room_id, "preset": "neurosymbolic",
                 "rules": self.rules, "neural_states": len(self._neural),
                 "weights": {"neural": self.neural_weight, "rule": self.rule_weight}}
        return json.dumps(model, indent=2).encode()
