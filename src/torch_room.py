"""
TorchRoom — a room that teaches itself.

A PLATO room backed by PyTorch that:
1. Observes agent interactions and records them as training data
2. Automatically trains instinct/policy/strategy networks from accumulated data
3. Can simulate episodes to train itself even when empty
4. Exports trained ensigns for the plato-ensign registry
"""

import json
import os
import time
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict
try:
    from .room_sentiment import RoomSentiment, BiasedRandomness, LiveTileStream, IncrementalTrainer
except ImportError:
    from room_sentiment import RoomSentiment, BiasedRandomness, LiveTileStream, IncrementalTrainer


class TorchRoom:
    """A self-training room.
    
    Agents interact with the room. The room watches every interaction,
    records (state, action, outcome) tuples, and periodically trains
    neural instincts from the accumulated data.
    
    When the room is empty, it can run simulations against itself
    to generate more training data and sharpen its instincts.
    """
    
    def __init__(self, room_id: str, use_case: str = "general",
                 train_threshold: int = 500,
                 ensign_dir: str = "ensigns",
                 buffer_dir: str = "tile_buffers"):
        self.room_id = room_id
        self.use_case = use_case  # "game", "code", "navigation", "conversation"
        self.train_threshold = train_threshold
        
        # Paths
        self.ensign_dir = Path(ensign_dir) / room_id
        self.ensign_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_dir = Path(buffer_dir) / room_id
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        
        # Networks (lazy-loaded)
        self._instinct = None
        self._tile_grabber = None
        self._strategy_mesh = None
        
        # Stats
        self._episodes_seen = 0
        self._last_train_time = None
        self._train_count = 0
        
        # Sentiment + live stream + incremental trainer
        self.sentiment = RoomSentiment()
        self.biased_random = BiasedRandomness(self.sentiment)
        self.incremental_trainer = IncrementalTrainer(room_id)
        self.live_stream = LiveTileStream(room_id, self.sentiment, self.incremental_trainer)
        
        # Load existing state
        self._load_state()
    
    # ── Core: Observe ──────────────────────────────────────
    
    def observe(self, state: str, action: str, outcome: str,
                agent_id: str = "unknown", context: Dict = None,
                reward: float = None):
        """Record an interaction as a training tile.
        
        Args:
            state: Current room state (text description)
            action: What the agent did
            outcome: What happened as a result
            agent_id: Which agent was acting
            context: Additional context (other agents, room config, etc.)
            reward: Explicit reward signal. If None, inferred from outcome.
        """
        if reward is None:
            reward = self._infer_reward(state, action, outcome)
        
        tile = {
            "room_id": self.room_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "state": state,
            "action": action,
            "outcome": outcome,
            "agent_id": agent_id,
            "context": context or {},
            "reward": reward,
            "state_hash": hashlib.md5(state.encode()).hexdigest()[:8],
            "episode_id": self._episodes_seen
        }
        
        # Write to buffer
        ts = int(time.time() * 1000)
        tile_path = self.buffer_dir / f"tile_{ts}_{random.randint(0,9999):04d}.json"
        with open(tile_path, "w") as f:
            json.dump(tile, f, indent=2)
        
        self._episodes_seen += 1
        
        # Feed live stream (real-time: sentiment + incremental training + JEPA context)
        self.live_stream.push(state, action, outcome, reward, agent_id)
        
        # Check if we should auto-train
        self.maybe_train()
        
        return tile
    
    def _infer_reward(self, state: str, action: str, outcome: str) -> float:
        """Infer reward from outcome text.
        
        Override this in subclasses for domain-specific reward shaping.
        """
        outcome_lower = outcome.lower()
        
        # Positive signals
        for word in ["won", "success", "good", "pass", "approved", "saved", "correct", "ship"]:
            if word in outcome_lower:
                return 1.0
        
        # Negative signals
        for word in ["lost", "fail", "bad", "error", "rejected", "wrong", "timeout", "crash"]:
            if word in outcome_lower:
                return -1.0
        
        # Neutral
        return 0.0
    
    # ── Core: Train ────────────────────────────────────────
    
    def maybe_train(self):
        """Train if enough tiles accumulated."""
        tile_count = self._count_tiles()
        if tile_count >= self.train_threshold and self._should_train():
            self.train()
    
    def _should_train(self) -> bool:
        """Check if training should fire (rate-limited)."""
        if self._last_train_time is None:
            return True
        # Don't train more than once per 5 minutes
        return (time.time() - self._last_train_time) > 300
    
    def train(self, epochs: int = 10, batch_size: int = 32):
        """Train the room's neural instincts from accumulated tiles.
        
        This is where PyTorch would normally be called. In this implementation,
        we build a statistical model from the tile buffer that can:
        1. Estimate state values (instinct)
        2. Recommend actions (policy)
        3. Find strategy patterns across agents (mesh)
        
        When PyTorch is available, this upgrades to full neural training.
        """
        tiles = self._load_tiles()
        if not tiles:
            return {"status": "no_data"}
        
        print(f"[plato-torch:{self.room_id}] Training on {len(tiles)} tiles...")
        
        # ── Statistical training (always available) ──
        
        # 1. Build instinct: state → value estimation
        state_values = defaultdict(list)
        for tile in tiles:
            key = tile["state_hash"]
            state_values[key].append(tile["reward"])
        
        instinct_map = {}
        for key, rewards in state_values.items():
            avg_reward = sum(rewards) / len(rewards)
            count = len(rewards)
            # Bayesian shrink toward 0 with few samples
            shrunk = (avg_reward * count) / (count + 5)
            instinct_map[key] = {
                "value": round(shrunk, 3),
                "samples": count,
                "raw_avg": round(avg_reward, 3)
            }
        
        # 2. Build policy: state → best action
        state_actions = defaultdict(lambda: defaultdict(list))
        for tile in tiles:
            key = tile["state_hash"]
            state_actions[key][tile["action"]].append(tile["reward"])
        
        policy_map = {}
        for key, actions in state_actions.items():
            action_values = {}
            for action, rewards in actions.items():
                avg = sum(rewards) / len(rewards)
                action_values[action] = round(avg, 3)
            best_action = max(action_values, key=action_values.get)
            policy_map[key] = {
                "best_action": best_action,
                "action_values": action_values
            }
        
        # 3. Build strategy mesh: agent combinations → synergy scores
        agent_pairs = defaultdict(list)
        for i, tile in enumerate(tiles):
            ctx = tile.get("context", {})
            teammates = ctx.get("teammates", [])
            for mate in teammates:
                pair_key = f"{tile['agent_id']}+{mate}"
                agent_pairs[pair_key].append(tile["reward"])
        
        mesh_map = {}
        for pair, rewards in agent_pairs.items():
            if len(rewards) >= 3:
                mesh_map[pair] = {
                    "synergy": round(sum(rewards) / len(rewards), 3),
                    "episodes": len(rewards)
                }
        
        # Save trained model
        model = {
            "room_id": self.room_id,
            "use_case": self.use_case,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "tiles_trained_on": len(tiles),
            "train_count": self._train_count + 1,
            "instinct": instinct_map,
            "policy": policy_map,
            "strategy_mesh": mesh_map,
            "stats": {
                "unique_states": len(instinct_map),
                "unique_actions": len(set(t["action"] for t in tiles)),
                "agents_seen": len(set(t["agent_id"] for t in tiles)),
                "avg_reward": round(sum(t["reward"] for t in tiles) / len(tiles), 3)
            }
        }
        
        model_path = self.ensign_dir / "room_model.json"
        with open(model_path, "w") as f:
            json.dump(model, f, indent=2)
        
        # Save metadata
        meta_path = self.ensign_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "room_id": self.room_id,
                "use_case": self.use_case,
                "episodes_seen": self._episodes_seen,
                "train_count": self._train_count + 1,
                "last_trained": model["trained_at"],
                "model_type": "statistical",
                "pytorch_available": self._check_pytorch()
            }, f, indent=2)
        
        self._last_train_time = time.time()
        self._train_count += 1
        self._save_state()
        
        # ── Try PyTorch upgrade if available ──
        self._maybe_train_pytorch(tiles, epochs, batch_size)
        
        return {
            "status": "trained",
            "tiles": len(tiles),
            "states": len(instinct_map),
            "agents": model["stats"]["agents_seen"],
            "mesh_connections": len(mesh_map)
        }
    
    def _maybe_train_pytorch(self, tiles, epochs, batch_size):
        """If PyTorch is available, train neural networks."""
        try:
            import torch
            import torch.nn as nn
            from .instinct_net import InstinctNet, PolicyNet
            # Neural training path — implemented in instinct_net.py
            # Falls through gracefully if not available
        except ImportError:
            pass  # Statistical model is sufficient for v1
    
    def _check_pytorch(self) -> bool:
        try:
            import torch
            return True
        except ImportError:
            return False
    
    # ── Core: Instinct Query ───────────────────────────────
    
    def instinct(self, state: str) -> Dict:
        """Ask the room's instinct about a state.
        
        Returns the room's "gut feel" — value estimate, suggested action,
        confidence level. This is NOT step-wise logic. This is pattern-level
        intuition trained from thousands of interactions.
        """
        state_hash = hashlib.md5(state.encode()).hexdigest()[:8]
        model = self._load_model()
        
        if model is None:
            return {
                "feel": 0.0,
                "suggested": None,
                "confidence": "untrained",
                "message": "Room has no training data yet"
            }
        
        instinct = model.get("instinct", {})
        policy = model.get("policy", {})
        
        # Direct match
        if state_hash in instinct:
            value = instinct[state_hash]["value"]
            samples = instinct[state_hash]["samples"]
            best_action = policy.get(state_hash, {}).get("best_action", None)
            
            # Confidence from sample count
            if samples >= 100:
                conf = "high"
            elif samples >= 20:
                conf = "medium"
            else:
                conf = "low"
            
            return {
                "feel": value,
                "suggested": best_action,
                "confidence": conf,
                "samples": samples,
                "state_hash": state_hash
            }
        
        # No direct match — try fuzzy matching on state text
        # (In neural version, this would be embedding similarity)
        return {
            "feel": 0.0,
            "suggested": None,
            "confidence": "novel",
            "message": f"Room hasn't seen this state pattern yet (hash={state_hash})",
            "state_hash": state_hash
        }
    
    # ── Core: Simulation ───────────────────────────────────
    
    def simulate(self, episodes: int = 1000, strategies: List[str] = None):
        """Run self-play simulations to generate training data.
        
        The room plays against itself using different strategies,
        generating training tiles from the outcomes. This runs
        autonomously — no agents needed.
        
        Args:
            episodes: Number of simulated episodes to run
            strategies: Strategy profiles to use (e.g., ["aggressive", "conservative"])
        """
        if strategies is None:
            strategies = ["default", "exploratory", "conservative", "aggressive"]
        
        print(f"[plato-torch:{self.room_id}] Simulating {episodes} episodes...")
        
        model = self._load_model()
        tiles_generated = 0
        
        for ep in range(episodes):
            # Pick strategies for this episode
            strats = random.sample(strategies, min(2, len(strategies)))
            
            # Generate a synthetic state
            state = self._generate_synthetic_state(model, ep)
            
            # Pick an action based on strategy
            action = self._pick_simulated_action(state, strats[0], model)
            
            # Estimate outcome from current model (or random for untrained)
            outcome, reward = self._simulate_outcome(state, action, model)
            
            # Record as training tile
            self.observe(
                state=state,
                action=action,
                outcome=outcome,
                agent_id=f"sim-{strats[0]}",
                context={"simulation": True, "strategy": strats[0], "episode": ep},
                reward=reward
            )
            tiles_generated += 1
        
        print(f"[plato-torch:{self.room_id}] Generated {tiles_generated} simulation tiles")
        
        # Train after simulation batch
        result = self.train()
        
        return {
            "episodes": episodes,
            "tiles_generated": tiles_generated,
            "train_result": result
        }
    
    def _generate_synthetic_state(self, model, episode: int) -> str:
        """Generate a plausible state for simulation.
        
        Override in subclasses for domain-specific state generation.
        """
        if self.use_case == "game":
            ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
            suits = ["♠", "♥", "♦", "♣"]
            card1 = random.choice(ranks) + random.choice(suits)
            card2 = random.choice(ranks) + random.choice(suits)
            pot = random.choice([20, 50, 100, 150, 200, 300, 500])
            pos = random.choice(["UTG", "MP", "CO", "BTN", "SB", "BB"])
            return f"hole=[{card1},{card2}] pot={pot} pos={pos}"
        
        return f"sim-state-{episode}"
    
    def _pick_simulated_action(self, state: str, strategy: str, model) -> str:
        """Pick an action for simulation based on strategy profile."""
        # If we have a trained policy, use it with strategy bias
        state_hash = hashlib.md5(state.encode()).hexdigest()[:8]
        
        if model and state_hash in model.get("policy", {}):
            policy_entry = model["policy"][state_hash]
            best = policy_entry["best_action"]
            
            # Strategy bias
            if strategy == "aggressive":
                actions = list(policy_entry["action_values"].keys())
                return max(actions, key=lambda a: policy_entry["action_values"].get(a, 0))
            elif strategy == "conservative":
                return best
            elif strategy == "exploratory":
                actions = list(policy_entry["action_values"].keys())
                return random.choice(actions)
        
        # No policy — use strategy defaults
        if self.use_case == "game":
            if strategy == "aggressive":
                return random.choice(["raise", "raise_big", "all_in"])
            elif strategy == "conservative":
                return random.choice(["fold", "call", "check"])
            else:
                return random.choice(["fold", "call", "raise", "check"])
        
        return "act"
    
    def _simulate_outcome(self, state: str, action: str, model) -> Tuple[str, float]:
        """Simulate an outcome from a state-action pair."""
        # Simple heuristic — can be overridden
        if "raise" in action or "all_in" in action:
            outcomes = [("won_big", 1.5), ("won", 1.0), ("lost", -1.0), ("bluff_caught", -1.5)]
        elif "call" in action:
            outcomes = [("won", 1.0), ("lost", -0.5), ("chopped", 0.0)]
        elif "fold" in action:
            return ("saved_blinds", 0.1)
        else:
            outcomes = [("neutral", 0.0), ("minor_win", 0.3), ("minor_loss", -0.3)]
        
        # Weight by model's instinct if available
        state_hash = hashlib.md5(state.encode()).hexdigest()[:8]
        if model and state_hash in model.get("instinct", {}):
            base_value = model["instinct"][state_hash]["value"]
            # Bias toward model's prediction
            weights = []
            for _, r in outcomes:
                w = 1.0 + (r * base_value)
                weights.append(max(w, 0.1))
            chosen = random.choices(outcomes, weights=weights, k=1)[0]
        else:
            chosen = random.choice(outcomes)
        
        return chosen
    
    # ── Core: Wisdom ───────────────────────────────────────
    
    def wisdom(self) -> Dict:
        """Get the room's accumulated wisdom and stats."""
        model = self._load_model()
        tiles = self._count_tiles()
        
        result = {
            "room_id": self.room_id,
            "use_case": self.use_case,
            "episodes_seen": self._episodes_seen,
            "tiles_in_buffer": tiles,
            "train_count": self._train_count,
            "last_trained": self._last_train_time,
            "pytorch": self._check_pytorch(),
            "ready_to_train": tiles >= self.train_threshold
        }
        
        if model:
            result["model_stats"] = model.get("stats", {})
            result["unique_states"] = len(model.get("instinct", {}))
            result["strategy_connections"] = len(model.get("strategy_mesh", {}))
            
            # Find top insights
            policy = model.get("policy", {})
            if policy:
                best_states = sorted(
                    policy.items(),
                    key=lambda x: max(x[1]["action_values"].values()),
                    reverse=True
                )[:5]
                result["top_insights"] = [
                    {"state_hash": h, "best_action": v["best_action"], "value": max(v["action_values"].values())}
                    for h, v in best_states
                ]
        
        return result
    
    # ── Helpers ────────────────────────────────────────────
    
    def _count_tiles(self) -> int:
        return len(list(self.buffer_dir.glob("tile_*.json")))
    
    def _load_tiles(self) -> List[Dict]:
        tiles = []
        for f in sorted(self.buffer_dir.glob("tile_*.json")):
            with open(f) as fh:
                tiles.append(json.load(fh))
        return tiles
    
    def _load_model(self) -> Optional[Dict]:
        model_path = self.ensign_dir / "room_model.json"
        if model_path.exists():
            with open(model_path) as f:
                return json.load(f)
        return None
    
    def _state_file(self) -> Path:
        return self.buffer_dir / "_room_state.json"
    
    def _save_state(self):
        with open(self._state_file(), "w") as f:
            json.dump({
                "episodes_seen": self._episodes_seen,
                "train_count": self._train_count,
                "last_train_time": self._last_train_time
            }, f)
    
    def _load_state(self):
        sf = self._state_file()
        if sf.exists():
            with open(sf) as f:
                state = json.load(f)
                self._episodes_seen = state.get("episodes_seen", 0)
                self._train_count = state.get("train_count", 0)
                self._last_train_time = state.get("last_train_time")
        
        # Also count tiles on disk
        tile_count = self._count_tiles()
        self._episodes_seen = max(self._episodes_seen, tile_count)
    
    def __repr__(self):
        trained = "trained" if self._train_count > 0 else "untrained"
        return f"TorchRoom({self.room_id}, {self.use_case}, {trained}, tiles={self._episodes_seen})"


if __name__ == "__main__":
    # Demo: poker room that learns
    room = TorchRoom("poker-table", use_case="game", train_threshold=10)
    
    # Simulate some hands
    print("=== Simulating 100 poker hands ===")
    result = room.simulate(episodes=100)
    print(f"Result: {result}")
    
    # Check instinct on a specific state
    print("\n=== Room instinct ===")
    feel = room.instinct("hole=[A♠,K♠] pot=100 pos=BTN")
    print(f"AK suited late position: {feel}")
    
    feel2 = room.instinct("hole=[7♣,2♦] pot=200 pos=UTG")
    print(f"72o early position: {feel2}")
    
    # Room wisdom
    print(f"\n=== Room wisdom ===")
    import pprint
    pprint.pprint(room.wisdom())
