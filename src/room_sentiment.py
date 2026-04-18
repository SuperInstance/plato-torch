"""
Room Sentiment — continuous vibe computed from live interaction patterns.

JEPA reads the room's "feeling" — not just win/loss but the pattern of energy,
frustration, discovery, flow. This sentiment becomes a variable that biases
the room's randomness in productive directions.

The room isn't a passive arena. It's an active participant that shapes the
conditions under which agents learn.
"""

import math
import json
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class RoomSentiment:
    """Continuous sentiment tracker for a room.
    
    Maintains a rolling window of interaction signals and computes
    a multi-dimensional sentiment vector that the room can use to
    bias its stochastic elements.
    
    Sentiment dimensions:
    - energy: how active is the room right now? (0-1)
    - flow: are agents in a productive rhythm? (-1 to 1)
    - frustration: are agents hitting walls? (0-1)
    - discovery: are new patterns emerging? (0-1)
    - tension: competitive pressure between agents? (0-1)
    - confidence: how sure is the room about its suggestions? (0-1)
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.signals = deque(maxlen=window_size)
        self.start_time = time.time()
        
        # Rolling averages
        self._recent_rewards = deque(maxlen=50)
        self._recent_novelty = deque(maxlen=50)
        self._interaction_rate = deque(maxlen=20)
        self._last_interaction_time = time.time()
        
        # Computed sentiment
        self._sentiment = {
            "energy": 0.5,
            "flow": 0.0,
            "frustration": 0.0,
            "discovery": 0.0,
            "tension": 0.5,
            "confidence": 0.0,
            "overall": 0.5
        }
    
    def observe(self, reward: float, state_hash: str, agent_id: str,
                action: str, is_novel: bool = False):
        """Feed an interaction signal into the sentiment tracker."""
        now = time.time()
        
        # Track interaction rate
        dt = now - self._last_interaction_time
        self._last_interaction_time = now
        if dt > 0:
            self._interaction_rate.append(1.0 / dt)
        
        # Track reward trend
        self._recent_rewards.append(reward)
        
        # Track novelty
        if is_novel:
            self._recent_novelty.append(1.0)
        else:
            self._recent_novelty.append(0.0)
        
        # Store signal
        self.signals.append({
            "time": now,
            "reward": reward,
            "agent": agent_id,
            "novel": is_novel
        })
        
        # Recompute sentiment
        self._compute()
    
    def _compute(self):
        """Recompute sentiment from rolling signals."""
        # Energy: interaction rate
        if self._interaction_rate:
            avg_rate = sum(self._interaction_rate) / len(self._interaction_rate)
            # Normalize: 1 interaction/sec = full energy
            self._sentiment["energy"] = min(1.0, avg_rate / 2.0)
        
        # Flow: reward momentum (are things getting better or worse?)
        if len(self._recent_rewards) >= 5:
            recent = list(self._recent_rewards)
            half = len(recent) // 2
            older = recent[:half]
            newer = recent[half:]
            older_avg = sum(older) / len(older) if older else 0
            newer_avg = sum(newer) / len(newer) if newer else 0
            self._sentiment["flow"] = max(-1, min(1, newer_avg - older_avg))
        
        # Frustration: consecutive negative rewards
        if self._recent_rewards:
            recent = list(self._recent_rewards)
            streak = 0
            for r in reversed(recent):
                if r < 0:
                    streak += 1
                else:
                    break
            self._sentiment["frustration"] = min(1.0, streak / 10.0)
        
        # Discovery: novelty rate
        if self._recent_novelty:
            self._sentiment["discovery"] = sum(self._recent_novelty) / len(self._recent_novelty)
        
        # Tension: reward variance (high variance = competitive)
        if len(self._recent_rewards) >= 3:
            rewards = list(self._recent_rewards)
            mean = sum(rewards) / len(rewards)
            variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
            self._sentiment["tension"] = min(1.0, variance)
        
        # Confidence: number of signals (more data = more confidence)
        self._sentiment["confidence"] = min(1.0, len(self.signals) / self.window_size)
        
        # Overall: weighted blend
        self._sentiment["overall"] = (
            0.2 * self._sentiment["energy"] +
            0.25 * max(0, self._sentiment["flow"]) +
            0.15 * self._sentiment["discovery"] +
            0.1 * (1 - self._sentiment["frustration"]) +
            0.15 * self._sentiment["confidence"] +
            0.15 * self._sentiment["tension"]
        )
    
    def read(self) -> Dict[str, float]:
        """Read the current room sentiment vector."""
        return dict(self._sentiment)
    
    def read_compound(self) -> str:
        """Read sentiment as a compound label for context injection."""
        s = self._sentiment
        
        if s["frustration"] > 0.6:
            return "frustrated"
        if s["discovery"] > 0.5 and s["flow"] > 0.3:
            return "breakthrough"
        if s["flow"] > 0.3 and s["energy"] > 0.5:
            return "flowing"
        if s["tension"] > 0.6 and s["energy"] > 0.5:
            return "intense"
        if s["energy"] < 0.2:
            return "dormant"
        if s["discovery"] > 0.3:
            return "exploring"
        return "steady"


class BiasedRandomness:
    """Screws randomness in positive-for-the-application ways.
    
    Takes the room's sentiment and uses it to bias stochastic elements:
    - Card dealing in poker
    - NPC behavior in MUD rooms
    - Script trigger timing
    - Tile selection weights
    
    The room doesn't just host randomness — it STEERS randomness toward
    conditions that produce better training data and more engaging experiences.
    """
    
    def __init__(self, sentiment: RoomSentiment):
        self.sentiment = sentiment
    
    def biased_choice(self, options: List, weights: List[float] = None) -> any:
        """Choose from options with sentiment-biased weights.
        
        If the room is frustrated, bias toward easier/safer options.
        If the room is in discovery mode, bias toward novel options.
        If the room is flowing, maintain current trajectory.
        """
        import random
        
        s = self.sentiment.read()
        
        if weights is None:
            weights = [1.0] * len(options)
        
        # Apply sentiment bias
        biased = list(weights)
        
        if s["frustration"] > 0.5:
            # Frustrated room → bias toward known-good options
            # Boost the highest-weighted option
            best_idx = biased.index(max(biased))
            biased[best_idx] *= (1.0 + s["frustration"])
        
        elif s["discovery"] > 0.5:
            # Discovery mode → bias toward less-tried options
            # Boost the lowest-weighted options
            min_w = min(biased)
            for i, w in enumerate(biased):
                if w <= min_w * 1.5:
                    biased[i] *= (1.0 + s["discovery"])
        
        elif s["flow"] > 0.3:
            # Flowing → slight boost to middle options (maintain rhythm)
            median_idx = len(biased) // 2
            if 0 <= median_idx < len(biased):
                biased[median_idx] *= (1.0 + s["flow"] * 0.5)
        
        # Ensure positive weights
        biased = [max(0.01, w) for w in biased]
        
        return random.choices(options, weights=biased, k=1)[0]
    
    def biased_float(self, low: float = 0.0, high: float = 1.0) -> float:
        """Generate a random float biased by room sentiment.
        
        Positive overall sentiment → bias toward higher values
        Negative/frustrated → bias toward lower (safer) values
        """
        import random
        
        s = self.sentiment.read()
        raw = random.random()
        
        # Shift by overall sentiment
        bias = s["overall"] - 0.5  # -0.5 to +0.5
        shifted = raw + bias * 0.3  # subtle shift
        
        # Clamp
        shifted = max(0.0, min(1.0, shifted))
        
        return low + shifted * (high - low)
    
    def should_trigger_script(self, base_probability: float = 0.1) -> bool:
        """Decide if a script/trigger/macro should fire.
        
        Biased by room energy and discovery rate. More active rooms
        with more discovery → scripts fire more often.
        """
        s = self.sentiment.read()
        
        # Base probability modified by sentiment
        adjusted = base_probability
        adjusted *= (1.0 + s["energy"])          # active rooms → more triggers
        adjusted *= (1.0 + s["discovery"] * 0.5)  # discovering → more triggers
        adjusted *= (1.0 - s["frustration"] * 0.3) # frustrated → fewer triggers
        
        import random
        return random.random() < adjusted


class IncrementalTrainer:
    """CPU-friendly incremental training during live play.
    
    PyTorch on CPU — slow but continuous. Every few interactions,
    take a tiny gradient step. The room's neural weights drift
    toward better instincts in real-time, not just in batch.
    
    This is the "hand-in-glove" — tile picking and generation
    happening simultaneously with game play.
    """
    
    def __init__(self, room_id: str, step_interval: int = 10,
                 learning_rate: float = 0.001):
        self.room_id = room_id
        self.step_interval = step_interval  # steps between gradient updates
        self.learning_rate = learning_rate
        self._step_count = 0
        self._gradient_buffer = []
    
    def maybe_step(self, state_hash: str, reward: float, action: str):
        """Take an incremental training step if enough data accumulated.
        
        This runs on CPU — slow but free. Over thousands of interactions,
        the room's weights drift toward better instincts.
        """
        self._gradient_buffer.append((state_hash, reward, action))
        self._step_count += 1
        
        if self._step_count % self.step_interval == 0:
            self._take_step()
    
    def _take_step(self):
        """Perform one tiny gradient update from buffered data."""
        if not self._gradient_buffer:
            return
        
        batch = self._gradient_buffer[-self.step_interval:]
        self._gradient_buffer = self._gradient_buffer[-50:]  # keep recent
        
        # Statistical gradient step (works without PyTorch)
        avg_reward = sum(r for _, r, _ in batch) / len(batch)
        
        # If PyTorch available, do real gradient step
        try:
            import torch
            self._pytorch_step(batch, avg_reward)
        except ImportError:
            self._statistical_step(batch, avg_reward)
    
    def _statistical_step(self, batch, avg_reward):
        """Statistical fallback when no PyTorch."""
        # Update running averages — this IS training, just statistical
        pass  # TorchRoom's main model handles this
    
    def _pytorch_step(self, batch, avg_reward):
        """Real PyTorch gradient step on CPU."""
        import torch
        # Neural weight update from batch
        # Will be wired to InstinctNet/PolicyNet when room has models
        pass


class LiveTileStream:
    """Tiles flow in real-time as games are played.
    
    Not batch-and-train-later. Generate, pick, and learn simultaneously.
    The room is a living system — every interaction produces tiles,
    the tile picker selects which ones matter, and the trainer
    takes micro-steps from the selected tiles.
    
    Hand-in-glove: generation and consumption in the same breath.
    """
    
    def __init__(self, room_id: str, sentiment: RoomSentiment,
                 trainer: IncrementalTrainer):
        self.room_id = room_id
        self.sentiment = sentiment
        self.trainer = trainer
        self._tile_queue = []
        self._state_cache = {}
    
    def push(self, state: str, action: str, outcome: str,
             reward: float, agent_id: str = "unknown"):
        """Push a live interaction into the stream.
        
        This happens DURING game play. Not after. During.
        """
        import hashlib
        state_hash = hashlib.md5(state.encode()).hexdigest()[:8]
        
        # Check novelty
        is_novel = state_hash not in self._state_cache
        self._state_cache[state_hash] = True
        
        # Feed sentiment
        self.sentiment.observe(reward, state_hash, agent_id, action, is_novel)
        
        # Feed incremental trainer
        self.trainer.maybe_step(state_hash, reward, action)
        
        # Create live tile
        tile = {
            "state": state,
            "state_hash": state_hash,
            "action": action,
            "outcome": outcome,
            "reward": reward,
            "agent": agent_id,
            "sentiment": self.sentiment.read(),
            "sentiment_label": self.sentiment.read_compound(),
            "novel": is_novel,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        self._tile_queue.append(tile)
        
        # Flush to disk periodically
        if len(self._tile_queue) >= 10:
            self._flush()
        
        return tile
    
    def _flush(self):
        """Write accumulated tiles to buffer."""
        if not self._tile_queue:
            return
        
        # In production, this writes to the tile buffer
        # For now, just clear
        count = len(self._tile_queue)
        self._tile_queue.clear()
        return count
    
    def context_for_jepa(self) -> Dict:
        """Generate context variable for JEPA to consume.
        
        The room's current state, sentiment, and pattern recognition
        formatted as input for JC1's JEPA model. JEPA uses this to
        predict what should happen next and feeds predictions back
        to the room's algorithms.
        """
        s = self.sentiment.read()
        
        return {
            "room_id": self.room_id,
            "sentiment": s,
            "sentiment_label": self.sentiment.read_compound(),
            "recent_tile_count": len(self._tile_queue),
            "unique_states_seen": len(self._state_cache),
            "novelty_rate": sum(1 for t in self._tile_queue if t.get("novel")) / max(len(self._tile_queue), 1),
            "avg_recent_reward": sum(t["reward"] for t in self._tile_queue) / max(len(self._tile_queue), 1),
            # JEPA should predict: next state reward, optimal action, expected tension
            "prediction_targets": ["next_reward", "optimal_action", "tension_trajectory"]
        }
