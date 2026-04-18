"""
Tile Grabber — which tiles should an agent reach for?

In a room with many tiles (system prompts, context fragments, skills),
the agent needs to know which ones are relevant RIGHT NOW. The Tile Grabber
is a learned attention mechanism over the room's tile space.

"Which tiles should I grab?" is a fundamentally different question from
"what's the next step?" It's about PATTERN RECOGNITION of what matters
in this exact situation, trained from thousands of prior interactions.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TileGrabber:
    """Learns which tiles are most relevant for given states.
    
    Trained from observation: when agents succeeded, which tiles did they
    have active? When they failed, which tiles were missing?
    """
    
    def __init__(self, room_id: str, ensign_dir: str = "ensigns"):
        self.room_id = room_id
        self.ensign_dir = Path(ensign_dir) / room_id
        self.tile_scores = defaultdict(lambda: defaultdict(float))
        self.tile_counts = defaultdict(lambda: defaultdict(int))
    
    def observe_grab(self, state: str, tiles_grabbed: List[str], reward: float):
        """Record which tiles were grabbed and how it turned out."""
        state_hash = hashlib.md5(state.encode()).hexdigest()[:8]
        for tile in tiles_grabbed:
            self.tile_scores[state_hash][tile] += reward
            self.tile_counts[state_hash][tile] += 1
    
    def recommend_tiles(self, state: str, available_tiles: List[str],
                        top_k: int = 5) -> List[Tuple[str, float]]:
        """Recommend which tiles to grab for this state.
        
        Returns list of (tile_id, relevance_score) sorted by relevance.
        """
        state_hash = hashlib.md5(state.encode()).hexdigest()[:8]
        
        recommendations = []
        for tile in available_tiles:
            count = self.tile_counts[state_hash].get(tile, 0)
            if count > 0:
                avg_score = self.tile_scores[state_hash][tile] / count
                recommendations.append((tile, avg_score))
            else:
                # No data — low relevance
                recommendations.append((tile, 0.0))
        
        # Sort by relevance
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]
    
    def tile_synergy(self, tile_a: str, tile_b: str) -> float:
        """How often do these two tiles appear together in successful outcomes?"""
        # Co-occurrence in high-reward states
        cooccur = 0
        total = 0
        for state_hash in self.tile_scores:
            a_score = self.tile_scores[state_hash].get(tile_a, 0)
            b_score = self.tile_scores[state_hash].get(tile_b, 0)
            if a_score > 0 and b_score > 0:
                cooccur += 1
            total += 1
        
        return cooccur / max(total, 1)
    
    def save(self):
        """Save tile grabber state."""
        path = self.ensign_dir / "tile_grabber.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdicts to regular dicts for serialization
        data = {
            "room_id": self.room_id,
            "tile_scores": {k: dict(v) for k, v in self.tile_scores.items()},
            "tile_counts": {k: dict(v) for k, v in self.tile_counts.items()}
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load tile grabber state."""
        path = self.ensign_dir / "tile_grabber.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                self.tile_scores = defaultdict(lambda: defaultdict(float),
                                               {k: defaultdict(float, v) 
                                                for k, v in data.get("tile_scores", {}).items()})
                self.tile_counts = defaultdict(lambda: defaultdict(int),
                                               {k: defaultdict(int, v) 
                                                for k, v in data.get("tile_counts", {}).items()})
