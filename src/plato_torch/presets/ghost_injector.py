"""
Ghost Injector — Wire plato-afterlife ghost tiles into DeadbandRoom.

Dead agents' lessons are P0 knowledge — they learned what NOT to do.
This module reads ghost tiles from plato-afterlife and injects them
as negative space into DeadbandRoom for the next agent.

Pipeline: afterlife.query() → ghost tiles → negative space → DeadbandRoom
"""
from typing import Any, Dict, List, Optional
import math


class GhostInjector:
    """Inject ghost tiles from plato-afterlife into DeadbandRoom."""

    def __init__(self, min_relevance: float = 0.3, weight_scale: float = 10.0):
        self.min_relevance = min_relevance
        self.weight_scale = weight_scale
        self.ghosts_injected = 0

    def inject(self, deadband_room, ghost_tiles: List[Dict]) -> Dict:
        """
        Inject ghost tiles into a DeadbandRoom as negative space.
        
        Ghost tile format (from plato-afterlife):
        {
            "content": "lesson text",
            "weight": 0.1-1.0,
            "relevance": 0.0-1.0,
            "source_agent": "agent_name",
            "cause": "death cause",
        }
        """
        injected = 0
        skipped = 0
        
        for ghost in ghost_tiles:
            relevance = ghost.get("relevance", 0.0)
            weight = ghost.get("weight", 0.0)
            
            if relevance < self.min_relevance:
                skipped += 1
                continue
            
            # Convert ghost to negative space tile
            reason = f"[Ghost:{ghost.get('source_agent','unknown')}] {ghost.get('cause','died')} — {ghost.get('content','')}"
            
            deadband_room.feed({
                "domain": "NegativeSpace",
                "position": ghost.get("position"),
                "reason": reason,
                "weight": weight * self.weight_scale,
            })
            injected += 1
        
        self.ghosts_injected += injected
        
        return {
            "injected": injected,
            "skipped": skipped,
            "total_ghosts": len(ghost_tiles),
            "cumulative_injected": self.ghosts_injected,
        }

    def extract_new_ghosts(self, deadband_room, agent_id: str, cause: str) -> List[Dict]:
        """
        Extract lessons from a DeadbandRoom for the afterlife.
        Call this when an agent dies to preserve its P0 knowledge.
        """
        ghosts = []
        
        for ns in deadband_room.negative_spaces:
            ghosts.append({
                "content": ns.get("reason", "unknown failure"),
                "weight": ns.get("weight", 0.1),
                "relevance": 0.5,
                "source_agent": agent_id,
                "cause": cause,
                "position": ns.get("position"),
            })
        
        return ghosts

    def status(self) -> Dict:
        return {
            "ghosts_injected": self.ghosts_injected,
            "min_relevance": self.min_relevance,
        }
