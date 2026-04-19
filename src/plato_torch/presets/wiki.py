"""
WikiRoom — big model compiles, cheap models consume.

Based on Casey's insight: the room IS the intelligence. Many rooms don't need
full training or ensigns. They need:
1. A wiki of compiled knowledge (schemas, plans, how-tos)
2. Cheap model workers that read the wiki and execute
3. A "ralph wiggum" pattern: try → get stuck → ask → continue

The WikiRoom IS the ship computer for rooms like:
- Slideshow maker (slides as rooms, assets as tiles)
- Code review workshop (PRs as rooms, review comments as tiles)
- Asset generation studio (ideas as rooms, variants as tiles)
"""

import json
import hashlib
import time
from collections import defaultdict
from typing import Dict, List, Optional, Any

try:
    from room_base import RoomBase
except ImportError:
    from .room_base import RoomBase


class WikiEntry:
    """A compiled knowledge entry in the room wiki."""
    
    def __init__(self, topic: str, content: str, abstraction_level: int = 0,
                 compiled_by: str = "unknown", source_tiles: List[str] = None):
        self.topic = topic
        self.content = content
        self.abstraction_level = abstraction_level  # 0=overview, 5=expert detail
        self.compiled_by = compiled_by  # which model/agent compiled this
        self.source_tiles = source_tiles or []
        self.created_at = time.time()
        self.access_count = 0
        self.helpfulness = 0.5  # tracks how often this wiki entry helped
    
    def record_access(self, was_helpful: bool):
        self.access_count += 1
        alpha = 0.1
        self.helpfulness = self.helpfulness * (1 - alpha) + (1.0 if was_helpful else 0.0) * alpha


class WikiRoom(RoomBase):
    """A room where the wiki IS the intelligence.
    
    Big models compile knowledge into wiki entries.
    Cheap models consume wiki entries to execute tasks.
    The ralph-wiggum pattern: try → stuck → wiki → continue.
    """
    
    def __init__(self, room_id: str = "wiki", **kwargs):
        super().__init__(room_id, preset="wiki", **kwargs)
        self._wiki: Dict[str, WikiEntry] = {}  # topic → entry
        self._schemas: Dict[str, Dict] = {}  # compiled task schemas
        self._stuck_log: List[Dict] = []  # ralph-wiggum stuck events
        self._task_queue: List[Dict] = []  # pending tasks for cheap models
        self._completed_tasks: List[Dict] = []
        self._stuck_threshold = kwargs.get("stuck_threshold", 0.3)
    
    def compile_wiki(self, topic: str, content: str, abstraction_level: int = 0,
                     compiled_by: str = "unknown", source_tiles: List[str] = None) -> Dict:
        """Big model compiles knowledge into the wiki.
        
        This is the 'captain explaining to the greenhorn' step.
        """
        entry = WikiEntry(topic, content, abstraction_level, compiled_by, source_tiles)
        self._wiki[topic] = entry
        return {"topic": topic, "level": abstraction_level, "compiled_by": compiled_by}
    
    def compile_schema(self, task_type: str, instructions: List[str],
                       prerequisites: List[str] = None, 
                       cheap_model_hints: List[str] = None) -> Dict:
        """Big model compiles a task schema for cheap model consumption.
        
        A schema is a simplified plan that a cheap model can follow:
        - Step-by-step instructions (no ambiguity)
        - Prerequisites (what must be true before starting)
        - Hints (what the cheap model should watch out for)
        """
        schema = {
            "task_type": task_type,
            "instructions": instructions,
            "prerequisites": prerequisites or [],
            "cheap_model_hints": cheap_model_hints or [],
            "compiled_at": time.time(),
        }
        self._schemas[task_type] = schema
        return schema
    
    def lookup(self, topic: str) -> Optional[str]:
        """Cheap model looks up wiki knowledge."""
        entry = self._wiki.get(topic)
        if entry:
            entry.record_access(True)
            return entry.content
        # Fuzzy: check if topic appears in any entry
        for t, entry in self._wiki.items():
            if topic.lower() in t.lower() or t.lower() in topic.lower():
                entry.record_access(True)
                return entry.content
        return None
    
    def lookup_schema(self, task_type: str) -> Optional[Dict]:
        """Cheap model looks up a compiled task schema."""
        return self._schemas.get(task_type)
    
    def report_stuck(self, agent_id: str, task: str, tried: str,
                     wiki_topics_checked: List[str] = None) -> Dict:
        """Ralph-wiggum pattern: cheap model reports it's stuck.
        
        This is the 'greenhorn asking the captain' step.
        """
        stuck_event = {
            "agent_id": agent_id,
            "task": task,
            "tried": tried,
            "wiki_checked": wiki_topics_checked or [],
            "timestamp": time.time(),
        }
        self._stuck_log.append(stuck_event)
        
        # Try to auto-resolve from wiki
        resolution = None
        for topic in (wiki_topics_checked or []):
            entry = self._wiki.get(topic)
            if entry and entry.helpfulness > 0.5:
                resolution = f"Wiki '{topic}' suggests: {entry.content[:200]}"
                break
        
        return {
            "stuck_reported": True,
            "auto_resolution": resolution,
            "needs_big_model": resolution is None,
        }
    
    def feed(self, data=None, **kwargs) -> Dict:
        if data is None: data = {}
        if isinstance(data, str): data = {"data": data}
        if isinstance(data, dict):
            # Wiki entries
            if "topic" in data and "content" in data:
                return self.compile_wiki(
                    data["topic"], data["content"],
                    data.get("abstraction_level", 0),
                    data.get("compiled_by", "unknown"),
                    data.get("source_tiles")
                )
            # Task schemas
            if "task_type" in data and "instructions" in data:
                return self.compile_schema(
                    data["task_type"], data["instructions"],
                    data.get("prerequisites"),
                    data.get("cheap_model_hints")
                )
            # Stuck reports
            if "task" in data and "tried" in data:
                return self.report_stuck(
                    data.get("agent_id", "unknown"),
                    data["task"], data["tried"],
                    data.get("wiki_topics_checked")
                )
            # Regular tile observation
            return self.observe(data.get("state",""), data.get("action",""), data.get("outcome",""))
        return {"status": "invalid"}
    
    def train_step(self, batch=None) -> Dict:
        if batch is None:
            return {"status": "ok", "message": "no batch", "preset": "wiki"}
        """Process tiles: extract knowledge, update wiki helpfulness."""
        for tile in batch:
            action = tile.get("action", "")
            reward = tile.get("reward", 0)
            # If an action referenced a wiki topic, track its helpfulness
            if reward > 0:
                for topic, entry in self._wiki.items():
                    if topic.lower() in action.lower():
                        entry.record_access(True)
            elif reward < 0:
                for topic, entry in self._wiki.items():
                    if topic.lower() in action.lower():
                        entry.record_access(False)
        
        return {
            "wiki_entries": len(self._wiki),
            "schemas": len(self._schemas),
            "stuck_events": len(self._stuck_log),
            "completed_tasks": len(self._completed_tasks),
        }
    
    def predict(self, input=None) -> Dict:
        """Look up wiki knowledge for the given input."""
        result = self.lookup(str(input))
        schema = self.lookup_schema(str(input))
        
        return {
            "wiki_answer": result,
            "task_schema": schema,
            "has_knowledge": result is not None or schema is not None,
        }
    
    def export_model(self, format: str = "json") -> Optional[bytes]:
        model = {
            "room_id": self.room_id,
            "preset": "wiki",
            "wiki_topics": list(self._wiki.keys()),
            "wiki_sizes": {t: len(e.content) for t, e in self._wiki.items()},
            "wiki_helpfulness": {t: round(e.helpfulness, 3) for t, e in self._wiki.items()},
            "schemas": list(self._schemas.keys()),
            "stuck_events": len(self._stuck_log),
            "completed_tasks": len(self._completed_tasks),
        }
        return json.dumps(model, indent=2).encode()
    
    def wiki_stats(self) -> Dict:
        """Get wiki statistics."""
        total_accesses = sum(e.access_count for e in self._wiki.values())
        avg_helpfulness = (sum(e.helpfulness for e in self._wiki.values()) / 
                          max(len(self._wiki), 1))
        return {
            "entries": len(self._wiki),
            "schemas": len(self._schemas),
            "total_accesses": total_accesses,
            "avg_helpfulness": round(avg_helpfulness, 3),
            "stuck_events": len(self._stuck_log),
            "stuck_rate": (len(self._stuck_log) / max(total_accesses, 1)),
        }


if __name__ == "__main__":
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    
    # Create a slideshow wiki room
    room = WikiRoom("slideshow-studio", ensign_dir=os.path.join(tmpdir, "e"),
                    buffer_dir=os.path.join(tmpdir, "t"))
    
    # Big model compiles knowledge
    room.compile_wiki("brand-colors", "Primary: #1a472a (forest green). Accent: #c9a227 (gold). Background: #f5f1eb (warm white).", 
                      compiled_by="glm-5.1")
    room.compile_wiki("slide-layout", "Title slides: centered, 48pt bold. Content: left-aligned, 32pt. Data: right-aligned chart + left caption.",
                      compiled_by="glm-5.1")
    room.compile_wiki("image-style", "Clean, minimal, lots of whitespace. Prefer nature metaphors. Avoid stock photos.",
                      compiled_by="glm-5.1")
    
    # Big model compiles a task schema
    room.compile_schema(
        "generate_slide_image",
        instructions=[
            "Read the slide title and content",
            "Identify the key metaphor or concept",
            "Generate a prompt for image generation",
            "Apply brand color palette",
            "Check image dimensions (16:9 ratio)",
        ],
        prerequisites=["slide title exists", "brand-colors wiki loaded"],
        cheap_model_hints=[
            "If no clear metaphor, use abstract geometric shapes",
            "Always include the gold accent color",
            "Stay minimal — less is more",
        ]
    )
    
    # Cheap model looks up wiki
    print("Lookup 'brand-colors':", room.lookup("brand-colors"))
    print("Lookup schema:", room.lookup_schema("generate_slide_image")["instructions"])
    
    # Cheap model gets stuck (ralph-wiggum)
    stuck = room.report_stuck("zeroclaw-1", "generate hero image", 
                              "tried geometric shapes but colors look wrong",
                              ["brand-colors", "image-style"])
    print(f"\nStuck: auto_resolution={stuck['auto_resolution']}, needs_big_model={stuck['needs_big_model']}")
    
    # Feed stuck as tile
    room.feed({"task": "generate hero image", "tried": "geometric shapes, wrong colors",
               "agent_id": "zeroclaw-1", "wiki_topics_checked": ["brand-colors"]})
    
    # Stats
    print(f"\nWiki stats: {room.wiki_stats()}")
    
    # Predict
    pred = room.predict("slide-layout")
    print(f"Predict: has_knowledge={pred['has_knowledge']}")
    
    print("\nWIKIROOM WORKS")
