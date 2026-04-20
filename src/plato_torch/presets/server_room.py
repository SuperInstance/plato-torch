"""
ServerRoom — Live room connected to PLATO Room Server.

Reads tiles from the local PLATO server (port 8847) and uses them
as training data. Zero-trust validated tiles only.
"""
from typing import Any, Dict, List, Optional
import json
import urllib.request

PLATO_URL = "http://localhost:8847"


class ServerRoom:
    """A room that reads validated tiles from the PLATO room server."""
    
    def __init__(self, domain: str = None):
        self.domain = domain
        self.tiles = []
        self._load_tiles()
    
    def _load_tiles(self):
        """Load tiles from PLATO server."""
        try:
            if self.domain:
                url = PLATO_URL + "/room/" + self.domain
            else:
                url = PLATO_URL + "/status"
            
            resp = urllib.request.urlopen(url, timeout=5)
            data = json.loads(resp.read())
            
            if self.domain:
                self.tiles = data.get("tiles", [])
            else:
                self.tiles = []
        except (urllib.error.URLError, json.JSONDecodeError, OSError):
            self.tiles = []
    
    def feed(self, data: Dict[str, Any]):
        """Submit a new tile to the server."""
        tile = {
            "domain": self.domain or data.get("domain", "general"),
            "question": data.get("question", ""),
            "answer": data.get("answer", ""),
            "tags": data.get("tags", []),
            "confidence": data.get("confidence", 0.5),
            "source": data.get("source", "server_room"),
        }
        
        try:
            body = json.dumps(tile).encode()
            req = urllib.request.Request(
                PLATO_URL + "/submit",
                data=body,
                headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=10)
            result = json.loads(resp.read())
            
            if result.get("status") == "accepted":
                self._load_tiles()  # Refresh
                return result
            return result
        except Exception as e:
            return {"status": "error", "reason": str(e)}
    
    def predict(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Search tiles for relevant answers."""
        question = query.get("question", "").lower()
        
        if not question or not self.tiles:
            return {"answer": None, "confidence": 0.0, "tile_count": len(self.tiles)}
        
        # Simple keyword matching
        best_tile = None
        best_score = 0
        
        query_words = set(question.split())
        
        for tile in self.tiles:
            tile_words = set(tile.get("question", "").lower().split())
            tile_words.update(set(tile.get("answer", "").lower().split()[:50]))
            
            overlap = len(query_words & tile_words)
            if overlap > best_score:
                best_score = overlap
                best_tile = tile
        
        if best_tile and best_score > 0:
            return {
                "answer": best_tile.get("answer", ""),
                "confidence": min(best_score / max(len(query_words), 1), 1.0),
                "tile_count": len(self.tiles),
                "domain": self.domain,
                "source": best_tile.get("source", ""),
            }
        
        return {"answer": None, "confidence": 0.0, "tile_count": len(self.tiles)}
    
    def train_step(self):
        """Refresh tiles from server."""
        self._load_tiles()
        return {"tiles_loaded": len(self.tiles)}
    
    def status(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "tile_count": len(self.tiles),
            "server": PLATO_URL,
        }
