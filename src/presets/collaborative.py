"""CollaborativeRoom — Multi-agent knowledge sharing."""
import json
from room_base import RoomBase


class CollaborativeRoom(RoomBase):
    """Merges knowledge dumps from multiple agents, produces consensus."""

    def __init__(self, name: str = "collaborative"):
        super().__init__(name)
        self.knowledge: dict[str, list[dict]] = {}  # agent_id -> dumps
        self.consensus: dict = {}

    # -- public API --

    def feed(self, data: dict) -> None:
        """Accept {agent_id, knowledge_dump}."""
        aid = data["agent_id"]
        self.knowledge.setdefault(aid, []).append(data["knowledge_dump"])

    def train_step(self) -> dict:
        """Merge knowledge across agents; cross-teach by majority vote on keys."""
        merged: dict[str, list] = {}
        for dumps in self.knowledge.values():
            for dump in dumps:
                if not isinstance(dump, dict):
                    continue
                for k, v in dump.items():
                    merged.setdefault(k, []).append(v)

        # Simple majority-consensus: pick most common value per key
        consensus: dict = {}
        for k, vals in merged.items():
            counts: dict[str, int] = {}
            for v in vals:
                key = json.dumps(v, sort_keys=True)
                counts[key] = counts.get(key, 0) + 1
            best = max(counts, key=counts.get)  # type: ignore[arg-type]
            consensus[k] = json.loads(best)

        self.consensus = consensus
        return {"agents": len(self.knowledge), "keys": len(consensus), "consensus": consensus}

    def predict(self) -> dict:
        """Return current consensus knowledge."""
        return self.consensus

    def export_model(self) -> str:
        """JSON dump of shared knowledge base."""
        payload = {
            "name": self.name,
            "agent_count": len(self.knowledge),
            "consensus": self.consensus,
            "raw_knowledge": self.knowledge,
        }
        return json.dumps(payload, indent=2)
