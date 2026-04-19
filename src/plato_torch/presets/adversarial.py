"""AdversarialRoom — Red team vs blue team training."""
import json
try:
    from room_base import RoomBase
except ImportError:
    from .room_base import RoomBase


class AdversarialRoom(RoomBase):
    """Tracks attack/defense rounds, computes success rates, surfaces weak inputs."""

    def __init__(self, room_id: str = "adversarial", **kwargs):
        super().__init__(room_id, **kwargs)
        self.rounds: list[dict] = []
        self.attack_success_rate: float = 0.0

    # -- public API --

    def feed(self, data=None) -> None:
        """Accept {attack_input, defense_response}."""
        if data is None: data = {}
        if isinstance(data, str): data = {"attack_input": data, "defense_response": data}
        data.setdefault("attack_input", "test")
        data.setdefault("defense_response", "test")
        self.rounds.append({
            "attack_input": data["attack_input"],
            "defense_response": data["defense_response"],
            "success": bool(data.get("success", False)),
        })

    def train_step(self) -> dict:
        """Compute attack success rate, identify weakest (most-penetrated) inputs."""
        if not self.rounds:
            return {"attack_success_rate": 0.0, "weakest": []}

        wins = sum(1 for r in self.rounds if r["success"])
        self.attack_success_rate = wins / len(self.rounds)

        # Group by attack_input, rank by success count
        scores: dict[str, int] = {}
        for r in self.rounds:
            key = r["attack_input"]
            scores[key] = scores.get(key, 0) + int(r["success"])

        weakest = sorted(scores, key=scores.get, reverse=True)[:5]  # type: ignore[arg-type]
        return {"attack_success_rate": self.attack_success_rate, "weakest": weakest}

    def predict(self, input=None) -> list[str]:
        """Return most vulnerable inputs for red-team testing."""
        stats = self.train_step()
        return stats["weakest"]

    def export_model(self) -> str:
        """JSON dump of the full attack/defense log."""
        payload = {
            "name": self.room_id,
            "rounds": self.rounds,
            "attack_success_rate": self.attack_success_rate,
        }
        return json.dumps(payload, indent=2)
