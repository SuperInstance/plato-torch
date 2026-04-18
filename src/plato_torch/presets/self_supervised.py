"""SelfSupervisedRoom — JEPA-style masked prediction for representation learning."""

import json
import random
import math
from ..room_base import RoomBase
class SelfSupervisedRoom(RoomBase):
    """Learns representations by masking input dimensions and predicting them."""

    def __init__(self, room_id: str = "self_supervised", mask_ratio: float = 0.3, lr: float = 0.01, **kwargs):
        super().__init__(room_id=room_id)
        self.mask_ratio = mask_ratio
        self.lr = lr
        # representation map: dim -> {mean, std, co_occurrence: {other_dim: cov}}
        self.repr_map: dict[int, dict] = {}
        self._buffer: list[dict[str, float]] = []

    # -- public API -----------------------------------------------------------

    def feed(self, states: list[dict[str, float]]) -> list[dict]:
        """Accept full states, auto-create masked versions, buffer for training."""
        results = []
        for state in states:
            keys = list(state.keys())
            n_mask = max(1, int(len(keys) * self.mask_ratio))
            masked_keys = random.sample(keys, n_mask)
            visible = {k: v for k, v in state.items() if k not in masked_keys}
            masked = {k: state[k] for k in masked_keys}
            self._buffer.append({"visible": visible, "masked": masked, "full": state})
            results.append({"visible_keys": list(visible.keys()), "masked_keys": masked_keys})
        return results

    def train_step(self) -> dict:
        """Learn to predict masked parts from visible parts (single gradient-free step)."""
        if not self._buffer:
            return {"status": "empty"}

        total_loss = 0.0
        n = 0
        for sample in self._buffer:
            full = sample["full"]
            # update per-dim statistics
            for k, v in full.items():
                if k not in self.repr_map:
                    self.repr_map[k] = {"mean": v, "m2": 0.0, "n": 1, "co": {}}
                entry = self.repr_map[k]
                entry["n"] += 1
                delta = v - entry["mean"]
                entry["mean"] += delta / entry["n"]
                entry["m2"] += delta * (v - entry["mean"])
                entry["std"] = math.sqrt(entry["m2"] / max(1, entry["n"] - 1)) if entry["n"] > 1 else 1.0
                # co-occurrence covariance
                for k2, v2 in full.items():
                    if k2 != k:
                        co = entry["co"].setdefault(k2, {"mean_xy": 0.0, "n": 0})
                        co["n"] += 1
                        co["mean_xy"] += (v * v2 - co["mean_xy"]) / co["n"]

            # pseudo-loss: squared error of mean-prediction for masked dims
            for mk in sample["masked"]:
                pred = self.repr_map.get(mk, {}).get("mean", 0.0)
                total_loss += (sample["full"][mk] - pred) ** 2
                n += 1

        self._buffer.clear()
        return {"status": "trained", "samples": n, "avg_loss": total_loss / max(1, n)}

    def predict(self, partial: dict[str, float]) -> dict[str, float]:
        """Fill in missing dimensions from a partial state using learned correlations."""
        result = dict(partial)
        known_keys = set(partial.keys())
        for dim, entry in self.repr_map.items():
            if dim in known_keys:
                continue
            # weighted prediction from known dims via covariance
            pred_num = 0.0
            pred_den = 0.0
            for known_dim, val in partial.items():
                if known_dim in entry.get("co", {}):
                    weight = abs(entry["co"][known_dim]["mean_xy"])
                    pred_num += weight * val
                    pred_den += weight
            if pred_den > 0:
                result[dim] = pred_num / pred_den
            else:
                result[dim] = entry["mean"]
        return result

    def export_model(self) -> str:
        """Return JSON of the representation map."""
        export = {}
        for dim, entry in self.repr_map.items():
            export[dim] = {
                "mean": round(entry["mean"], 6),
                "std": round(entry.get("std", 0.0), 6),
                "n": entry["n"],
                "top_correlations": dict(
                    sorted(
                        ((k, round(v["mean_xy"], 6)) for k, v in entry.get("co", {}).items()),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )[:5]
                ),
            }
        return json.dumps(export, indent=2)
