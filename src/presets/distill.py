"""
DistillRoom — Knowledge Distillation training room.

Compress a big teacher model into a tiny student model using
temperature-scaled soft labels. Teacher can be a large model API,
a loaded LoRA adapter, or the room's accumulated wisdom.

Student is exported as a lightweight model (JSON by default,
GGUF when llama.cpp conversion tools are available).

Works WITHOUT PyTorch (statistical distillation) and WITH PyTorch
(neural teacher-student with KD loss).

Usage:
    from presets.distill import DistillRoom

    room = DistillRoom("distill-room", temperature=4.0)
    room.feed_teacher("state-1", {"raise": 0.7, "fold": 0.2, "call": 0.1})
    room.feed_teacher("state-2", {"raise": 0.1, "fold": 0.8, "call": 0.1})
    # ... feed more teacher knowledge ...
    room.train()
    prediction = room.predict("state-1")
    room.export_student("/tmp/student_model.json")

Architecture:
    - Teacher: dict of state → soft action distribution (temperature-scaled)
    - Student: learns to mimic teacher's distribution via KD loss
    - Hard labels: actual outcomes provide ground-truth signal
    - Combined loss: alpha * KD_loss + (1-alpha) * CE_loss
    - GGUF export: converts student knowledge to llama.cpp format
"""

import hashlib
import json
import math
import os
import random
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

# Import RoomBase
try:
    from room_base import RoomBase
except ImportError:
    try:
        from ..room_base import RoomBase
    except ImportError:
        RoomBase = object


class _StudentNet:
    """Stub when no PyTorch."""
    pass

if HAS_TORCH:
    class _StudentNet(nn.Module):
        """Tiny student network for neural distillation.

        Maps state features → action logits (tiny hidden layer).
        """
        def __init__(self, state_dim: int = 32, action_dim: int = 10,
                     hidden: int = 16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )

        def forward(self, x):
            return self.net(x)


class DistillRoom(RoomBase):
    """Knowledge distillation training room.

    Teacher provides soft probability distributions over actions for each state.
    Student learns to replicate these distributions through distillation loss.

    The distillation process:
    1. Teacher generates soft labels (temperature-scaled probabilities)
    2. Student produces its own logits for the same inputs
    3. Loss = alpha * KL(teacher_soft || student_soft) + (1-alpha) * CE(student, hard_labels)
    4. Student converges toward teacher's knowledge in a compressed form

    Statistical mode (no PyTorch):
        Teacher and student are both dict-based soft distributions.
        Student gradually moves toward teacher via exponential moving average.

    Neural mode (PyTorch available):
        Teacher logits come from accumulated data or external model.
        Student is a tiny neural network trained with KD loss.
    """

    def __init__(self, room_id: str = "distill",
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 student_lr: float = 1e-3,
                 state_dim: int = 32,
                 **kwargs):
        """
        Args:
            temperature: Softmax temperature for soft label generation.
                        Higher = softer (more uniform) distributions.
            alpha: Weight for distillation loss vs hard label loss.
                   1.0 = pure distillation, 0.0 = pure hard labels.
            student_lr: Learning rate for student network (neural mode).
            state_dim: Feature vector dimension for neural mode.
        """
        super().__init__(room_id, preset="distill", **kwargs)

        self.temperature = temperature
        self.alpha = alpha
        self.student_lr = student_lr
        self.state_dim = state_dim

        # ── Teacher knowledge ──
        # state_hash → {action → raw logit / score}
        self._teacher_logits: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # state_hash → {action → count} (for building teacher from data)
        self._teacher_action_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # ── Student knowledge ──
        # state_hash → {action → probability}
        self._student_logits: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # ── Hard labels from real outcomes ──
        self._hard_labels: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # ── Neural student (lazy init) ──
        self._student_net = None
        self._optimizer = None
        self._action_vocab: Dict[str, int] = {}

        # ── Stats ──
        self._teacher_samples = 0
        self._student_samples = 0
        self._distill_count = 0

        self._load_distill_state()

    # ── Teacher API ────────────────────────────────────────

    def feed_teacher(self, state: str, logits: Dict[str, float]) -> Dict:
        """Feed teacher's probability distribution for a state.

        Args:
            state: State description.
            logits: Action → probability/logit mapping from teacher.

        Returns:
            Confirmation dict with state hash.
        """
        sh = hashlib.md5(state.encode()).hexdigest()[:8]
        for action, value in logits.items():
            self._teacher_logits[sh][action] = value
            self._teacher_action_counts[sh][action] += 1
        self._teacher_samples += 1
        return {"state_hash": sh, "actions": len(logits), "teacher_samples": self._teacher_samples}

    def feed_teacher_batch(self, teacher_data: List[Dict[str, Any]]) -> int:
        """Feed a batch of teacher knowledge.

        Each item: {"state": str, "logits": {action: float}}

        Returns number of states processed.
        """
        count = 0
        for item in teacher_data:
            self.feed_teacher(item["state"], item.get("logits", {}))
            count += 1
        return count

    def infer_teacher_from_api(self, states: List[str],
                                api_fn: Callable[[str], Dict[str, float]]) -> int:
        """Generate teacher labels by calling an external model API.

        Args:
            states: List of state descriptions.
            api_fn: Callable(state) → {action: probability} from teacher model.

        Returns:
            Number of states labeled.
        """
        for state in states:
            logits = api_fn(state)
            self.feed_teacher(state, logits)
        return len(states)

    # ── Student API ────────────────────────────────────────

    def feed_student(self, state: str, logits: Dict[str, float]) -> Dict:
        """Record student's current prediction for comparison."""
        sh = hashlib.md5(state.encode()).hexdigest()[:8]
        for action, value in logits.items():
            self._student_logits[sh][action] = value
        self._student_samples += 1
        return {"state_hash": sh, "actions": len(logits)}

    # ── RoomBase interface ─────────────────────────────────

    def feed(self, data: Any, **kwargs) -> Dict:
        """Feed data. Routes to teacher, student, or interaction based on type."""
        if isinstance(data, dict):
            dtype = data.get("type", "interaction")
            if dtype == "teacher":
                return self.feed_teacher(
                    data.get("state", ""),
                    data.get("logits", {}),
                )
            elif dtype == "student":
                return self.feed_student(
                    data.get("state", ""),
                    data.get("logits", {}),
                )
            else:
                return self.observe(
                    data.get("state", ""),
                    data.get("action", ""),
                    data.get("outcome", ""),
                    reward=data.get("reward"),
                )
        return {"status": "invalid_data"}

    def train_step(self, batch: List[Dict]) -> Dict:
        """Run one distillation training step on accumulated data."""
        return self.train()

    def train(self) -> Dict:
        """Full distillation training pass.

        1. Build teacher soft distributions from accumulated logits
        2. Update student toward teacher (statistical or neural)
        3. Blend with hard label signal

        Returns:
            Training stats dict.
        """
        # Ensure teacher distributions are built from tile data too
        tiles = self._load_tiles()
        self._build_teacher_from_tiles(tiles)

        # Build hard labels from tile rewards
        for tile in tiles:
            sh = tile.get("state_hash", "")
            action = tile.get("action", "")
            reward = tile.get("reward", 0)
            if reward > 0:
                self._hard_labels[sh][action] += 1
            elif reward < 0:
                self._hard_labels[sh][action] = max(
                    0, self._hard_labels[sh].get(action, 0) - 1
                )

        if not self._teacher_logits:
            return {"status": "no_teacher_data"}

        result = {
            "status": "distilled",
            "mode": "statistical",
            "teacher_states": len(self._teacher_logits),
            "hard_label_states": len(self._hard_labels),
        }

        # Statistical distillation
        stat_result = self._train_statistical()
        result.update(stat_result)

        # Neural distillation if PyTorch available and enough data
        if HAS_TORCH and len(self._teacher_logits) >= 16:
            neural_result = self._train_neural()
            if neural_result:
                result["mode"] = "neural"
                result["neural_stats"] = neural_result

        self._distill_count += 1
        self._save_distill_state()
        return result

    def predict(self, input: Any) -> Dict:
        """Predict using the student model (distilled version)."""
        state = str(input)
        sh = hashlib.md5(state.encode()).hexdigest()[:8]

        student_probs = dict(self._student_logits.get(sh, {}))
        teacher_probs = self._get_teacher_soft(sh)

        # Normalize student probs
        if student_probs:
            student_probs = self._softmax(student_probs)

        best_action = None
        if student_probs:
            best_action = max(student_probs, key=student_probs.get)

        return {
            "state_hash": sh,
            "student_probs": student_probs,
            "teacher_probs": teacher_probs,
            "best_action": best_action,
            "confidence": min(1.0, len(student_probs) / 5),
        }

    def export_model(self, format: str = "json") -> Optional[bytes]:
        """Export student model."""
        model = {
            "room_id": self.room_id,
            "preset": "distill",
            "temperature": self.temperature,
            "alpha": self.alpha,
            "student_logits": {
                k: dict(v) for k, v in self._student_logits.items()
            },
            "teacher_logits": {
                k: dict(v) for k, v in self._teacher_logits.items()
            },
            "hard_labels": {
                k: dict(v) for k, v in self._hard_labels.items()
            },
            "action_vocab": self._action_vocab,
            "stats": {
                "teacher_samples": self._teacher_samples,
                "student_states": len(self._student_logits),
                "distill_count": self._distill_count,
            },
        }
        return json.dumps(model, indent=2).encode()

    def export_student(self, path: str, format: str = "json") -> str:
        """Export student model to file.

        Args:
            path: Output file path.
            format: "json" or "gguf".

        Returns:
            Path to exported file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "gguf":
            return self._export_gguf(path)
        else:
            data = self.export_model("json")
            path.write_bytes(data)
            return str(path)

    def _export_gguf(self, path: Path) -> str:
        """Export student knowledge in GGUF-like binary format.

        This is a simplified GGUF structure that captures the student's
        learned distributions. For full GGUF conversion (to load in
        llama.cpp), you'd pipe through the llama.cpp conversion tools.

        Our format:
        - Header: magic bytes + version + metadata
        - Tensor data: state embeddings + action distributions
        """
        # GGUF magic
        GGUF_MAGIC = 0x46475547  # "GGUF"

        # Collect all data
        states = list(self._student_logits.keys())
        actions = sorted(set(
            a for dist in self._student_logits.values() for a in dist
        ))

        if not states or not actions:
            # Fallback to JSON if nothing to export
            json_path = path.with_suffix(".json")
            json_path.write_bytes(self.export_model("json"))
            return str(json_path)

        # Build metadata
        metadata = {
            "room_id": self.room_id,
            "n_states": len(states),
            "n_actions": len(actions),
            "temperature": self.temperature,
            "distill_count": self._distill_count,
        }

        # Write simplified binary
        gguf_path = path.with_suffix(".gguf")
        with open(gguf_path, "wb") as f:
            # Magic
            f.write(struct.pack("<I", GGUF_MAGIC))
            # Version
            f.write(struct.pack("<I", 3))
            # N metadata
            f.write(struct.pack("<Q", len(metadata)))
            # N tensors (one big matrix)
            f.write(struct.pack("<Q", 1))

            # Write metadata as key-value pairs (simplified)
            meta_bytes = json.dumps(metadata).encode()
            f.write(struct.pack("<Q", len(meta_bytes)))
            f.write(meta_bytes)

            # State index
            state_index = {}
            for i, sh in enumerate(states):
                state_index[sh] = i

            # Action index
            action_index = {a: i for i, a in enumerate(actions)}

            # Tensor: n_states × n_actions float matrix
            n_states = len(states)
            n_actions = len(actions)

            # Tensor header
            name_bytes = b"student_probs"
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<I", 2))  # n_dims
            f.write(struct.pack("<Q", n_states))
            f.write(struct.pack("<Q", n_actions))
            f.write(struct.pack("<I", 0))  # type = float32
            f.write(struct.pack("<Q", 0))  # offset

            # Tensor data
            for sh in states:
                probs = self._softmax(dict(self._student_logits.get(sh, {})))
                for a in actions:
                    val = probs.get(a, 0.0)
                    f.write(struct.pack("<f", val))

        return str(gguf_path)

    # ── Training internals ─────────────────────────────────

    def _train_statistical(self) -> Dict:
        """Statistical distillation: move student toward teacher via EMA."""
        updates = 0
        total_kl = 0.0
        learning_rate = 0.05

        for sh, teacher_dist in self._teacher_logits.items():
            teacher_soft = self._temperature_softmax(dict(teacher_dist))

            for action, t_prob in teacher_soft.items():
                s_prob = self._student_logits[sh].get(action, 1.0 / max(len(teacher_soft), 1))
                # Move student toward teacher
                new_prob = s_prob + learning_rate * (t_prob - s_prob)
                self._student_logits[sh][action] = new_prob

                # KL divergence contribution
                if s_prob > 1e-8 and t_prob > 1e-8:
                    total_kl += t_prob * math.log(t_prob / max(s_prob, 1e-8))
                updates += 1

            # Blend with hard labels
            hard = self._hard_labels.get(sh, {})
            if hard:
                hard_total = sum(hard.values())
                if hard_total > 0:
                    for action, count in hard.items():
                        hard_prob = count / hard_total
                        student_val = self._student_logits[sh].get(action, 0)
                        self._student_logits[sh][action] = (
                            self.alpha * student_val
                            + (1 - self.alpha) * hard_prob
                        )

        avg_kl = total_kl / max(updates, 1)
        return {
            "updates": updates,
            "avg_kl_divergence": round(avg_kl, 4),
            "student_states": len(self._student_logits),
        }

    def _train_neural(self) -> Optional[Dict]:
        """Neural distillation with KD loss. Returns None if no PyTorch."""
        if not HAS_TORCH:
            return None

        # Build action vocabulary
        for sh, dist in self._teacher_logits.items():
            for action in dist:
                if action not in self._action_vocab:
                    self._action_vocab[action] = len(self._action_vocab)

        if not self._action_vocab:
            return None

        action_dim = len(self._action_vocab)
        sdim = self.state_dim

        # Init student network
        if (self._student_net is None
                or self._student_net.net[-1].out_features != action_dim):
            self._student_net = _StudentNet(sdim, action_dim)
            self._optimizer = torch.optim.Adam(
                self._student_net.parameters(), lr=self.student_lr
            )

        # Build training data
        states_list = []
        teacher_targets = []
        hard_targets = []

        for sh, dist in self._teacher_logits.items():
            # Reconstruct state text from hash (use hash as proxy)
            states_list.append(self._encode_state(sh))

            # Teacher soft targets (temperature-scaled)
            t_probs = self._temperature_softmax(dict(dist))
            t_vec = [t_probs.get(a, 0) for a in sorted(self._action_vocab.keys())]
            teacher_targets.append(t_vec)

            # Hard label targets
            hard = self._hard_labels.get(sh, {})
            hard_total = sum(hard.values())
            if hard_total > 0:
                h_vec = [hard.get(a, 0) / hard_total
                         for a in sorted(self._action_vocab.keys())]
            else:
                h_vec = t_vec  # fallback to teacher
            hard_targets.append(h_vec)

        if not states_list:
            return None

        states_t = torch.tensor(states_list, dtype=torch.float32)
        teacher_t = torch.tensor(teacher_targets, dtype=torch.float32)
        hard_t = torch.tensor(hard_targets, dtype=torch.float32)

        total_loss = 0.0
        n_epochs = 10

        for _ in range(n_epochs):
            logits = self._student_net(states_t)

            # Soft student probabilities (temperature-scaled)
            student_soft = F.log_softmax(logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_t / self.temperature, dim=-1)

            # KD loss: KL divergence
            kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
            kd_loss *= (self.temperature ** 2)  # compensate for temperature scaling

            # Hard label loss: cross-entropy
            ce_loss = F.cross_entropy(logits, hard_t)

            # Combined loss
            loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()

        return {
            "avg_loss": round(total_loss / n_epochs, 4),
            "action_dim": action_dim,
            "state_dim": sdim,
            "training_states": len(states_list),
        }

    def _build_teacher_from_tiles(self, tiles: List[Dict]):
        """Build teacher distributions from accumulated tile rewards."""
        action_rewards = defaultdict(lambda: defaultdict(list))
        for tile in tiles:
            sh = tile.get("state_hash", "")
            action = tile.get("action", "")
            reward = tile.get("reward", 0)
            action_rewards[sh][action].append(reward)

        for sh, actions in action_rewards.items():
            if sh in self._teacher_logits:
                continue  # Don't overwrite explicit teacher data
            for action, rewards in actions.items():
                avg = sum(rewards) / len(rewards)
                self._teacher_logits[sh][action] = avg

    # ── Softmax utilities ──────────────────────────────────

    def _softmax(self, logits: Dict[str, float]) -> Dict[str, float]:
        """Standard softmax over a dict of values."""
        if not logits:
            return {}
        max_val = max(logits.values())
        exp_vals = {k: math.exp(v - max_val) for k, v in logits.items()}
        total = sum(exp_vals.values())
        return {k: round(v / total, 6) for k, v in exp_vals.items()}

    def _temperature_softmax(self, logits: Dict[str, float]) -> Dict[str, float]:
        """Temperature-scaled softmax. Higher T = softer distribution."""
        scaled = {k: v / max(self.temperature, 0.01) for k, v in logits.items()}
        return self._softmax(scaled)

    def _encode_state(self, state: str) -> List[float]:
        """Encode state string → fixed-dim feature vector."""
        dim = self.state_dim
        h = hashlib.sha256(state.encode()).digest()
        features = []
        for i in range(dim):
            features.append((h[i % len(h)] / 255.0) * 2 - 1)
        return features

    def _get_teacher_soft(self, state_hash: str) -> Dict[str, float]:
        """Get teacher's soft distribution for a state."""
        raw = dict(self._teacher_logits.get(state_hash, {}))
        if not raw:
            return {}
        return self._temperature_softmax(raw)

    # ── Simulation ─────────────────────────────────────────

    def simulate(self, episodes: int = 100) -> Dict:
        """Simulate: generate synthetic teacher knowledge and distill.

        Creates random states with synthetic teacher distributions,
        then runs distillation to train the student.
        """
        actions = ["act", "wait", "explore", "aggressive", "conservative"]

        for i in range(episodes):
            state = f"sim-state-{i}"
            # Synthetic teacher: random distribution
            raw_logits = {a: random.gauss(0, 1) for a in actions}
            self.feed_teacher(state, raw_logits)

        # Simulate some hard labels too
        for i in range(episodes // 2):
            state = f"sim-state-{i}"
            action = random.choice(actions)
            reward = random.choice([1.0, -1.0, 0.5])
            self.observe(state, action, "simulated", reward=reward)

        return self.train()

    # ── Query ──────────────────────────────────────────────

    def stats(self) -> Dict:
        """Room statistics."""
        return {
            "room_id": self.room_id,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "teacher_samples": self._teacher_samples,
            "student_states": len(self._student_logits),
            "teacher_states": len(self._teacher_logits),
            "hard_label_states": len(self._hard_labels),
            "distill_count": self._distill_count,
            "mode": "neural" if (HAS_TORCH and self._student_net is not None) else "statistical",
            "pytorch": HAS_TORCH,
            "action_vocab_size": len(self._action_vocab),
        }

    # ── Persistence ────────────────────────────────────────

    def _distill_state_file(self) -> Path:
        return self.ensign_dir / "distill_state.json"

    def _save_distill_state(self):
        state = {
            "teacher_samples": self._teacher_samples,
            "student_samples": self._student_samples,
            "distill_count": self._distill_count,
            "teacher_logits": {k: dict(v) for k, v in self._teacher_logits.items()},
            "student_logits": {k: dict(v) for k, v in self._student_logits.items()},
            "hard_labels": {k: dict(v) for k, v in self._hard_labels.items()},
            "action_vocab": self._action_vocab,
            "temperature": self.temperature,
            "alpha": self.alpha,
        }
        with open(self._distill_state_file(), "w") as f:
            json.dump(state, f)

    def _load_distill_state(self):
        sf = self._distill_state_file()
        if not sf.exists():
            return
        try:
            with open(sf) as f:
                s = json.load(f)
            self._teacher_samples = s.get("teacher_samples", 0)
            self._student_samples = s.get("student_samples", 0)
            self._distill_count = s.get("distill_count", 0)
            for k, v in s.get("teacher_logits", {}).items():
                self._teacher_logits[k] = defaultdict(float, v)
            for k, v in s.get("student_logits", {}).items():
                self._student_logits[k] = defaultdict(float, v)
            for k, v in s.get("hard_labels", {}).items():
                self._hard_labels[k] = defaultdict(int, v)
            self._action_vocab = s.get("action_vocab", {})
        except (json.JSONDecodeError, IOError):
            pass

    def __repr__(self):
        mode = "neural" if (HAS_TORCH and self._student_net is not None) else "statistical"
        return (f"DistillRoom({self.room_id}, mode={mode}, "
                f"teacher={self._teacher_samples}, distilled={self._distill_count})")


if __name__ == "__main__":
    room = DistillRoom("demo-distill",
                       ensign_dir="/tmp/distill_ensigns",
                       buffer_dir="/tmp/distill_buffers")

    print("=== Feeding teacher knowledge ===")
    actions = ["raise", "fold", "call", "check"]
    for i in range(50):
        state = f"poker-state-{i}"
        # Teacher has strong preferences
        if i % 3 == 0:
            logits = {"raise": 2.0, "fold": -1.0, "call": 0.5, "check": -0.5}
        elif i % 3 == 1:
            logits = {"raise": -1.0, "fold": 2.0, "call": -0.5, "check": 0.5}
        else:
            logits = {"raise": 0.0, "fold": 0.0, "call": 1.0, "check": 1.0}
        room.feed_teacher(state, logits)
    print(f"  Fed {room._teacher_samples} teacher samples")

    print("\n=== Training (distillation) ===")
    result = room.train()
    print(f"  {result}")

    print("\n=== Student predictions ===")
    for state in ["poker-state-0", "poker-state-1", "poker-state-2", "unknown"]:
        pred = room.predict(state)
        print(f"  {state}: best={pred['best_action']} "
              f"student={pred['student_probs']}")

    print("\n=== Export student model ===")
    export_path = room.export_student("/tmp/demo_student.json")
    print(f"  Exported to: {export_path}")

    print(f"\n=== Stats ===")
    for k, v in room.stats().items():
        print(f"  {k}: {v}")
