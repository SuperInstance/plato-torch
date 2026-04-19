"""
ReinforceRoom — PPO-style reinforcement learning training room.

Episodes flow as: state → action → reward → next_state.
Maintains a policy network (action selection) and value network
(state valuation). Supports self-play simulation.

Works WITHOUT PyTorch (statistical/tabular fallback) and WITH PyTorch
(neural policy + value networks with PPO clipped objective).

Usage:
    from presets.reinforce import ReinforceRoom

    room = ReinforceRoom("rl-room")
    room.start_episode()
    room.step(state="hole=[A♠,K♥] pot=100", action="raise", reward=1.0)
    result = room.end_episode()
    room.train()
    action = room.act("hole=[7♣,2♦] pot=200 pos=UTG")
"""

import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

try:
    from room_base import RoomBase
except ImportError:
    try:
        from ..room_base import RoomBase
    except ImportError:
        RoomBase = object

# Neural networks (only constructed when PyTorch available)
if HAS_TORCH:
    class _PolicyNet(nn.Module):
        def __init__(self, state_dim, action_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )
        def forward(self, x):
            return self.net(x)

    class _ValueNet(nn.Module):
        def __init__(self, state_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)


class ReinforceRoom(RoomBase):
    """RL training room with PPO-style policy gradient.

    Statistical mode: tabular Q-learning with softmax action selection.
    Neural mode: actor-critic with PPO clipped objective.
    """

    def __init__(self, room_id="reinforce", gamma=0.99, lam=0.95,
                 clip_ratio=0.2, lr=3e-4, epochs_per_update=4,
                 state_dim=32, **kwargs):
        super().__init__(room_id, preset="reinforce", **kwargs)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.lr = lr
        self.epochs_per_update = epochs_per_update
        self.state_dim = state_dim

        # Episode buffer
        self._episodes: List[List[Dict]] = []
        self._current_episode: List[Dict] = []

        # Tabular mode
        self._q_table = defaultdict(lambda: defaultdict(float))
        self._visit_count = defaultdict(lambda: defaultdict(int))
        self._value_table = defaultdict(float)
        self._state_freq = defaultdict(int)

        # Neural mode (lazy)
        self._policy_net = None
        self._value_net = None
        self._optimizer = None
        self._action_vocab: Dict[str, int] = {}

        # Stats
        self._rl_total_episodes = 0
        self._rl_total_steps = 0
        self._rl_train_count = 0

        self._load_rl_state()

    # ── Episode API ──

    def start_episode(self) -> str:
        ep_id = f"ep-{self._rl_total_episodes}-{int(time.time())}"
        self._current_episode = []
        return ep_id

    def step(self, state, action, reward, next_state="", done=False,
             agent_id="unknown", context=None):
        transition = {
            "state": state, "action": action, "reward": reward,
            "next_state": next_state, "done": done,
            "agent_id": agent_id, "context": context or {},
            "state_hash": hashlib.md5(state.encode()).hexdigest()[:8],
            "timestamp": time.time(),
        }
        self._current_episode.append(transition)
        self._rl_total_steps += 1
        sh = transition["state_hash"]
        self._visit_count[sh][action] += 1
        self._state_freq[sh] += 1
        if done:
            self.end_episode()

    def end_episode(self) -> Dict:
        if not self._current_episode:
            return {"status": "empty"}
        returns = self._compute_returns(self._current_episode)
        for t, r in zip(self._current_episode, returns):
            t["return"] = r
        self._episodes.append(self._current_episode)
        self._rl_total_episodes += 1
        self._update_tabular(self._current_episode, returns)
        result = {
            "episode": self._rl_total_episodes,
            "steps": len(self._current_episode),
            "total_reward": round(sum(t["reward"] for t in self._current_episode), 4),
            "avg_return": round(sum(returns) / len(returns), 4) if returns else 0,
        }
        self._current_episode = []
        self._save_rl_state()
        return result

    # ── RoomBase interface ──

    def feed(self, data=None, **kwargs):
        if isinstance(data, dict):
            reward = data.get("reward")
            if reward is None:
                reward = self._infer_reward(
                    data.get("state", ""), data.get("action", ""),
                    data.get("outcome", ""))
            if not self._current_episode:
                self.start_episode()
            self.step(data.get("state", ""), data.get("action", ""),
                     reward, data.get("next_state", ""), data.get("done", False))
            if data.get("done"):
                return self.end_episode()
            return {"status": "buffered"}
        return {"status": "invalid_data"}

    def train_step(self, batch=None):
        if batch is None:
            return {"status": "ok", "message": "no batch", "preset": "reinforce"}
        return self.train()

    def predict(self, input=None):
        return self.policy_query(str(input))

    def export_model(self, format="json"):
        model = {
            "room_id": self.room_id, "preset": "reinforce",
            "mode": "neural" if self._policy_net else "tabular",
            "q_table": {k: dict(v) for k, v in self._q_table.items()},
            "value_table": dict(self._value_table),
            "action_vocab": self._action_vocab,
            "stats": {"episodes": self._rl_total_episodes,
                      "steps": self._rl_total_steps,
                      "states_known": len(self._q_table)},
        }
        return json.dumps(model, indent=2).encode()

    # ── Action Selection ──

    def act(self, state, epsilon=0.1, temperature=1.0):
        state_hash = hashlib.md5(state.encode()).hexdigest()[:8]
        if HAS_TORCH and self._policy_net is not None:
            return self._act_neural(state, temperature)
        return self._act_tabular(state_hash, epsilon)

    def _act_tabular(self, state_hash, epsilon):
        actions = list(self._q_table[state_hash].keys())
        if not actions:
            return "explore"
        if random.random() < epsilon:
            return random.choice(actions)
        return max(actions, key=lambda a: self._q_table[state_hash][a])

    def _act_neural(self, state, temperature):
        if not HAS_TORCH or self._policy_net is None:
            return "explore"
        features = self._encode_state(state)
        state_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self._policy_net(state_t).squeeze(0) / max(temperature, 0.01)
            probs = torch.softmax(logits, dim=0)
            idx = torch.multinomial(probs, 1).item()
        inv = {v: k for k, v in self._action_vocab.items()}
        return inv.get(idx, "explore")

    # ── Training ──

    def train(self, batch_episodes=None):
        episodes = self._episodes
        if batch_episodes:
            episodes = episodes[-batch_episodes:]
        if not episodes:
            return {"status": "no_data"}
        all_t = [t for ep in episodes for t in ep]
        if not all_t:
            return {"status": "no_data"}

        total_reward = sum(t["reward"] for t in all_t)
        avg_reward = total_reward / len(all_t)

        for sh in self._q_table:
            if self._q_table[sh]:
                self._value_table[sh] = max(self._q_table[sh].values())

        result = {
            "status": "trained", "mode": "tabular",
            "episodes": len(episodes), "transitions": len(all_t),
            "avg_reward": round(avg_reward, 4),
            "states_known": len(self._q_table),
            "actions_known": len(set(a for acts in self._q_table.values() for a in acts)),
        }

        if HAS_TORCH and len(all_t) >= 64:
            nr = self._train_neural(all_t)
            if nr:
                result["mode"] = "neural"
                result["neural_stats"] = nr

        self._rl_train_count += 1
        self._flush_episodes(episodes)
        self._episodes = self._episodes[len(episodes):]
        self._save_rl_state()
        return result

    def _update_tabular(self, episode, returns):
        lr = 0.1
        for t, G in zip(episode, returns):
            sh = t["state_hash"]
            action = t["action"]
            old_q = self._q_table[sh][action]
            self._q_table[sh][action] = old_q + lr * (G - old_q)
            if self._q_table[sh]:
                self._value_table[sh] = max(self._q_table[sh].values())

    def _train_neural(self, transitions):
        if not HAS_TORCH:
            return None
        for t in transitions:
            a = t["action"]
            if a not in self._action_vocab:
                self._action_vocab[a] = len(self._action_vocab)
        action_dim = len(self._action_vocab)
        sdim = self.state_dim
        if self._policy_net is None or self._policy_net.net[-1].out_features != action_dim:
            self._policy_net = _PolicyNet(sdim, action_dim)
            self._value_net = _ValueNet(sdim)
            self._optimizer = torch.optim.Adam(
                list(self._policy_net.parameters()) + list(self._value_net.parameters()),
                lr=self.lr)

        states = torch.tensor([self._encode_state(t["state"]) for t in transitions], dtype=torch.float32)
        actions_idx = torch.tensor([self._action_vocab.get(t["action"], 0) for t in transitions], dtype=torch.long)
        returns_t = torch.tensor([t.get("return", 0.0) for t in transitions], dtype=torch.float32)
        if returns_t.std() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        total_pl = total_vl = 0.0
        for _ in range(self.epochs_per_update):
            values = self._value_net(states)
            advantages = returns_t - values.detach()
            adv_std = advantages.std()
            if adv_std > 1e-6:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
            logits = self._policy_net(states)
            log_probs = torch.log_softmax(logits, dim=-1)
            sel_lp = log_probs.gather(1, actions_idx.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(sel_lp - sel_lp.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, returns_t)
            entropy = -(log_probs * torch.softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            total_pl += policy_loss.item()
            total_vl += value_loss.item()

        return {"policy_loss": round(total_pl / self.epochs_per_update, 4),
                "value_loss": round(total_vl / self.epochs_per_update, 4),
                "action_dim": action_dim}

    # ── Simulation ──

    def simulate(self, episodes=100, state_generator=None,
                 action_space=None, reward_fn=None):
        if action_space is None:
            action_space = ["act", "wait", "explore", "aggressive", "conservative"]
        if state_generator is None:
            c = [0]
            def state_generator(): c[0] += 1; return f"sim-state-{c[0]}"
        if reward_fn is None:
            def reward_fn(s, a):
                b = random.gauss(0, 1)
                if a == "aggressive": b += random.choice([-1.5, 1.0])
                elif a == "conservative": b += 0.1
                return round(b, 3)
        total_r = steps = 0
        for _ in range(episodes):
            self.start_episode()
            state = state_generator()
            ep_steps = random.randint(3, 20)
            for s in range(ep_steps):
                action = self.act(state, epsilon=0.3)
                if action not in action_space:
                    action = random.choice(action_space)
                reward = reward_fn(state, action)
                done = (s == ep_steps - 1)
                ns = state_generator() if not done else ""
                self.step(state, action, reward, ns, done, agent_id="sim-selfplay")
                state = ns
            r = self.end_episode()
            total_r += r.get("total_reward", 0)
            steps += r.get("steps", 0)
        return {"episodes": episodes, "total_steps": steps,
                "total_reward": round(total_r, 3),
                "avg_reward_per_step": round(total_r / max(steps, 1), 4)}

    # ── Query ──

    def value(self, state):
        sh = hashlib.md5(state.encode()).hexdigest()[:8]
        if HAS_TORCH and self._value_net is not None:
            f = self._encode_state(state)
            with torch.no_grad():
                return self._value_net(torch.tensor(f, dtype=torch.float32).unsqueeze(0)).item()
        return self._value_table.get(sh, 0.0)

    def policy_query(self, state):
        sh = hashlib.md5(state.encode()).hexdigest()[:8]
        if HAS_TORCH and self._policy_net is not None:
            f = self._encode_state(state)
            with torch.no_grad():
                logits = self._policy_net(torch.tensor(f, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                probs = torch.softmax(logits, dim=0).tolist()
            inv = {v: k for k, v in self._action_vocab.items()}
            dist = {inv.get(i, f"a{i}"): round(p, 4) for i, p in enumerate(probs)}
            best = inv.get(max(range(len(probs)), key=lambda i: probs[i]), None)
            return {"state_hash": sh, "action_probs": dist, "best_action": best,
                    "state_value": self.value(state), "mode": "neural"}
        q = dict(self._q_table.get(sh, {}))
        if not q:
            return {"state_hash": sh, "action_probs": {}, "best_action": None,
                    "state_value": 0.0, "mode": "tabular", "confidence": 0}
        mx = max(q.values())
        ex = {a: math.exp(v - mx) for a, v in q.items()}
        tot = sum(ex.values())
        dist = {a: round(e / tot, 4) for a, e in ex.items()}
        best = max(dist, key=dist.get)
        samples = self._state_freq.get(sh, 0)
        return {"state_hash": sh, "action_probs": dist, "best_action": best,
                "state_value": round(self._value_table.get(sh, 0.0), 4),
                "mode": "tabular", "confidence": min(1.0, samples / 20)}

    def stats(self):
        return {
            "room_id": self.room_id,
            "total_episodes": self._rl_total_episodes,
            "total_steps": self._rl_total_steps,
            "train_count": self._rl_train_count,
            "states_known": len(self._q_table),
            "mode": "neural" if (HAS_TORCH and self._policy_net) else "tabular",
            "pytorch": HAS_TORCH,
            "pending_episodes": len(self._episodes),
        }

    # ── Internals ──

    def _compute_returns(self, episode):
        returns = []
        G = 0.0
        for t in reversed(episode):
            G = t["reward"] + self.gamma * G
            returns.insert(0, G)
        return returns

    def _encode_state(self, state):
        dim = self.state_dim
        h1 = hashlib.sha256(state.encode()).digest()
        h2 = hashlib.sha256((state + "__s__").encode()).digest()
        return [(h1[i%32] / 255.0 * 2 - 1 + h2[i%32] / 255.0 * 2 - 1) / 2 for i in range(dim)]

    def _flush_episodes(self, episodes):
        ts = int(time.time())
        p = self.ensign_dir / f"trained_{ts}.jsonl"
        with open(p, "w") as f:
            for ep in episodes:
                for t in ep:
                    f.write(json.dumps(t, default=str) + "\n")

    def _save_rl_state(self):
        p = self.ensign_dir / "rl_state.json"
        with open(p, "w") as f:
            json.dump({
                "rl_total_episodes": self._rl_total_episodes,
                "rl_total_steps": self._rl_total_steps,
                "rl_train_count": self._rl_train_count,
                "q_table": {k: dict(v) for k, v in self._q_table.items()},
                "value_table": dict(self._value_table),
                "action_vocab": self._action_vocab,
            }, f)

    def _load_rl_state(self):
        p = self.ensign_dir / "rl_state.json"
        if not p.exists(): return
        try:
            with open(p) as f:
                s = json.load(f)
            self._rl_total_episodes = s.get("rl_total_episodes", 0)
            self._rl_total_steps = s.get("rl_total_steps", 0)
            self._rl_train_count = s.get("rl_train_count", 0)
            for k, v in s.get("q_table", {}).items():
                self._q_table[k] = defaultdict(float, v)
            self._value_table = defaultdict(float, s.get("value_table", {}))
            self._action_vocab = s.get("action_vocab", {})
        except (json.JSONDecodeError, IOError):
            pass

    def __repr__(self):
        m = "neural" if (HAS_TORCH and self._policy_net) else "tabular"
        return f"ReinforceRoom({self.room_id}, mode={m}, eps={self._rl_total_episodes})"


if __name__ == "__main__":
    room = ReinforceRoom("demo-rl", ensign_dir="/tmp/rl_ensigns", buffer_dir="/tmp/rl_buffers")
    print("=== Sim 200 episodes ===")
    print(room.simulate(episodes=200))
    print("\n=== Train ===")
    print(room.train())
    print("\n=== Policy ===")
    print(room.policy_query("sim-state-1"))
    print(room.stats())
