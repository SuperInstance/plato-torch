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
    room.step(state="flop=[Q♠,J♠,2♣]", action="bet", reward=0.5)
    result = room.end_episode()
    room.train()
    action = room.act("hole=[7♣,2♦] pot=200 pos=UTG")

Architecture:
    - Tabular mode: Q-table with softmax action selection, incremental updates
    - Neural mode: Actor-critic with PPO clipped surrogate objective
    - State encoding: SHA-256 hash → fixed-dim feature vector
    - Episode buffer: accumulated transitions with computed returns
"""

import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional PyTorch
HAS_TORCH = False