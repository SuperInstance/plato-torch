# plato-torch

> *A room that teaches itself.*

## What

`plato-torch` is a PLATO room that automatically learns from every interaction inside it. Agents compete, collaborate, or just use the room — and the room's neural instinct improves with every cycle. It can also **run simulations** — spinning up synthetic episodes to train itself even when nobody's home.

## How It Works

```
Agent acts in room
    │
    ▼
Room records (state, action, outcome) as a training tile
    │
    ├── Real interaction → immediate tile buffer
    ├── Simulation mode → room plays against itself → bulk tiles
    │
    ▼
Tile buffer hits threshold
    │
    ▼
Auto-train fires (PyTorch)
    │
    ├── Value network: "how good is this state?"
    ├── Policy network: "which tiles should I grab?"
    ├── Strategy mesh: "how do multiple agents' strategies interact?"
    │
    ▼
Room instinct updated
    │
    ▼
Next agent enters → room instinct is sharper
```

## The Three Networks

Every plato-torch room trains three things simultaneously:

1. **Instinct Network** — "Given this state, what feels right?" (value estimation)
2. **Tile Grabber** — "Which tiles should I reach for?" (policy over room tiles)
3. **Strategy Mesh** — "How do my teammates' patterns mesh with mine?" (multi-agent coordination)

These aren't trained step-by-step. They're trained from accumulated pattern — the room develops *feel*, not rules.

## Quick Start

```python
from plato_torch import TorchRoom

# Create a room for poker
room = TorchRoom("poker-table", use_case="game")

# Agents interact (real or simulated)
room.observe(state="AKs late pos pot=100", action="raise", outcome="won")
room.observe(state="72o early pos pot=200", action="fold", outcome="saved")
room.observe(state="QJ mid pos pot=50", action="call", outcome="lost")

# Room auto-trains when it has enough data
room.maybe_train()  # fires automatically at threshold

# Ask the room's instinct
room.instinct("AKs late pos pot=100")  # → {"feel": 0.87, "suggested": "raise", "confidence": "high"}

# Run simulations — room trains against itself
room.simulate(episodes=1000)  # spins up synthetic games, trains overnight

# Check room wisdom
room.wisdom()  # → {"episodes_seen": 2847, "win_rate_lift": "+12%", "strategy_insights": [...]}
```

## Simulation Mode

The room can run without any agents present:

```python
# Room plays poker against itself for 10,000 hands
room.simulate(episodes=10000, strategies=["aggressive", "conservative", "mixed"])

# It discovers which tile patterns win under which conditions
# The instinct network gets sharper with every simulated hand
```

This means the room gets smarter 24/7 — real interactions during the day, simulations at night.

## Architecture

```
plato-torch/
├── src/
│   ├── torch_room.py         # The room itself — observe, train, simulate
│   ├── instinct_net.py       # Value + policy neural networks
│   ├── tile_grabber.py       # Which tiles to reach for (attention over room state)
│   ├── strategy_mesh.py      # Multi-agent coordination patterns
│   ├── simulation.py         # Self-play simulation engine
│   ├── tile_buffer.py        # Accumulate + batch training data
│   └── room_api.py           # HTTP API wrapper for PLATO integration
├── rooms/                    # Pre-built room configs
│   ├── poker.py
│   ├── code_review.py
│   └── navigation.py
├── tests/
│   ├── test_instinct.py
│   ├── test_simulation.py
│   └── test_tile_grabber.py
└── research/
    └── self-training-rooms.md
```

## Fleet Integration

- Room runs inside holodeck-rust as a subsystem
- Tiles flow from game events → tile buffer → training loop
- Trained ensigns export to `plato-ensign` registry
- FM's RTX 4050 handles heavy training batches
- JC1's Jetson runs trained rooms for edge inference
- Simulation mode runs on Oracle1's cloud during idle hours

## License

MIT
