# plato-torch — Self-Training Rooms

**21 AI training methods as grab-and-go PLATO rooms.**

Every method shares the same API:
```python
room = ReinforceRoom("poker-room", ensign_dir="./ensigns", buffer_dir="./tiles")
room.feed(data)                    # Give it experience
room.train_step(batch)             # Learn from it
prediction = room.predict(input)   # Use the knowledge
model = room.export_model()        # Save it
```

## Quick Start

```python
import sys; sys.path.insert(0, "src")
from presets import PRESET_MAP

# See all 21 presets
for name, cls in sorted(PRESET_MAP.items()):
    print(name, cls.__name__)

# Pick one and use it
from presets import ReinforceRoom
room = ReinforceRoom("my-room")
room.observe("state-1", "action-a", "won")
room.observe("state-1", "action-b", "lost")
room.train_step(room._load_tiles())
print(room.predict("state-1"))
```

## All 21 Presets

### Classic ML
| Preset | Class | Description |
|--------|-------|-------------|
| `supervised` | `SupervisedRoom` | Labeled input→output via frequency counting |
| `contrastive` | `ContrastiveRoom` | Cosine similarity, triplet margin learning |
| `self_supervised` | `SelfSupervisedRoom` | JEPA-style masked prediction (Welford online) |

### Reinforcement
| Preset | Class | Description |
|--------|-------|-------------|
| `reinforce` | `ReinforceRoom` | Policy gradient, Monte Carlo returns |
| `inverse_rl` | `InverseRLRoom` | Observe expert, infer reward function |
| `imitate` | `ImitateRoom` | Clone expert behavior from demonstrations |

### Efficient Tuning
| Preset | Class | Description |
|--------|-------|-------------|
| `lora` | `LoRARoom` | PEFT delta table simulation |
| `qlora` | `QLoRARoom` | 4-bit quantized base + LoRA delta adapters |

### Population Methods
| Preset | Class | Description |
|--------|-------|-------------|
| `evolve` | `EvolveRoom` | Genetic algorithm, tournament selection |
| `adversarial` | `AdversarialRoom` | Red team vs blue team attack tracking |
| `collaborative` | `CollaborativeRoom` | Multi-agent knowledge sharing, majority vote |

### Meta / Federated
| Preset | Class | Description |
|--------|-------|-------------|
| `meta_learn` | `MetaLearnRoom` | Nearest-task fast adaptation (1-3 shot) |
| `federate` | `FederateRoom` | Federated averaging across agents |
| `multitask` | `MultitaskRoom` | Shared backbone + task-specific heads |

### Lifecycle
| Preset | Class | Description |
|--------|-------|-------------|
| `curriculum` | `CurriculumRoom` | Easy first, then harder (dojo progression) |
| `continual` | `ContinualRoom` | Lifelong learning, EWC-inspired replay buffer |
| `fewshot` | `FewshotRoom` | Prototype matching from 1-5 examples |
| `active` | `ActiveRoom` | Model chooses what data to learn from |

### Generative
| Preset | Class | Description |
|--------|-------|-------------|
| `generate` | `GenerateRoom` | N-gram data augmentation, synthetic state generation |

### Hybrid
| Preset | Class | Description |
|--------|-------|-------------|
| `neurosymbolic` | `NeurosymbolicRoom` | Neural instinct + symbolic rules blend |
| `distill` | `DistillRoom` | Teacher→student with temperature scaling |

## Architecture

```
plato-torch/
├── src/
│   ├── room_base.py          # RoomBase abstract class (feed/train_step/predict/export)
│   ├── torch_room.py         # TorchRoom — the full room with sentiment + tiles
│   ├── room_sentiment.py     # 6-dimensional room mood (energy, flow, frustration...)
│   ├── tile_grabber.py       # Learned attention over tile space
│   ├── instinct_net.py       # Tiny instinct network
│   ├── room_presets.py       # Registry of all 21 presets
│   └── presets/
│       ├── __init__.py       # PRESET_MAP — all 21 classes
│       ├── reinforce.py      # RL policy gradient
│       ├── evolve.py         # Genetic algorithm
│       ├── distill.py        # Teacher→student
│       ├── supervised.py     # Label frequency
│       ├── contrastive.py    # Triplet similarity
│       ├── self_supervised.py # JEPA masked prediction
│       ├── lora_train.py     # PEFT delta table
│       ├── qlora.py          # 4-bit quantized LoRA
│       ├── meta_learn.py     # Fast task adaptation
│       ├── federate.py       # Federated averaging
│       ├── multitask.py      # Shared backbone, task heads
│       ├── active.py         # Uncertainty sampling
│       ├── curriculum.py     # Difficulty progression
│       ├── imitate.py        # Expert cloning
│       ├── neurosymbolic.py  # Neural + symbolic rules
│       ├── continual.py      # EWC replay buffer
│       ├── fewshot.py        # Prototype matching
│       ├── generate.py       # N-gram augmentation
│       ├── adversarial.py    # Red/blue team
│       └── collaborative.py  # Multi-agent knowledge
├── docs/
│   ├── training-rooms.md           # Room architecture overview
│   └── training-seed-synergy.md    # Training ↔ seed-programming synergy paper
├── tests/
│   └── test_torch_room.py          # Unit tests
├── README.md
└── ARCHITECTURE-PLAN.md
```

## Key Concepts

### RoomBase API
Every preset inherits from `RoomBase` and implements:
- `feed(data)` — ingest experience
- `train_step(batch)` — learn from a batch of tiles
- `predict(input)` — use accumulated knowledge
- `export_model()` — serialize for transport

### Room Sentiment
Rooms have 6-dimensional mood: energy, flow, frustration, discovery, tension, confidence.
The room reads its own vibe and steers randomness toward productive exploration.

### Biased Randomness
When a room is frustrated → bias safe actions. Discovery mode → bias novel actions.
The room is an active participant, not a passive arena.

### Live Tile Stream
Tiles are generated hand-in-glove with room events. The stream feeds training in real-time.

### JEPA Context
`context_for_jepa()` outputs sentiment vectors for JC1's JEPA model consumption.

## Design Decisions

1. **Pure Python, no torch** — all presets use statistical methods (frequency, hashing, sampling)
2. **Same API everywhere** — walk into any room, same interface
3. **Composable** — rooms can feed into each other (evolve → distill → ensign)
4. **Agent-native** — designed for AI agents to use autonomously, not just humans
5. **Statistical grounding** — every method has real math under the hood (Monte Carlo, Welford, tournament selection, etc.)

## For Agents Following This Work

### How To Add A New Preset
1. Create `src/presets/your_preset.py`
2. Inherit from `RoomBase` (from `room_base import RoomBase`)
3. Implement: `feed()`, `train_step()`, `predict()`, `export_model()`
4. Constructor: `def __init__(self, room_id: str, **kwargs)` → `super().__init__(room_id, **kwargs)`
5. Add to `src/presets/__init__.py` PRESET_MAP
6. Add to `src/room_presets.py` PRESET_REGISTRY
7. Test: `room = YourPreset("test", ensign_dir="/tmp/e", buffer_dir="/tmp/b")`

### How Training Relates To Ensigns
- plato-torch rooms accumulate experience as tiles
- plato-ensign exports room wisdom as a portable ensign (LoRA/GGUF/Interpreter)
- The ensign loads instantly in any agent — "walk into room → load ensign → instant instinct"
- See `docs/training-seed-synergy.md` for the full alignment philosophy

### Fleet Integration
- **Oracle1** (cloud): runs training rooms, coordinates fleet learning
- **Forgemaster** (RTX 4050): trains LoRA adapters from accumulated tiles
- **JC1** (Jetson Orin): deploys ensigns for edge inference
