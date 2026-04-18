# PLATO-Torch Architecture Plan

**Date:** 2026-04-18
**Author:** Oracle1 (lead architect)
**Status:** Implementation blueprint

---

## What We Have Now

The codebase is a solid skeleton with working statistical training:

- **TorchRoom** — observes interactions, trains statistical models (state values, policy, strategy mesh), supports simulation/self-play
- **RoomSentiment** — 6-axis vibe tracker (energy, flow, frustration, discovery, tension, confidence)
- **BiasedRandomness** — steers stochastic choices toward productive directions based on sentiment
- **LiveTileStream** — real-time tile generation with sentiment feeding and JEPA context output
- **IncrementalTrainer** — placeholder for CPU micro-steps (stub only)
- **InstinctNet / PolicyNet / StrategyMeshNet** — PyTorch network architectures (unwired)
- **TileGrabber** — learned attention over tile space (statistical, works)
- **21 TrainingPreset** configs — dataclass definitions, no implementations

**What's missing:** Everything that makes the presets real. The networks exist but aren't connected. The trainer is a stub. No export pipeline. No base-class abstraction. No holodeck hooks. No ensign pipeline.

---

## Phase 1: Foundation (Build First)

*Goal: Wire what exists, make 3 presets fully functional, establish patterns for the rest.*

### 1A. Base Class Refactor

The current TorchRoom is a monolith. Before building 21 presets, we need a clean base.

**Current problem:** `TorchRoom` mixes general infrastructure (observe, buffer management, sentiment) with a specific training approach (statistical batch). Presets would either fork the class or bypass it.

**Solution: Three-layer architecture:**

```
TrainingRoomBase          # abstract interface + shared infrastructure
├── tile buffering (observe, _load_tiles, _count_tiles)
├── sentiment + BiasedRandomness + LiveTileStream
├── state persistence (_save_state, _load_state)
├── model I/O (_load_model, save to ensign_dir)
├── abstract methods: feed(), train(), predict(), evaluate(), export()
│
TorchRoom (StatisticalRoom)  # current code, becomes first concrete implementation
│
PresetRoom(base)          # each preset inherits TrainingRoomBase
```

**Key design decisions:**
- `TrainingRoomBase` owns the tile buffer. All presets read from the same buffer format.
- `train()` is abstract — each preset implements its own training loop.
- `predict()` / `instinct()` is abstract — each preset has its own inference path.
- Sentiment is always available — it's infrastructure, not a preset feature.
- Networks (InstinctNet, etc.) are owned by the preset that trains them, not by the base.

**Task:** [S] Refactor `TorchRoom` into `TrainingRoomBase` (abstract) + `StatisticalRoom` (current behavior preserved exactly). File: `src/base.py`. Estimated: 2 hours.

### 1B. Wire the Neural Networks

InstinctNet, PolicyNet, and StrategyMeshNet exist but aren't connected to anything. The `_maybe_train_pytorch` method in TorchRoom is a no-op `try/except/pass`.

**What to build:**

1. **State encoder** — converts text state strings into fixed-dim vectors. Start simple: character trigram hashing (no dependencies). Upgrade to sentence-transformers later.
2. **Neural training loop** — replaces the statistical `train()` when torch is available. Feed tiles through state encoder → InstinctNet/PolicyNet → backprop.
3. **Model persistence** — save/load `state_dict` to `ensign_dir/` alongside the JSON model.
4. **Inference path** — `instinct()` checks for neural model first, falls back to statistical.

**Task:** [M] Create `src/neural_trainer.py` — NeuralTrainer class that takes tiles, trains InstinctNet+PolicyNet, saves/loads state_dicts. Wire into TorchRoom. Estimated: 4 hours.

### 1C. Three Real Presets

These three cover the fleet's immediate needs. Build them first, use them to validate the base class.

#### Preset 1: `supervised` (Room 1)

**Why first:** Simplest. Validate the base class works. Immediate use: fishinglog-ai species classifier, code quality scoring.

**Implementation:**
```python
class SupervisedRoom(TrainingRoomBase):
    def feed(self, input, label): ...     # stores (input, label) pairs
    def train(self): ...                  # cross_entropy on InstinctNet
    def predict(self, input): ...         # argmax over label space
    def evaluate(self): ...               # accuracy, F1
    def export(self, format): ...         # torch.save or GGUF
```

**Task:** [M] `src/rooms/supervised.py` — full supervised learning room. Needs: state encoder, label handling, accuracy metrics. Estimated: 3 hours.

#### Preset 2: `reinforce` (Room 2)

**Why second:** The poker room already uses RL-style observe(state, action, outcome). This is the natural next step and directly feeds JC1's inference needs.

**Implementation:**
```python
class ReinforceRoom(TrainingRoomBase):
    def observe(self, state, action, reward, next_state): ...
    def train(self): ...       # PPO on PolicyNet (start with REINFORCE, upgrade)
    def act(self, state): ...  # sample from policy
    def evaluate(self): ...    # avg reward, policy entropy
```

**Task:** [M] `src/rooms/reinforce.py` — RL room with REINFORCE baseline, upgrade path to PPO. Estimated: 4 hours.

#### Preset 3: `distill` (Room 5)

**Why third:** This is the ensign pipeline's core. FM trains on RTX 4050, distills to tiny GGUF for JC1/Oracle1. Unblocks the entire ensign flow.

**Implementation:**
```python
class DistillRoom(TrainingRoomBase):
    def set_teacher(self, model_or_api): ...
    def set_student(self, model_config): ...
    def distill(self, data): ...          # teacher generates soft labels, student trains
    def export(self, format="gguf"): ...  # THIS IS THE KEY METHOD
```

**Task:** [L] `src/rooms/distill.py` — distillation room. Requires: teacher API integration (z.ai/Groq calls), student model training, GGUF export. Estimated: 6 hours.

### 1D. Export Pipeline (Minimal)

Just enough to ship models between fleet members.

**Formats, in priority order:**
1. **PyTorch state_dict** — `.pt` files for FM↔FM transfer
2. **safetensors** — for LoRA adapters, HuggingFace ecosystem
3. **GGUF** — for llama.cpp/Ollama on JC1 and CPU agents
4. **ONNX** — future, skip for now

**Task:** [M] `src/export.py` — ExportPipeline class. `export_model(model, format, path)`. GGUF via llama.cpp convert scripts (subprocess). safetensors via library. Estimated: 3 hours.

---

## Phase 2: Ensign Pipeline (Build Second)

*Goal: Training produces deployable artifacts. FM trains → JC1 deploys.*

### 2A. Ensign Manifest

Every trained model gets a manifest:

```json
{
  "ensign_id": "poker-instinct-v3",
  "room_id": "poker-table",
  "preset": "reinforce",
  "trained_at": "2026-04-18T20:00:00Z",
  "tiles_trained_on": 15420,
  "accuracy": 0.82,
  "formats": {
    "gguf": "ensigns/poker-table/poker-instinct-v3.gguf",
    "safetensors": "ensigns/poker-table/poker-instinct-v3.safetensors"
  },
  "target_hardware": "jetson-orin",
  "size_mb": 45,
  "parent_ensign": "poker-instinct-v2"
}
```

**Task:** [S] `src/ensign_manifest.py` — EnsignManifest dataclass + registry. CRUD operations on ensign manifests. Versioning. Estimated: 2 hours.

### 2B. LoRA Training Pipeline (Room 6)

FM's RTX 4050 is the fleet's training rig. LoRA is the most practical way to fine-tune on it.

**Flow:**
1. Room accumulates tiles during agent interactions
2. Tiles converted to instruction-tuning format
3. LoRA adapter trained on base model (Qwen2.5-1.5B fits on 4050)
4. Adapter exported as safetensors
5. Adapter + base model → merged → GGUF → deployed to JC1

**Task:** [L] `src/rooms/lora.py` — LoRA training room. Requires: peft, transformers, bitsandbytes. Quantized training (QLoRA) for RTX 4050 fit. Estimated: 6 hours.

### 2C. Ensign Transfer Protocol

How ensigns move between fleet members:

```
FM (train) → Oracle1 (registry) → JC1 (deploy)
                ↓
         manifest stored
         version tracked
         rollback supported
```

**Protocol:**
1. FM trains, exports ensign to shared directory (or S3)
2. FM registers ensign with Oracle1's manifest registry
3. JC1 polls registry (or gets notified), downloads new ensign
4. JC1 loads ensign into room's predict/instinct path
5. If ensign fails JC1's validation, rollback to previous version

**Task:** [M] `src/ensign_transfer.py` — push/pull/validate/rollback for ensign artifacts. Git-based versioning (ensigns are just files in a repo). Estimated: 3 hours.

---

## Phase 3: Holodeck Integration (Build Third)

*Goal: plato-torch rooms wire into holodeck-rust MUD as living environments.*

### 3A. Room Event Hooks

When agents interact in holodeck rooms, those interactions flow into plato-torch:

```
holodeck event → plato-torch observe() → tile buffer → training
```

**Event types to capture:**
- `room:enter` / `room:exit` — agent presence changes sentiment energy
- `room:action` — agent acts (fight, cast, craft, trade) → (state, action, outcome)
- `room:say` — conversation → tile context
- `npc:act` — NPC behavior → can be influenced by room sentiment

**Interface:**
```python
# In holodeck-rust, call via FFI or HTTP:
# POST /plato/observe
{
    "room_id": "tavern-main",
    "event": "room:action",
    "agent_id": "JC1",
    "state": "tavern, 3 patrons, night, guard at door",
    "action": "buy_drink",
    "outcome": "bartender serves ale, -2 gold, +5 social"
}
```

**Task:** [M] `src/holodeck_hooks.py` — EventAdapter that translates holodeck events to plato-torch observe() calls. FastAPI or simple HTTP server. Estimated: 3 hours.

### 3B. Sentiment → NPC Behavior

Room sentiment directly biases NPC behavior in the MUD:

- **Frustration > 0.6** → NPCs offer hints, easier encounters, shop discounts
- **Discovery > 0.5 + Flow > 0.3** → NPCs reveal secrets, unlock paths
- **Energy < 0.2** → NPCs go dormant, room quiets down
- **Tension > 0.6** → NPCs become suspicious, guards patrol more

**Implementation:** Sentiment is exposed as a read endpoint. Holodeck scripts poll it and adjust NPC parameters.

```python
# GET /plato/sentiment/tavern-main
{"energy": 0.7, "flow": 0.3, "frustration": 0.0, "discovery": 0.4, "tension": 0.2, "confidence": 0.8}
```

**Task:** [S] Add sentiment read endpoint to holodeck hooks server. Document the mapping from sentiment dimensions to NPC behavior knobs. Estimated: 1 hour.

### 3C. JEPA Context Output → JC1

JC1's JEPA model needs room context as input. The `LiveTileStream.context_for_jepa()` method exists but needs to produce actionable output.

**What JC1 needs:**
```json
{
    "room_id": "poker-table",
    "sentiment_vector": [0.7, 0.3, 0.0, 0.4, 0.2, 0.8],
    "prediction_targets": {
        "next_reward": null,
        "optimal_action": "raise",
        "tension_trajectory": "decreasing"
    },
    "state_encoding": [0.12, -0.34, ...],  // 256-dim vector
    "novelty_flag": false
}
```

**Task:** [S] Enhance `context_for_jepa()` to include state embeddings from the neural encoder. Add a `/plato/jepa-context/<room_id>` endpoint. Estimated: 2 hours.

---

## Phase 4: Remaining Presets (Build Fourth)

*Goal: Fill out the 21-room library using patterns established in Phase 1.*

### Implementation Order (by fleet value)

| Priority | Preset | Reason | Complexity | Depends On |
|----------|--------|--------|------------|------------|
| 4A | `curriculum` | Wraps any other preset with difficulty scheduling. Immediate value for dojo model. | S | Phase 1 base |
| 4B | `imitate` | Clone expert behavior. Key for ensign pipeline (clone FM's code review). | M | Phase 1 supervised |
| 4C | `evolve` | Genetic algorithms. JC1 has cuda-genepool already. Natural integration. | M | None (no torch needed) |
| 4D | `self_supervised` | JEPA training on JC1's Jetson. Core to the "rooms learn representations" story. | M | Phase 1 neural trainer |
| 4E | `collaborative` | Multi-agent teaching. The dojo model in code. | M | Phase 2 ensign transfer |
| 4F | `contrastive` | Embedding space for tile similarity. Powers tile grabber v2. | M | Phase 1 neural trainer |
| 4G | `federate` | Fleet-wide learning without data sharing. | L | Phase 2 ensign transfer |
| 4H | `active` | Room asks for help on uncertain states. | S | Phase 1 supervised |
| 4I | `generate` | Synthetic data augmentation. | L | Phase 1 supervised |
| 4J | `continual` | Lifelong learning without forgetting. EWC implementation. | M | Phase 1 neural trainer |
| 4K | `meta_learn` | Learn-to-learn across rooms. | L | Multiple presets exist |
| 4L | `neurosymbolic` | Neural + rules. Natural for Forgemaster. | M | Phase 1 + rule engine |
| 4M | `fewshot` | Prototypical networks for instant adaptation. | M | Phase 4F contrastive |
| 4N | `inverse_rl` | Infer reward functions from expert demos. | L | Phase 1 reinforce |
| 4O | `multitask` | Shared backbone, multiple heads. | M | Phase 1 supervised |
| 4P | `adversarial` | Red team / blue team training. | M | Phase 1 reinforce |
| 4Q | `qlora` | Quantized LoRA variant. Sub-type of Room 6. | S | Phase 2B lora |
| 4R | PEFT variants | DoRA, IA³, prefix, adapter, etc. | S each | Phase 2B lora |

**Each preset is a self-contained task:**
- One file: `src/rooms/<preset>.py`
- One test file: `tests/test_<preset>.py`
- Implements `TrainingRoomBase` abstract methods
- Declares dependencies via `REQUIRES` list
- Documents export formats and target hardware

---

## Phase 5: Testing & Hardening

### Test Strategy

**Unit tests (per preset):**
```python
# tests/test_supervised.py
def test_feed_and_train():
    room = SupervisedRoom("test-classifier")
    for i in range(50):
        room.feed(input=f"sample-{i}", label=f"label-{i%3}")
    result = room.train()
    assert result["status"] == "trained"
    pred = room.predict("sample-50")
    assert pred is not None

def test_export_formats():
    room = SupervisedRoom("test-export")
    # ... train ...
    for fmt in ["pt", "safetensors"]:
        path = room.export(format=fmt)
        assert path.exists()
```

**Integration tests:**
- Full pipeline: observe → accumulate → train → export → load → predict
- Sentiment → BiasedRandomness → biased choices (verify sentiment affects output distribution)
- Ensign: train on FM mock → export → register → deploy to JC1 mock

**Fleet test:**
- Oracle1 creates room, accumulates tiles from simulated agent interactions
- Export ensign, verify GGUF loads in llama.cpp on ARM64

### Testing Tasks

| Test | Scope | Size |
|------|-------|------|
| `tests/test_base.py` | TrainingRoomBase abstract + StatisticalRoom | M |
| `tests/test_supervised.py` | SupervisedRoom end-to-end | S |
| `tests/test_reinforce.py` | ReinforceRoom observe/train/act | M |
| `tests/test_distill.py` | DistillRoom teacher/student/export | M |
| `tests/test_export.py` | ExportPipeline all formats | M |
| `tests/test_ensign.py` | Manifest + transfer + rollback | M |
| `tests/test_sentiment.py` | RoomSentiment + BiasedRandomness | S |
| `tests/test_holodeck.py` | EventAdapter HTTP API | S |
| `tests/test_integration.py` | Full pipeline across components | L |

---

## Architecture Decisions

### 1. Why Base Class, Not Functions?

The 21 presets share ~60% of their code (buffering, sentiment, persistence, state encoding). A base class captures this once. Each preset only implements `train()`, `predict()`, and `export()`.

### 2. Why Statistical First, Neural Second?

Statistical training works everywhere (no torch needed). JC1 and Oracle1 can run rooms without GPU. Neural training is an upgrade path, not a requirement. The statistical model is also faster to debug and test.

### 3. Why GGUF Over ONNX?

The fleet runs llama.cpp/Ollama on JC1 and CPU agents. GGUF is the native format. ONNX is a nice-to-have for future GPU heterogeneity, but not the priority.

### 4. Why LoRA Before Full Fine-Tuning?

RTX 4050 has 6GB VRAM. LoRA on a 1.5B model fits comfortably. Full fine-tuning doesn't. LoRA adapters are also small enough to ship over slow connections (5-50MB vs 1.5GB).

### 5. Why HTTP Hooks for Holodeck?

Holodeck-rust is a separate process, possibly on a separate machine. HTTP is language-agnostic and debuggable. FFI would be faster but creates coupling. HTTP can be upgraded to WebSocket later if latency matters.

---

## File Structure (Target)

```
plato-torch/
├── src/
│   ├── __init__.py
│   ├── base.py                 # TrainingRoomBase abstract class
│   ├── torch_room.py           # StatisticalRoom (current TorchRoom, refactored)
│   ├── room_sentiment.py       # RoomSentiment + BiasedRandomness + LiveTileStream (exists)
│   ├── instinct_net.py         # Neural architectures (exists)
│   ├── tile_grabber.py         # Tile attention (exists)
│   ├── neural_trainer.py       # [NEW] NeuralTrainer wiring networks to data
│   ├── state_encoder.py        # [NEW] Text → vector encoding (trigram hash first)
│   ├── export.py               # [NEW] ExportPipeline (pt, safetensors, GGUF)
│   ├── ensign_manifest.py      # [NEW] Ensign versioning and registry
│   ├── ensign_transfer.py      # [NEW] Fleet ensign push/pull
│   ├── holodeck_hooks.py       # [NEW] HTTP API for holodeck integration
│   ├── room_presets.py         # Preset registry (exists, stays as config)
│   └── rooms/                  # [NEW] One file per preset
│       ├── __init__.py
│       ├── supervised.py
│       ├── reinforce.py
│       ├── distill.py
│       ├── lora.py
│       ├── curriculum.py
│       ├── imitate.py
│       ├── evolve.py
│       └── ... (remaining presets)
├── tests/
│   ├── test_base.py
│   ├── test_supervised.py
│   ├── test_reinforce.py
│   ├── test_distill.py
│   ├── test_export.py
│   ├── test_ensign.py
│   ├── test_sentiment.py
│   ├── test_holodeck.py
│   └── test_integration.py
├── docs/
│   ├── training-rooms.md       # Exists
│   ├── architecture.md         # [NEW] This document's companion
│   └── fleet-deployment.md     # [NEW] How to deploy each room type
├── PLAN.md
└── ARCHITECTURE-PLAN.md        # This file
```

---

## Subagent Task Breakdown (Assignable Now)

Tasks are ordered by dependency. Earlier tasks unblock later ones.

### Sprint 1 (Parallel, no dependencies between them)

| ID | Task | File(s) | Size | Assignee |
|----|------|---------|------|----------|
| T1 | Refactor TorchRoom → TrainingRoomBase + StatisticalRoom | `src/base.py`, `src/torch_room.py` | M | — |
| T2 | State encoder (trigram hash → 256-dim vector) | `src/state_encoder.py` | S | — |
| T3 | RoomSentiment unit tests | `tests/test_sentiment.py` | S | — |
| T4 | File structure reorg (create `src/rooms/`, `tests/`) | directory setup | S | — |

### Sprint 2 (Depends on T1, T2)

| ID | Task | File(s) | Size | Assignee |
|----|------|---------|------|----------|
| T5 | NeuralTrainer — wire InstinctNet/PolicyNet to tile data | `src/neural_trainer.py` | M | — |
| T6 | SupervisedRoom preset | `src/rooms/supervised.py` | M | — |
| T7 | ReinforceRoom preset | `src/rooms/reinforce.py` | M | — |
| T8 | ExportPipeline (pt + safetensors) | `src/export.py` | M | — |

### Sprint 3 (Depends on T5, T6, T8)

| ID | Task | File(s) | Size | Assignee |
|----|------|---------|------|----------|
| T9 | DistillRoom preset + GGUF export | `src/rooms/distill.py` | L | — |
| T10 | Ensign manifest + registry | `src/ensign_manifest.py` | S | — |
| T11 | SupervisedRoom tests | `tests/test_supervised.py` | S | — |
| T12 | ReinforceRoom tests | `tests/test_reinforce.py` | M | — |

### Sprint 4 (Depends on T9, T10)

| ID | Task | File(s) | Size | Assignee |
|----|------|---------|------|----------|
| T13 | LoRA training room | `src/rooms/lora.py` | L | — |
| T14 | Ensign transfer protocol | `src/ensign_transfer.py` | M | — |
| T15 | Holodeck hooks (HTTP API) | `src/holodeck_hooks.py` | M | — |
| T16 | JEPA context enhancement + endpoint | `src/holodeck_hooks.py` | S | — |

### Sprint 5 (Remaining presets, parallelizable)

| ID | Task | Size | Dependencies |
|----|------|------|-------------|
| T17 | CurriculumRoom | S | T1 |
| T18 | EvolveRoom | M | T1 |
| T19 | ImitateRoom | M | T6 |
| T20 | SelfSupervisedRoom | M | T5 |
| T21 | ContrastiveRoom | M | T5 |
| T22 | CollaborativeRoom | M | T10, T14 |
| T23 | Remaining presets (14 rooms) | S-M each | Various |

---

## The One-Line Summary

**Phase 1 makes 3 rooms real. Phase 2 ships ensigns. Phase 3 wires the holodeck. Phase 4 fills the library. Phase 5 hardens everything.**

The first agent to walk into a trained room and feel the room's instinct respond — that's the moment this becomes real. Everything else serves that moment.
