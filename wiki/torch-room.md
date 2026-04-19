# TorchRoom — The Training Engine

## Why
TorchRoom wraps RoomBase with actual training mechanics: observe→train→simulate→instinct.
This is the engine that makes rooms learn from experience.

## Pipeline
1. **Observe**: Agent acts in room → tile recorded with state, action, outcome, reward
2. **Train**: Accumulated tiles → train_step() → model update
3. **Simulate**: Trained model → predict() → simulated outcomes for evaluation
4. **Instinct**: Compressed wisdom → export_model() → ensign package

## Key Classes
- `LiveTileStream`: Generates tiles during live play, hand-in-glove with agent actions
- `RoomSentiment`: 6-dimensional mood tracker (energy, flow, frustration, discovery, tension, confidence)
- `BiasedRandomness`: Steers stochastic elements toward productive exploration
- `TileGrabber`: Learned attention over room tile space

## Line Reference
- L1-50: TorchRoom constructor, parameter setup
- L51-100: LiveTileStream — tile generation during interaction
- L101-150: RoomSentiment — 6D mood computation with EMA
- L151-200: BiasedRandomness — sentiment-aware noise injection
- L201-250: TileGrabber — attention-weighted tile selection
- L251-300: train() — full training loop with curriculum
- L301-350: simulate() — generate outcomes from trained model
- L351-400: instinct() — compress training into instinct net
