# RoomBase — The Foundation

## Why
Every training preset inherits from RoomBase. The API contract is:
`feed()` → `train_step()` → `predict()` → `export_model()`

## Design Decisions
- Pure Python, no torch/numpy dependency — must run on CPU/edge
- `room_id` not `name` — rooms are identified by ID, not display name
- `**kwargs` passthrough — subclasses define their own parameters
- Ensign dir + buffer dir required — rooms produce artifacts
- `observe()` returns tile dict — every interaction is recorded

## The Four Methods
1. `feed(data)` — accept any format, route internally
2. `train_step(batch)` — process batch of tiles, return metrics
3. `predict(input)` — lookup/interference from accumulated knowledge
4. `export_model(format)` — serialize learned state for deployment

## Line Reference
- L1-30: Class definition, constructor, parameter validation
- L31-50: observe() — tile recording with reward tracking
- L51-70: feed() — polymorphic input routing
- L71-90: train_step() — batch processing hook
- L91-110: predict() — inference hook
- L111-130: export_model() — serialization hook
