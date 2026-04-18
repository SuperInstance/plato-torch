# PLATO-Torch Implementation Plan

## Priority Order (what to build first)

### Phase 1: Core Infrastructure (this session)
1. **Base class refactor** — abstract RoomBase that all 21 presets inherit from
2. **Reinforce preset** — the poker room needs RL, this is the most immediate use case
3. **Evolve preset** — pure Python, no deps, maps directly to JC1's cuda-genepool
4. **Distill preset** — needed for ensign export pipeline

### Phase 2: PEFT + Integration (next session)
5. **LoRA/QLoRA preset** — FM's training work
6. **Holodeck-rust hooks** — room events → tile generation
7. **Ensign export pipeline** — train → export → register in plato-ensign

### Phase 3: Advanced Presets
8. **Curriculum** — dojo-style progressive difficulty
9. **Self-supervised (JEPA)** — JC1's lane
10. **Neurosymbolic** — constraint-theory + neural hybrid (Forgemaster's lane)

## Subagent Task Breakdown

| # | Task | Size | Agent | File |
|---|------|------|-------|------|
| 1 | Base class refactor | M | subagent | room_base.py + torch_room.py refactor |
| 2 | Reinforce preset | L | subagent | presets/reinforce.py |
| 3 | Evolve preset | M | subagent | presets/evolve.py |
| 4 | Distill preset | L | subagent | presets/distill.py |
| 5 | Holodeck event hooks | S | Oracle1 | hooks into holodeck-rust |
| 6 | Tests for all presets | M | subagent | tests/test_presets.py |
