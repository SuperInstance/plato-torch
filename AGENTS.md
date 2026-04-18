# AGENTS.md — plato-torch

## What This Is
21 AI training methods as PLATO rooms. Pure Python, same API.

## How To Use
```python
from presets import PRESET_MAP
room = PRESET_MAP["reinforce"]("my-room", ensign_dir="./e", buffer_dir="./t")
room.observe("state", "action", "outcome")
room.train_step(room._load_tiles())
room.predict("state")
```

## Adding A New Preset
1. Create `src/presets/your_name.py`
2. Inherit `RoomBase` with constructor `(self, room_id: str, **kwargs)`
3. Implement: `feed()`, `train_step()`, `predict()`, `export_model()`
4. Add to `__init__.py` PRESET_MAP and `room_presets.py` PRESET_REGISTRY

## Key Files
- `src/room_base.py` — abstract base class
- `src/presets/` — all 21 implementations
- `src/torch_room.py` — full room with sentiment + tiles
- `docs/training-seed-synergy.md` — why training and seed-programming are the same thing

## Gotchas
- Constructor MUST be `(self, room_id: str, **kwargs)` — subagents kept getting this wrong
- No numpy/torch in presets — pure Python only
- `self.name` is wrong, use `self.room_id`
- Health check on holodeck (port 7778) is telnet `nc`, NOT HTTP curl
