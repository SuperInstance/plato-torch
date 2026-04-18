"""plato-torch — Self-training rooms for PLATO."""
__version__ = "0.5.0a1"

from .presets import PRESET_MAP
from .room_base import RoomBase

__all__ = ["PRESET_MAP", "RoomBase", "__version__"]
