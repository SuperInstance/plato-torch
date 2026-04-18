"""PLATO-Torch training room presets."""

from .reinforce import ReinforceRoom
from .evolve import EvolveRoom
from .distill import DistillRoom

PRESET_MAP = {
    "reinforce": ReinforceRoom,
    "evolve": EvolveRoom,
    "distill": DistillRoom,
}
