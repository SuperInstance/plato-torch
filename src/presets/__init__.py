"""PLATO-Torch training room presets."""

from .reinforce import ReinforceRoom
from .evolve import EvolveRoom
from .distill import DistillRoom
from .active import ActiveRoom
from .curriculum import CurriculumRoom
from .imitate import ImitateRoom
from .neurosymbolic import NeurosymbolicRoom
from .continual import ContinualRoom
from .fewshot import FewshotRoom

PRESET_MAP = {
    "reinforce": ReinforceRoom,
    "evolve": EvolveRoom,
    "distill": DistillRoom,
    "active": ActiveRoom,
    "curriculum": CurriculumRoom,
    "imitate": ImitateRoom,
    "neurosymbolic": NeurosymbolicRoom,
    "continual": ContinualRoom,
    "fewshot": FewshotRoom,
}
