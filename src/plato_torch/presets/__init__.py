from .reinforce import ReinforceRoom
from .evolve import EvolveRoom
from .distill import DistillRoom
from .active import ActiveRoom
from .curriculum import CurriculumRoom
from .imitate import ImitateRoom
from .neurosymbolic import NeurosymbolicRoom
from .continual import ContinualRoom
from .fewshot import FewshotRoom
from .supervised import SupervisedRoom
from .contrastive import ContrastiveRoom
from .self_supervised import SelfSupervisedRoom
from .lora_train import LoRARoom
from .meta_learn import MetaLearnRoom
from .federate import FederateRoom
from .inverse_rl import InverseRLRoom
from .multitask import MultitaskRoom
from .qlora import QLoRARoom
from .generate import GenerateRoom
from .adversarial import AdversarialRoom
from .collaborative import CollaborativeRoom

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
    "supervised": SupervisedRoom,
    "contrastive": ContrastiveRoom,
    "self_supervised": SelfSupervisedRoom,
    "lora": LoRARoom,
    "meta_learn": MetaLearnRoom,
    "federate": FederateRoom,
    "inverse_rl": InverseRLRoom,
    "multitask": MultitaskRoom,
    "qlora": QLoRARoom,
    "generate": GenerateRoom,
    "adversarial": AdversarialRoom,
    "collaborative": CollaborativeRoom,
}
