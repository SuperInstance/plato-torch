"""PLATO Room Preset Registry — 21 training methods, same API."""

PRESET_REGISTRY = {
    # Classic ML
    "supervised":       {"class": "SupervisedRoom",       "file": "supervised.py",
                         "desc": "Labeled input→output via frequency counting"},
    "contrastive":      {"class": "ContrastiveRoom",      "file": "contrastive.py",
                         "desc": "Cosine similarity, triplet margin learning"},
    "self_supervised":  {"class": "SelfSupervisedRoom",   "file": "self_supervised.py",
                         "desc": "JEPA-style masked prediction (Welford online)"},

    # Reinforcement
    "reinforce":        {"class": "ReinforceRoom",        "file": "reinforce.py",
                         "desc": "Policy gradient, Monte Carlo returns"},
    "inverse_rl":       {"class": "InverseRLRoom",        "file": "inverse_rl.py",
                         "desc": "Observe expert, infer reward function"},
    "imitate":          {"class": "ImitateRoom",          "file": "imitate.py",
                         "desc": "Clone expert behavior from demonstrations"},

    # Efficient Tuning
    "lora":             {"class": "LoRARoom",             "file": "lora_train.py",
                         "desc": "PEFT delta table simulation"},
    "qlora":            {"class": "QLoRARoom",            "file": "qlora.py",
                         "desc": "4-bit quantized base + LoRA delta adapters"},

    # Population Methods
    "evolve":           {"class": "EvolveRoom",           "file": "evolve.py",
                         "desc": "Genetic algorithm, tournament selection"},
    "adversarial":      {"class": "AdversarialRoom",      "file": "adversarial.py",
                         "desc": "Red team vs blue team attack tracking"},
    "collaborative":    {"class": "CollaborativeRoom",    "file": "collaborative.py",
                         "desc": "Multi-agent knowledge sharing, majority vote"},

    # Meta / Federated
    "meta_learn":       {"class": "MetaLearnRoom",        "file": "meta_learn.py",
                         "desc": "Nearest-task fast adaptation (1-3 shot)"},
    "federate":         {"class": "FederateRoom",         "file": "federate.py",
                         "desc": "Federated averaging across agents"},
    "multitask":        {"class": "MultitaskRoom",        "file": "multitask.py",
                         "desc": "Shared backbone + task-specific heads"},

    # Lifecycle
    "curriculum":       {"class": "CurriculumRoom",       "file": "curriculum.py",
                         "desc": "Easy first, then harder (dojo progression)"},
    "continual":        {"class": "ContinualRoom",        "file": "continual.py",
                         "desc": "Lifelong learning, EWC-inspired replay buffer"},
    "fewshot":          {"class": "FewshotRoom",          "file": "fewshot.py",
                         "desc": "Prototype matching from 1-5 examples"},
    "active":           {"class": "ActiveRoom",           "file": "active.py",
                         "desc": "Model chooses what data to learn from"},

    # Generative
    "generate":         {"class": "GenerateRoom",         "file": "generate.py",
                         "desc": "N-gram data augmentation, synthetic state generation"},

    # Hybrid
    "neurosymbolic":    {"class": "NeurosymbolicRoom",    "file": "neurosymbolic.py",
                         "desc": "Neural instinct + symbolic rules blend"},
    "distill":          {"class": "DistillRoom",          "file": "distill.py",
                         "desc": "Teacher→student with temperature scaling"},
}

# Quick lookup
def get_preset(name: str):
    """Import and return the preset class by name."""
    from presets import PRESET_MAP
    return PRESET_MAP.get(name)

def list_presets():
    """Return all preset names and descriptions."""
    return {k: v["desc"] for k, v in PRESET_REGISTRY.items()}

if __name__ == "__main__":
    for name, info in sorted(PRESET_REGISTRY.items()):
        print(f"  {name:20s} — {info['desc']}")
    print(f"\n{len(PRESET_REGISTRY)} presets registered")
