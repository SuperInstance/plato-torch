"""
Training Room Registry — grab-and-go training presets.

Every training method is a preset. Walk into the room with the right
preset, feed data, train, export. The room handles the complexity.

Usage:
    room = TorchRoom("my-room", preset="reinforce")
    room.observe(state="...", action="...", outcome="...")
    room.train()
    result = room.instinct("what should I do here?")
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class TrainingPreset:
    """Configuration for a training room type."""
    name: str
    description: str
    training_paradigm: str
    requires: List[str] = field(default_factory=list)
    default_params: Dict[str, Any] = field(default_factory=dict)
    reward_type: str = "inferred"  # "inferred", "explicit", "self", "adversarial"
    data_flow: str = "batch"       # "batch", "stream", "episode", "generation"
    supports_simulation: bool = True
    supports_sentiment: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "gguf", "safetensors"])


PRESETS: Dict[str, TrainingPreset] = {
    # ── The 20 Training Rooms ──────────────────────────────
    
    "supervised": TrainingPreset(
        name="Supervised Learning",
        description="Learn from labeled input→output pairs. Classic classification/regression.",
        training_paradigm="supervised",
        requires=["torch"],
        default_params={"loss": "cross_entropy", "optimizer": "adam", "lr": 0.001},
        reward_type="explicit",
        data_flow="batch",
    ),
    
    "reinforce": TrainingPreset(
        name="Reinforcement Learning",
        description="Learn from reward signals. State→Action→Reward→Next State loop.",
        training_paradigm="reinforcement",
        requires=["torch"],
        default_params={"algorithm": "PPO", "gamma": 0.99, "clip_ratio": 0.2},
        reward_type="inferred",
        data_flow="episode",
    ),
    
    "self_supervised": TrainingPreset(
        name="Self-Supervised / JEPA",
        description="Learn representations by predicting missing parts. No labels needed.",
        training_paradigm="self_supervised",
        requires=["torch"],
        default_params={"method": "jepa", "mask_ratio": 0.5, "latent_dim": 256},
        reward_type="self",
        data_flow="batch",
    ),
    
    "contrastive": TrainingPreset(
        name="Contrastive Learning",
        description="Learn by contrasting similar vs dissimilar pairs.",
        training_paradigm="contrastive",
        requires=["torch"],
        default_params={"loss": "InfoNCE", "temperature": 0.07, "embedding_dim": 128},
        reward_type="self",
        data_flow="batch",
    ),
    
    "distill": TrainingPreset(
        name="Knowledge Distillation",
        description="Compress a big teacher into a tiny student.",
        training_paradigm="distillation",
        requires=["torch", "transformers"],
        default_params={"temperature": 4.0, "alpha": 0.7, "student_model": "Qwen2.5-0.5B"},
        reward_type="self",
        data_flow="batch",
        export_formats=["gguf", "onnx", "safetensors"],
    ),
    
    "lora": TrainingPreset(
        name="LoRA / PEFT Fine-Tuning",
        description="Freeze base model, train small adapter weights.",
        training_paradigm="peft",
        requires=["torch", "peft", "transformers"],
        default_params={"method": "lora", "rank": 16, "alpha": 32, "target_modules": ["q_proj", "v_proj"]},
        reward_type="inferred",
        data_flow="batch",
        export_formats=["safetensors"],
    ),
    
    "qlora": TrainingPreset(
        name="QLoRA (Quantized LoRA)",
        description="4-bit quantized base + LoRA. Train big models on consumer GPUs.",
        training_paradigm="peft",
        requires=["torch", "peft", "transformers", "bitsandbytes"],
        default_params={"quantization": "4bit", "rank": 16, "alpha": 32},
        reward_type="inferred",
        data_flow="batch",
        export_formats=["safetensors"],
    ),
    
    "evolve": TrainingPreset(
        name="Evolutionary / Genetic",
        description="Evolve a population through selection, mutation, crossover.",
        training_paradigm="evolutionary",
        requires=[],
        default_params={"population_size": 50, "mutation_rate": 0.1, "crossover_rate": 0.7, "generations": 100},
        reward_type="explicit",
        data_flow="generation",
    ),
    
    "meta_learn": TrainingPreset(
        name="Meta-Learning (MAML)",
        description="Learn to learn. Adapt to new tasks in 1-3 gradient steps.",
        training_paradigm="meta",
        requires=["torch"],
        default_params={"method": "MAML", "inner_lr": 0.01, "outer_lr": 0.001, "n_shots": 5},
        reward_type="explicit",
        data_flow="episode",
    ),
    
    "federate": TrainingPreset(
        name="Federated Learning",
        description="Train across agents without sharing raw data. Privacy-preserving.",
        training_paradigm="federated",
        requires=["torch"],
        default_params={"aggregation": "fedavg", "min_agents": 2, "rounds": 50},
        reward_type="self",
        data_flow="batch",
    ),
    
    "active": TrainingPreset(
        name="Active Learning",
        description="Model chooses which data to learn from. Asks for labels on uncertain cases.",
        training_paradigm="active",
        requires=["torch", "sklearn"],
        default_params={"strategy": "uncertainty", "query_size": 10, "budget": 100},
        reward_type="explicit",
        data_flow="batch",
    ),
    
    "generate": TrainingPreset(
        name="Generative (GAN / Diffusion)",
        description="Generate synthetic training data that looks real.",
        training_paradigm="generative",
        requires=["torch"],
        default_params={"method": "diffusion", "steps": 1000, "guidance_scale": 7.5},
        reward_type="adversarial",
        data_flow="generation",
    ),
    
    "curriculum": TrainingPreset(
        name="Curriculum Learning",
        description="Easy examples first, then progressively harder. Like a dojo.",
        training_paradigm="curriculum",
        requires=["torch"],
        default_params={
            "stages": [
                {"difficulty": "easy", "threshold": 0.9},
                {"difficulty": "medium", "threshold": 0.8},
                {"difficulty": "hard", "threshold": 0.7},
            ]
        },
        reward_type="inferred",
        data_flow="batch",
    ),
    
    "imitate": TrainingPreset(
        name="Imitation Learning",
        description="Clone expert behavior from demonstrations.",
        training_paradigm="imitation",
        requires=["torch"],
        default_params={"method": "behavioral_cloning", "dagger": False},
        reward_type="explicit",
        data_flow="episode",
    ),
    
    "inverse_rl": TrainingPreset(
        name="Inverse Reinforcement Learning",
        description="Observe expert, infer their reward function.",
        training_paradigm="inverse_rl",
        requires=["torch"],
        default_params={"method": "max_entropy_irl", "episodes": 100},
        reward_type="self",
        data_flow="episode",
    ),
    
    "multitask": TrainingPreset(
        name="Multi-Task Learning",
        description="One model, multiple related tasks. Shared backbone, task-specific heads.",
        training_paradigm="multitask",
        requires=["torch"],
        default_params={"tasks": [], "shared_layers": 4, "task_layers": 2},
        reward_type="explicit",
        data_flow="batch",
    ),
    
    "continual": TrainingPreset(
        name="Continual / Lifelong Learning",
        description="Learn forever without forgetting. Resist catastrophic forgetting.",
        training_paradigm="continual",
        requires=["torch"],
        default_params={"method": "EWC", "ewc_lambda": 0.4, "memory_size": 1000},
        reward_type="inferred",
        data_flow="stream",
    ),
    
    "fewshot": TrainingPreset(
        name="Few-Shot / Zero-Shot",
        description="Adapt to new tasks from 1-5 examples. Generalize to unknowns.",
        training_paradigm="fewshot",
        requires=["torch"],
        default_params={"method": "prototypical", "n_shots": 5, "n_queries": 15},
        reward_type="explicit",
        data_flow="batch",
    ),
    
    "neurosymbolic": TrainingPreset(
        name="Neurosymbolic Hybrid",
        description="Neural instinct + symbolic rules. Best of both worlds.",
        training_paradigm="hybrid",
        requires=["torch"],
        default_params={"neural_weight": 0.6, "rule_weight": 0.4, "rules": []},
        reward_type="inferred",
        data_flow="stream",
    ),
    
    "adversarial": TrainingPreset(
        name="Adversarial Training",
        description="Red team vs blue team. Attack and defend simultaneously.",
        training_paradigm="adversarial",
        requires=["torch"],
        default_params={"attack_steps": 10, "epsilon": 0.03, "pgd": True},
        reward_type="adversarial",
        data_flow="episode",
    ),
    
    "collaborative": TrainingPreset(
        name="Collaborative Multi-Agent",
        description="Agents teach each other. Knowledge sharing across the fleet.",
        training_paradigm="collaborative",
        requires=["torch"],
        default_params={"n_agents": 3, "share_interval": 10, "distillation": True},
        reward_type="self",
        data_flow="episode",
    ),
}

# PEFT method variants (Room 6 sub-types)
PEFT_METHODS = {
    "lora": {"rank": 16, "alpha": 32, "target": ["q_proj", "v_proj"]},
    "qlora": {"rank": 16, "alpha": 32, "quant": "4bit"},
    "dora": {"rank": 16, "alpha": 32, "decompose": True},
    "ia3": {"target": ["k_proj", "v_proj", "mlp"]},
    "prefix": {"prefix_length": 20},
    "prompt": {"n_virtual_tokens": 20},
    "adapter": {"bottleneck_size": 64},
    "bitfit": {},  # only bias terms
    "loha": {"rank": 8, "alpha": 16},
    "lokr": {"rank": 8, "alpha": 16, "decomposition": "kronecker"},
    "x-lora": {"n_experts": 4, "rank": 8},
}


def list_presets() -> List[Dict]:
    """List all available training presets."""
    return [
        {
            "id": key,
            "name": p.name,
            "description": p.description,
            "requires": p.requires,
            "supports_sim": p.supports_simulation,
        }
        for key, p in PRESETS.items()
    ]


def get_preset(preset_id: str) -> Optional[TrainingPreset]:
    """Get a specific preset configuration."""
    return PRESETS.get(preset_id)


def check_dependencies(preset_id: str) -> Dict[str, bool]:
    """Check which dependencies are available for a preset."""
    preset = PRESETS.get(preset_id)
    if not preset:
        return {}
    
    available = {}
    for dep in preset.requires:
        try:
            __import__(dep)
            available[dep] = True
        except ImportError:
            available[dep] = False
    
    return available


if __name__ == "__main__":
    import json
    
    print("PLATO Training Room Registry")
    print("=" * 50)
    
    presets = list_presets()
    for p in presets:
        print(f"\n  {p['id']:15s} — {p['name']}")
        print(f"  {'':15s}   {p['description'][:60]}")
        if p['requires']:
            print(f"  {'':15s}   requires: {', '.join(p['requires'])}")
    
    print(f"\n  Total: {len(presets)} training room types")
    print(f"  PEFT variants: {len(PEFT_METHODS)} methods")
    
    # Check what's available
    print("\nDependency check:")
    for key in ["torch", "sklearn", "transformers", "peft", "bitsandbytes", "deap"]:
        try:
            __import__(key)
            print(f"  ✅ {key}")
        except ImportError:
            print(f"  ❌ {key}")
