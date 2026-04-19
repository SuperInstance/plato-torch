# Training Presets — 22 Grab-and-Go Rooms

## Why
Every AI training method should be accessible as a walk-into-room experience.
Same API: feed() → train_step() → predict() → export_model().

## Preset Map
| Preset | Method | Key Feature |
|--------|--------|-------------|
| supervised | Labeled learning | Error-driven weight updates |
| reinforce | Policy gradient | Monte Carlo returns, epsilon-greedy |
| evolve | Genetic algorithm | Tournament selection, 50-genome pop |
| distill | Teacher→Student | Temperature scaling, KL divergence |
| contrastive | Comparison learning | Similarity/difference pairs |
| self_supervised | JEPA | Predict missing from context |
| lora | Low-rank adaptation | Rank-k weight decomposition |
| meta_learn | Learn to learn | Task-distribution training |
| federate | Distributed | Cross-room knowledge averaging |
| generate | Generative | Pattern synthesis |
| adversarial | GAN-style | Generator vs discriminator |
| collaborative | Multi-agent | Shared reward optimization |
| active | Strategic queries | Uncertainty-based sampling |
| curriculum | Easy→hard | Progressive difficulty scheduling |
| imitate | Behavioral cloning | Expert trajectory matching |
| neurosymbolic | Neural + symbolic | Hybrid reasoning |
| continual | Lifelong learning | Catastrophic forgetting prevention |
| fewshot | 3-5 examples | Rapid adaptation |
| inverse_rl | Reward inference | Learn from observed behavior |
| multitask | Multi-objective | Shared representation learning |
| qlora | Quantized LoRA | 4-bit quantization + LoRA |
| wiki | Knowledge compile | Big→wiki→cheap model pipeline |

## Registry
All presets registered in `src/presets/__init__.py` as `PRESET_MAP`.
Import: `from presets import PRESET_MAP`

## Bug Notes
- Subagents always use `name` not `room_id` — fixed post-delivery
- Subagents add numpy dependency — stripped to pure Python
- Must copy `room_base.py` into `presets/` for pip install relative imports
