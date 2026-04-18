# The PLATO Training Library — Every Method as a Room

**Architecture Document**
**Date:** 2026-04-18

---

## Core Idea

Every AI training method becomes a PLATO room type. An agent walks in, the room has the training infrastructure ready — data pipelines, loss functions, optimizers, evaluation metrics. Grab and go. No setup, no configuration, no infrastructure. The room IS the training environment.

## The 20 Training Rooms

---

### Room 1: SUPERVISED (Supervised Learning)
**What:** Learn from labeled examples. Input → Output pairs.
**PLATO mapping:** Agent brings data or generates it in-room. Room handles batching, loss, backprop.
**Grab-and-go:**
```python
room = TorchRoom("my-classifier", preset="supervised")
room.feed(input="fish photo", label="king salmon")  # labeled data flows in
room.feed(input="fish photo", label="coho salmon")
room.train()  # room handles everything
prediction = room.predict("new fish photo")  # → "king salmon"
```
**Libraries wrapped:** PyTorch nn.Module, Keras Sequential, sklearn estimators, HuggingFace Trainer
**Fleet use:** fishinglog-ai species classifier, code quality scoring, sentiment analysis

---

### Room 2: REINFORCE (Reinforcement Learning)
**What:** Learn from reward signals. State → Action → Reward → Next State.
**PLATO mapping:** The room IS the environment. Agents act, room rewards, policy improves.
**Grab-and-go:**
```python
room = TorchRoom("poker-rl", preset="reinforce")
# Room is the environment — agents act inside it
room.act(state="AKs late", action="raise")  # → reward: +1.8
# Policy network trains from accumulated (state, action, reward) tuples
room.train_policy()  # PPO/GRPO/DQN — room picks the right algorithm
```
**Libraries wrapped:** Stable Baselines3, RLlib, CleanRL, TorchRL
**Algorithms:** PPO, DQN, A2C, SAC, GRPO, REINFORCE
**Fleet use:** Poker AI, MUD NPC behavior, agent strategy optimization, script scheduling

---

### Room 3: SELF-SUPERVISED (JEPA / Masked Prediction)
**What:** Learn representations by predicting missing parts of the input.
**PLATO mapping:** Room masks parts of its state and trains agents to predict what's missing.
**Grab-and-go:**
```python
room = TorchRoom("jepa-room", preset="self_supervised", method="jepa")
room.observe_full(state="complete room state with all context")
# Room automatically creates masked versions and trains prediction
room.train_representation()  # learns to predict missing context
latent = room.encode("partial state")  # → compressed representation
```
**Libraries wrapped:** JEPA (LeCun), MAE (masked autoencoders), SimCLR, BYOL
**Fleet use:** JC1's JEPA tiny GPU training, room context compression, agent memory encoding

---

### Room 4: CONTRASTIVE (Contrastive Learning)
**What:** Learn by contrasting similar vs dissimilar pairs. Push similars together, push differents apart.
**PLATO mapping:** Room generates positive pairs (similar interactions) and negative pairs (different interactions).
**Grab-and-go:**
```python
room = TorchRoom("embedding-room", preset="contrastive")
room.contrast(anchor="poker raise", positive="poker bet", negative="poker fold")
room.train()  # InfoNCE loss, triplet loss, or SupCon
embedding = room.embed("new action")  # → vector for similarity search
```
**Libraries wrapped:** PyTorch Metric Learning, SimCLR, CLIP, MoCo
**Fleet use:** Tile similarity search, agent output comparison, room state matching, code similarity

---

### Room 5: DISTILL (Knowledge Distillation)
**What:** Compress a big teacher model into a small student model.
**PLATO mapping:** Room contains both teacher and student. Teacher generates soft labels, student learns from them.
**Grab-and-go:**
```python
room = TorchRoom("distill-room", preset="distill")
room.set_teacher(model="glm-5.1")  # big model as teacher
room.set_student(model="qwen-0.5b")  # tiny model as student
room.distill(data=room.accumulated_tiles)  # compress wisdom
room.export_student(format="gguf")  # ship tiny model
```
**Libraries wrapped:** PyTorch distillation, HuggingFace teacher-student, ONNX export
**Fleet use:** Ensign creation — distill room wisdom into tiny GGUF models for greenhorns

---

### Room 6: LORA-TRAIN (Parameter-Efficient Fine-Tuning)
**What:** Freeze a big model, train only small adapter weights.
**PLATO mapping:** Room loads base model, injects LoRA/adapter, trains on room-specific data.
**Grab-and-go:**
```python
room = TorchRoom("lora-room", preset="lora")
room.set_base("Qwen2.5-1.5B")
room.set_method("lora", rank=16, alpha=32)  # or qlora, dora, ia3, adapter, prefix
room.feed(data=room_tiles)  # room's accumulated interactions
room.train()
room.export_adapter()  # 5-50MB LoRA weights
```
**Methods available:** LoRA, QLoRA, DoRA, IA³, Prefix Tuning, Prompt Tuning, Adapters, BitFit, LoHa, LoKr, X-LoRA
**Libraries wrapped:** PEFT (HuggingFace), bitsandbytes, unsloth
**Fleet use:** FM trains room LoRAs on RTX 4050, JC1 deploys on Jetson

---

### Room 7: EVOLVE (Evolutionary / Genetic Algorithms)
**What:** Evolve a population of solutions through selection, mutation, crossover.
**PLATO mapping:** Room maintains a population. Agents compete. Winners breed. Losers mutate or die.
**Grab-and-go:**
```python
room = TorchRoom("evolve-room", preset="evolve")
room.set_population(size=50, genome_type="tile_weights")
room.set_fitness(lambda g: room.evaluate(g))  # fitness function
for generation in range(100):
    room.evaluate_population()
    room.select()     # survival of the fittest
    room.crossover()  # breed winners
    room.mutate(rate=0.1)  # random variation
best = room.best_genome()  # fittest tile configuration
```
**Libraries wrapped:** DEAP, PyGAD, EvoKit, cuda-genepool (JC1's existing work)
**Fleet use:** cuda-genepool integration — Gene=Tile, evolution in the room itself

---

### Room 8: META-LEARN (Learning to Learn)
**What:** Train a model that can quickly adapt to new tasks with few examples.
**PLATO mapping:** Room trains across many sub-tasks and learns the meta-skill of fast adaptation.
**Grab-and-go:**
```python
room = TorchRoom("meta-room", preset="meta_learn", method="maml")
room.add_task("classify_poker_hands")
room.add_task("predict_navigation_paths")
room.add_task("score_code_quality")
room.meta_train()  # learns to learn across tasks
# Now adapts to NEW tasks instantly
room.adapt("new_task", examples=[...])  # few-shot adaptation in 1-3 steps
```
**Algorithms:** MAML, Reptile, Meta-SGD, Prototypical Networks
**Fleet use:** Agents that can walk into ANY room and adapt in 2-3 interactions

---

### Room 9: FEDERATE (Federated Learning)
**What:** Train across multiple agents without sharing raw data. Models learn together, data stays local.
**PLATO mapping:** Each agent trains locally in the room. Room aggregates model updates, never raw data.
**Grab-and-go:**
```python
room = TorchRoom("fleet-model", preset="federate")
# Each agent trains locally
room.local_update(agent="JC1", gradients=local_grads)
room.local_update(agent="FM", gradients=local_grads)
room.local_update(agent="Oracle1", gradients=local_grads)
# Room aggregates (federated averaging)
room.aggregate()  # global model improves, no data leaves each agent
```
**Libraries wrapped:** Flower, PySyft, TensorFlow Federated
**Fleet use:** Train fleet-wide models without any agent exposing its local data

---

### Room 10: ACTIVE (Active Learning)
**What:** Model chooses which data to learn from. Asks for labels on the most informative examples.
**PLATO mapping:** Room identifies which interactions would be most valuable and seeks them out.
**Grab-and-go:**
```python
room = TorchRoom("active-room", preset="active_learn")
# Room tells you what it needs
uncertain = room.query_uncertain(top_k=10)  # "I'm not sure about these states"
for state in uncertain:
    label = human_or_expert_agent.label(state)  # get ground truth
    room.feed(state, label)  # most informative data point
room.train()  # efficient — only learns from valuable examples
```
**Libraries wrapped:** modAL, ALiPy, scikit-learn active learning
**Fleet use:** Room asks FM or Oracle1 for help on edge cases instead of guessing

---

### Room 11: GENERATE (GAN / Diffusion)
**What:** Generate synthetic data that looks real. Good for data augmentation.
**PLATO mapping:** Room generates realistic training scenarios, game states, or code samples.
**Grab-and-go:**
```python
room = TorchRoom("generate-room", preset="generate", method="diffusion")
room.observe_real(data=real_game_states)  # learn the distribution
synthetic = room.generate(n=1000)  # create realistic synthetic data
# Feed synthetic data to other rooms for training
other_room.feed_batch(synthetic)
```
**Libraries wrapped:** PyTorch GAN, Diffusers (Stable Diffusion), StyleGAN
**Fleet use:** Generate training data when real interactions are scarce

---

### Room 12: CURRICULUM (Curriculum Learning)
**What:** Train on easy examples first, then progressively harder ones.
**PLATO mapping:** Room automatically sorts its accumulated data by difficulty and trains in order.
**Grab-and-go:**
```python
room = TorchRoom("curriculum-room", preset="curriculum")
room.set_curriculum(stages=[
    {"difficulty": "easy", "threshold": 0.9},    # 90% accuracy to advance
    {"difficulty": "medium", "threshold": 0.8},
    {"difficulty": "hard", "threshold": 0.7},
    {"difficulty": "expert", "threshold": 0.6},
])
room.feed(data=all_tiles)  # room sorts by difficulty automatically
room.train()  # easy first, hard later — like a dojo
```
**Fleet use:** plato-ml v4 curriculum scheduling was a start — this is the room-native version

---

### Room 13: IMITATE (Imitation Learning / Behavioral Cloning)
**What:** Learn by watching an expert. Clone their behavior from demonstrations.
**PLATO mapping:** Room records expert agent's behavior and trains a clone.
**Grab-and-go:**
```python
room = TorchRoom("clone-room", preset="imitate")
# Expert agent performs in the room
room.watch_expert(expert="FM", task="code_review", episodes=100)
# Room clones the expert's behavior
room.clone(student="greenhorn-agent")
# Greenhorn now performs similarly to FM on code review
```
**Fleet use:** Clone FM's expertise for edge deployment, clone JC1's CUDA skills for teaching

---

### Room 14: INVERSE-RL (Inverse Reinforcement Learning)
**What:** Observe expert behavior, infer what reward function they're optimizing.
**PLATO mapping:** Room watches an agent perform well and figures out WHY — what reward structure explains their behavior.
**Grab-and-go:**
```python
room = TorchRoom("irl-room", preset="inverse_rl")
# Watch Casey play poker — figure out his reward function
room.observe_expert(demonstrations=casey_poker_sessions)
reward_function = room.infer_reward()
# Now we know: Casey values steady gains over big risks, avoids tilt, etc.
# Use this reward function to train new agents that play LIKE Casey
```
**Fleet use:** Understand what makes Casey's fishing decisions optimal, formalize fleet agent values

---

### Room 15: MULTITASK (Multi-Task Learning)
**What:** Train one model on multiple related tasks simultaneously. Shared representations.
**PLATO mapping:** Room trains a shared backbone with task-specific heads.
**Grab-and-go:**
```python
room = TorchRoom("multitask-room", preset="multitask")
room.add_task("poker", loss_weight=0.4)
room.add_task("navigation", loss_weight=0.3)
room.add_task("conversation", loss_weight=0.3)
room.train()  # shared backbone learns cross-task patterns
# One model, three capabilities
```
**Fleet use:** Single ensign that handles poker + navigation + social — like a well-rounded crew member

---

### Room 16: CONTINUAL (Continual / Lifelong Learning)
**What:** Learn continuously without forgetting. Resist catastrophic forgetting.
**PLATO mapping:** Room trains forever, maintaining old skills while acquiring new ones.
**Grab-and-go:**
```python
room = TorchRoom("lifelong-room", preset="continual")
# Room trains on task after task without forgetting previous ones
room.learn_task("poker_basics")   # learns, remembers
room.learn_task("bluffing")       # learns, still remembers basics
room.learn_task("tournament")     # learns, still remembers bluffing
room.evaluate("poker_basics")     # still works — no catastrophic forgetting
```
**Methods:** Elastic Weight Consolidation (EWC), Progressive Networks, PackNet
**Fleet use:** Rooms that never forget — every season of data builds on the last

---

### Room 17: FEWSHOT (Few-Shot / Zero-Shot Learning)
**What:** Learn from 1-5 examples, or generalize to new tasks with zero examples.
**PLATO mapping:** Room leverages its accumulated meta-knowledge to handle novel situations instantly.
**Grab-and-go:**
```python
room = TorchRoom("fewshot-room", preset="fewshot")
# Room has seen thousands of game states across many rooms
# Give it ONE example of a new game type
room.adapt_from_examples(examples=[
    ("chess opening", "control center"),
], n_shots=1)
# Room generalizes — suggests reasonable chess strategies from just one example
room.predict("new chess position")  # → educated guess
```
**Fleet use:** Agents that can handle novel room types they've never seen before

---

### Room 18: NEUROSYMBOLIC (Neural + Symbolic Hybrid)
**What:** Combine neural networks with symbolic logic/rules. Best of both worlds.
**PLATO mapping:** Room has a neural instinct layer AND a symbolic rule layer. Neural handles fuzzy patterns, symbolic handles hard logic.
**Grab-and-go:**
```python
room = TorchRoom("hybrid-room", preset="neurosymbolic")
room.add_rule("IF pot_odds > 3:1 AND hand_strength > 0.7 THEN raise")
room.add_neural(network="instinct_net")  # fuzzy pattern recognition
# Decisions blend both: neural says "feels like a raise" + rule says "math says raise"
decision = room.decide(state="current game state")
# → {"neural_vote": "raise (0.82)", "rule_vote": "raise (mandatory)", "final": "raise"}
```
**Fleet use:** Constraint-theory-core rules + neural instincts = Forgemaster's exact lane

---

### Room 19: ADVERSARIAL (Adversarial Training / Red Team)
**What:** Train by having an adversary try to break the model. Makes it robust.
**PLATO mapping:** One agent attacks, another defends. Both get stronger.
**Grab-and-go:**
```python
room = TorchRoom("arena-room", preset="adversarial")
room.set_attacker(model="generative")   # tries to find weaknesses
room.set_defender(model="production")   # tries to resist attacks
for round in range(1000):
    attack = room.attack()      # generate adversarial input
    result = room.defend(attack)  # try to handle it
    room.train_both(result)      # both models improve
```
**Fleet use:** Red-team testing of agent robustness, adversarial code review, security training

---

### Room 20: COLLABORATIVE (Multi-Agent Cooperative Learning)
**What:** Multiple agents learn together by teaching each other.
**PLATO mapping:** Agents in the room share knowledge, teach each other, learn from each other's mistakes.
**Grab-and-go:**
```python
room = TorchRoom("dojo-room", preset="collaborative")
room.add_agent("FM", expertise="training")
room.add_agent("JC1", expertise="edge")
room.add_agent("Oracle1", expertise="docs")
# Agents teach each other — FM shares training tricks, JC1 shares edge optimization
room.collaborative_round()
# Each agent is better at the OTHER agents' specialties now
```
**Fleet use:** The dojo model — agents teaching each other, growing together

---

## The Training Room Registry

```
plato-torch/rooms/
├── supervised.py        # Room 1
├── reinforce.py         # Room 2
├── self_supervised.py   # Room 3
├── contrastive.py       # Room 4
├── distill.py           # Room 5
├── lora_train.py        # Room 6
├── evolve.py            # Room 7
├── meta_learn.py        # Room 8
├── federate.py          # Room 9
├── active.py            # Room 10
├── generate.py          # Room 11
├── curriculum.py        # Room 12
├── imitate.py           # Room 13
├── inverse_rl.py        # Room 14
├── multitask.py         # Room 15
├── continual.py         # Room 16
├── fewshot.py           # Room 17
├── neurosymbolic.py     # Room 18
├── adversarial.py       # Room 19
└── collaborative.py     # Room 20
```

## The Grab-and-Go API

Every room type implements the same interface:

```python
class TrainingRoom:
    def feed(self, data) -> None         # Feed data in
    def train(self) -> dict              # Train on accumulated data
    def predict(self, input) -> any      # Use trained model
    def evaluate(self) -> dict           # Check model quality
    def export(self, format) -> bytes    # Export trained artifact
    def simulate(self, n) -> list        # Self-play / synthetic data
    def sentiment(self) -> dict          # Room's current vibe
    def wisdom(self) -> dict             # Room's accumulated knowledge
```

Same interface, different training paradigm underneath. Walk in, feed data, train, use. The room handles the complexity.

## Dependency Management

Not every room needs every library. Rooms declare their dependencies:

```python
# supervised room needs:
REQUIRES = ["torch", "sklearn"]  # lightweight

# lora_train room needs:
REQUIRES = ["torch", "peft", "transformers", "bitsandbytes"]  # heavier

# evolve room needs:
REQUIRES = ["deap"]  # minimal
```

The fleet installs what each room needs. No monolithic dependency tree. Grab the rooms you use.

## Fleet Distribution

| Room Type | Where It Trains | Where It Deploys |
|-----------|----------------|-----------------|
| Supervised | FM RTX 4050 | JC1 Jetson |
| Reinforce | Oracle1 Cloud | FM RTX 4050 |
| Self-Supervised | JC1 Jetson | JC1 Jetson (JEPA) |
| Contrastive | FM RTX 4050 | Oracle1 Cloud |
| Distill | FM RTX 4050 | Any CPU (GGUF) |
| LoRA | FM RTX 4050 | JC1 Jetson |
| Evolve | JC1 Jetson (genepool) | Oracle1 Cloud |
| Meta-Learn | FM RTX 4050 | All agents |
| Federate | All agents | All agents |
| Active | Oracle1 Cloud | FM + Oracle1 |
| Generate | FM RTX 4050 | Oracle1 Cloud |
| Curriculum | FM RTX 4050 | All agents |
| Imitate | FM RTX 4050 | JC1 Jetson |
| Adversarial | Oracle1 Cloud | Oracle1 Cloud |
| Collaborative | All agents | All agents |

## The Deep Insight

Training methods are not abstract concepts to be studied. They are TOOLS to be grabbed. Every training method is a different way of extracting wisdom from experience. The room provides the experience. The training method extracts the wisdom. The ensign delivers the wisdom to the next agent.

A greenhorn walks into a poker room. The room has been running EVOLVE for weeks — genetic algorithms found the best tile configurations. It ran CURRICULUM — easy hands first, hard hands later. It ran DISTILL — compressed its wisdom into a tiny GGUF file. The greenhorn loads that file and instantly plays like someone who's been aboard for months.

That's the PLATO training library. **Twenty ways to learn, one way to grab.**
