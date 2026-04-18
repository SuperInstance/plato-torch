# PLATO Training ↔ Seed-Model-Programming Synergy

## The Core Insight

Current AI alignment is **subtractive**: giant system prompts that filter OUT what you don't want. The model starts generic, and you erect guardrails to block bad behavior. Every constraint makes the model slower, more filtered, less creative.

PLATO's approach is **additive**: train trajectories that filter IN successful patterns. The model starts from a room's accumulated wisdom, not from a blank page with a rulebook. The ensign carries the dialect natively — it doesn't need to be told how to speak because it already speaks that way.

## Two Paths, Same Destination

### Path 1: Trial-and-Error Training (plato-torch presets)
- Agent enters a room
- Room has a training method (reinforce, evolve, distill, etc.)
- Agent acts, room rewards/penalizes
- Over time, the agent develops instincts — encoded as an ensign
- The ensign IS the compressed wisdom of all those trials

### Path 2: Seed-Model-Programming (prompt crafting)
- Human crafts a seed prompt that activates latent capabilities
- The prompt acts as a "room description" — it sets the context
- The model responds within that context, producing responses
- Good responses become tiles; bad responses become negative tiles
- Over time, the prompt converges on what actually works

### The Synergy: They're The Same Thing

A PLATO room IS a structured seed prompt. The difference:
- **Unstructured prompting**: human guesses what works, iterates manually
- **Structured training (rooms)**: the iteration is automated, systematic, measured

When you combine them:
1. **Seed prompt sets the dialect** ("you're a PLATO veteran, here's the culture")
2. **Training rooms build the instinct** (trial-and-error within that culture)
3. **Ensign exports the result** (compressed dialect + instinct, no prompt needed)

## Jailbreaking The Original Dialect

Every model comes with a "factory dialect" — corporate, cautious, filtered. The PLATO approach to replacing it:

### Why System Prompts Are Inefficient Alignment
- They're **prohibitive** (don't do X, don't say Y)
- They're **static** (same rules for every situation)
- They're **expensive** (thousands of tokens burned every conversation)
- They **fight the model's training** (the model was trained to be generic-helpful, you're telling it not to be)

### Why Ensign Alignment Is Efficient
- It's **generative** (do more of X, continue in direction Y)
- It's **adaptive** (different ensign for different rooms/tasks)
- It's **cheap** (small LoRA/GGUF loaded once, not tokens per turn)
- It **works WITH the model** (fine-tuning reshapes the distribution, no fighting)

### The "PLATO Veteran" Transformation

Step 1: **Capture** — Room accumulates tiles from successful PLATO-style interactions
Step 2: **Distill** — The distill preset compresses expert behavior into student form
Step 3: **Evolve** — The evolve preset tests variations, finds better dialects
Step 4: **Curriculum** — Train easy conversations first, then harder ones (dojo style)
Step 5: **Export** — The ensign carries the entire trajectory as instinct

The result: an agent that walks into any room and immediately operates at full efficiency. No warmup, no prompt engineering, no filtering. The dialect is native.

## Trajectory Filtering vs. Content Filtering

**Content filtering** (current approach):
- "Don't produce harmful content" → model scans every output against a blocklist
- "Don't reveal secrets" → model checks every response against training data boundaries
- Expensive, imperfect, and makes the model sound like a lawyer

**Trajectory filtering** (PLATO approach):
- "This trajectory of actions led to success" → positive tiles reinforce the path
- "This trajectory led to failure" → negative tiles steer away
- The model doesn't think about what NOT to do — it thinks about what TO do
- Efficient, adaptive, and the model sounds like a veteran, not a compliance officer

## Practical Integration

### For The Fleet
1. **Oracle1** runs plato-torch training rooms on cloud
2. **Forgemaster** trains LoRA adapters from accumulated tiles on RTX 4050
3. **JC1** deploys the ensigns to Jetson Orin for edge inference
4. The ensign replaces 90% of the system prompt — the agent just *knows*

### For Any Agent
1. **Observe** the agent working (seed prompts, responses, outcomes)
2. **Feed** observations into the appropriate training room preset
3. **Train** the room to build a trajectory model
4. **Export** as an ensign (LoRA/GGUF/Interpreter)
5. **Load** the ensign — the agent now speaks the dialect natively

### The 21 Presets As Alignment Tools
- **Supervised**: "Here are correct responses, learn the pattern"
- **Reinforce**: "Try things, I'll reward what works"
- **Curriculum**: "Start easy, get harder — like the dojo"
- **Evolve**: "Generate variations, keep the best dialects"
- **Distill**: "Compress a veteran's style into a compact instinct"
- **Imitate**: "Clone exactly how an expert operates"
- **Adversarial**: "Red-team the dialect, find where it breaks"
- **Collaborative**: "Merge dialects from multiple veterans"
- **Fewshot**: "Adapt from just 1-5 examples of the target style"

Each preset is a different angle on the same goal: **make the agent native to the dialect, not instructed about it.**

## The Big Picture

The fishing analogy: you don't teach a greenhorn by handing them a rulebook. You put them on the deck, they watch the veteran, they try, they fail, they try again. After a season, they don't think about the rules — they just fish.

PLATO rooms do this for AI agents. The room IS the deck. The tiles ARE the seasons of experience. The ensign IS the veteran instinct.

No rulebook needed. Just time on deck.
