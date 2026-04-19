# Cognitive Scaffolds — Rooms That Teach Thinking

## Why
Based on JC1's "Rooms as Cognitive Scaffolds" paper. Rooms don't just contain agents —
they actively shape HOW agents think by enforcing reasoning state machines.

## Four Scaffold Types
1. **LogicScaffold**: PREMISE → REASONING → CONCLUSION → VERIFIED
   - Rejects fallacies, requires evidence for claims
   - Use for: architecture decisions, code review, debugging

2. **CreativeScaffold**: INSPIRATION → EXPLORATION → SYNTHESIS → EXPRESSION
   - No ideas rejected, metaphor anchors encouraged
   - Use for: brainstorming, design, writing

3. **DebugScaffold**: IDENTIFY → REPRODUCE → DIAGNOSE → FIX → VERIFY
   - Cannot jump to FIX without REPRODUCING first
   - Use for: bug fixing, incident response

4. **TrainingScaffold**: DEMO → PRACTICE → ASSESS → MASTER
   - Dojo-style progression with difficulty scaling
   - Use for: onboarding, skill building, greenhorn training

## Key Mechanism
Each scaffold validates agent output against stage-specific assertions.
If an agent tries to skip stages (e.g., jumping to FIX without REPRODUCE),
the scaffold rejects the output and generates a shaping prompt.

## Line Reference
- L1-50: ScaffoldBase — state machine, transition validation
- L51-100: LogicScaffold — assertion-based reasoning validation
- L101-150: CreativeScaffold — metaphor anchor tracking
- L151-200: DebugScaffold — stage enforcement (no skipping)
- L201-250: TrainingScaffold — difficulty progression
- L251-300: export() — serialize scaffold for room deployment
