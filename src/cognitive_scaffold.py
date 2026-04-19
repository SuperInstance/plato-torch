"""
Cognitive Scaffold — rooms that teach agents how to think.

Based on JC1's insight: "The room is a cognitive scaffold. It's not passive space.
It's an active participant in thinking."

ref: wiki/cognitive-scaffold.md — four scaffold types, stage enforcement, shaping prompts
"""

Three scaffold types:
- LogicScaffold: teaches causality, rigor, step-by-step reasoning
- CreativeScaffold: teaches metaphor, association, lateral thinking
- DebugScaffold: teaches causality tracing, hypothesis testing, verification

Each scaffold wraps a room preset and adds cognitive shaping: assertions that
reject invalid outputs, state machines that enforce reasoning steps, and
word anchors that focus attention.
"""

import json
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Any
from enum import Enum


class ScaffoldType(Enum):
    LOGIC = "logic"
    CREATIVE = "creative"
    DEBUG = "debug"
    TRAINING = "training"


class CognitiveScaffold:
    """A room that actively shapes agent thinking through assertions, 
    state machines, and word anchors."""
    
    def __init__(self, scaffold_type: ScaffoldType, room_id: str = ""):
        self.room_id = room_id or scaffold_type.value
        self.scaffold_type = scaffold_type
        self.state = "initial"
        self.assertion_log = []
        self.state_history = []
        self.episodes = []  # past successes/failures
        self.word_anchors = []
        self._transition_count = 0
        self._rejection_count = 0
        
        # Configure based on type
        self._setup_scaffold()
        self.state = self.states[0]  # Start at first defined state
    
    def _setup_scaffold(self):
        """Configure assertions, state machine, and anchors for the scaffold type."""
        if self.scaffold_type == ScaffoldType.LOGIC:
            self.states = ["PREMISE", "REASONING", "CONCLUSION", "VERIFIED"]
            self.assertions = [
                "Every claim must be supported by evidence",
                "Logical fallacies are rejected (ad hominem, straw man, false dichotomy)",
                "Conclusions must follow from premises",
            ]
            self.word_anchors = ["therefore", "because", "implies", "if-then", "proven"]
            
        elif self.scaffold_type == ScaffoldType.CREATIVE:
            self.states = ["INSPIRATION", "EXPLORATION", "SYNTHESIS", "EXPRESSION"]
            self.assertions = [
                "No idea is rejected in the exploration phase",
                "Metaphor and analogy are encouraged",
                "Synthesis must combine at least 2 distinct concepts",
            ]
            self.word_anchors = ["like", "imagine", "what if", "suppose", "analogous"]
            
        elif self.scaffold_type == ScaffoldType.DEBUG:
            self.states = ["IDENTIFY", "REPRODUCE", "DIAGNOSE", "FIX", "VERIFY"]
            self.assertions = [
                "Every bug report must include reproduction steps",
                "Root cause must be identified before fixing",
                "Fix must be verified with a test case",
                "No jumping to FIX without REPRODUCE",
            ]
            self.word_anchors = ["reproduce", "stack_trace", "root_cause", "verify", "test"]
            
        elif self.scaffold_type == ScaffoldType.TRAINING:
            self.states = ["DEMO", "PRACTICE", "ASSESS", "MASTER"]
            self.assertions = [
                "Demonstrate before practicing",
                "Assessment before mastery",
                "Celebrate small wins",
            ]
            self.word_anchors = ["watch", "try", "show me", "again", "got it"]
    
    def validate(self, agent_output: str, current_state: str = None) -> Dict:
        """Validate agent output against scaffold assertions.
        
        Returns validation result with rejection reason if any assertion fails.
        """
        state = current_state or self.state
        rejections = []
        
        # Check state machine
        if state in self.states:
            state_idx = self.states.index(state)
            
            # Logic: must have evidence for claims
            if self.scaffold_type == ScaffoldType.LOGIC:
                if state == "CONCLUSION" and "therefore" not in agent_output.lower() and "because" not in agent_output.lower():
                    rejections.append("Logic scaffold: conclusion must follow from reasoning (use 'therefore' or 'because')")
            
            # Debug: can't jump to FIX without REPRODUCE
            if self.scaffold_type == ScaffoldType.DEBUG:
                if state == "IDENTIFY" and ("fix" in agent_output.lower() or "solution" in agent_output.lower()):
                    rejections.append("Debug scaffold: must REPRODUCE before FIX. Don't jump ahead.")
            
            # Creative: synthesis must combine concepts
            if self.scaffold_type == ScaffoldType.CREATIVE:
                if state == "SYNTHESIS":
                    concepts = [w for w in agent_output.lower().split() if len(w) > 4]
                    if len(set(concepts)) < 2:
                        rejections.append("Creative scaffold: synthesis must combine at least 2 distinct concepts")
            
            # Check word anchors — reward outputs that use them
            anchors_used = [a for a in self.word_anchors if a.lower() in agent_output.lower()]
        
        result = {
            "valid": len(rejections) == 0,
            "rejections": rejections,
            "state": state,
            "scaffold": self.scaffold_type.value,
            "anchors_used": anchors_used if 'anchors_used' in dir() else [],
        }
        
        if not result["valid"]:
            self._rejection_count += 1
            self.assertion_log.append({
                "state": state, "rejection": rejections[0],
                "output_preview": agent_output[:100]
            })
        
        return result
    
    def advance(self, agent_output: str = "") -> Dict:
        """Try to advance to the next state in the state machine."""
        current_idx = self.states.index(self.state) if self.state in self.states else 0
        
        # Validation before advancing
        validation = self.validate(agent_output)
        if not validation["valid"]:
            return {
                "advanced": False,
                "state": self.state,
                "validation": validation,
                "message": f"Cannot advance: {validation['rejections'][0]}"
            }
        
        if current_idx < len(self.states) - 1:
            old_state = self.state
            self.state = self.states[current_idx + 1]
            self._transition_count += 1
            self.state_history.append((old_state, self.state))
            return {
                "advanced": True,
                "from": old_state,
                "to": self.state,
                "message": f"Transitioned: {old_state} → {self.state}"
            }
        
        return {"advanced": False, "state": self.state, "message": "Already at final state"}
    
    def record_episode(self, success: bool, context: str):
        """Record a success/failure episode for future learning."""
        self.episodes.append({
            "state": self.state,
            "success": success,
            "context": context[:200],
            "transition_count": self._transition_count,
        })
    
    def get_scaffold_prompt(self) -> str:
        """Generate a system prompt that embeds the scaffold's cognitive shaping."""
        assertions_text = "\n".join(f"- {a}" for a in self.assertions)
        anchors_text = ", ".join(self.word_anchors)
        
        return f"""You are in a {self.scaffold_type.value} cognitive scaffold room.

Current reasoning stage: {self.state}
Available stages: {' → '.join(self.states)}

Assertions (you MUST follow these):
{assertions_text}

Word anchors (using these shows you're thinking in the right direction):
{anchors_text}

Past episodes in this room: {len(self.episodes)} ({sum(1 for e in self.episodes if e['success'])} successful)

The room is teaching you how to think. Follow the stages. Respect the assertions. Use the anchors."""
    
    def export(self) -> Dict:
        """Export scaffold state for persistence."""
        return {
            "room_id": self.room_id,
            "scaffold_type": self.scaffold_type.value,
            "state": self.state,
            "states": self.states,
            "assertions": self.assertions,
            "word_anchors": self.word_anchors,
            "episodes": len(self.episodes),
            "transitions": self._transition_count,
            "rejections": self._rejection_count,
        }


if __name__ == "__main__":
    # Demo: logic scaffold
    print("=== Logic Scaffold ===")
    logic = CognitiveScaffold(ScaffoldType.LOGIC, "logic-room")
    print(logic.get_scaffold_prompt()[:200])
    
    # Bad output (no reasoning)
    result = logic.validate("The answer is 42.")
    print(f"Bad: valid={result['valid']} rejection={result.get('rejections', [])}")
    
    # Good output (with reasoning)
    result = logic.validate("Because all men are mortal and Socrates is a man, therefore Socrates is mortal.")
    print(f"Good: valid={result['valid']}")
    
    # Advance
    adv = logic.advance("Premise: all men are mortal. Evidence: every observed human has died.")
    print(f"Advance: {adv}")
    
    # Debug scaffold
    print("\n=== Debug Scaffold ===")
    debug = CognitiveScaffold(ScaffoldType.DEBUG, "debug-room")
    
    # Bad: try to fix without reproducing
    result = debug.validate("Just restart the server, that usually fixes it.")
    print(f"Jump to fix: valid={result['valid']} rejection={result.get('rejections', [])}")
    
    # Good: identify first
    result = debug.validate("I see a 500 error when hitting /api/users. Let me check the logs.")
    print(f"Identify: valid={result['valid']}")
    adv = debug.advance("I see a 500 error when hitting /api/users. Error trace shows null pointer at line 42.")
    print(f"Advance: {adv}")
    
    # Creative scaffold
    print("\n=== Creative Scaffold ===")
    creative = CognitiveScaffold(ScaffoldType.CREATIVE, "creative-room")
    print(f"States: {' → '.join(creative.states)}")
    result = creative.validate("What if we imagine the data flowing like a river through the system?")
    print(f"Creative: valid={result['valid']} anchors={result.get('anchors_used', [])}")
    
    # Export
    print(f"\nExport: {json.dumps(logic.export(), indent=2)}")
