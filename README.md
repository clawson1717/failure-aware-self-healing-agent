# Failure-Aware Self-Healing Agent (FASHA)

An agent framework that doesn't just detect and recover from reasoning failures—it *diagnoses* them. When a failure occurs, the system performs root-cause analysis to understand *why* it happened, selects an appropriate recovery strategy based on the diagnosis, and learns from the episode to prevent similar failures in the future.

## Problem Solved

Existing recovery systems (ATR, Flow-Heal, Watchdog) patch symptoms without understanding causes. FASHA adds a diagnostic layer: **failure classification → root-cause inference → strategy selection → pattern learning**. Think "doctor who diagnoses before prescribing" rather than "firefighter who just puts out flames."

## Architecture

```
Failure Event → Diagnostic Engine → Root Cause → Recovery Strategy → Verification
                    ↓                                ↓
              Investigation               Pattern Memory (learns)
```

## Implementation Roadmap

- [x] **Step 1:** Project Scaffold ✅
- [x] **Step 2:** Failure Event Record ✅
- [x] **Step 3:** Diagnostic Engine ✅
- [ ] **Step 4:** Recovery Strategy Library
- [ ] **Step 5:** Self-Healing Loop
- [ ] **Step 6:** Failure Pattern Memory
- [ ] **Step 7:** Confidence Tracker
- [ ] **Step 8:** Mock Agent Interface
- [ ] **Step 9:** CLI Interface
- [ ] **Step 10:** Benchmark Suite
- [ ] **Step 11:** Integration Tests
- [ ] **Step 12:** Documentation & Demo

## Installation

```bash
pip install -e .
```

## Usage

```python
from fasha import FailureEvent
from fasha.diagnostic import DiagnosticEngine
from datetime import datetime

# Create a failure event
event = FailureEvent(
    timestamp=datetime.now(),
    failure_type="knowledge_gap",
    context="Model attempted to answer a question about an unseen domain",
    session_id="session-001",
    outcome="failed"
)

# Diagnose the root cause
engine = DiagnosticEngine()
result = engine.diagnose(event)

print(f"Root cause: {result.root_cause}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Evidence: {result.evidence}")
print(f"Investigation hints: {result.investigation_hints}")
```

## Root Cause Categories

1. **Knowledge Gap** — Missing information in training data
2. **Reasoning Error** — Logical flaw in reasoning chain
3. **Context Loss** — Forgot important prior context
4. **Strategy Mismatch** — Wrong approach for the problem type
5. **External Failure** — Environment or external dependency issue

## Comparison with Related Projects

| Project | Focus | FASHA Advantage |
|---------|-------|----------------|
| ATR | Detection + Pruning | Adds diagnosis + strategy selection |
| Flow-Heal | Recovery | Adds root-cause understanding |
| Watchdog | Monitoring | Adds actionable recovery + learning |
| Reasoning-Watchdog | Quality monitoring | Adds self-healing with diagnosis |

## Key Innovations

1. **Diagnostic layer** — First system to classify root-causes, not just detect symptoms
2. **Strategy selection** — Recovery method chosen based on diagnosis, not one-size-fits-all
3. **Pattern learning** — Historical failures inform future prevention

## Prior Art

- Box Maze (arXiv:2603.19182) — process-control boundary enforcement
- Uncertainty Estimation Scaling (arXiv:2603.19118) — hybrid 2-sample AUROC
- OS-Themis (arXiv:2603.19191) — multi-agent critic with milestone decomposition
