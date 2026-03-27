"""FASHA - Failure-Aware Self-Healing Agent.

A Python library for failure detection, diagnosis, and recovery in AI agents.
"""

from failure_event import FailureEvent
from diagnostic import DiagnosticEngine, DiagnosisResult, RootCause
from recovery import (
    RecoveryResult,
    RecoveryStrategy,
    RecoveryStrategyLibrary,
    ReReadStrategy,
    AskClarificationStrategy,
    BacktrackStrategy,
    RetryDifferentApproachStrategy,
    SimplifyStrategy,
    EscalateStrategy,
)

__all__ = [
    # Failure Event
    "FailureEvent",
    # Diagnostic
    "DiagnosticEngine",
    "DiagnosisResult",
    "RootCause",
    # Recovery
    "RecoveryResult",
    "RecoveryStrategy",
    "RecoveryStrategyLibrary",
    "ReReadStrategy",
    "AskClarificationStrategy",
    "BacktrackStrategy",
    "RetryDifferentApproachStrategy",
    "SimplifyStrategy",
    "EscalateStrategy",
]
