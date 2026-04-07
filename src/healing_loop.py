"""Self-Healing Loop for FASHA.

Implements the core state machine that orchestrates failure detection, diagnosis,
recovery strategy selection, execution, and verification. The loop processes
FailureEvents through a well-defined state machine:

    HEALTHY → FAILING → DIAGNOSING → RECOVERING → VERIFYING → HEALTHY

Each transition corresponds to a phase of the self-healing process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from failure_event import FailureEvent
from diagnostic import DiagnosticEngine, DiagnosisResult, RootCause
from recovery import RecoveryStrategyLibrary, RecoveryResult


class HealingState(str, Enum):
    """States of the self-healing loop state machine."""

    HEALTHY = "healthy"
    FAILING = "failing"
    DIAGNOSING = "diagnosing"
    RECOVERING = "recovering"
    VERIFYING = "verifying"


@dataclass
class HealingRecord:
    """Record of a single healing attempt.

    Attributes:
        event: The original failure event that triggered healing.
        diagnosis: The diagnosis result from the diagnostic engine.
        strategy_name: Name of the recovery strategy selected.
        recovery_result: Result of executing the recovery strategy.
        healed: Whether the healing attempt was ultimately successful.
        attempt_number: Which attempt this was (1-based) for this event.
        timestamp: When this healing record was created.
    """

    event: FailureEvent
    diagnosis: Optional[DiagnosisResult] = None
    strategy_name: Optional[str] = None
    recovery_result: Optional[RecoveryResult] = None
    healed: bool = False
    attempt_number: int = 1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert the healing record to a dictionary."""
        return {
            "event": self.event.to_dict(),
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
            "strategy_name": self.strategy_name,
            "recovery_result": self.recovery_result.to_dict() if self.recovery_result else None,
            "healed": self.healed,
            "attempt_number": self.attempt_number,
            "timestamp": self.timestamp.isoformat(),
        }


class SelfHealingLoop:
    """Orchestrates the detect → diagnose → select → execute → verify → learn cycle.

    The loop maintains a state machine that transitions through well-defined
    phases when processing failure events. It tracks history and statistics
    across all healing attempts.

    Args:
        max_recovery_attempts: Maximum retries per event before giving up.
        confidence_threshold: Minimum diagnosis confidence to proceed with recovery.
            Below this threshold the loop will select the Escalate strategy.
        verification_timeout: Seconds allowed for the verification phase (reserved
            for future async verification).
        diagnostic_engine: Optional pre-configured DiagnosticEngine instance.
        strategy_library: Optional pre-configured RecoveryStrategyLibrary instance.

    Example:
        >>> from datetime import datetime, timezone
        >>> from failure_event import FailureEvent
        >>> loop = SelfHealingLoop()
        >>> event = FailureEvent(
        ...     timestamp=datetime.now(timezone.utc),
        ...     failure_type="context_loss",
        ...     context="Agent forgot earlier instructions",
        ...     session_id="s1",
        ...     outcome="failed",
        ... )
        >>> loop.step(event)
        >>> loop.get_state()
        <HealingState.HEALTHY: 'healthy'>
    """

    def __init__(
        self,
        max_recovery_attempts: int = 3,
        confidence_threshold: float = 0.4,
        verification_timeout: float = 30.0,
        diagnostic_engine: Optional[DiagnosticEngine] = None,
        strategy_library: Optional[RecoveryStrategyLibrary] = None,
    ) -> None:
        self.max_recovery_attempts = max_recovery_attempts
        self.confidence_threshold = confidence_threshold
        self.verification_timeout = verification_timeout

        self._engine = diagnostic_engine or DiagnosticEngine()
        self._library = strategy_library or RecoveryStrategyLibrary()

        self._state: HealingState = HealingState.HEALTHY
        self._history: list[HealingRecord] = []
        self._current_record: Optional[HealingRecord] = None
        self._current_attempt: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self) -> HealingState:
        """Return the current state of the healing loop."""
        return self._state

    def step(self, event: FailureEvent) -> HealingRecord:
        """Process a single failure event through the full state machine.

        Transitions: HEALTHY → FAILING → DIAGNOSING → RECOVERING → VERIFYING → HEALTHY.
        If recovery fails, retries up to ``max_recovery_attempts`` times before
        recording a failed healing attempt.

        Args:
            event: The failure event to process.

        Returns:
            The HealingRecord created for this event.
        """
        # Phase 1: Detect — transition to FAILING
        self._state = HealingState.FAILING
        self._current_attempt = 0

        while self._current_attempt < self.max_recovery_attempts:
            self._current_attempt += 1
            record = HealingRecord(event=event, attempt_number=self._current_attempt)

            # Phase 2: Diagnose — transition to DIAGNOSING
            self._state = HealingState.DIAGNOSING
            diagnosis = self._engine.diagnose(event)
            record.diagnosis = diagnosis

            # Phase 3: Select strategy — transition to RECOVERING
            self._state = HealingState.RECOVERING
            if diagnosis.confidence < self.confidence_threshold:
                # Low confidence — escalate
                strategy = self._library.get_strategy("escalate")
            else:
                strategy = self._library.select_strategy(diagnosis)
            record.strategy_name = strategy.name

            # Phase 4: Execute recovery
            context = self._build_recovery_context(event, diagnosis)
            recovery_result = strategy.execute(context)
            record.recovery_result = recovery_result

            # Phase 5: Verify — transition to VERIFYING
            self._state = HealingState.VERIFYING
            verified = self._verify(recovery_result)
            record.healed = verified

            self._history.append(record)

            if verified:
                # Success — return to HEALTHY
                self._state = HealingState.HEALTHY
                return record

        # Exhausted attempts — return to HEALTHY with last (failed) record
        self._state = HealingState.HEALTHY
        return self._history[-1]

    def run(self, events: list[FailureEvent]) -> list[HealingRecord]:
        """Process a batch of failure events sequentially.

        Args:
            events: List of failure events to process.

        Returns:
            List of HealingRecords, one per event (the final attempt's record).
        """
        results: list[HealingRecord] = []
        for event in events:
            record = self.step(event)
            results.append(record)
        return results

    def abort(self) -> None:
        """Force the loop back to HEALTHY and clear transient state."""
        self._state = HealingState.HEALTHY
        self._current_record = None
        self._current_attempt = 0

    def get_history(self) -> list[HealingRecord]:
        """Return the full list of healing records."""
        return list(self._history)

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate recovery statistics.

        Returns:
            Dictionary with keys: total, successful, failed, by_cause.
        """
        total = len(self._history)
        successful = sum(1 for r in self._history if r.healed)
        failed = total - successful

        by_cause: dict[str, dict[str, int]] = {}
        for record in self._history:
            cause = (
                record.diagnosis.root_cause.value
                if record.diagnosis
                else RootCause.UNKNOWN.value
            )
            if cause not in by_cause:
                by_cause[cause] = {"total": 0, "successful": 0, "failed": 0}
            by_cause[cause]["total"] += 1
            if record.healed:
                by_cause[cause]["successful"] += 1
            else:
                by_cause[cause]["failed"] += 1

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "by_cause": by_cause,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recovery_context(
        event: FailureEvent, diagnosis: DiagnosisResult
    ) -> dict[str, Any]:
        """Build the context dict handed to a recovery strategy's execute()."""
        return {
            "failure_type": event.failure_type,
            "session_id": event.session_id,
            "evidence": diagnosis.evidence,
            "confidence": diagnosis.confidence,
            "timestamp": event.timestamp.isoformat(),
            "context": event.context,
            # Provide minimal session-like data so strategies that need
            # history / checkpoints can still run (they degrade gracefully).
            "session_history": [event.context],
            "last_context": event.context,
        }

    @staticmethod
    def _verify(result: RecoveryResult) -> bool:
        """Verify a recovery result.

        The current implementation treats the strategy's own success flag as
        the verification signal.  A future version could run domain-specific
        checks or wait for external confirmation.
        """
        return result.success
