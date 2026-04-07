"""Self-Healing Loop for FASHA.

Orchestrates the full detect -> diagnose -> recover -> verify pipeline
using a state machine with states: HEALTHY, FAILING, DIAGNOSING, RECOVERING, VERIFYING.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from diagnostic import DiagnosticEngine, DiagnosisResult
from failure_event import FailureEvent
from recovery import RecoveryResult, RecoveryStrategyLibrary


class HealingState(str, Enum):
    """States of the self-healing loop state machine."""

    HEALTHY = "healthy"
    FAILING = "failing"
    DIAGNOSING = "diagnosing"
    RECOVERING = "recovering"
    VERIFYING = "verifying"


@dataclass
class HealingRecord:
    """Record of a single healing cycle.

    Attributes:
        event: The failure event that triggered healing.
        diagnosis: The diagnosis result, if diagnosis completed.
        recovery_result: The recovery result, if recovery completed.
        final_state: The state after the healing cycle completed.
    """

    event: FailureEvent
    diagnosis: Optional[DiagnosisResult] = None
    recovery_result: Optional[RecoveryResult] = None
    final_state: HealingState = HealingState.HEALTHY


class SelfHealingLoop:
    """Orchestrates the detect -> diagnose -> recover -> verify pipeline.

    Uses a state machine to track progress through the healing cycle.
    Configurable thresholds control behavior at each phase.

    Example:
        >>> from datetime import datetime, timezone
        >>> from failure_event import FailureEvent
        >>> from diagnostic import DiagnosticEngine
        >>> from recovery import RecoveryStrategyLibrary
        >>> loop = SelfHealingLoop(DiagnosticEngine(), RecoveryStrategyLibrary())
        >>> event = FailureEvent(
        ...     timestamp=datetime.now(timezone.utc),
        ...     failure_type="context_loss",
        ...     context="Agent forgot earlier instructions",
        ...     session_id="s1",
        ...     outcome="failed",
        ... )
        >>> result = loop.step(event)
        >>> result.success
        True
    """

    def __init__(
        self,
        diagnostic_engine: DiagnosticEngine,
        recovery_library: RecoveryStrategyLibrary,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the self-healing loop.

        Args:
            diagnostic_engine: Engine for diagnosing failure root causes.
            recovery_library: Library of recovery strategies.
            config: Optional configuration dict with keys:
                - max_retries (int): Max recovery retries per event (default 3).
                - confidence_threshold (float): Min diagnosis confidence to
                  attempt recovery; below this, escalate (default 0.5).
                - auto_escalate (bool): Whether to auto-escalate on low
                  confidence or repeated failures (default True).
        """
        self._engine = diagnostic_engine
        self._library = recovery_library
        self._config = config or {}
        self._state = HealingState.HEALTHY
        self._history: list[HealingRecord] = []
        self._aborted = False

    @property
    def max_retries(self) -> int:
        return self._config.get("max_retries", 3)

    @property
    def confidence_threshold(self) -> float:
        return self._config.get("confidence_threshold", 0.5)

    @property
    def auto_escalate(self) -> bool:
        return self._config.get("auto_escalate", True)

    def get_state(self) -> HealingState:
        """Return the current state of the healing loop."""
        return self._state

    def get_history(self) -> list[HealingRecord]:
        """Return the history of all healing cycles."""
        return list(self._history)

    def abort(self) -> None:
        """Abort any in-progress healing and reset to HEALTHY."""
        self._aborted = True
        self._state = HealingState.HEALTHY

    def step(self, event: FailureEvent) -> RecoveryResult:
        """Process a single failure event through the full healing pipeline.

        Transitions: HEALTHY -> FAILING -> DIAGNOSING -> RECOVERING -> VERIFYING -> HEALTHY

        If the loop was aborted, returns a failed result immediately.
        If diagnosis confidence is below threshold and auto_escalate is enabled,
        escalates instead of attempting the top-scored strategy.

        Args:
            event: The failure event to process.

        Returns:
            RecoveryResult from the recovery attempt.
        """
        if self._aborted:
            return RecoveryResult(
                success=False,
                strategy_used="none",
                outcome_message="Healing loop was aborted",
            )

        record = HealingRecord(event=event)

        # FAILING
        self._state = HealingState.FAILING

        # DIAGNOSING
        self._state = HealingState.DIAGNOSING
        diagnosis = self._engine.diagnose(event)
        record.diagnosis = diagnosis

        # Check confidence threshold
        if diagnosis.confidence < self.confidence_threshold and self.auto_escalate:
            self._state = HealingState.RECOVERING
            escalate = self._library.get_strategy("escalate")
            context = self._build_recovery_context(event, diagnosis)
            result = escalate.execute(context)
            result.recovery_data["escalation_reason"] = "low_confidence"
        else:
            # RECOVERING
            self._state = HealingState.RECOVERING
            strategy = self._library.select_strategy(diagnosis)
            context = self._build_recovery_context(event, diagnosis)
            result = strategy.execute(context)

            # Retry logic if recovery fails
            retries = 0
            while not result.success and retries < self.max_retries:
                retries += 1
                if self._aborted:
                    result = RecoveryResult(
                        success=False,
                        strategy_used="none",
                        outcome_message="Healing loop was aborted during retry",
                    )
                    break
                result = strategy.execute(context)

            if not result.success and self.auto_escalate:
                escalate = self._library.get_strategy("escalate")
                result = escalate.execute(context)
                result.recovery_data["escalation_reason"] = "retries_exhausted"

        # VERIFYING
        self._state = HealingState.VERIFYING
        record.recovery_result = result

        # Update event with diagnosis and recovery info
        event.diagnosis = diagnosis.root_cause.value
        event.recovery_action = result.strategy_used
        if result.success:
            event.outcome = "recovered"
        else:
            event.outcome = "escalated" if result.strategy_used == "escalate" else "failed"

        # Back to HEALTHY
        self._state = HealingState.HEALTHY
        record.final_state = HealingState.HEALTHY
        self._history.append(record)

        return result

    def run(self, failure_events: list[FailureEvent]) -> list[RecoveryResult]:
        """Process a batch of failure events through the healing pipeline.

        Args:
            failure_events: List of failure events to process.

        Returns:
            List of RecoveryResults, one per event.
        """
        results: list[RecoveryResult] = []
        for event in failure_events:
            if self._aborted:
                results.append(
                    RecoveryResult(
                        success=False,
                        strategy_used="none",
                        outcome_message="Healing loop was aborted",
                    )
                )
                continue
            results.append(self.step(event))
        return results

    def _build_recovery_context(
        self, event: FailureEvent, diagnosis: DiagnosisResult
    ) -> dict[str, Any]:
        """Build the context dict passed to recovery strategy execute().

        Args:
            event: The failure event being processed.
            diagnosis: The diagnosis result.

        Returns:
            Context dictionary for recovery strategy execution.
        """
        return {
            "failure_type": event.failure_type,
            "context": event.context,
            "session_id": event.session_id,
            "evidence": diagnosis.evidence,
            "confidence": diagnosis.confidence,
            "root_cause": diagnosis.root_cause.value,
            "timestamp": event.timestamp.isoformat(),
        }
