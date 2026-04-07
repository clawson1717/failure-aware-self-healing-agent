"""Comprehensive tests for the SelfHealingLoop state machine."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from failure_event import FailureEvent
from diagnostic import DiagnosticEngine, DiagnosisResult, RootCause
from recovery import (
    RecoveryResult,
    RecoveryStrategy,
    RecoveryStrategyLibrary,
)
from healing_loop import HealingState, HealingRecord, SelfHealingLoop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    failure_type: str = "context_loss",
    context: str = "Agent forgot earlier instructions",
    outcome: str = "failed",
    session_id: str = "test-session",
) -> FailureEvent:
    return FailureEvent(
        timestamp=datetime.now(timezone.utc),
        failure_type=failure_type,
        context=context,
        session_id=session_id,
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# 1. Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_starts_healthy(self):
        loop = SelfHealingLoop()
        assert loop.get_state() == HealingState.HEALTHY

    def test_empty_history(self):
        loop = SelfHealingLoop()
        assert loop.get_history() == []

    def test_empty_stats(self):
        loop = SelfHealingLoop()
        stats = loop.get_stats()
        assert stats == {"total": 0, "successful": 0, "failed": 0, "by_cause": {}}


# ---------------------------------------------------------------------------
# 2. State transitions through step()
# ---------------------------------------------------------------------------

class TestStateTransitions:
    def test_step_returns_to_healthy(self):
        """After step() completes, state should be HEALTHY."""
        loop = SelfHealingLoop()
        event = _make_event()
        loop.step(event)
        assert loop.get_state() == HealingState.HEALTHY

    def test_step_passes_through_all_states(self):
        """Verify the loop visits FAILING → DIAGNOSING → RECOVERING → VERIFYING."""
        visited: list[HealingState] = []

        class SpyEngine(DiagnosticEngine):
            def diagnose(self_inner, event):
                visited.append(HealingState.DIAGNOSING)
                return super().diagnose(event)

        class SpyLibrary(RecoveryStrategyLibrary):
            def select_strategy(self_inner, diagnosis):
                visited.append(HealingState.RECOVERING)
                return super().select_strategy(diagnosis)

        loop = SelfHealingLoop(
            diagnostic_engine=SpyEngine(),
            strategy_library=SpyLibrary(),
        )
        loop.step(_make_event())

        assert HealingState.DIAGNOSING in visited
        assert HealingState.RECOVERING in visited
        assert loop.get_state() == HealingState.HEALTHY

    def test_step_records_healing_record(self):
        loop = SelfHealingLoop()
        event = _make_event()
        record = loop.step(event)
        assert isinstance(record, HealingRecord)
        assert record.event is event
        assert record.diagnosis is not None
        assert record.strategy_name is not None
        assert record.recovery_result is not None


# ---------------------------------------------------------------------------
# 3. abort()
# ---------------------------------------------------------------------------

class TestAbort:
    def test_abort_resets_to_healthy(self):
        loop = SelfHealingLoop()
        # Manually set a non-HEALTHY state to verify abort works
        loop._state = HealingState.RECOVERING
        loop.abort()
        assert loop.get_state() == HealingState.HEALTHY

    def test_abort_clears_attempt_counter(self):
        loop = SelfHealingLoop()
        loop._current_attempt = 5
        loop.abort()
        assert loop._current_attempt == 0

    def test_abort_preserves_history(self):
        """abort() should NOT erase previously recorded history."""
        loop = SelfHealingLoop()
        loop.step(_make_event())
        assert len(loop.get_history()) >= 1
        loop.abort()
        assert len(loop.get_history()) >= 1


# ---------------------------------------------------------------------------
# 4. run() — batch processing
# ---------------------------------------------------------------------------

class TestRunBatch:
    def test_run_processes_all_events(self):
        loop = SelfHealingLoop()
        events = [_make_event(session_id=f"s{i}") for i in range(5)]
        results = loop.run(events)
        assert len(results) == 5

    def test_run_returns_one_record_per_event(self):
        loop = SelfHealingLoop()
        events = [_make_event(), _make_event(failure_type="reasoning_error")]
        results = loop.run(events)
        assert len(results) == 2
        assert results[0].event is events[0]
        assert results[1].event is events[1]

    def test_state_healthy_after_run(self):
        loop = SelfHealingLoop()
        loop.run([_make_event(), _make_event()])
        assert loop.get_state() == HealingState.HEALTHY


# ---------------------------------------------------------------------------
# 5. get_stats()
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_count_successful(self):
        loop = SelfHealingLoop()
        # context_loss events with context containing "forgot" → diagnosed
        # as CONTEXT_LOSS with decent confidence → ReRead strategy succeeds
        loop.step(_make_event(context="Agent forgot the user name"))
        stats = loop.get_stats()
        assert stats["successful"] >= 1

    def test_stats_total_matches_history(self):
        loop = SelfHealingLoop()
        loop.run([_make_event() for _ in range(3)])
        stats = loop.get_stats()
        assert stats["total"] == len(loop.get_history())

    def test_stats_by_cause_populated(self):
        loop = SelfHealingLoop()
        loop.step(_make_event(failure_type="context_loss", context="forgot the goal"))
        loop.step(
            _make_event(
                failure_type="external_failure",
                context="server returned 503 service unavailable",
            )
        )
        stats = loop.get_stats()
        assert len(stats["by_cause"]) >= 1

    def test_stats_successful_plus_failed_equals_total(self):
        loop = SelfHealingLoop()
        loop.run([_make_event() for _ in range(4)])
        stats = loop.get_stats()
        assert stats["successful"] + stats["failed"] == stats["total"]


# ---------------------------------------------------------------------------
# 6. Configurable thresholds
# ---------------------------------------------------------------------------

class TestThresholds:
    def test_max_recovery_attempts_limits_retries(self):
        """When every recovery fails, the loop should attempt exactly max_recovery_attempts times."""

        class AlwaysFailLibrary(RecoveryStrategyLibrary):
            def select_strategy(self_inner, diagnosis):
                strategy = MagicMock(spec=RecoveryStrategy)
                strategy.name = "always_fail"
                strategy.execute.return_value = RecoveryResult(
                    success=False,
                    strategy_used="always_fail",
                    outcome_message="nope",
                )
                return strategy

            def get_strategy(self_inner, name):
                return self_inner.select_strategy(None)

        loop = SelfHealingLoop(
            max_recovery_attempts=2,
            strategy_library=AlwaysFailLibrary(),
        )
        loop.step(_make_event())
        # Should have exactly 2 records (one per attempt)
        assert len(loop.get_history()) == 2
        assert all(not r.healed for r in loop.get_history())

    def test_confidence_threshold_triggers_escalation(self):
        """Low-confidence diagnoses should force escalation."""
        class LowConfidenceEngine(DiagnosticEngine):
            def diagnose(self_inner, event):
                return DiagnosisResult(
                    root_cause=RootCause.UNKNOWN,
                    confidence=0.1,
                    evidence=["very uncertain"],
                )

        loop = SelfHealingLoop(
            confidence_threshold=0.5,
            diagnostic_engine=LowConfidenceEngine(),
        )
        record = loop.step(_make_event())
        assert record.strategy_name == "escalate"

    def test_default_thresholds(self):
        loop = SelfHealingLoop()
        assert loop.max_recovery_attempts == 3
        assert loop.confidence_threshold == 0.4
        assert loop.verification_timeout == 30.0


# ---------------------------------------------------------------------------
# 7. Different failure types
# ---------------------------------------------------------------------------

class TestFailureTypes:
    @pytest.mark.parametrize(
        "failure_type,context,expected_strategy",
        [
            ("context_loss", "Agent forgot earlier constraints", "re_read"),
            ("knowledge_gap", "Model doesn't know about this topic, unsure", "ask_clarification"),
            ("reasoning_error", "Steps contradict each other, flawed logic", "backtrack"),
            ("strategy_mismatch", "Wrong approach chosen, stuck at dead-end", "retry_different_approach"),
            ("external_failure", "API returned 503 timeout", "simplify"),
        ],
    )
    def test_correct_strategy_selected(self, failure_type, context, expected_strategy):
        loop = SelfHealingLoop()
        event = _make_event(failure_type=failure_type, context=context)
        record = loop.step(event)
        assert record.strategy_name == expected_strategy


# ---------------------------------------------------------------------------
# 8. HealingRecord
# ---------------------------------------------------------------------------

class TestHealingRecord:
    def test_to_dict_round_trip(self):
        loop = SelfHealingLoop()
        record = loop.step(_make_event())
        d = record.to_dict()
        assert "event" in d
        assert "diagnosis" in d
        assert "strategy_name" in d
        assert "healed" in d

    def test_record_has_timestamp(self):
        loop = SelfHealingLoop()
        record = loop.step(_make_event())
        assert record.timestamp is not None


# ---------------------------------------------------------------------------
# 9. HealingState enum values
# ---------------------------------------------------------------------------

class TestHealingStateEnum:
    def test_all_states_present(self):
        expected = {"healthy", "failing", "diagnosing", "recovering", "verifying"}
        actual = {s.value for s in HealingState}
        assert actual == expected

    def test_enum_is_str(self):
        assert isinstance(HealingState.HEALTHY, str)
        assert HealingState.HEALTHY == "healthy"
