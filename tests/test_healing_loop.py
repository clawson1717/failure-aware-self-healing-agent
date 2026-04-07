"""Tests for the SelfHealingLoop module."""

from datetime import datetime, timezone

import pytest

from diagnostic import DiagnosticEngine, DiagnosisResult, RootCause
from failure_event import FailureEvent
from healing_loop import HealingRecord, HealingState, SelfHealingLoop
from recovery import RecoveryResult, RecoveryStrategyLibrary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    failure_type: str = "context_loss",
    context: str = "Agent forgot earlier instructions",
    session_id: str = "test-session",
    outcome: str = "failed",
) -> FailureEvent:
    return FailureEvent(
        timestamp=datetime.now(timezone.utc),
        failure_type=failure_type,
        context=context,
        session_id=session_id,
        outcome=outcome,
    )


def _make_loop(config: dict | None = None) -> SelfHealingLoop:
    return SelfHealingLoop(
        DiagnosticEngine(),
        RecoveryStrategyLibrary(),
        config=config,
    )


# ---------------------------------------------------------------------------
# State machine basics
# ---------------------------------------------------------------------------

class TestStateMachine:
    def test_initial_state_is_healthy(self):
        loop = _make_loop()
        assert loop.get_state() == HealingState.HEALTHY

    def test_state_returns_to_healthy_after_step(self):
        loop = _make_loop()
        loop.step(_make_event())
        assert loop.get_state() == HealingState.HEALTHY

    def test_state_enum_values(self):
        assert HealingState.HEALTHY.value == "healthy"
        assert HealingState.FAILING.value == "failing"
        assert HealingState.DIAGNOSING.value == "diagnosing"
        assert HealingState.RECOVERING.value == "recovering"
        assert HealingState.VERIFYING.value == "verifying"


# ---------------------------------------------------------------------------
# Single step processing
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_recovery_result(self):
        loop = _make_loop()
        result = loop.step(_make_event())
        assert isinstance(result, RecoveryResult)

    def test_step_context_loss_uses_re_read(self):
        loop = _make_loop(config={"auto_escalate": False})
        result = loop.step(_make_event(
            failure_type="context_loss",
            context="Agent forgot the user's earlier request, lost context, no longer recall the thread",
        ))
        assert result.strategy_used == "re_read"

    def test_step_knowledge_gap_uses_ask_clarification(self):
        loop = _make_loop()
        result = loop.step(_make_event(
            failure_type="knowledge_gap",
            context="I don't know about this topic, unsure",
        ))
        assert result.strategy_used == "ask_clarification"

    def test_step_reasoning_error_uses_backtrack(self):
        loop = _make_loop(config={"auto_escalate": False})
        result = loop.step(_make_event(
            failure_type="reasoning_error",
            context="Steps contradicted each other, inconsistent flawed logic, wrong conclusion from false assumption",
        ))
        assert result.strategy_used == "backtrack"

    def test_step_strategy_mismatch_uses_retry_different(self):
        loop = _make_loop()
        result = loop.step(_make_event(
            failure_type="strategy_mismatch",
            context="Wrong approach chosen, need to try different method",
        ))
        assert result.strategy_used == "retry_different_approach"

    def test_step_external_failure_uses_simplify(self):
        loop = _make_loop()
        result = loop.step(_make_event(
            failure_type="external_failure",
            context="Server returned 503 service unavailable timeout",
        ))
        assert result.strategy_used == "simplify"

    def test_step_updates_event_diagnosis(self):
        event = _make_event()
        loop = _make_loop()
        loop.step(event)
        assert event.diagnosis is not None

    def test_step_updates_event_recovery_action(self):
        event = _make_event()
        loop = _make_loop()
        loop.step(event)
        assert event.recovery_action is not None


# ---------------------------------------------------------------------------
# Batch run processing
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_empty_list(self):
        loop = _make_loop()
        results = loop.run([])
        assert results == []

    def test_run_single_event(self):
        loop = _make_loop()
        results = loop.run([_make_event()])
        assert len(results) == 1
        assert isinstance(results[0], RecoveryResult)

    def test_run_multiple_events(self):
        loop = _make_loop(config={"auto_escalate": False})
        events = [
            _make_event(failure_type="context_loss", context="forgot earlier info, lost context, no longer recall thread"),
            _make_event(failure_type="knowledge_gap", context="don't know this, unsure, uncertain, no idea"),
            _make_event(failure_type="reasoning_error", context="contradicted steps, inconsistent flawed logic, wrong conclusion"),
        ]
        results = loop.run(events)
        assert len(results) == 3
        assert results[0].strategy_used == "re_read"
        assert results[1].strategy_used == "ask_clarification"
        assert results[2].strategy_used == "backtrack"

    def test_run_state_healthy_after_batch(self):
        loop = _make_loop()
        loop.run([_make_event(), _make_event()])
        assert loop.get_state() == HealingState.HEALTHY


# ---------------------------------------------------------------------------
# Abort
# ---------------------------------------------------------------------------

class TestAbort:
    def test_abort_resets_to_healthy(self):
        loop = _make_loop()
        loop.abort()
        assert loop.get_state() == HealingState.HEALTHY

    def test_step_after_abort_returns_failed(self):
        loop = _make_loop()
        loop.abort()
        result = loop.step(_make_event())
        assert not result.success
        assert result.strategy_used == "none"
        assert "aborted" in result.outcome_message.lower()

    def test_run_after_abort_returns_all_failed(self):
        loop = _make_loop()
        loop.abort()
        results = loop.run([_make_event(), _make_event()])
        assert len(results) == 2
        assert all(not r.success for r in results)

    def test_abort_mid_run(self):
        """Abort during a batch run stops further processing."""
        loop = _make_loop()
        events = [_make_event() for _ in range(5)]
        # Process first event normally, then abort
        result_first = loop.step(events[0])
        loop.abort()
        remaining = loop.run(events[1:])
        assert result_first.strategy_used != "none"
        assert all(not r.success for r in remaining)


# ---------------------------------------------------------------------------
# Config thresholds
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_max_retries(self):
        loop = _make_loop()
        assert loop.max_retries == 3

    def test_default_confidence_threshold(self):
        loop = _make_loop()
        assert loop.confidence_threshold == 0.5

    def test_default_auto_escalate(self):
        loop = _make_loop()
        assert loop.auto_escalate is True

    def test_custom_config(self):
        loop = _make_loop(config={
            "max_retries": 5,
            "confidence_threshold": 0.8,
            "auto_escalate": False,
        })
        assert loop.max_retries == 5
        assert loop.confidence_threshold == 0.8
        assert loop.auto_escalate is False

    def test_high_confidence_threshold_triggers_escalation(self):
        """When confidence_threshold is very high, most diagnoses escalate."""
        loop = _make_loop(config={"confidence_threshold": 0.99})
        # A generic event will have low confidence
        event = _make_event(
            failure_type="unknown_type",
            context="something happened",
        )
        result = loop.step(event)
        assert result.strategy_used == "escalate"
        assert result.recovery_data.get("escalation_reason") == "low_confidence"

    def test_auto_escalate_false_skips_escalation(self):
        """When auto_escalate is False, low confidence does not trigger escalation."""
        loop = _make_loop(config={
            "confidence_threshold": 0.99,
            "auto_escalate": False,
        })
        event = _make_event(
            failure_type="context_loss",
            context="forgot earlier instructions",
        )
        result = loop.step(event)
        # Should use the normal strategy, not escalate
        assert result.strategy_used != "escalate"


# ---------------------------------------------------------------------------
# History tracking
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_empty_initially(self):
        loop = _make_loop()
        assert loop.get_history() == []

    def test_history_grows_after_step(self):
        loop = _make_loop()
        loop.step(_make_event())
        history = loop.get_history()
        assert len(history) == 1
        assert isinstance(history[0], HealingRecord)

    def test_history_records_event(self):
        event = _make_event()
        loop = _make_loop()
        loop.step(event)
        record = loop.get_history()[0]
        assert record.event is event

    def test_history_records_diagnosis(self):
        loop = _make_loop()
        loop.step(_make_event())
        record = loop.get_history()[0]
        assert record.diagnosis is not None
        assert isinstance(record.diagnosis, DiagnosisResult)

    def test_history_records_recovery_result(self):
        loop = _make_loop()
        loop.step(_make_event())
        record = loop.get_history()[0]
        assert record.recovery_result is not None
        assert isinstance(record.recovery_result, RecoveryResult)

    def test_history_final_state_healthy(self):
        loop = _make_loop()
        loop.step(_make_event())
        record = loop.get_history()[0]
        assert record.final_state == HealingState.HEALTHY

    def test_history_multiple_events(self):
        loop = _make_loop()
        loop.run([_make_event(), _make_event(), _make_event()])
        assert len(loop.get_history()) == 3

    def test_history_is_copy(self):
        """get_history returns a copy, not a mutable reference."""
        loop = _make_loop()
        loop.step(_make_event())
        h1 = loop.get_history()
        h1.clear()
        assert len(loop.get_history()) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_step_with_minimal_event(self):
        """Event with minimal context still processes."""
        loop = _make_loop()
        event = _make_event(context="", failure_type="unknown")
        result = loop.step(event)
        assert isinstance(result, RecoveryResult)

    def test_multiple_sequential_steps(self):
        """Loop can process many events sequentially."""
        loop = _make_loop()
        for _ in range(10):
            result = loop.step(_make_event())
            assert isinstance(result, RecoveryResult)
        assert len(loop.get_history()) == 10
        assert loop.get_state() == HealingState.HEALTHY

    def test_healing_state_is_str_enum(self):
        assert isinstance(HealingState.HEALTHY, str)
        assert HealingState.HEALTHY == "healthy"
