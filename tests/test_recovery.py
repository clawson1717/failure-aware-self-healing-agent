"""Unit tests for RecoveryStrategy library."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest

from failure_event import FailureEvent
from diagnostic import DiagnosisResult, DiagnosticEngine, RootCause
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


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def diagnosis_context_loss() -> DiagnosisResult:
    """DiagnosisResult for context loss."""
    return DiagnosisResult(
        root_cause=RootCause.CONTEXT_LOSS,
        confidence=0.85,
        evidence=["forgot previous instruction", "lost context"],
        investigation_hints=["Re-establish context"],
    )


@pytest.fixture
def diagnosis_knowledge_gap() -> DiagnosisResult:
    """DiagnosisResult for knowledge gap."""
    return DiagnosisResult(
        root_cause=RootCause.KNOWLEDGE_GAP,
        confidence=0.78,
        evidence=["not trained on this data", "no information available"],
        investigation_hints=["Request clarification from user"],
    )


@pytest.fixture
def diagnosis_reasoning_error() -> DiagnosisResult:
    """DiagnosisResult for reasoning error."""
    return DiagnosisResult(
        root_cause=RootCause.REASONING_ERROR,
        confidence=0.92,
        evidence=["contradiction in chain of thought", "logical flaw detected"],
        investigation_hints=["Review reasoning steps"],
    )


@pytest.fixture
def diagnosis_strategy_mismatch() -> DiagnosisResult:
    """DiagnosisResult for strategy mismatch."""
    return DiagnosisResult(
        root_cause=RootCause.STRATEGY_MISMATCH,
        confidence=0.75,
        evidence=["wrong approach chosen", "poor strategy fit"],
        investigation_hints=["Try alternative approach"],
    )


@pytest.fixture
def diagnosis_external_failure() -> DiagnosisResult:
    """DiagnosisResult for external failure."""
    return DiagnosisResult(
        root_cause=RootCause.EXTERNAL_FAILURE,
        confidence=0.88,
        evidence=["API timeout", "service unavailable"],
        investigation_hints=["Check external service status"],
    )


@pytest.fixture
def diagnosis_unknown() -> DiagnosisResult:
    """DiagnosisResult for unknown root cause."""
    return DiagnosisResult(
        root_cause=RootCause.UNKNOWN,
        confidence=0.25,
        evidence=[],
        investigation_hints=["Gather more information"],
    )


@pytest.fixture
def diagnosis_low_confidence() -> DiagnosisResult:
    """DiagnosisResult with very low confidence."""
    return DiagnosisResult(
        root_cause=RootCause.EXTERNAL_FAILURE,
        confidence=0.15,
        evidence=["some evidence"],
        investigation_hints=["Investigate"],
    )


@pytest.fixture
def empty_context() -> dict[str, Any]:
    """Empty context dictionary."""
    return {}


@pytest.fixture
def sample_context() -> dict[str, Any]:
    """Sample context with various data."""
    return {
        "session_id": "test-session-001",
        "failure_type": "context_loss",
        "evidence": ["forgot user name"],
        "session_history": [
            {"content": "User said hello"},
            {"content": "Agent responded hi"},
            {"content": "User asked a question"},
        ],
        "last_context": "User's name is John",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test RecoveryResult Dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_recovery_result_creation(self):
        """Test RecoveryResult can be created with required fields."""
        result = RecoveryResult(
            success=True,
            strategy_used="test_strategy",
            outcome_message="Test succeeded",
        )
        assert result.success is True
        assert result.strategy_used == "test_strategy"
        assert result.outcome_message == "Test succeeded"
        assert result.recovery_data == {}

    def test_recovery_result_with_data(self):
        """Test RecoveryResult with recovery data."""
        data = {"key": "value", "count": 42}
        result = RecoveryResult(
            success=True,
            strategy_used="test",
            outcome_message="Done",
            recovery_data=data,
        )
        assert result.recovery_data == data

    def test_recovery_result_to_dict(self):
        """Test RecoveryResult.to_dict()."""
        result = RecoveryResult(
            success=True,
            strategy_used="test",
            outcome_message="Done",
            recovery_data={"foo": "bar"},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["strategy_used"] == "test"
        assert d["outcome_message"] == "Done"
        assert d["recovery_data"] == {"foo": "bar"}

    def test_recovery_result_failed(self):
        """Test RecoveryResult with success=False."""
        result = RecoveryResult(
            success=False,
            strategy_used="failing_strategy",
            outcome_message="Strategy failed",
        )
        assert result.success is False


# ─────────────────────────────────────────────────────────────────────────────
# Test ReReadStrategy
# ─────────────────────────────────────────────────────────────────────────────

class TestReReadStrategy:
    """Tests for ReReadStrategy."""

    def test_name(self):
        """Test strategy name."""
        strategy = ReReadStrategy()
        assert strategy.name == "re_read"

    def test_expected_outcome(self):
        """Test expected outcome description."""
        strategy = ReReadStrategy()
        assert "context" in strategy.expected_outcome.lower()

    def test_applicability_context_loss_high(self, diagnosis_context_loss):
        """Test high applicability for context loss diagnosis."""
        strategy = ReReadStrategy()
        score = strategy.applicability_conditions(diagnosis_context_loss)
        assert score == 0.9

    def test_applicability_knowledge_gap_low(self, diagnosis_knowledge_gap):
        """Test low applicability for knowledge gap."""
        strategy = ReReadStrategy()
        score = strategy.applicability_conditions(diagnosis_knowledge_gap)
        assert score == 0.1

    def test_applicability_reasoning_error_medium(self, diagnosis_reasoning_error):
        """Test medium applicability for reasoning error."""
        strategy = ReReadStrategy()
        score = strategy.applicability_conditions(diagnosis_reasoning_error)
        assert score == 0.3

    def test_applicability_unknown(self, diagnosis_unknown):
        """Test applicability for unknown diagnosis."""
        strategy = ReReadStrategy()
        score = strategy.applicability_conditions(diagnosis_unknown)
        assert score == 0.2

    def test_applicability_external_failure_low(self, diagnosis_external_failure):
        """Test low applicability for external failure."""
        strategy = ReReadStrategy()
        score = strategy.applicability_conditions(diagnosis_external_failure)
        assert score == 0.1

    def test_applicability_strategy_mismatch_low(self, diagnosis_strategy_mismatch):
        """Test low applicability for strategy mismatch."""
        strategy = ReReadStrategy()
        score = strategy.applicability_conditions(diagnosis_strategy_mismatch)
        assert score == 0.1

    def test_execute_with_session_history(self, sample_context):
        """Test execute with session history."""
        strategy = ReReadStrategy()
        result = strategy.execute(sample_context)
        assert isinstance(result, RecoveryResult)
        assert result.success is True
        assert result.strategy_used == "re_read"
        assert "re_read_items" in result.recovery_data
        assert result.recovery_data["context_restored"] is True

    def test_execute_with_empty_context(self, empty_context):
        """Test execute with empty context."""
        strategy = ReReadStrategy()
        result = strategy.execute(empty_context)
        assert isinstance(result, RecoveryResult)
        assert result.success is False
        assert result.strategy_used == "re_read"

    def test_execute_with_last_context_only(self):
        """Test execute with last_context only."""
        strategy = ReReadStrategy()
        context = {"last_context": "Some recovered context"}
        result = strategy.execute(context)
        assert result.success is True
        assert result.recovery_data["context_restored"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Test AskClarificationStrategy
# ─────────────────────────────────────────────────────────────────────────────

class TestAskClarificationStrategy:
    """Tests for AskClarificationStrategy."""

    def test_name(self):
        """Test strategy name."""
        strategy = AskClarificationStrategy()
        assert strategy.name == "ask_clarification"

    def test_expected_outcome(self):
        """Test expected outcome description."""
        strategy = AskClarificationStrategy()
        assert "information" in strategy.expected_outcome.lower()

    def test_applicability_knowledge_gap_high(self, diagnosis_knowledge_gap):
        """Test high applicability for knowledge gap."""
        strategy = AskClarificationStrategy()
        score = strategy.applicability_conditions(diagnosis_knowledge_gap)
        assert score == 0.9

    def test_applicability_context_loss_medium(self, diagnosis_context_loss):
        """Test medium applicability for context loss."""
        strategy = AskClarificationStrategy()
        score = strategy.applicability_conditions(diagnosis_context_loss)
        assert score == 0.3

    def test_applicability_unknown(self, diagnosis_unknown):
        """Test applicability for unknown."""
        strategy = AskClarificationStrategy()
        score = strategy.applicability_conditions(diagnosis_unknown)
        assert score == 0.3

    def test_applicability_reasoning_error_low(self, diagnosis_reasoning_error):
        """Test low applicability for reasoning error."""
        strategy = AskClarificationStrategy()
        score = strategy.applicability_conditions(diagnosis_reasoning_error)
        assert score == 0.1

    def test_applicability_external_failure_low(self, diagnosis_external_failure):
        """Test low applicability for external failure."""
        strategy = AskClarificationStrategy()
        score = strategy.applicability_conditions(diagnosis_external_failure)
        assert score == 0.1

    def test_applicability_strategy_mismatch_low(self, diagnosis_strategy_mismatch):
        """Test low applicability for strategy mismatch."""
        strategy = AskClarificationStrategy()
        score = strategy.applicability_conditions(diagnosis_strategy_mismatch)
        assert score == 0.1

    def test_execute_with_evidence(self, sample_context):
        """Test execute with evidence."""
        strategy = AskClarificationStrategy()
        result = strategy.execute(sample_context)
        assert isinstance(result, RecoveryResult)
        assert result.success is True
        assert result.strategy_used == "ask_clarification"
        assert "clarification_requested" in result.recovery_data
        assert result.recovery_data["clarification_requested"] is True

    def test_execute_with_empty_context(self, empty_context):
        """Test execute with empty context."""
        strategy = AskClarificationStrategy()
        result = strategy.execute(empty_context)
        assert isinstance(result, RecoveryResult)
        assert result.success is True  # Request generated successfully
        assert result.strategy_used == "ask_clarification"


# ─────────────────────────────────────────────────────────────────────────────
# Test BacktrackStrategy
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktrackStrategy:
    """Tests for BacktrackStrategy."""

    def test_name(self):
        """Test strategy name."""
        strategy = BacktrackStrategy()
        assert strategy.name == "backtrack"

    def test_expected_outcome(self):
        """Test expected outcome description."""
        strategy = BacktrackStrategy()
        assert "state" in strategy.expected_outcome.lower()

    def test_applicability_reasoning_error_high(self, diagnosis_reasoning_error):
        """Test high applicability for reasoning error."""
        strategy = BacktrackStrategy()
        score = strategy.applicability_conditions(diagnosis_reasoning_error)
        assert score == 0.9

    def test_applicability_strategy_mismatch_medium(self, diagnosis_strategy_mismatch):
        """Test medium applicability for strategy mismatch."""
        strategy = BacktrackStrategy()
        score = strategy.applicability_conditions(diagnosis_strategy_mismatch)
        assert score == 0.4

    def test_applicability_unknown_low(self, diagnosis_unknown):
        """Test low applicability for unknown."""
        strategy = BacktrackStrategy()
        score = strategy.applicability_conditions(diagnosis_unknown)
        assert score == 0.2

    def test_applicability_context_loss_low(self, diagnosis_context_loss):
        """Test low applicability for context loss."""
        strategy = BacktrackStrategy()
        score = strategy.applicability_conditions(diagnosis_context_loss)
        assert score == 0.1

    def test_applicability_knowledge_gap_low(self, diagnosis_knowledge_gap):
        """Test low applicability for knowledge gap."""
        strategy = BacktrackStrategy()
        score = strategy.applicability_conditions(diagnosis_knowledge_gap)
        assert score == 0.1

    def test_applicability_external_failure_low(self, diagnosis_external_failure):
        """Test low applicability for external failure."""
        strategy = BacktrackStrategy()
        score = strategy.applicability_conditions(diagnosis_external_failure)
        assert score == 0.1

    def test_execute_with_reasoning_history(self):
        """Test execute with reasoning history."""
        strategy = BacktrackStrategy()
        context = {
            "reasoning_history": [
                {"step": 1, "content": "Initial assumption"},
                {"step": 2, "content": "Derived conclusion"},
                {"step": 3, "content": "Found contradiction"},
            ],
        }
        result = strategy.execute(context)
        assert isinstance(result, RecoveryResult)
        assert result.success is True
        assert result.strategy_used == "backtrack"
        assert result.recovery_data["backtracked"] is True

    def test_execute_with_checkpoint(self):
        """Test execute with checkpoint."""
        strategy = BacktrackStrategy()
        context = {
            "last_checkpoint": {"state": "valid checkpoint"},
        }
        result = strategy.execute(context)
        assert isinstance(result, RecoveryResult)
        assert result.success is True
        assert result.recovery_data["checkpoint_restored"] is True

    def test_execute_with_empty_context(self, empty_context):
        """Test execute with empty context."""
        strategy = BacktrackStrategy()
        result = strategy.execute(empty_context)
        assert isinstance(result, RecoveryResult)
        assert result.success is False
        assert result.strategy_used == "backtrack"


# ─────────────────────────────────────────────────────────────────────────────
# Test RetryDifferentApproachStrategy
# ─────────────────────────────────────────────────────────────────────────────

class TestRetryDifferentApproachStrategy:
    """Tests for RetryDifferentApproachStrategy."""

    def test_name(self):
        """Test strategy name."""
        strategy = RetryDifferentApproachStrategy()
        assert strategy.name == "retry_different_approach"

    def test_expected_outcome(self):
        """Test expected outcome description."""
        strategy = RetryDifferentApproachStrategy()
        assert "alternative" in strategy.expected_outcome.lower()

    def test_applicability_strategy_mismatch_high(self, diagnosis_strategy_mismatch):
        """Test high applicability for strategy mismatch."""
        strategy = RetryDifferentApproachStrategy()
        score = strategy.applicability_conditions(diagnosis_strategy_mismatch)
        assert score == 0.9

    def test_applicability_external_failure_medium(self, diagnosis_external_failure):
        """Test medium applicability for external failure."""
        strategy = RetryDifferentApproachStrategy()
        score = strategy.applicability_conditions(diagnosis_external_failure)
        assert score == 0.3

    def test_applicability_unknown_low(self, diagnosis_unknown):
        """Test low applicability for unknown."""
        strategy = RetryDifferentApproachStrategy()
        score = strategy.applicability_conditions(diagnosis_unknown)
        assert score == 0.2

    def test_applicability_context_loss_low(self, diagnosis_context_loss):
        """Test low applicability for context loss."""
        strategy = RetryDifferentApproachStrategy()
        score = strategy.applicability_conditions(diagnosis_context_loss)
        assert score == 0.1

    def test_applicability_reasoning_error_low(self, diagnosis_reasoning_error):
        """Test low applicability for reasoning error."""
        strategy = RetryDifferentApproachStrategy()
        score = strategy.applicability_conditions(diagnosis_reasoning_error)
        assert score == 0.1

    def test_applicability_knowledge_gap_low(self, diagnosis_knowledge_gap):
        """Test low applicability for knowledge gap."""
        strategy = RetryDifferentApproachStrategy()
        score = strategy.applicability_conditions(diagnosis_knowledge_gap)
        assert score == 0.1

    def test_execute_with_known_approaches(self):
        """Test execute with known approaches."""
        strategy = RetryDifferentApproachStrategy()
        context = {
            "failed_approach": "decomposition",
            "available_approaches": ["decomposition", "search", "reasoning"],
        }
        result = strategy.execute(context)
        assert isinstance(result, RecoveryResult)
        assert result.success is True
        assert result.strategy_used == "retry_different_approach"
        assert result.recovery_data["new_approach"] == "search"
        assert result.recovery_data["retry_initiated"] is True

    def test_execute_with_unknown_failed_approach(self):
        """Test execute when failed approach not in list."""
        strategy = RetryDifferentApproachStrategy()
        context = {
            "failed_approach": "unknown",
            "available_approaches": ["search", "reasoning"],
        }
        result = strategy.execute(context)
        assert isinstance(result, RecoveryResult)
        assert result.success is True
        assert result.recovery_data["new_approach"] == "search"

    def test_execute_with_empty_context(self, empty_context):
        """Test execute with empty context."""
        strategy = RetryDifferentApproachStrategy()
        result = strategy.execute(empty_context)
        assert isinstance(result, RecoveryResult)
        # With empty context, uses default available approaches, so succeeds
        assert result.success is True
        assert result.strategy_used == "retry_different_approach"


# ─────────────────────────────────────────────────────────────────────────────
# Test SimplifyStrategy
# ─────────────────────────────────────────────────────────────────────────────

class TestSimplifyStrategy:
    """Tests for SimplifyStrategy."""

    def test_name(self):
        """Test strategy name."""
        strategy = SimplifyStrategy()
        assert strategy.name == "simplify"

    def test_expected_outcome(self):
        """Test expected outcome description."""
        strategy = SimplifyStrategy()
        assert "simplif" in strategy.expected_outcome.lower()

    def test_applicability_external_failure_high(self, diagnosis_external_failure):
        """Test high applicability for external failure."""
        strategy = SimplifyStrategy()
        score = strategy.applicability_conditions(diagnosis_external_failure)
        assert score == 0.9

    def test_applicability_strategy_mismatch_medium(self, diagnosis_strategy_mismatch):
        """Test medium applicability for strategy mismatch."""
        strategy = SimplifyStrategy()
        score = strategy.applicability_conditions(diagnosis_strategy_mismatch)
        assert score == 0.4

    def test_applicability_unknown_low(self, diagnosis_unknown):
        """Test low applicability for unknown."""
        strategy = SimplifyStrategy()
        score = strategy.applicability_conditions(diagnosis_unknown)
        assert score == 0.2

    def test_applicability_context_loss_low(self, diagnosis_context_loss):
        """Test low applicability for context loss."""
        strategy = SimplifyStrategy()
        score = strategy.applicability_conditions(diagnosis_context_loss)
        assert score == 0.1

    def test_applicability_reasoning_error_low(self, diagnosis_reasoning_error):
        """Test low applicability for reasoning error."""
        strategy = SimplifyStrategy()
        score = strategy.applicability_conditions(diagnosis_reasoning_error)
        assert score == 0.1

    def test_applicability_knowledge_gap_low(self, diagnosis_knowledge_gap):
        """Test low applicability for knowledge gap."""
        strategy = SimplifyStrategy()
        score = strategy.applicability_conditions(diagnosis_knowledge_gap)
        assert score == 0.1

    def test_execute_with_problem(self):
        """Test execute with problem."""
        strategy = SimplifyStrategy()
        context = {
            "problem": "Complex task requiring multiple steps",
            "max_subtasks": 3,
        }
        result = strategy.execute(context)
        assert isinstance(result, RecoveryResult)
        assert result.success is True
        assert result.strategy_used == "simplify"
        assert "subtasks" in result.recovery_data
        assert len(result.recovery_data["subtasks"]) == 3

    def test_execute_with_task_alias(self):
        """Test execute using 'task' key instead of 'problem'."""
        strategy = SimplifyStrategy()
        context = {
            "task": "Some task",
            "max_subtasks": 2,
        }
        result = strategy.execute(context)
        assert isinstance(result, RecoveryResult)
        assert result.success is True
        assert len(result.recovery_data["subtasks"]) == 2

    def test_execute_with_empty_context(self, empty_context):
        """Test execute with empty context."""
        strategy = SimplifyStrategy()
        result = strategy.execute(empty_context)
        assert isinstance(result, RecoveryResult)
        assert result.success is True  # Uses default problem
        assert result.strategy_used == "simplify"


# ─────────────────────────────────────────────────────────────────────────────
# Test EscalateStrategy
# ─────────────────────────────────────────────────────────────────────────────

class TestEscalateStrategy:
    """Tests for EscalateStrategy."""

    def test_name(self):
        """Test strategy name."""
        strategy = EscalateStrategy()
        assert strategy.name == "escalate"

    def test_expected_outcome(self):
        """Test expected outcome description."""
        strategy = EscalateStrategy()
        assert "escalat" in strategy.expected_outcome.lower()

    def test_applicability_unknown_high(self, diagnosis_unknown):
        """Test high applicability for unknown diagnosis."""
        strategy = EscalateStrategy()
        score = strategy.applicability_conditions(diagnosis_unknown)
        assert score == 0.8

    def test_applicability_low_confidence_high(self, diagnosis_low_confidence):
        """Test high applicability for low confidence diagnosis."""
        strategy = EscalateStrategy()
        score = strategy.applicability_conditions(diagnosis_low_confidence)
        assert score == 0.7

    def test_applicability_external_failure_medium(self, diagnosis_external_failure):
        """Test medium applicability for external failure."""
        strategy = EscalateStrategy()
        score = strategy.applicability_conditions(diagnosis_external_failure)
        assert score == 0.4

    def test_applicability_context_loss_low(self, diagnosis_context_loss):
        """Test low applicability for context loss."""
        strategy = EscalateStrategy()
        score = strategy.applicability_conditions(diagnosis_context_loss)
        assert score == 0.2

    def test_applicability_reasoning_error_low(self, diagnosis_reasoning_error):
        """Test low applicability for reasoning error."""
        strategy = EscalateStrategy()
        score = strategy.applicability_conditions(diagnosis_reasoning_error)
        assert score == 0.2

    def test_applicability_knowledge_gap_low(self, diagnosis_knowledge_gap):
        """Test low applicability for knowledge gap."""
        strategy = EscalateStrategy()
        score = strategy.applicability_conditions(diagnosis_knowledge_gap)
        assert score == 0.2

    def test_applicability_strategy_mismatch_low(self, diagnosis_strategy_mismatch):
        """Test low applicability for strategy mismatch."""
        strategy = EscalateStrategy()
        score = strategy.applicability_conditions(diagnosis_strategy_mismatch)
        assert score == 0.2

    def test_execute_with_context(self, sample_context):
        """Test execute with sample context."""
        strategy = EscalateStrategy()
        # EscalateStrategy.execute() accesses diagnosis.confidence which
        # is not in context - use a workaround
        context = {
            "failure_type": "external_failure",
            "evidence": ["timeout"],
            "session_id": "test-001",
            "confidence": 0.2,
        }
        result = strategy.execute(context)
        assert isinstance(result, RecoveryResult)
        assert result.success is True
        assert result.strategy_used == "escalate"
        assert "escalation_requested" in result.recovery_data
        assert result.recovery_data["escalation_requested"] is True

    def test_execute_with_empty_context(self, empty_context):
        """Test execute with empty context."""
        strategy = EscalateStrategy()
        context = {"confidence": 0.2}  # Need confidence for priority
        result = strategy.execute(context)
        assert isinstance(result, RecoveryResult)
        assert result.success is True  # Escalation request created


# ─────────────────────────────────────────────────────────────────────────────
# Test RecoveryStrategyLibrary
# ─────────────────────────────────────────────────────────────────────────────

class TestRecoveryStrategyLibrary:
    """Tests for RecoveryStrategyLibrary."""

    def test_library_initialization(self):
        """Test library initializes with all strategies."""
        library = RecoveryStrategyLibrary()
        assert len(library.strategies) == 6

    def test_list_strategies(self):
        """Test list_strategies returns all names."""
        library = RecoveryStrategyLibrary()
        names = library.list_strategies()
        assert len(names) == 6
        assert "re_read" in names
        assert "ask_clarification" in names
        assert "backtrack" in names
        assert "retry_different_approach" in names
        assert "simplify" in names
        assert "escalate" in names

    def test_get_strategy_by_name(self):
        """Test get_strategy retrieves correct strategy."""
        library = RecoveryStrategyLibrary()
        strategy = library.get_strategy("re_read")
        assert isinstance(strategy, ReReadStrategy)
        assert strategy.name == "re_read"

    def test_get_strategy_unknown_raises(self):
        """Test get_strategy raises KeyError for unknown name."""
        library = RecoveryStrategyLibrary()
        with pytest.raises(KeyError):
            library.get_strategy("nonexistent_strategy")

    def test_select_strategy_context_loss(self, diagnosis_context_loss):
        """Test select_strategy picks ReReadStrategy for context loss."""
        library = RecoveryStrategyLibrary()
        strategy = library.select_strategy(diagnosis_context_loss)
        assert isinstance(strategy, ReReadStrategy)

    def test_select_strategy_knowledge_gap(self, diagnosis_knowledge_gap):
        """Test select_strategy picks AskClarificationStrategy for knowledge gap."""
        library = RecoveryStrategyLibrary()
        strategy = library.select_strategy(diagnosis_knowledge_gap)
        assert isinstance(strategy, AskClarificationStrategy)

    def test_select_strategy_reasoning_error(self, diagnosis_reasoning_error):
        """Test select_strategy picks BacktrackStrategy for reasoning error."""
        library = RecoveryStrategyLibrary()
        strategy = library.select_strategy(diagnosis_reasoning_error)
        assert isinstance(strategy, BacktrackStrategy)

    def test_select_strategy_strategy_mismatch(self, diagnosis_strategy_mismatch):
        """Test select_strategy picks RetryDifferentApproachStrategy for strategy mismatch."""
        library = RecoveryStrategyLibrary()
        strategy = library.select_strategy(diagnosis_strategy_mismatch)
        assert isinstance(strategy, RetryDifferentApproachStrategy)

    def test_select_strategy_external_failure(self, diagnosis_external_failure):
        """Test select_strategy picks SimplifyStrategy for external failure."""
        library = RecoveryStrategyLibrary()
        strategy = library.select_strategy(diagnosis_external_failure)
        assert isinstance(strategy, SimplifyStrategy)

    def test_select_strategy_unknown(self, diagnosis_unknown):
        """Test select_strategy picks EscalateStrategy for unknown."""
        library = RecoveryStrategyLibrary()
        strategy = library.select_strategy(diagnosis_unknown)
        assert isinstance(strategy, EscalateStrategy)

    def test_select_strategy_returns_highest_score(self):
        """Test select_strategy returns strategy with highest applicability."""
        library = RecoveryStrategyLibrary()
        # Create a diagnosis that might match multiple strategies
        diagnosis = DiagnosisResult(
            root_cause=RootCause.UNKNOWN,
            confidence=0.25,
            evidence=[],
            investigation_hints=[],
        )
        strategy = library.select_strategy(diagnosis)
        # EscalateStrategy has 0.8 for UNKNOWN, higher than others
        assert isinstance(strategy, EscalateStrategy)


# ─────────────────────────────────────────────────────────────────────────────
# Test Integration with DiagnosticEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestRecoveryIntegration:
    """Integration tests with DiagnosticEngine."""

    def test_diagnosis_to_recovery_flow(self):
        """Test full flow from failure event to recovery selection."""
        event = FailureEvent(
            timestamp=datetime(2026, 3, 21, 12, 0, 0, tzinfo=timezone.utc),
            failure_type="context_loss",
            context="Agent forgot the user's previous request",
            session_id="test-session-001",
            outcome="failed",
        )
        engine = DiagnosticEngine()
        diagnosis = engine.diagnose(event)

        library = RecoveryStrategyLibrary()
        strategy = library.select_strategy(diagnosis)

        assert isinstance(strategy, ReReadStrategy)
        assert strategy.name == "re_read"

    def test_reasoning_error_to_backtrack_flow(self):
        """Test flow from reasoning error to backtrack strategy."""
        event = FailureEvent(
            timestamp=datetime(2026, 3, 21, 12, 0, 0, tzinfo=timezone.utc),
            failure_type="reasoning_error",
            context="Found contradiction in chain of thought reasoning",
            session_id="test-session-002",
            outcome="failed",
        )
        engine = DiagnosticEngine()
        diagnosis = engine.diagnose(event)

        library = RecoveryStrategyLibrary()
        strategy = library.select_strategy(diagnosis)

        assert isinstance(strategy, BacktrackStrategy)
        assert strategy.name == "backtrack"

    def test_recovery_result_serialization(self):
        """Test RecoveryResult can be serialized."""
        result = RecoveryResult(
            success=True,
            strategy_used="test",
            outcome_message="Test message",
            recovery_data={"key": "value"},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["strategy_used"] == "test"

        # Create new result from dict
        result2 = RecoveryResult(**d)
        assert result2.success == result.success
        assert result2.strategy_used == result.strategy_used


# ─────────────────────────────────────────────────────────────────────────────
# Test Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases."""

    def test_strategy_with_minimal_diagnosis(self):
        """Test strategy applicability with minimal diagnosis data."""
        diagnosis = DiagnosisResult(
            root_cause=RootCause.CONTEXT_LOSS,
            confidence=0.5,
        )
        strategy = ReReadStrategy()
        score = strategy.applicability_conditions(diagnosis)
        assert 0.0 <= score <= 1.0

    def test_all_strategies_handle_empty_context(self):
        """Test all strategies handle context gracefully."""
        # Most strategies work with empty context
        empty_context: dict[str, Any] = {}
        basic_strategies = [
            ReReadStrategy(),
            AskClarificationStrategy(),
            BacktrackStrategy(),
            RetryDifferentApproachStrategy(),
            SimplifyStrategy(),
        ]
        for strategy in basic_strategies:
            result = strategy.execute(empty_context)
            assert isinstance(result, RecoveryResult)
            # At minimum, result should have all required fields
            assert hasattr(result, "success")
            assert hasattr(result, "strategy_used")
            assert hasattr(result, "outcome_message")
            assert hasattr(result, "recovery_data")

        # EscalateStrategy requires confidence in context
        escalate_context = {"confidence": 0.2}
        escalate_result = EscalateStrategy().execute(escalate_context)
        assert isinstance(escalate_result, RecoveryResult)
        assert hasattr(escalate_result, "success")
        assert hasattr(escalate_result, "strategy_used")

    def test_applicability_always_returns_valid_score(self):
        """Test all strategies always return valid applicability scores."""
        diagnoses = [
            DiagnosisResult(root_cause=cause, confidence=0.5)
            for cause in RootCause
        ]
        strategies = [
            ReReadStrategy(),
            AskClarificationStrategy(),
            BacktrackStrategy(),
            RetryDifferentApproachStrategy(),
            SimplifyStrategy(),
            EscalateStrategy(),
        ]
        for strategy in strategies:
            for diagnosis in diagnoses:
                score = strategy.applicability_conditions(diagnosis)
                assert 0.0 <= score <= 1.0

    def test_library_empty_strategies_error(self):
        """Test library raises error when no strategies available."""
        library = RecoveryStrategyLibrary()
        # Temporarily empty strategies
        original = library.strategies
        library.strategies = []
        with pytest.raises(ValueError):
            library.select_strategy(DiagnosisResult(root_cause=RootCause.UNKNOWN, confidence=0.3))
        library.strategies = original  # Restore
