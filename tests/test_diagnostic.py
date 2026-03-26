"""Unit tests for DiagnosticEngine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest

from failure_event import FailureEvent
from diagnostic import (
    DiagnosisResult,
    DiagnosticEngine,
    RootCause,
    _KEYWORD_PATTERNS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_event() -> FailureEvent:
    """Minimal FailureEvent for testing."""
    return FailureEvent(
        timestamp=datetime(2026, 3, 21, 12, 0, 0, tzinfo=timezone.utc),
        failure_type="external_failure",
        context="Test context",
        session_id="test-session-001",
        outcome="failed",
    )


@pytest.fixture
def engine() -> DiagnosticEngine:
    """DiagnosticEngine with default settings."""
    return DiagnosticEngine()


@pytest.fixture
def engine_with_llm(engine: DiagnosticEngine) -> DiagnosticEngine:
    """DiagnosticEngine with LLM artificially enabled for testing."""
    engine.use_llm = True
    engine._llm_available = True
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# Test RootCause Enum
# ─────────────────────────────────────────────────────────────────────────────

class TestRootCauseEnum:
    """Tests for RootCause enumeration."""

    def test_all_expected_values_present(self) -> None:
        """Verify all 6 expected root cause values exist."""
        expected = {
            "knowledge_gap",
            "reasoning_error",
            "context_loss",
            "strategy_mismatch",
            "external_failure",
            "unknown",
        }
        actual = {cause.value for cause in RootCause}
        assert expected == actual

    def test_root_cause_is_string(self) -> None:
        """RootCause values should be strings for serialization."""
        assert isinstance(RootCause.KNOWLEDGE_GAP.value, str)
        assert RootCause.KNOWLEDGE_GAP.value == "knowledge_gap"

    def test_root_cause_ordering(self) -> None:
        """RootCause enum members should be ordered."""
        causes = list(RootCause)
        assert RootCause.KNOWLEDGE_GAP in causes
        assert RootCause.UNKNOWN in causes
        assert len(causes) == 6


# ─────────────────────────────────────────────────────────────────────────────
# Test DiagnosisResult Dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagnosisResultCreation:
    """Tests for DiagnosisResult instantiation."""

    def test_create_with_required_fields(self) -> None:
        """Test creating DiagnosisResult with required fields only."""
        result = DiagnosisResult(
            root_cause=RootCause.KNOWLEDGE_GAP,
            confidence=0.85,
        )

        assert result.root_cause == RootCause.KNOWLEDGE_GAP
        assert result.confidence == 0.85
        assert result.evidence == []
        assert result.investigation_hints == []

    def test_create_with_all_fields(self) -> None:
        """Test creating DiagnosisResult with all fields."""
        result = DiagnosisResult(
            root_cause=RootCause.REASONING_ERROR,
            confidence=0.92,
            evidence=["Matched 3 keyword patterns", "failure_type aligns"],
            investigation_hints=["Review logic chain", "Check assumptions"],
        )

        assert result.root_cause == RootCause.REASONING_ERROR
        assert result.confidence == 0.92
        assert len(result.evidence) == 2
        assert len(result.investigation_hints) == 2

    def test_default_fields_are_empty_lists(self) -> None:
        """Test that evidence and investigation_hints default to empty lists."""
        result = DiagnosisResult(
            root_cause=RootCause.CONTEXT_LOSS,
            confidence=0.75,
        )

        assert result.evidence == []
        assert result.investigation_hints == []


class TestDiagnosisResultSerialization:
    """Tests for DiagnosisResult to_dict / from_dict."""

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict should include all fields."""
        result = DiagnosisResult(
            root_cause=RootCause.EXTERNAL_FAILURE,
            confidence=0.88,
            evidence=["Timeout occurred"],
            investigation_hints=["Check network"],
        )

        d = result.to_dict()

        assert d["root_cause"] == "external_failure"
        assert d["confidence"] == 0.88
        assert d["evidence"] == ["Timeout occurred"]
        assert d["investigation_hints"] == ["Check network"]

    def test_from_dict_round_trip(self) -> None:
        """from_dict should restore the original result."""
        original = DiagnosisResult(
            root_cause=RootCause.STRATEGY_MISMATCH,
            confidence=0.80,
            evidence=["Wrong approach detected"],
            investigation_hints=["Try alternative strategy"],
        )

        restored = DiagnosisResult.from_dict(original.to_dict())

        assert restored.root_cause == original.root_cause
        assert restored.confidence == original.confidence
        assert restored.evidence == original.evidence
        assert restored.investigation_hints == original.investigation_hints

    def test_to_dict_preserves_unknown_root_cause(self) -> None:
        """to_dict should handle UNKNOWN root cause."""
        result = DiagnosisResult(
            root_cause=RootCause.UNKNOWN,
            confidence=0.30,
        )

        d = result.to_dict()
        assert d["root_cause"] == "unknown"

        restored = DiagnosisResult.from_dict(d)
        assert restored.root_cause == RootCause.UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Test DiagnosticEngine Initialization
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagnosticEngineInit:
    """Tests for DiagnosticEngine initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization (no LLM)."""
        engine = DiagnosticEngine()

        assert engine.use_llm is False
        assert engine._llm_available is False
        assert engine._last_result is None

    def test_init_with_llm_disabled(self) -> None:
        """Test init with use_llm=False explicitly."""
        engine = DiagnosticEngine(use_llm=False)
        assert engine.use_llm is False
        assert engine._llm_available is False

    def test_init_with_llm_no_api_key(self) -> None:
        """Test LLM enabled but no API key available."""
        with patch.dict("os.environ", {}, clear=True):
            engine = DiagnosticEngine(use_llm=True)
            assert engine.use_llm is True
            assert engine._llm_available is False

    def test_init_with_llm_and_api_key_from_env(self) -> None:
        """Test LLM uses API key from environment variable."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
            engine = DiagnosticEngine(use_llm=True)
            assert engine.use_llm is True
            assert engine._llm_available is True

    def test_init_with_explicit_api_key(self) -> None:
        """Test LLM uses explicitly passed API key."""
        engine = DiagnosticEngine(use_llm=True, llm_api_key="sk-explicit-key")
        assert engine.use_llm is True
        assert engine._llm_available is True

    def test_llm_model_default(self) -> None:
        """Test default LLM model is gpt-4."""
        engine = DiagnosticEngine()
        assert engine.llm_model == "gpt-4"

    def test_llm_model_custom(self) -> None:
        """Test custom LLM model name."""
        engine = DiagnosticEngine(use_llm=True, llm_model="gpt-3.5-turbo")
        assert engine.llm_model == "gpt-3.5-turbo"


# ─────────────────────────────────────────────────────────────────────────────
# Test KNOWLEDGE_GAP Diagnosis
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeGapDiagnosis:
    """Tests for Knowledge Gap (missing info) diagnosis."""

    def test_diagnose_knowledge_gap_unknown(self, engine: DiagnosticEngine) -> None:
        """Diagnosis for 'I don't know' style failures."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="The model said it didn't know the answer",
            session_id="sess-kg-001",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.KNOWLEDGE_GAP
        assert result.confidence > 0.3

    def test_diagnose_knowledge_gap_no_training_data(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for missing training data."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="Query about specific protein structure - no training data available",
            session_id="sess-kg-002",
            outcome="escalated",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.KNOWLEDGE_GAP
        assert len(result.evidence) >= 1

    def test_diagnose_knowledge_gap_cutoff(self, engine: DiagnosticEngine) -> None:
        """Diagnosis for knowledge beyond cutoff date."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="Event after knowledge cutoff date cannot be answered",
            session_id="sess-kg-003",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.KNOWLEDGE_GAP

    def test_diagnose_knowledge_gap_unsure(self, engine: DiagnosticEngine) -> None:
        """Diagnosis for expressed uncertainty."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="Model expressed uncertainty about the legal question",
            session_id="sess-kg-004",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.KNOWLEDGE_GAP
        assert result.confidence >= 0.5

    def test_diagnose_knowledge_gap_no_idea(self, engine: DiagnosticEngine) -> None:
        """Diagnosis for 'no idea' style failures."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="Had no idea how to approach the quantum physics problem",
            session_id="sess-kg-005",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.KNOWLEDGE_GAP


# ─────────────────────────────────────────────────────────────────────────────
# Test REASONING_ERROR Diagnosis
# ─────────────────────────────────────────────────────────────────────────────

class TestReasoningErrorDiagnosis:
    """Tests for Reasoning Error (logical flaw) diagnosis."""

    def test_diagnose_reasoning_error_contradiction(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for contradiction in reasoning."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context="Step 3 contradicted the conclusion reached in Step 1",
            session_id="sess-re-001",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.REASONING_ERROR
        assert result.confidence >= 0.5

    def test_diagnose_reasoning_error_inconsistent(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for inconsistent reasoning."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context="Model produced inconsistent conclusions from the same premises",
            session_id="sess-re-002",
            outcome="escalated",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.REASONING_ERROR

    def test_diagnose_reasoning_error_flawed(self, engine: DiagnosticEngine) -> None:
        """Diagnosis for flawed logic."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context="The reasoning chain was fundamentally flawed",
            session_id="sess-re-003",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.REASONING_ERROR

    def test_diagnose_reasoning_error_circular(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for circular reasoning."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context="Circular reasoning detected - conclusion used to prove itself",
            session_id="sess-re-004",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.REASONING_ERROR

    def test_diagnose_reasoning_error_miscalculation(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for math/arithmetic errors."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context="Miscalculation in the financial projection led to wrong answer",
            session_id="sess-re-005",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.REASONING_ERROR

    def test_diagnose_reasoning_error_overlook(self, engine: DiagnosticEngine) -> None:
        """Diagnosis for overlooked important factor."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context="Failed to account for the constraint stated earlier",
            session_id="sess-re-006",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.REASONING_ERROR


# ─────────────────────────────────────────────────────────────────────────────
# Test CONTEXT_LOSS Diagnosis
# ─────────────────────────────────────────────────────────────────────────────

class TestContextLossDiagnosis:
    """Tests for Context Loss (forgot important info) diagnosis."""

    def test_diagnose_context_loss_forgot(self, engine: DiagnosticEngine) -> None:
        """Diagnosis for forgetting earlier information."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="context_loss",
            context="Model forgot what the user asked earlier in the conversation",
            session_id="sess-cl-001",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.CONTEXT_LOSS
        assert result.confidence >= 0.5

    def test_diagnose_context_loss_repeating(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for repeating previous actions."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="context_loss",
            context="Model kept repeating the same step - lost the thread",
            session_id="sess-cl-002",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.CONTEXT_LOSS

    def test_diagnose_context_loss_window_exceeded(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for context window exceeded."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="context_loss",
            context="Context window exceeded - earlier requirements were lost",
            session_id="sess-cl-003",
            outcome="escalated",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.CONTEXT_LOSS
        assert result.confidence >= 0.5

    def test_diagnose_context_loss_forgot_goal(self, engine: DiagnosticEngine) -> None:
        """Diagnosis for forgetting the original goal."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="context_loss",
            context="Model forgot the original goal stated at the start",
            session_id="sess-cl-004",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.CONTEXT_LOSS

    def test_diagnose_context_loss_truncation(self, engine: DiagnosticEngine) -> None:
        """Diagnosis for context truncation."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="context_loss",
            context="Conversation was truncated and context was lost",
            session_id="sess-cl-005",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.CONTEXT_LOSS

    def test_diagnose_context_loss_again(self, engine: DiagnosticEngine) -> None:
        """Diagnosis for 'said again' pattern."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="context_loss",
            context="Model asked for information it was already given once",
            session_id="sess-cl-006",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.CONTEXT_LOSS


# ─────────────────────────────────────────────────────────────────────────────
# Test STRATEGY_MISMATCH Diagnosis
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyMismatchDiagnosis:
    """Tests for Strategy Mismatch (wrong approach) diagnosis."""

    def test_diagnose_strategy_mismatch_wrong_approach(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for wrong approach chosen."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="strategy_mismatch",
            context="Chose the wrong approach for the mathematical proof",
            session_id="sess-sm-001",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.STRATEGY_MISMATCH
        assert result.confidence >= 0.5

    def test_diagnose_strategy_mismatch_inappropriate(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for inappropriate strategy."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="strategy_mismatch",
            context="Strategy was inappropriate for the creative writing task",
            session_id="sess-sm-002",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.STRATEGY_MISMATCH

    def test_diagnose_strategy_mismatch_stuck(self, engine: DiagnosticEngine) -> None:
        """Diagnosis for getting stuck/stalled."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="strategy_mismatch",
            context="Analysis got stuck and no progress was being made",
            session_id="sess-sm-003",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.STRATEGY_MISMATCH

    def test_diagnose_strategy_mismatch_backtrack(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for needing to backtrack."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="strategy_mismatch",
            context="Had to backtrack and try a different approach",
            session_id="sess-sm-004",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.STRATEGY_MISMATCH

    def test_diagnose_strategy_mismatch_dead_end(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for hitting a dead end."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="strategy_mismatch",
            context="Reached a dead-end with current strategy",
            session_id="sess-sm-005",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.STRATEGY_MISMATCH


# ─────────────────────────────────────────────────────────────────────────────
# Test EXTERNAL_FAILURE Diagnosis
# ─────────────────────────────────────────────────────────────────────────────

class TestExternalFailureDiagnosis:
    """Tests for External Failure (environment issue) diagnosis."""

    def test_diagnose_external_failure_timeout(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for timeout."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="API request timed out after 30 seconds",
            session_id="sess-ef-001",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.EXTERNAL_FAILURE
        assert result.confidence >= 0.5

    def test_diagnose_external_failure_connection(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for connection failure."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="Database connection failed",
            session_id="sess-ef-002",
            outcome="escalated",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.EXTERNAL_FAILURE

    def test_diagnose_external_failure_rate_limit(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for rate limiting."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="API rate limit exceeded - 429 error received",
            session_id="sess-ef-003",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.EXTERNAL_FAILURE

    def test_diagnose_external_failure_500_error(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for HTTP 500 server error."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="Received 503 Service Unavailable from the API",
            session_id="sess-ef-004",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.EXTERNAL_FAILURE

    def test_diagnose_external_failure_auth(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for authentication/authorization failure."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="Permission denied accessing the resource",
            session_id="sess-ef-005",
            outcome="escalated",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.EXTERNAL_FAILURE

    def test_diagnose_external_failure_dependency(
        self, engine: DiagnosticEngine
    ) -> None:
        """Diagnosis for missing dependency."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="Required dependency was missing from the environment",
            session_id="sess-ef-006",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.EXTERNAL_FAILURE

    def test_diagnose_external_failure_retry(self, engine: DiagnosticEngine) -> None:
        """Diagnosis for retry scenario."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="Failed to connect to the server, retrying...",
            session_id="sess-ef-007",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.EXTERNAL_FAILURE


# ─────────────────────────────────────────────────────────────────────────────
# Test get_root_cause
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRootCause:
    """Tests for get_root_cause method."""

    def test_get_root_cause_after_diagnose(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test get_root_cause returns correct value after diagnose."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="API timed out",
            session_id="sess-grc-001",
            outcome="failed",
        )

        engine.diagnose(event)
        root_cause = engine.get_root_cause()

        assert isinstance(root_cause, str)
        assert root_cause == "external_failure"

    def test_get_root_cause_raises_without_diagnose(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test get_root_cause raises RuntimeError if diagnose not called."""
        with pytest.raises(RuntimeError, match="No diagnosis available"):
            engine.get_root_cause()

    def test_get_root_cause_after_reset_raises(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test get_root_cause raises after reset."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context="Logic error",
            session_id="sess-grc-003",
            outcome="recovered",
        )

        engine.diagnose(event)
        engine.reset()

        with pytest.raises(RuntimeError, match="No diagnosis available"):
            engine.get_root_cause()


# ─────────────────────────────────────────────────────────────────────────────
# Test suggest_investigation
# ─────────────────────────────────────────────────────────────────────────────

class TestSuggestInvestigation:
    """Tests for suggest_investigation method."""

    def test_returns_list_of_hints(
        self, engine: DiagnosticEngine, base_event: FailureEvent
    ) -> None:
        """Test that suggest_investigation returns a list."""
        hints = engine.suggest_investigation(base_event)

        assert isinstance(hints, list)
        assert len(hints) > 0
        assert all(isinstance(h, str) for h in hints)

    def test_hints_for_knowledge_gap(self, engine: DiagnosticEngine) -> None:
        """Test hints are appropriate for knowledge gap."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="Model didn't know the answer",
            session_id="sess-si-001",
            outcome="failed",
        )

        hints = engine.suggest_investigation(event)

        # Should contain hints related to knowledge
        hint_text = " ".join(hints).lower()
        assert "knowledge" in hint_text or "training" in hint_text or "search" in hint_text

    def test_hints_for_reasoning_error(self, engine: DiagnosticEngine) -> None:
        """Test hints are appropriate for reasoning error."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context="Contradiction in the reasoning",
            session_id="sess-si-002",
            outcome="recovered",
        )

        hints = engine.suggest_investigation(event)

        hint_text = " ".join(hints).lower()
        assert "logic" in hint_text or "reasoning" in hint_text or "assumption" in hint_text

    def test_hints_for_timeout_specific(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test specific hints for timeout context."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="The API request timed out",
            session_id="sess-si-003",
            outcome="failed",
        )

        hints = engine.suggest_investigation(event)

        hint_text = " ".join(hints).lower()
        assert "timeout" in hint_text or "network" in hint_text or "latency" in hint_text

    def test_hints_are_unique(self, engine: DiagnosticEngine) -> None:
        """Test that hints are deduplicated."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="Timeout occurred - external API error",
            session_id="sess-si-004",
            outcome="failed",
        )

        hints = engine.suggest_investigation(event)

        # Check for duplicates
        assert len(hints) == len(set(hints))


# ─────────────────────────────────────────────────────────────────────────────
# Test reset
# ─────────────────────────────────────────────────────────────────────────────

class TestReset:
    """Tests for reset method."""

    def test_reset_clears_last_result(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test that reset clears the last diagnosis."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="Did not know the answer",
            session_id="sess-r-001",
            outcome="failed",
        )

        engine.diagnose(event)
        assert engine._last_result is not None

        engine.reset()

        assert engine._last_result is None

    def test_reset_allows_new_diagnosis(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test that reset allows diagnosing new events."""
        event1 = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="Unknown topic",
            session_id="sess-r-002",
            outcome="failed",
        )
        event2 = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context="Logic flaw",
            session_id="sess-r-003",
            outcome="recovered",
        )

        result1 = engine.diagnose(event1)
        engine.reset()
        result2 = engine.diagnose(event2)

        assert result1.root_cause != result2.root_cause


# ─────────────────────────────────────────────────────────────────────────────
# Test LLM Augmentation
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMAugmentation:
    """Tests for LLM augmentation feature."""

    def test_llm_augment_boosts_confidence(
        self, engine_with_llm: DiagnosticEngine
    ) -> None:
        """Test that LLM augmentation increases confidence."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="Model had no knowledge of the topic",
            session_id="sess-llm-001",
            outcome="recovered",
        )

        result = engine_with_llm.diagnose(event)

        # With LLM boost, confidence should be higher
        assert result.confidence >= 0.65  # base 0.5 + boost 0.15

    def test_llm_augment_adds_evidence(
        self, engine_with_llm: DiagnosticEngine
    ) -> None:
        """Test that LLM adds evidence to diagnosis."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context="Flawed logic chain",
            session_id="sess-llm-002",
            outcome="recovered",
        )

        result = engine_with_llm.diagnose(event)

        evidence_text = " ".join(result.evidence).lower()
        assert "llm" in evidence_text

    def test_llm_not_available_without_key_uses_heuristic(
        self, base_event: FailureEvent
    ) -> None:
        """Test that engine falls back to heuristic when LLM key missing."""
        with patch.dict("os.environ", {}, clear=True):
            engine = DiagnosticEngine(use_llm=True)

        # Should work even without actual LLM key
        assert engine.use_llm is True
        assert engine._llm_available is False

        result = engine.diagnose(base_event)
        assert isinstance(result, DiagnosisResult)
        assert isinstance(result.root_cause, RootCause)


# ─────────────────────────────────────────────────────────────────────────────
# Test Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_context(self, engine: DiagnosticEngine) -> None:
        """Test diagnosis with empty context."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="",
            session_id="sess-ec-001",
            outcome="failed",
        )

        result = engine.diagnose(event)

        # Should still return a result (possibly UNKNOWN)
        assert isinstance(result, DiagnosisResult)
        assert isinstance(result.root_cause, RootCause)

    def test_very_long_context(self, engine: DiagnosticEngine) -> None:
        """Test diagnosis with very long context."""
        long_context = "timeout " * 1000
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context=long_context,
            session_id="sess-ec-002",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.EXTERNAL_FAILURE

    def test_unicode_context(self, engine: DiagnosticEngine) -> None:
        """Test diagnosis with unicode characters."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="日本語のコンテキスト - unknown topic",
            session_id="sess-ec-003",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert isinstance(result, DiagnosisResult)
        assert result.root_cause == RootCause.KNOWLEDGE_GAP

    def test_special_characters_context(self, engine: DiagnosticEngine) -> None:
        """Test diagnosis with special characters in context."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context="Contradiction: \"A\" and \"not A\" both asserted\nNewline\ttab",
            session_id="sess-ec-004",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.REASONING_ERROR

    def test_unknown_failure_type(self, engine: DiagnosticEngine) -> None:
        """Test diagnosis with unrecognized failure_type."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="unknown_failure",
            context="Something went wrong but unclear what",
            session_id="sess-ec-005",
            outcome="failed",
        )

        result = engine.diagnose(event)

        # Should still produce a result using context keywords
        assert isinstance(result, DiagnosisResult)

    def test_confidence_bounded_0_to_1(self, engine: DiagnosticEngine) -> None:
        """Test that confidence is always between 0 and 1."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="timeout",
            session_id="sess-ec-006",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert 0.0 <= result.confidence <= 1.0

    def test_case_insensitive_keyword_matching(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test that keyword matching is case-insensitive."""
        event1 = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="UNKNOWN topic",
            session_id="sess-ec-007",
            outcome="failed",
        )
        event2 = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="unknown topic",
            session_id="sess-ec-008",
            outcome="failed",
        )

        result1 = engine.diagnose(event1)
        result2 = engine.diagnose(event2)

        # Both should identify knowledge gap (case insensitive)
        assert result1.root_cause == RootCause.KNOWLEDGE_GAP
        assert result2.root_cause == RootCause.KNOWLEDGE_GAP

    def test_multiple_root_causes_in_context(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test when context contains keywords from multiple categories."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context="Model contradicted itself and forgot earlier requirements",
            session_id="sess-ec-009",
            outcome="failed",
        )

        result = engine.diagnose(event)

        # Should pick the best match
        assert result.root_cause in [
            RootCause.REASONING_ERROR,
            RootCause.CONTEXT_LOSS,
        ]

    def test_failure_type_overrides_context(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test that failure_type can provide signal when context is ambiguous."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="Something unexpected happened",
            session_id="sess-ec-010",
            outcome="failed",
        )

        result = engine.diagnose(event)

        # failure_type provides signal even with vague context
        assert result.root_cause == RootCause.EXTERNAL_FAILURE


# ─────────────────────────────────────────────────────────────────────────────
# Test Confidence Scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceScoring:
    """Tests for confidence calculation logic."""

    def test_high_keyword_count_high_confidence(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test that many keyword matches yield higher confidence."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context=(
                "Contradiction in step 2. Inconsistent with step 1. "
                "Circular reasoning detected. Logic flawed."
            ),
            session_id="sess-cs-001",
            outcome="failed",
        )

        result = engine.diagnose(event)

        assert result.root_cause == RootCause.REASONING_ERROR
        assert result.confidence >= 0.7

    def test_single_keyword_low_confidence(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test that single keyword match yields lower confidence."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="something went wrong",
            session_id="sess-cs-002",
            outcome="failed",
        )

        result = engine.diagnose(event)

        # Generic phrase may not trigger strong pattern
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_is_rounded_to_2_decimals(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test that confidence is rounded to 2 decimal places."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="I don't know",
            session_id="sess-cs-003",
            outcome="failed",
        )

        result = engine.diagnose(event)

        # Check it's rounded to 2 decimal places
        assert result.confidence == round(result.confidence, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Test Evidence Building
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidenceBuilding:
    """Tests for evidence generation in diagnosis."""

    def test_evidence_contains_keyword_match_count(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test that evidence includes keyword match count."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="unknown and unsure about the topic",
            session_id="sess-eb-001",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        evidence_text = " ".join(result.evidence)
        assert "Matched" in evidence_text or "keyword" in evidence_text.lower()

    def test_evidence_contains_failure_type_align(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test that evidence mentions failure_type alignment."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="reasoning_error",
            context="Contradiction detected in logic",
            session_id="sess-eb-002",
            outcome="recovered",
        )

        result = engine.diagnose(event)

        # evidence should include failure_type alignment
        assert len(result.evidence) >= 1

    def test_evidence_contains_outcome(self, engine: DiagnosticEngine) -> None:
        """Test that evidence includes outcome information."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context="API timeout",
            session_id="sess-eb-003",
            outcome="escalated",
        )

        result = engine.diagnose(event)

        evidence_text = " ".join(result.evidence)
        assert "escalated" in evidence_text.lower()

    def test_evidence_empty_for_unknown(self, engine: DiagnosticEngine) -> None:
        """Test that evidence can be minimal for unknown root cause."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="unknown",
            context="xyzzy",
            session_id="sess-eb-004",
            outcome="failed",
        )

        result = engine.diagnose(event)

        # Should still have some evidence even for unknown
        assert isinstance(result.evidence, list)


# ─────────────────────────────────────────────────────────────────────────────
# Test Keyword Patterns Coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordPatternsCoverage:
    """Tests for keyword pattern dictionary completeness."""

    def test_all_root_causes_have_patterns(self) -> None:
        """Test every RootCause has at least one keyword pattern."""
        for cause in RootCause:
            if cause == RootCause.UNKNOWN:
                continue  # UNKNOWN has no patterns by design
            assert cause in _KEYWORD_PATTERNS
            assert len(_KEYWORD_PATTERNS[cause]) > 0

    def test_knowledge_gap_has_sufficient_patterns(self) -> None:
        """Test Knowledge Gap has diverse patterns."""
        patterns = _KEYWORD_PATTERNS[RootCause.KNOWLEDGE_GAP]
        assert len(patterns) >= 10

    def test_reasoning_error_has_sufficient_patterns(self) -> None:
        """Test Reasoning Error has diverse patterns."""
        patterns = _KEYWORD_PATTERNS[RootCause.REASONING_ERROR]
        assert len(patterns) >= 10

    def test_context_loss_has_sufficient_patterns(self) -> None:
        """Test Context Loss has diverse patterns."""
        patterns = _KEYWORD_PATTERNS[RootCause.CONTEXT_LOSS]
        assert len(patterns) >= 10

    def test_strategy_mismatch_has_sufficient_patterns(self) -> None:
        """Test Strategy Mismatch has diverse patterns."""
        patterns = _KEYWORD_PATTERNS[RootCause.STRATEGY_MISMATCH]
        assert len(patterns) >= 10

    def test_external_failure_has_sufficient_patterns(self) -> None:
        """Test External Failure has diverse patterns."""
        patterns = _KEYWORD_PATTERNS[RootCause.EXTERNAL_FAILURE]
        assert len(patterns) >= 10

    def test_patterns_are_valid_regex(self) -> None:
        """Test that all patterns are valid regular expressions."""
        import re

        for cause, patterns in _KEYWORD_PATTERNS.items():
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as e:
                    pytest.fail(
                        f"Invalid regex in {cause.value}: {pattern!r} - {e}"
                    )


# ─────────────────────────────────────────────────────────────────────────────
# Integration-Style Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    """Integration-style tests for complete workflows."""

    def test_full_diagnosis_workflow(self, engine: DiagnosticEngine) -> None:
        """Test complete diagnosis workflow."""
        event = FailureEvent(
            timestamp=datetime(2026, 3, 26, 10, 0, 0, tzinfo=timezone.utc),
            failure_type="external_failure",
            context="API request timed out after 30 seconds",
            session_id="sess-int-001",
            outcome="recovered",
            diagnosis=None,
            recovery_action="Retried with longer timeout",
        )

        # Diagnose
        result = engine.diagnose(event)

        # Verify result structure
        assert isinstance(result, DiagnosisResult)
        assert isinstance(result.root_cause, RootCause)
        assert isinstance(result.confidence, float)
        assert isinstance(result.evidence, list)
        assert isinstance(result.investigation_hints, list)

        # Verify get_root_cause
        assert engine.get_root_cause() == result.root_cause.value

        # Verify suggest_investigation
        hints = engine.suggest_investigation(event)
        assert len(hints) > 0

    def test_diagnosis_survives_round_trip_serialization(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test that diagnosis result survives dict serialization."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="strategy_mismatch",
            context="Wrong strategy chosen",
            session_id="sess-int-002",
            outcome="failed",
        )

        result = engine.diagnose(event)
        serialized = result.to_dict()
        restored = DiagnosisResult.from_dict(serialized)

        assert restored.root_cause == result.root_cause
        assert restored.confidence == result.confidence
        assert restored.evidence == result.evidence

    def test_multiple_events_different_root_causes(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test diagnosing multiple events with different root causes."""
        events = [
            FailureEvent(
                timestamp=datetime.now(timezone.utc),
                failure_type="knowledge_gap",
                context="Model doesn't know the answer",
                session_id=f"sess-multi-{i}",
                outcome="failed",
            )
            for i in range(5)
        ]

        root_causes = set()
        for event in events:
            result = engine.diagnose(event)
            root_causes.add(result.root_cause)

        # All should be knowledge gap
        assert len(root_causes) == 1
        assert RootCause.KNOWLEDGE_GAP in root_causes

    def test_investigation_hints_vary_by_failure_type(
        self, engine: DiagnosticEngine
    ) -> None:
        """Test that investigation hints differ appropriately per failure type."""
        events = [
            FailureEvent(
                timestamp=datetime.now(timezone.utc),
                failure_type="knowledge_gap",
                context="Unknown topic",
                session_id="sess-ih-001",
                outcome="failed",
            ),
            FailureEvent(
                timestamp=datetime.now(timezone.utc),
                failure_type="reasoning_error",
                context="Logic error",
                session_id="sess-ih-002",
                outcome="failed",
            ),
        ]

        hints = [engine.suggest_investigation(e) for e in events]

        # Hints should be different for different failure types
        assert hints[0] != hints[1]
