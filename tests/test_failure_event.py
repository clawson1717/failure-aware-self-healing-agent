"""Unit tests for FailureEvent dataclass."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from failure_event import FailureEvent


class TestFailureEventCreation:
    """Tests for FailureEvent instantiation."""

    def test_create_with_required_fields_only(self) -> None:
        """Test creating a FailureEvent with only required fields."""
        timestamp = datetime(2026, 3, 21, 12, 0, 0, tzinfo=timezone.utc)
        event = FailureEvent(
            timestamp=timestamp,
            failure_type="knowledge_gap",
            context="Model lacked training data for specific query",
            session_id="session-abc-123",
            outcome="recovered",
        )

        assert event.timestamp == timestamp
        assert event.failure_type == "knowledge_gap"
        assert event.context == "Model lacked training data for specific query"
        assert event.session_id == "session-abc-123"
        assert event.outcome == "recovered"
        assert event.diagnosis is None
        assert event.recovery_action is None

    def test_create_with_all_fields(self) -> None:
        """Test creating a FailureEvent with all fields populated."""
        timestamp = datetime(2026, 3, 21, 14, 30, 0, tzinfo=timezone.utc)
        event = FailureEvent(
            timestamp=timestamp,
            failure_type="reasoning_error",
            context="Model produced inconsistent reasoning steps",
            session_id="session-xyz-789",
            outcome="recovered",
            diagnosis="Flawed chain-of-thought logic",
            recovery_action="Restarted reasoning with corrected prompt",
        )

        assert event.timestamp == timestamp
        assert event.failure_type == "reasoning_error"
        assert event.context == "Model produced inconsistent reasoning steps"
        assert event.session_id == "session-xyz-789"
        assert event.outcome == "recovered"
        assert event.diagnosis == "Flawed chain-of-thought logic"
        assert event.recovery_action == "Restarted reasoning with corrected prompt"

    def test_create_with_all_failure_types(self) -> None:
        """Test creating events with all defined failure type categories."""
        valid_types = [
            "knowledge_gap",
            "reasoning_error",
            "context_loss",
            "strategy_mismatch",
            "external_failure",
        ]
        for failure_type in valid_types:
            event = FailureEvent(
                timestamp=datetime.now(timezone.utc),
                failure_type=failure_type,
                context="Test context",
                session_id="session-test",
                outcome="recovered",
            )
            assert event.failure_type == failure_type

    def test_create_with_all_outcomes(self) -> None:
        """Test creating events with all valid outcome values."""
        valid_outcomes = ["recovered", "failed", "escalated"]
        for outcome in valid_outcomes:
            event = FailureEvent(
                timestamp=datetime.now(timezone.utc),
                failure_type="external_failure",
                context="Test context",
                session_id="session-test",
                outcome=outcome,
            )
            assert event.outcome == outcome


class TestFieldAccess:
    """Tests for field access patterns."""

    def test_field_access_via_attribute(self) -> None:
        """Test accessing fields as object attributes."""
        event = FailureEvent(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            failure_type="context_loss",
            context="Context window exceeded",
            session_id="sess-001",
            outcome="escalated",
            diagnosis="Context overflow",
            recovery_action="Summarized and continued",
        )

        assert event.timestamp.year == 2026
        assert event.failure_type == "context_loss"
        assert event.context == "Context window exceeded"
        assert event.session_id == "sess-001"
        assert event.outcome == "escalated"
        assert event.diagnosis == "Context overflow"
        assert event.recovery_action == "Summarized and continued"

    def test_optional_fields_default_to_none(self) -> None:
        """Test that optional fields default to None when not provided."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context="Missing info",
            session_id="sess-002",
            outcome="failed",
        )
        assert event.diagnosis is None
        assert event.recovery_action is None


class TestToDict:
    """Tests for to_dict serialization."""

    def test_to_dict_with_all_fields(self) -> None:
        """Test to_dict with all fields populated."""
        timestamp = datetime(2026, 3, 21, 10, 0, 0, tzinfo=timezone.utc)
        event = FailureEvent(
            timestamp=timestamp,
            failure_type="strategy_mismatch",
            context="Chosen strategy inappropriate for task",
            session_id="sess-003",
            outcome="recovered",
            diagnosis="Mismatch between strategy and problem type",
            recovery_action="Switched to alternative strategy",
        )

        result = event.to_dict()

        assert isinstance(result, dict)
        assert result["timestamp"] == "2026-03-21T10:00:00+00:00"
        assert result["failure_type"] == "strategy_mismatch"
        assert result["context"] == "Chosen strategy inappropriate for task"
        assert result["session_id"] == "sess-003"
        assert result["outcome"] == "recovered"
        assert result["diagnosis"] == "Mismatch between strategy and problem type"
        assert result["recovery_action"] == "Switched to alternative strategy"

    def test_to_dict_with_optional_fields_none(self) -> None:
        """Test to_dict when optional fields are None."""
        event = FailureEvent(
            timestamp=datetime(2026, 3, 21, tzinfo=timezone.utc),
            failure_type="external_failure",
            context="External API timeout",
            session_id="sess-004",
            outcome="escalated",
        )

        result = event.to_dict()

        assert result["diagnosis"] is None
        assert result["recovery_action"] is None

    def test_to_dict_timestamp_is_iso_format(self) -> None:
        """Test that timestamp in to_dict is ISO 8601 formatted."""
        timestamp = datetime(2026, 3, 21, 15, 45, 30, 123000, tzinfo=timezone.utc)
        event = FailureEvent(
            timestamp=timestamp,
            failure_type="reasoning_error",
            context="Test",
            session_id="sess-005",
            outcome="recovered",
        )

        result = event.to_dict()

        assert result["timestamp"] == "2026-03-21T15:45:30.123000+00:00"
        # Verify it's parseable
        parsed = datetime.fromisoformat(result["timestamp"])
        assert parsed == timestamp


class TestFromDict:
    """Tests for from_dict deserialization."""

    def test_from_dict_with_all_fields(self) -> None:
        """Test creating FailureEvent from dict with all fields."""
        data: dict[str, Any] = {
            "timestamp": "2026-03-21T16,00:00+00:00",
            "failure_type": "context_loss",
            "context": "Context was truncated",
            "session_id": "sess-006",
            "outcome": "recovered",
            "diagnosis": "Exceeded context window",
            "recovery_action": "Reloaded context",
        }

        # Note: using valid ISO format here
        data["timestamp"] = "2026-03-21T16:00:00+00:00"
        event = FailureEvent.from_dict(data)

        assert event.timestamp == datetime(2026, 3, 21, 16, 0, 0, tzinfo=timezone.utc)
        assert event.failure_type == "context_loss"
        assert event.context == "Context was truncated"
        assert event.session_id == "sess-006"
        assert event.outcome == "recovered"
        assert event.diagnosis == "Exceeded context window"
        assert event.recovery_action == "Reloaded context"

    def test_from_dict_with_missing_optional_fields(self) -> None:
        """Test from_dict when optional fields are absent."""
        data: dict[str, Any] = {
            "timestamp": "2026-03-21T17:00:00+00:00",
            "failure_type": "knowledge_gap",
            "context": "Unknown domain",
            "session_id": "sess-007",
            "outcome": "failed",
        }

        event = FailureEvent.from_dict(data)

        assert event.diagnosis is None
        assert event.recovery_action is None

    def test_from_dict_with_none_optional_fields(self) -> None:
        """Test from_dict when optional fields are explicitly None."""
        data: dict[str, Any] = {
            "timestamp": "2026-03-21T18:00:00+00:00",
            "failure_type": "reasoning_error",
            "context": "Bad logic",
            "session_id": "sess-008",
            "outcome": "escalated",
            "diagnosis": None,
            "recovery_action": None,
        }

        event = FailureEvent.from_dict(data)

        assert event.diagnosis is None
        assert event.recovery_action is None

    def test_from_dict_with_datetime_timestamp(self) -> None:
        """Test from_dict accepts datetime objects for timestamp."""
        dt = datetime(2026, 3, 21, 19, 0, 0, tzinfo=timezone.utc)
        data: dict[str, Any] = {
            "timestamp": dt,
            "failure_type": "external_failure",
            "context": "Service unavailable",
            "session_id": "sess-009",
            "outcome": "recovered",
        }

        event = FailureEvent.from_dict(data)

        assert event.timestamp == dt


class TestToJson:
    """Tests for to_json serialization."""

    def test_to_json_produces_valid_json(self) -> None:
        """Test that to_json produces parseable JSON."""
        event = FailureEvent(
            timestamp=datetime(2026, 3, 21, 20, 0, 0, tzinfo=timezone.utc),
            failure_type="strategy_mismatch",
            context="Wrong strategy selected",
            session_id="sess-010",
            outcome="recovered",
        )

        json_str = event.to_json()

        # Should not raise
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["failure_type"] == "strategy_mismatch"

    def test_to_json_contains_all_fields(self) -> None:
        """Test that JSON output contains all fields."""
        event = FailureEvent(
            timestamp=datetime(2026, 3, 21, 21, 0, 0, tzinfo=timezone.utc),
            failure_type="knowledge_gap",
            context="Missing knowledge",
            session_id="sess-011",
            outcome="failed",
            diagnosis="Training data gap",
            recovery_action="External lookup",
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        expected_keys = {"timestamp", "failure_type", "context", "session_id",
                         "outcome", "diagnosis", "recovery_action"}
        assert set(parsed.keys()) == expected_keys


class TestFromJson:
    """Tests for from_json deserialization."""

    def test_from_json_round_trip(self) -> None:
        """Test that from_json undoes to_json."""
        original = FailureEvent(
            timestamp=datetime(2026, 3, 21, 22, 0, 0, tzinfo=timezone.utc),
            failure_type="context_loss",
            context="Context overflow",
            session_id="sess-012",
            outcome="escalated",
            diagnosis="Window exceeded",
            recovery_action="Summarized history",
        )

        json_str = original.to_json()
        restored = FailureEvent.from_json(json_str)

        assert restored.timestamp == original.timestamp
        assert restored.failure_type == original.failure_type
        assert restored.context == original.context
        assert restored.session_id == original.session_id
        assert restored.outcome == original.outcome
        assert restored.diagnosis == original.diagnosis
        assert restored.recovery_action == original.recovery_action

    def test_from_json_preserves_nested_data(self) -> None:
        """Test that complex context survives round-trip."""
        original = FailureEvent(
            timestamp=datetime(2026, 3, 21, 23, 0, 0, tzinfo=timezone.utc),
            failure_type="reasoning_error",
            context="Step 3 contradicted Step 1",
            session_id="sess-013",
            outcome="recovered",
            diagnosis="Logic error in step 3",
            recovery_action="Corrected reasoning chain",
        )

        json_str = original.to_json()
        restored = FailureEvent.from_json(json_str)

        assert restored.context == "Step 3 contradicted Step 1"
        assert restored.diagnosis == "Logic error in step 3"
        assert restored.recovery_action == "Corrected reasoning chain"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_fields(self) -> None:
        """Test that empty strings are valid for string fields."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="",
            context="",
            session_id="",
            outcome="recovered",
        )

        assert event.failure_type == ""
        assert event.context == ""
        assert event.session_id == ""

    def test_unicode_in_fields(self) -> None:
        """Test that unicode characters are handled correctly."""
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="外部障害",
            context="日本語のコンテキスト",
            session_id="sess-日本語-001",
            outcome="恢復",
            diagnosis="根本原因",
            recovery_action="復旧手順",
        )

        json_str = event.to_json()
        restored = FailureEvent.from_json(json_str)

        assert restored.failure_type == "外部障害"
        assert restored.context == "日本語のコンテキスト"
        assert restored.session_id == "sess-日本語-001"

    def test_special_characters_in_context(self) -> None:
        """Test context with special characters survives round-trip."""
        # Use explicit escape sequences to avoid confusion
        original_context = 'Special chars: "quotes" \\ backslash \\n newline \t tab'
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="external_failure",
            context=original_context,
            session_id="sess-special",
            outcome="recovered",
        )

        json_str = event.to_json()
        restored = FailureEvent.from_json(json_str)

        # The context has literal \n (backslash + n) and \t (tab)
        assert restored.context == original_context

    def test_very_long_field_values(self) -> None:
        """Test that very long field values work correctly."""
        long_context = "x" * 10000
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type="knowledge_gap",
            context=long_context,
            session_id="sess-long",
            outcome="recovered",
        )

        json_str = event.to_json()
        restored = FailureEvent.from_json(json_str)

        assert restored.context == long_context

    def test_naive_datetime(self) -> None:
        """Test that naive datetime (no timezone) is accepted."""
        naive_dt = datetime(2026, 3, 21, 12, 0, 0)
        event = FailureEvent(
            timestamp=naive_dt,
            failure_type="knowledge_gap",
            context="Test",
            session_id="sess-naive",
            outcome="recovered",
        )

        assert event.timestamp == naive_dt
        assert event.timestamp.tzinfo is None
