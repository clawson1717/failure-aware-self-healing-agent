"""FailureEvent dataclass for tracking and serializing failure occurrences."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class FailureEvent:
    """Record of a failure occurrence in the self-healing agent system.

    Attributes:
        timestamp: Datetime when the failure occurred.
        failure_type: Category of failure (e.g., "knowledge_gap", "reasoning_error",
            "context_loss", "strategy_mismatch", "external_failure").
        context: Description of what was happening when failure occurred.
        session_id: Identifier for the reasoning session.
        outcome: Result of the recovery attempt ("recovered", "failed", "escalated").
        diagnosis: Root cause diagnosis (filled after diagnostic step).
        recovery_action: Recovery action taken.
    """

    timestamp: datetime
    failure_type: str
    context: str
    session_id: str
    outcome: str
    diagnosis: Optional[str] = None
    recovery_action: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the FailureEvent to a dictionary.

        Returns:
            Dictionary representation with ISO-formatted timestamp.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "failure_type": self.failure_type,
            "context": self.context,
            "session_id": self.session_id,
            "outcome": self.outcome,
            "diagnosis": self.diagnosis,
            "recovery_action": self.recovery_action,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureEvent:
        """Create a FailureEvent from a dictionary.

        Args:
            data: Dictionary with failure event data. Timestamp should be
                ISO-formatted string or a datetime object.

        Returns:
            A new FailureEvent instance.
        """
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            timestamp=timestamp,
            failure_type=data["failure_type"],
            context=data["context"],
            session_id=data["session_id"],
            outcome=data["outcome"],
            diagnosis=data.get("diagnosis"),
            recovery_action=data.get("recovery_action"),
        )

    def to_json(self) -> str:
        """Serialize the FailureEvent to a JSON string.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> FailureEvent:
        """Deserialize a FailureEvent from a JSON string.

        Args:
            json_str: JSON string representation of a FailureEvent.

        Returns:
            A new FailureEvent instance.
        """
        return cls.from_dict(json.loads(json_str))
