"""Recovery Strategy Library for FASHA.

This module provides recovery strategies for handling failures diagnosed by the
Failure-Aware Self-Healing Agent (FASHA), enabling targeted recovery actions based
on the root cause diagnosis.

Recovery Strategies:
    - ReReadStrategy: Re-reads context to recover from Context Loss
    - AskClarificationStrategy: Requests user clarification for Knowledge Gaps
    - BacktrackStrategy: Returns to previous reasoning state for Reasoning Errors
    - RetryDifferentApproachStrategy: Tries alternative approach for Strategy Mismatch
    - SimplifyStrategy: Simplifies the problem/subtask for External Failures
    - EscalateStrategy: Escalates to human or higher authority for unrecoverable failures
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from diagnostic import DiagnosisResult, RootCause


@dataclass
class RecoveryResult:
    """Result of a recovery action.

    Attributes:
        success: Whether the recovery was successful.
        strategy_used: Name of the recovery strategy that was applied.
        outcome_message: Human-readable description of the outcome.
        recovery_data: Additional data from the recovery action.
    """

    success: bool
    strategy_used: str
    outcome_message: str
    recovery_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert recovery result to a dictionary."""
        return {
            "success": self.success,
            "strategy_used": self.strategy_used,
            "outcome_message": self.outcome_message,
            "recovery_data": self.recovery_data,
        }


class RecoveryStrategy(ABC):
    """Abstract base class for recovery strategies.

    All concrete recovery strategies must inherit from this class and implement
    the applicability_conditions() and execute() methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this strategy."""
        ...

    @property
    @abstractmethod
    def expected_outcome(self) -> str:
        """Return a description of the expected outcome of this strategy."""
        ...

    @abstractmethod
    def applicability_conditions(self, diagnosis: DiagnosisResult) -> float:
        """Evaluate how applicable this strategy is to the given diagnosis.

        Args:
            diagnosis: The diagnosis result containing the root cause analysis.

        Returns:
            A float between 0.0 and 1.0 indicating how well this strategy
            fits the diagnosed failure. Higher values indicate better fit.
        """
        ...

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> RecoveryResult:
        """Execute the recovery action.

        Args:
            context: Dictionary containing context needed for recovery,
                such as the original failure event, session state, etc.

        Returns:
            RecoveryResult indicating whether the recovery was successful
            and any relevant recovery data.
        """
        ...


class ReReadStrategy(RecoveryStrategy):
    """Recovery strategy for Context Loss failures.

    This strategy addresses context loss by re-reading or refreshing the context
    to recover lost information. It is most applicable when the diagnosis indicates
    that the agent forgot or lost important prior information.
    """

    name: str = "re_read"
    expected_outcome: str = (
        "Context is re-established and agent can continue with recovered information"
    )

    def applicability_conditions(self, diagnosis: DiagnosisResult) -> float:
        """Check applicability for context loss.

        Returns a high score (0.9) for CONTEXT_LOSS diagnoses, as this strategy
        directly addresses forgetting or losing context. Returns lower scores
        for other failure types.

        Args:
            diagnosis: The diagnosis result.

        Returns:
            Applicability score between 0.0 and 1.0.
        """
        if diagnosis.root_cause == RootCause.CONTEXT_LOSS:
            return 0.9
        elif diagnosis.root_cause == RootCause.UNKNOWN:
            return 0.2
        elif diagnosis.root_cause == RootCause.REASONING_ERROR:
            # Reasoning errors can sometimes be caused by missing context
            return 0.3
        return 0.1

    def execute(self, context: dict[str, Any]) -> RecoveryResult:
        """Execute context re-reading recovery.

        Retrieves and re-establishes context from session state or history.

        Args:
            context: Dictionary that should contain 'session_history' or
                similar context data.

        Returns:
            RecoveryResult with success status and recovered context data.
        """
        session_history = context.get("session_history", [])
        recovered_info = context.get("last_context", "")

        # Simulate re-reading context from history
        if session_history:
            # Extract key information from history
            recovered_items = []
            for item in session_history[-5:]:  # Last 5 items
                if isinstance(item, dict) and "content" in item:
                    recovered_items.append(item["content"])
                elif isinstance(item, str):
                    recovered_items.append(item)

            recovery_data = {
                "re_read_items": recovered_items,
                "context_restored": len(recovered_items) > 0,
            }
            success = len(recovered_items) > 0
            outcome_message = (
                f"Re-read {len(recovered_items)} context items from history"
                if success
                else "No context history available to re-read"
            )
        elif recovered_info:
            recovery_data = {
                "recovered_context": recovered_info,
                "context_restored": True,
            }
            success = True
            outcome_message = "Re-established context from last known state"
        else:
            recovery_data = {
                "context_restored": False,
            }
            success = False
            outcome_message = "No context available to re-read"

        return RecoveryResult(
            success=success,
            strategy_used=self.name,
            outcome_message=outcome_message,
            recovery_data=recovery_data,
        )


class AskClarificationStrategy(RecoveryStrategy):
    """Recovery strategy for Knowledge Gap failures.

    This strategy addresses knowledge gaps by requesting clarification or
    additional information from the user. It is most applicable when the
    diagnosis indicates missing information that the user can provide.
    """

    name: str = "ask_clarification"
    expected_outcome: str = (
        "User provides needed information and agent can proceed with complete knowledge"
    )

    def applicability_conditions(self, diagnosis: DiagnosisResult) -> float:
        """Check applicability for knowledge gaps.

        Returns a high score (0.9) for KNOWLEDGE_GAP diagnoses, as this strategy
        directly addresses missing information by requesting it from the user.

        Args:
            diagnosis: The diagnosis result.

        Returns:
            Applicability score between 0.0 and 1.0.
        """
        if diagnosis.root_cause == RootCause.KNOWLEDGE_GAP:
            return 0.9
        elif diagnosis.root_cause == RootCause.UNKNOWN:
            return 0.3
        elif diagnosis.root_cause == RootCause.CONTEXT_LOSS:
            # Context loss might need user clarification too
            return 0.3
        return 0.1

    def execute(self, context: dict[str, Any]) -> RecoveryResult:
        """Execute clarification request recovery.

        Generates a clarification request to the user.

        Args:
            context: Dictionary containing failure context and evidence.

        Returns:
            RecoveryResult with clarification request details.
        """
        evidence = context.get("evidence", [])
        failure_type = context.get("failure_type", "unknown")

        # Generate clarification question based on evidence
        clarification_prompt = (
            "I need more information to help with this. "
            "Could you please clarify: What specific information would help resolve this?"
        )

        if evidence:
            clarification_prompt = (
                f"I encountered a knowledge gap. Based on the evidence: "
                f"{'; '.join(evidence[:2])} - Could you provide more details or "
                f"clarify your expectations?"
            )

        recovery_data = {
            "clarification_requested": True,
            "clarification_prompt": clarification_prompt,
            "knowledge_gap_type": failure_type,
        }

        return RecoveryResult(
            success=True,  # Request was generated successfully
            strategy_used=self.name,
            outcome_message="Clarification request generated for user",
            recovery_data=recovery_data,
        )


class BacktrackStrategy(RecoveryStrategy):
    """Recovery strategy for Reasoning Error failures.

    This strategy addresses reasoning errors by returning to a previous
    reasoning state and attempting to correct the logical flaw. It is most
    applicable when the diagnosis indicates a logical inconsistency or
    flawed reasoning chain.
    """

    name: str = "backtrack"
    expected_outcome: str = (
        "Agent returns to previous valid state and can re-reason without error"
    )

    def applicability_conditions(self, diagnosis: DiagnosisResult) -> float:
        """Check applicability for reasoning errors.

        Returns a high score (0.9) for REASONING_ERROR diagnoses, as this
        strategy directly addresses logical flaws by backtracking to a
        valid previous state.

        Args:
            diagnosis: The diagnosis result.

        Returns:
            Applicability score between 0.0 and 1.0.
        """
        if diagnosis.root_cause == RootCause.REASONING_ERROR:
            return 0.9
        elif diagnosis.root_cause == RootCause.STRATEGY_MISMATCH:
            # Strategy mismatch can sometimes be fixed by backtracking
            return 0.4
        elif diagnosis.root_cause == RootCause.UNKNOWN:
            return 0.2
        return 0.1

    def execute(self, context: dict[str, Any]) -> RecoveryResult:
        """Execute backtracking recovery.

        Returns to a previous reasoning state and re-attempts reasoning.

        Args:
            context: Dictionary containing reasoning history.

        Returns:
            RecoveryResult with backtracking details.
        """
        reasoning_history = context.get("reasoning_history", [])
        checkpoint = context.get("last_checkpoint", None)

        if reasoning_history and len(reasoning_history) > 1:
            # Go back to previous state
            previous_state = reasoning_history[-2] if len(reasoning_history) >= 2 else None
            recovery_data = {
                "backtracked": True,
                "previous_state": previous_state,
                "states_removed": len(reasoning_history) - 1 if previous_state else 0,
                "checkpoint_restored": checkpoint is not None,
            }
            success = previous_state is not None
            outcome_message = (
                f"Backtracked to previous reasoning state, "
                f"removed {recovery_data['states_removed']} erroneous steps"
                if success
                else "No previous state available to backtrack to"
            )
        elif checkpoint:
            recovery_data = {
                "backtracked": True,
                "checkpoint_restored": True,
                "checkpoint": checkpoint,
            }
            success = True
            outcome_message = "Restored from last checkpoint"
        else:
            recovery_data = {
                "backtracked": False,
            }
            success = False
            outcome_message = "No reasoning history or checkpoint available to backtrack"

        return RecoveryResult(
            success=success,
            strategy_used=self.name,
            outcome_message=outcome_message,
            recovery_data=recovery_data,
        )


class RetryDifferentApproachStrategy(RecoveryStrategy):
    """Recovery strategy for Strategy Mismatch failures.

    This strategy addresses strategy mismatches by attempting a different
    approach to solve the problem. It is most applicable when the diagnosis
    indicates the wrong approach was chosen for the problem.
    """

    name: str = "retry_different_approach"
    expected_outcome: str = (
        "Agent attempts problem with alternative strategy and achieves better results"
    )

    def applicability_conditions(self, diagnosis: DiagnosisResult) -> float:
        """Check applicability for strategy mismatches.

        Returns a high score (0.9) for STRATEGY_MISMATCH diagnoses, as this
        strategy directly addresses wrong approach selection by trying
        a different strategy.

        Args:
            diagnosis: The diagnosis result.

        Returns:
            Applicability score between 0.0 and 1.0.
        """
        if diagnosis.root_cause == RootCause.STRATEGY_MISMATCH:
            return 0.9
        elif diagnosis.root_cause == RootCause.EXTERNAL_FAILURE:
            # External failures might benefit from retrying
            return 0.3
        elif diagnosis.root_cause == RootCause.UNKNOWN:
            return 0.2
        return 0.1

    def execute(self, context: dict[str, Any]) -> RecoveryResult:
        """Execute retry with different approach.

        Selects and attempts an alternative approach.

        Args:
            context: Dictionary containing failed approach info.

        Returns:
            RecoveryResult with alternative approach details.
        """
        failed_approach = context.get("failed_approach", "unknown")
        available_approaches = context.get(
            "available_approaches",
            ["decomposition", "search", "reasoning", "simplification"],
        )

        # Select a different approach
        if failed_approach in available_approaches:
            alternative_approaches = [a for a in available_approaches if a != failed_approach]
        else:
            alternative_approaches = available_approaches

        if alternative_approaches:
            new_approach = alternative_approaches[0]
            recovery_data = {
                "new_approach": new_approach,
                "previous_approach": failed_approach,
                "retry_initiated": True,
            }
            success = True
            outcome_message = (
                f"Switching from '{failed_approach}' to alternative approach: '{new_approach}'"
            )
        else:
            recovery_data = {
                "retry_initiated": False,
            }
            success = False
            outcome_message = "No alternative approaches available"

        return RecoveryResult(
            success=success,
            strategy_used=self.name,
            outcome_message=outcome_message,
            recovery_data=recovery_data,
        )


class SimplifyStrategy(RecoveryStrategy):
    """Recovery strategy for External Failure situations.

    This strategy addresses external failures by simplifying the problem
    or breaking it into smaller, more manageable subtasks. It is most
    applicable when external dependencies are unavailable or unreliable.
    """

    name: str = "simplify"
    expected_outcome: str = (
        "Problem is simplified or decomposed into smaller tasks that can be completed"
    )

    def applicability_conditions(self, diagnosis: DiagnosisResult) -> float:
        """Check applicability for external failures.

        Returns a high score (0.9) for EXTERNAL_FAILURE diagnoses, as this
        strategy addresses external dependency issues by simplifying
        requirements.

        Args:
            diagnosis: The diagnosis result.

        Returns:
            Applicability score between 0.0 and 1.0.
        """
        if diagnosis.root_cause == RootCause.EXTERNAL_FAILURE:
            return 0.9
        elif diagnosis.root_cause == RootCause.STRATEGY_MISMATCH:
            # Simplification can help when current approach is too complex
            return 0.4
        elif diagnosis.root_cause == RootCause.UNKNOWN:
            return 0.2
        return 0.1

    def execute(self, context: dict[str, Any]) -> RecoveryResult:
        """Execute problem simplification.

        Simplifies the problem or creates subtasks.

        Args:
            context: Dictionary containing problem details.

        Returns:
            RecoveryResult with simplification details.
        """
        problem = context.get("problem", context.get("task", "unknown task"))
        max_subtasks = context.get("max_subtasks", 3)

        # Create simplified subtasks
        subtasks = []
        for i in range(1, max_subtasks + 1):
            subtasks.append(f"Subtask {i}: Simplified portion of '{problem}'")

        if subtasks:
            recovery_data = {
                "original_problem": problem,
                "subtasks": subtasks,
                "simplification_applied": True,
                "subtask_count": len(subtasks),
            }
            success = True
            outcome_message = (
                f"Simplified '{problem}' into {len(subtasks)} manageable subtasks"
            )
        else:
            recovery_data = {
                "simplification_applied": False,
            }
            success = False
            outcome_message = "Failed to simplify problem"

        return RecoveryResult(
            success=success,
            strategy_used=self.name,
            outcome_message=outcome_message,
            recovery_data=recovery_data,
        )


class EscalateStrategy(RecoveryStrategy):
    """Recovery strategy for unrecoverable failures.

    This strategy is used when all other recovery attempts have failed
    or when the failure requires human intervention. It escalates to
    a human or higher authority for resolution.
    """

    name: str = "escalate"
    expected_outcome: str = (
        "Issue is escalated to human or higher authority for resolution"
    )

    def applicability_conditions(self, diagnosis: DiagnosisResult) -> float:
        """Check applicability for escalation.

        Returns a moderate-high score (0.7) for UNKNOWN diagnoses, as
        escalation is appropriate when the root cause cannot be determined.
        Also returns high scores when confidence is very low, indicating
        the agent cannot reliably recover.

        Args:
            diagnosis: The diagnosis result.

        Returns:
            Applicability score between 0.0 and 1.0.
        """
        if diagnosis.root_cause == RootCause.UNKNOWN:
            return 0.8
        elif diagnosis.confidence < 0.3:
            # Low confidence suggests agent cannot handle this
            return 0.7
        elif diagnosis.root_cause == RootCause.EXTERNAL_FAILURE:
            # External failures may need human intervention
            return 0.4
        return 0.2

    def execute(self, context: dict[str, Any]) -> RecoveryResult:
        """Execute escalation.

        Creates an escalation request for human or higher authority.

        Args:
            context: Dictionary containing failure details for escalation.

        Returns:
            RecoveryResult with escalation details.
        """
        failure_type = context.get("failure_type", "unknown")
        evidence = context.get("evidence", [])
        session_id = context.get("session_id", "unknown")

        escalation_request = {
            "session_id": session_id,
            "failure_type": failure_type,
            "evidence": evidence,
            "priority": "high" if context.get("confidence", 0.5) < 0.3 else "medium",
        }

        recovery_data = {
            "escalation_requested": True,
            "escalation_request": escalation_request,
            "escalation_timestamp": context.get("timestamp"),
        }

        return RecoveryResult(
            success=True,  # Escalation request was created
            strategy_used=self.name,
            outcome_message="Escalated to human support for resolution",
            recovery_data=recovery_data,
        )


class RecoveryStrategyLibrary:
    """Library of available recovery strategies.

    This class manages all recovery strategies and provides methods to
    select the best strategy for a given diagnosis or to retrieve
    strategies by name.

    Example:
        >>> from diagnostic import DiagnosticEngine, DiagnosisResult, RootCause
        >>> from failure_event import FailureEvent
        >>> from datetime import datetime, timezone
        >>> from recovery import RecoveryStrategyLibrary
        >>>
        >>> # Create diagnosis
        >>> event = FailureEvent(
        ...     timestamp=datetime.now(timezone.utc),
        ...     failure_type="context_loss",
        ...     context="Agent forgot the user's name",
        ...     session_id="test-001",
        ...     outcome="failed",
        ... )
        >>> engine = DiagnosticEngine()
        >>> diagnosis = engine.diagnose(event)
        >>>
        >>> # Select recovery strategy
        >>> library = RecoveryStrategyLibrary()
        >>> strategy = library.select_strategy(diagnosis)
        >>> print(strategy.name)
        re_read
    """

    def __init__(self) -> None:
        """Initialize the strategy library with all available strategies."""
        self.strategies: list[RecoveryStrategy] = [
            ReReadStrategy(),
            AskClarificationStrategy(),
            BacktrackStrategy(),
            RetryDifferentApproachStrategy(),
            SimplifyStrategy(),
            EscalateStrategy(),
        ]

    def select_strategy(self, diagnosis: DiagnosisResult) -> RecoveryStrategy:
        """Select the best-fit recovery strategy for the given diagnosis.

        Evaluates all strategies' applicability conditions and returns
        the strategy with the highest score.

        Args:
            diagnosis: The diagnosis result to select a strategy for.

        Returns:
            The RecoveryStrategy with the highest applicability score.

        Raises:
            ValueError: If no strategies are available.
        """
        if not self.strategies:
            raise ValueError("No recovery strategies available")

        best_strategy: RecoveryStrategy | None = None
        best_score = -1.0

        for strategy in self.strategies:
            score = strategy.applicability_conditions(diagnosis)
            if score > best_score:
                best_score = score
                best_strategy = strategy

        # best_strategy is guaranteed to be non-None when strategies is non-empty
        return best_strategy

    def get_strategy(self, name: str) -> RecoveryStrategy:
        """Get a strategy by its name.

        Args:
            name: The name of the strategy to retrieve.

        Returns:
            The RecoveryStrategy with the given name.

        Raises:
            KeyError: If no strategy with the given name exists.
        """
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy
        raise KeyError(f"Strategy '{name}' not found")

    def list_strategies(self) -> list[str]:
        """List all available strategy names.

        Returns:
            List of strategy names in the order they were added.
        """
        return [strategy.name for strategy in self.strategies]
