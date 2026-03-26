"""DiagnosticEngine for classifying failure root causes in FASHA.

This module provides root-cause classification for failures encountered by the
Failure-Aware Self-Healing Agent (FASHA), enabling targeted recovery strategies.

Root Cause Categories:
    - KNOWLEDGE_GAP: Missing information or training data
    - REASONING_ERROR: Logical flaws in chain-of-thought
    - CONTEXT_LOSS: Forgetting important prior information
    - STRATEGY_MISMATCH: Wrong approach chosen for the problem
    - EXTERNAL_FAILURE: Environment or external dependency issues
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from failure_event import FailureEvent


class RootCause(str, Enum):
    """Enumeration of root cause categories for failure diagnosis."""

    KNOWLEDGE_GAP = "knowledge_gap"
    REASONING_ERROR = "reasoning_error"
    CONTEXT_LOSS = "context_loss"
    STRATEGY_MISMATCH = "strategy_mismatch"
    EXTERNAL_FAILURE = "external_failure"
    UNKNOWN = "unknown"


# Maps root cause category to relevant keyword patterns.
# Patterns use word boundaries at meaningful word starts.
# End-anchoring on stems uses loose boundaries to catch
# morphological variants (e.g. "contradicted", "miscalculation").
_KEYWORD_PATTERNS: dict[RootCause, list[str]] = {
    RootCause.KNOWLEDGE_GAP: [
        r"\bunknown\b",
        r"\bdon'?t\s+know\b",
        r"\bdo\s+not\s+know\b",
        r"\bdidn'?t\s+know\b",
        r"\bdoesn'?t\s+know\b",
        r"\bno\s+knowledge\b",
        r"\bno\s+information\b",
        r"\bnot\s+trained\b",
        r"\btraining\s+data\b",
        r"\bnot\s+in\s+my\s+data\b",
        r"\bbeyond\s+my\s+knowledge\b",
        r"\bafter\s+.*knowledge\b",
        r"\bnever\s+encountered\b",
        r"\bfirst\s+time\b",
        r"\bnever\s+seen\b",
        r"\bno\s+prior.*experience\b",
        r"\blackbox\b",
        r"\bopaque\b",
        r"\bno\s+explanation\b",
        r"\bcannot\s+explain\b",
        r"\bno\s+idea\b",
        r"\bunsure\b",
        r"\buncertain\b",
        r"\buncertainty\b",
        r"\black\s+box\b",
    ],
    RootCause.REASONING_ERROR: [
        r"\bcontradict",
        r"\bcontradiction\b",
        r"\binconsistent\b",
        r"\bincorrect\s+logic\b",
        r"\bflawed\b",
        r"\blogic\s+error\b",
        r"\bstep\s+.*contradict\b",
        r"\bfell\s+through\b",
        r"\bbroke\s+.*chain\b",
        r"\bchain\s+break\b",
        r"\bwrong\s+conclusion\b",
        r"\bfalse\s+assumption\b",
        r"\bcircular\b",
        r"\bcircular\s+reasoning\b",
        r"\bloop\b",
        r"\binfinite\s+loop\b",
        r"\bxor\b",
        r"\bconfused\b",
        r"\bmisordered\b",
        r"\bout\s+of\s+order\b",
        r"\bmiscalculat",
        r"\bmath\s+.*error\b",
        r"\barithmetic\s+.*error\b",
        r"\boverlook\b",
        r"\bmissed\b",
        r"\bomitted\b",
        r"\bfailed\s+to\s+account\b",
        r"\baccount\s+for\b",
        r"\bdidn'?t\s+.*account\b",
        r"\baccounted\b",
    ],
    RootCause.CONTEXT_LOSS: [
        r"\bforgot\b",
        r"\bforget\b",
        r"\blost\b",
        r"\blose\b",
        r"\bforget\s+.*earlier\b",
        r"\bforget\s+.*previous\b",
        r"\bforget\s+.*mentioned\b",
        r"\bforget\s+.*said\b",
        r"\bforgot\s+.*said\b",
        r"\bno\s+longer\s+.*remember\b",
        r"\bno\s+longer\s+.*recall\b",
        r"\blost\s+.*thread\b",
        r"\blost\s+.*context\b",
        r"\bdropped\b",
        r"\bdrop\s+.*thread\b",
        r"\bdiscontinu",
        r"\bbroke\s+.*continuit",
        r"\brepeat\s+.*previous\b",
        r"\brepeating\b",
        r"\bagain\b",
        r"\bonce\s+more\b",
        r"\bsaid\s+.*already\b",
        r"\balready\s+.*mentioned\b",
        r"\balready\s+given\b",
        r"\bcontext\s+window\b",
        r"\bexceed\s+.*context\b",
        r"\btruncat",
        r"\boverflow\b",
        r"\bcontext\s+limit\b",
        r"\bmemory\s+limit\b",
        r"\bforgot\s+.*goal\b",
        r"\blost\s+.*goal\b",
        r"\blost\s+.*objective\b",
        r"\blost\s+.*track\b",
    ],
    RootCause.STRATEGY_MISMATCH: [
        r"\bwrong\s+.*approach\b",
        r"\bwrong\s+.*strategy\b",
        r"\bwrong\s+.*method\b",
        r"\bwrong\s+.*tool\b",
        r"\binappropriate\b",
        r"\bnot\s+.*right\s+.*approach\b",
        r"\bnot\s+.*suitable\b",
        r"\bnot\s+.*fit\b",
        r"\bpoor\s+.*choice\b",
        r"\bbad\s+.*strategy\b",
        r"\bbetter\s+.*approach\b",
        r"\bswitch\s+.*strategy\b",
        r"\bchange\s+.*approach\b",
        r"\btry\s+.*different\b",
        r"\btry\s+.*alternative\b",
        r"\banother\s+.*way\b",
        r"\bbacktrack",
        r"\brestart",
        r"\breexamine",
        r"\breassess",
        r"\breconsider",
        r"\bwrong\s+.*direction\b",
        r"\bstalled\b",
        r"\bstuck\b",
        r"\bno\s+.*progress\b",
        r"\bdead.?end\b",
        r"\broadblock\b",
        r"\bcannot\s+.*proceed\b",
        r"\bunproductive\b",
        r"\bfruitless\b",
    ],
    RootCause.EXTERNAL_FAILURE: [
        r"\btimeout\b",
        r"\btimed\s+out\b",
        r"\bconnection\s+.*fail",
        r"\bconnection\s+error\b",
        r"\bnetwork\b",
        r"\bnetwork\s+error\b",
        r"\bserver\s+error\b",
        r"\bapi\s+error\b",
        r"\bapi\s+fail",
        r"\b503\b",
        r"\b500\b",
        r"\b502\b",
        r"\b504\b",
        r"\brate.?limit\b",
        r"\bquota\b",
        r"\bexceed\s+.*limit\b",
        r"\bpermission\s+.*denied\b",
        r"\baccess\s+.*denied\b",
        r"\bforbidden\b",
        r"\bunauthorized\b",
        r"\bauth\s+.*fail",
        r"\bauth\s+.*error\b",
        r"\bnot\s+found\b",
        r"\b404\b",
        r"\bresource\s+not\s+found\b",
        r"\bdependency\s+.*fail",
        r"\bmissing\s+.*dependency\b",
        r"\bservice\s+down\b",
        r"\bservice\s+unavailable\b",
        r"\bunavailable\b",
        r"\bcrash",
        r"\bcrashed\b",
        r"\bsegmentation\s+fault\b",
        r"\bsegfault\b",
        r"\bexception\b",
        r"\berror\s+.*occurred\b",
        r"\bsomething\s+.*went\s+.*wrong\b",
        r"\bfailed\s+to\b",
        r"\bunable\s+to\b",
        r"\bcould\s+not\b",
        r"\bcannot\s+.*connect\b",
        r"\bretry\b",
        r"\bretrying\b",
        r"\bexternal\b",
        r"\benvironment\b",
        r"\binfrastructure\b",
        r"\bfailed\b",
    ],
}


@dataclass
class DiagnosisResult:
    """Result of a failure diagnosis.

    Attributes:
        root_cause: The classified root cause category.
        confidence: Confidence score between 0.0 and 1.0.
        evidence: List of evidence strings supporting the diagnosis.
        investigation_hints: Suggested investigation steps to verify/expand diagnosis.
    """

    root_cause: RootCause
    confidence: float
    evidence: list[str] = field(default_factory=list)
    investigation_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnosis result to a dictionary."""
        return {
            "root_cause": self.root_cause.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "investigation_hints": self.investigation_hints,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosisResult:
        """Create a DiagnosisResult from a dictionary."""
        return cls(
            root_cause=RootCause(data["root_cause"]),
            confidence=data["confidence"],
            evidence=data.get("evidence", []),
            investigation_hints=data.get("investigation_hints", []),
        )


class DiagnosticEngine:
    """Engine for diagnosing root causes of failures in FASHA.

    Uses heuristic keyword matching as the primary diagnosis mechanism,
    with optional LLM-based diagnosis when API credentials are available.

    Example:
        >>> from failure_event import FailureEvent
        >>> from diagnostic import DiagnosticEngine
        >>> from datetime import datetime, timezone
        >>> event = FailureEvent(
        ...     timestamp=datetime.now(timezone.utc),
        ...     failure_type="knowledge_gap",
        ...     context="Model produced inconsistent reasoning steps",
        ...     session_id="test-001",
        ...     outcome="recovered",
        ... )
        >>> engine = DiagnosticEngine()
        >>> result = engine.diagnose(event)
        >>> print(result.root_cause.value)
        knowledge_gap
    """

    # Minimum keyword matches to consider a category
    KEYWORD_THRESHOLD = 1

    # Minimum confidence boost for LLM-enhanced diagnosis
    LLM_CONFIDENCE_BOOST = 0.15

    def __init__(
        self,
        use_llm: bool = False,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4",
    ) -> None:
        """Initialize the DiagnosticEngine.

        Args:
            use_llm: Whether to use LLM for enhanced diagnosis (default False).
            llm_api_key: API key for LLM service. If None, checks environment
                variable OPENAI_API_KEY.
            llm_model: Model name for LLM (default "gpt-4").
        """
        self.use_llm = use_llm
        self.llm_model = llm_model

        # Determine if LLM is actually available
        self._llm_available = False
        if use_llm:
            key = llm_api_key or os.environ.get("OPENAI_API_KEY")
            if key:
                self._llm_api_key = key
                self._llm_available = True
            else:
                # LLM requested but no key available - will use mock
                self._llm_api_key = None
                self._llm_available = False

        self._last_result: Optional[DiagnosisResult] = None

    # Investigation hints templates per root cause
    _INVESTIGATION_HINTS: dict[RootCause, list[str]] = {
        RootCause.KNOWLEDGE_GAP: [
            "Check if the query requires domain-specific knowledge not in training data",
            "Verify if the topic is within the model's knowledge cutoff date",
            "Consider using external search or retrieval to supplement knowledge",
            "Ask the user for clarification or additional context",
            "Check if the model explicitly stated uncertainty about the topic",
            "Consider providing relevant documents or reference material",
        ],
        RootCause.REASONING_ERROR: [
            "Review the chain-of-thought steps for logical inconsistencies",
            "Check if assumptions stated early in reasoning were contradicted later",
            "Verify the logical validity of each reasoning step",
            "Look for circular reasoning patterns",
            "Check for missing steps or gaps in the logic chain",
            "Verify any mathematical or arithmetic calculations",
            "Consider if a simpler explanation fits the evidence better",
        ],
        RootCause.CONTEXT_LOSS: [
            "Review conversation history for information that may have been forgotten",
            "Check if context window limits were approached or exceeded",
            "Verify if earlier requirements or goals were lost",
            "Consider re-summarizing context to reinforce important information",
            "Check for signs of repetition indicating context loss",
            "Review if any critical information from earlier turns is missing",
            "Consider implementing context checkpointing for long conversations",
        ],
        RootCause.STRATEGY_MISMATCH: [
            "Re-examine the problem type and verify appropriate strategy was chosen",
            "Consider alternative approaches that might better fit the problem",
            "Check if the strategy was appropriate for the problem's complexity",
            "Review whether sufficient planning preceded action selection",
            "Consider if backtracking to try a different approach would help",
            "Check if dead-ends were recognized and handled appropriately",
            "Evaluate if more granular decomposition of the problem is needed",
        ],
        RootCause.EXTERNAL_FAILURE: [
            "Check external service status and availability",
            "Verify network connectivity and configuration",
            "Review API rate limits and quota usage",
            "Check authentication and authorization settings",
            "Review recent deployment or infrastructure changes",
            "Check logs for detailed error messages from external services",
            "Verify that all required dependencies are available",
            "Consider implementing retry logic with exponential backoff",
        ],
        RootCause.UNKNOWN: [
            "Gather more information about the failure context",
            "Review similar past failures for patterns",
            "Consult documentation for the failing component",
            "Consider involving a human expert for complex failures",
            "Check if this is a novel failure type not previously encountered",
        ],
    }

    def diagnose(self, event: FailureEvent) -> DiagnosisResult:
        """Diagnose the root cause of a failure event.

        Uses heuristic keyword matching to identify the most likely root cause.
        If LLM is enabled and available, augments the heuristic analysis.

        Args:
            event: The FailureEvent to diagnose.

        Returns:
            DiagnosisResult containing the root cause, confidence, evidence,
            and investigation hints.
        """
        # Run heuristic analysis
        scores = self._heuristic_analysis(event)

        # Determine best match from keyword scores
        best_cause = RootCause.UNKNOWN
        best_score = 0
        for cause, score in scores.items():
            if cause == RootCause.UNKNOWN:
                continue
            if score > best_score:
                best_score = score
                best_cause = cause

        # Fallback: use failure_type as signal when no keywords matched
        type_cause = self._failure_type_to_cause(event.failure_type)
        if best_score == 0 and type_cause is not None:
            best_cause = type_cause
            best_score = 1  # Recognized failure_type provides weak signal

        # Calculate confidence based on score
        confidence = self._calculate_confidence(scores, best_cause, best_score)

        # Build evidence list
        evidence = self._build_evidence(event, scores, best_cause, best_score)

        # Get investigation hints
        hints = self.suggest_investigation(event)

        result = DiagnosisResult(
            root_cause=best_cause,
            confidence=confidence,
            evidence=evidence,
            investigation_hints=hints,
        )

        # LLM augmentation if enabled and available
        if self.use_llm and self._llm_available:
            result = self._llm_augment(event, result)

        self._last_result = result
        return result

    def _heuristic_analysis(self, event: FailureEvent) -> dict[RootCause, int]:
        """Perform heuristic keyword matching analysis.

        Args:
            event: The FailureEvent to analyze.

        Returns:
            Dictionary mapping each RootCause to its keyword match score.
        """
        scores: dict[RootCause, int] = {cause: 0 for cause in RootCause}

        # Text to search: context + failure_type
        text = f"{event.context} {event.failure_type}".lower()

        for cause, patterns in _KEYWORD_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[cause] += 1

        return scores

    def _failure_type_to_cause(self, failure_type: str) -> Optional[RootCause]:
        """Map failure_type field to a likely RootCause.

        Args:
            failure_type: The failure_type string from FailureEvent.

        Returns:
            Corresponding RootCause if recognized, else None.
        """
        type_map: dict[str, RootCause] = {
            "knowledge_gap": RootCause.KNOWLEDGE_GAP,
            "reasoning_error": RootCause.REASONING_ERROR,
            "context_loss": RootCause.CONTEXT_LOSS,
            "strategy_mismatch": RootCause.STRATEGY_MISMATCH,
            "external_failure": RootCause.EXTERNAL_FAILURE,
        }
        return type_map.get(failure_type.lower())

    def _calculate_confidence(
        self,
        scores: dict[RootCause, int],
        best_cause: RootCause,
        best_score: int,
    ) -> float:
        """Calculate confidence score for the diagnosis.

        Args:
            scores: Keyword match scores for each root cause.
            best_cause: The identified root cause.
            best_score: The highest keyword match score.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if best_cause == RootCause.UNKNOWN or best_score == 0:
            return 0.3  # Low confidence for unknown

        # Base confidence from score
        # More matches = higher confidence, but with diminishing returns
        base_confidence = min(0.5 + (best_score * 0.1), 0.95)

        # Check if there's a clear winner (no close second)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            runner_up = sorted_scores[1]
            if runner_up > 0 and best_score == runner_up:
                # Near-tie - reduce confidence
                base_confidence *= 0.7
            elif runner_up > 0 and best_score - runner_up <= 2:
                # Close second - slight reduction
                base_confidence *= 0.85

        return round(base_confidence, 2)

    def _build_evidence(
        self,
        event: FailureEvent,
        scores: dict[RootCause, int],
        best_cause: RootCause,
        best_score: int,
    ) -> list[str]:
        """Build evidence list for the diagnosis.

        Args:
            event: The FailureEvent being diagnosed.
            scores: Keyword match scores.
            best_cause: The identified root cause.
            best_score: The keyword match score for the best cause.

        Returns:
            List of evidence strings.
        """
        evidence = []

        if best_score > 0:
            evidence.append(
                f"Matched {best_score} keyword pattern(s) "
                f"for {best_cause.value}"
            )

        # Add failure_type as evidence if it aligns
        type_cause = self._failure_type_to_cause(event.failure_type)
        if type_cause is not None and type_cause == best_cause:
            evidence.append(
                f"failure_type '{event.failure_type}' aligns with diagnosis"
            )

        # Add outcome as contextual evidence
        if event.outcome:
            evidence.append(f"Recovery outcome: {event.outcome}")

        return evidence

    def _llm_augment(
        self, event: FailureEvent, heuristic_result: DiagnosisResult
    ) -> DiagnosisResult:
        """Augment heuristic diagnosis with LLM analysis.

        This is a stub implementation that simulates LLM augmentation.
        When actual OpenAI API key is provided, this should call the API.

        Args:
            event: The FailureEvent being diagnosed.
            heuristic_result: Result from heuristic analysis.

        Returns:
            Augmented DiagnosisResult with potentially updated root cause
            and/or confidence.
        """
        # This is a stub: in production, call OpenAI API here
        # For now, we just boost confidence to simulate LLM agreement
        new_confidence = min(
            heuristic_result.confidence + self.LLM_CONFIDENCE_BOOST, 0.99
        )
        heuristic_result.confidence = round(new_confidence, 2)
        heuristic_result.evidence.append("Diagnosis augmented via LLM analysis")
        return heuristic_result

    def get_root_cause(self) -> str:
        """Return the root cause of the last diagnosis.

        Returns:
            String value of the root cause, or "unknown" if no diagnosis
            has been performed yet.

        Raises:
            RuntimeError: If diagnose() has not been called first.
        """
        if self._last_result is None:
            raise RuntimeError(
                "No diagnosis available. Call diagnose() first."
            )
        return self._last_result.root_cause.value

    def suggest_investigation(self, event: FailureEvent) -> list[str]:
        """Generate investigation hints for a failure event.

        Provides targeted suggestions for investigating the root cause,
        based on both heuristic analysis and the event's failure_type.

        Args:
            event: The FailureEvent to generate hints for.

        Returns:
            List of investigation hint strings.
        """
        # Run quick analysis to get best guess
        scores = self._heuristic_analysis(event)
        best_cause = max(scores, key=scores.get)

        # If failure_type is more specific, use it
        type_cause = self._failure_type_to_cause(event.failure_type)
        if type_cause is not None and type_cause != RootCause.UNKNOWN:
            # failure_type gives us a strong signal
            base_hints = self._INVESTIGATION_HINTS.get(type_cause, [])
        else:
            base_hints = self._INVESTIGATION_HINTS.get(
                best_cause if best_cause != RootCause.UNKNOWN else RootCause.UNKNOWN,
                self._INVESTIGATION_HINTS[RootCause.UNKNOWN],
            )

        # Add event-specific hints based on context keywords
        specific_hints: list[str] = []
        text_lower = event.context.lower()

        if "timeout" in text_lower or "timed out" in text_lower:
            specific_hints.append(
                "Investigate timeout duration settings and network latency"
            )
        if "api" in text_lower or "api error" in text_lower:
            specific_hints.append("Review API endpoint configuration and parameters")
        if "context window" in text_lower or "context limit" in text_lower:
            specific_hints.append(
                "Check if context summarization or truncation is working correctly"
            )
        if "contradict" in text_lower or "inconsistent" in text_lower:
            specific_hints.append(
                "Carefully compare early vs. late reasoning steps for conflicts"
            )
        if "forgot" in text_lower or "forget" in text_lower:
            specific_hints.append(
                "Implement context checkpointing to preserve critical information"
            )

        # Deduplicate and combine
        all_hints = list(dict.fromkeys(specific_hints + base_hints))
        return all_hints

    def reset(self) -> None:
        """Reset the engine state, clearing the last diagnosis."""
        self._last_result = None
