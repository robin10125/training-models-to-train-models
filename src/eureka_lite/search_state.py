from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .rewards import RewardCandidate


class GenerationPhase(str, Enum):
    NEEDS_POPULATION = "needs_population"
    EVALUATING = "evaluating"


@dataclass
class GenerationState:
    index: int
    phase: GenerationPhase
    candidates: list[RewardCandidate] = field(default_factory=list)
    raw_results: list[Any] = field(default_factory=list)
    rejected_candidates: list[tuple[RewardCandidate, str]] = field(default_factory=list)


@dataclass
class SearchState:
    finalized_results: list[Any] = field(default_factory=list)
    generation: GenerationState | None = None
    best_expression: str | None = None
    best_score: float | None = None
    elite_context: list[dict[str, Any]] = field(default_factory=list)
    evolution_feedback: str | None = None
    completed: bool = False

