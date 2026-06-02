from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalResult:
    name: str        # maps to gen_ai.evaluation.name
    score: float     # 0.0 - 1.0
    passed: bool     # score >= pass_threshold
    reason: Optional[str] = None


class BaseEval(ABC):
    """Base class for all evaluators."""

    name: str  # must match gen_ai.evaluation.name values Splunk recognizes

    @abstractmethod
    def evaluate(self, input: str, output: str, context: dict) -> EvalResult:
        """
        Args:
            input: the user prompt / agent input
            output: the LLM / agent response
            context: span attributes (tool calls, agent name, model, etc.)
        Returns:
            EvalResult with score 0.0-1.0
        """
