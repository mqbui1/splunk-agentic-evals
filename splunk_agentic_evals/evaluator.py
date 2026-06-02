import logging
import random
from typing import List, Optional

from .config import EvalConfig
from .judge import LLMJudge
from .emitter import EvalMetricEmitter
from .evals.base import BaseEval, EvalResult
from .evals.builtin import HallucinationEval, RelevanceEval, ToxicityEval, BiasEval, SentimentEval
from .evals.custom import ToolSelectionEval, GoalCompletionEval, PromptInjectionEval

logger = logging.getLogger(__name__)

_EVAL_REGISTRY = {
    "hallucination": HallucinationEval,
    "relevance": RelevanceEval,
    "toxicity": ToxicityEval,
    "bias": BiasEval,
    "sentiment": SentimentEval,
    "tool_selection": ToolSelectionEval,
    "goal_completion": GoalCompletionEval,
    "prompt_injection": PromptInjectionEval,
}


class GenAIEvaluator:
    """
    Main entry point. Runs enabled evals against a span's input/output
    and emits gen_ai.evaluation.score metrics to Splunk via OTel.
    """

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        if self.config.custom_judge is not None:
            self._judge = self.config.custom_judge
        else:
            self._judge = LLMJudge(
                model_id=self.config.judge_model,
                aws_region=self.config.aws_region,
                bedrock_profile_arn=self.config.bedrock_profile_arn,
            )
        self._emitter = EvalMetricEmitter(
            otlp_endpoint=self.config.otlp_endpoint,
            otlp_headers=self.config.otlp_headers,
            otlp_insecure=self.config.otlp_insecure,
            service_name=self.config.service_name,
            deployment_environment=self.config.deployment_environment,
        )
        self._evals: List[BaseEval] = self._build_evals()

    def _build_evals(self) -> List[BaseEval]:
        evals = []
        for name in self.config.enabled_evals:
            cls = _EVAL_REGISTRY.get(name)
            if cls:
                evals.append(cls(judge=self._judge, pass_threshold=self.config.pass_threshold))
            else:
                logger.warning("Unknown eval '%s' — skipping", name)
        return evals

    def register_eval(self, eval_instance: BaseEval):
        """Register a custom eval at runtime."""
        self._evals.append(eval_instance)

    def run(self, input: str, output: str, span_attributes: dict) -> List[EvalResult]:
        """
        Run all enabled evals and emit results as OTel metrics.

        Args:
            input: the prompt/user message sent to the LLM or agent
            output: the response from the LLM or agent
            span_attributes: dict of OTel span attributes from the gen_ai span
        Returns:
            list of EvalResult (also emitted as metrics)
        """
        if random.random() > self.config.sample_rate:
            logger.debug("Span skipped due to sample_rate=%.2f", self.config.sample_rate)
            return []

        results = []
        for ev in self._evals:
            try:
                result = ev.evaluate(input=input, output=output, context=span_attributes)
                results.append(result)
            except Exception as e:
                logger.warning("Eval '%s' raised an exception: %s", ev.name, e)

        if results:
            self._emitter.emit(results, span_attributes)

        return results
