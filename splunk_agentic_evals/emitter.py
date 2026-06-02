import logging
from typing import List, Optional

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

from .evals.base import EvalResult

logger = logging.getLogger(__name__)


class EvalMetricEmitter:
    """
    Emits gen_ai.evaluation.score metrics to the OTel endpoint.
    These map directly to Splunk's quality score charts.
    """

    METRIC_NAME = "gen_ai.evaluation.score"

    def __init__(
        self,
        otlp_endpoint: str,
        otlp_headers: dict = None,
        otlp_insecure: bool = True,
        service_name: Optional[str] = None,
        deployment_environment: Optional[str] = None,
    ):
        exporter = OTLPMetricExporter(
            endpoint=otlp_endpoint,
            headers=otlp_headers or {},
        )
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
        resource_attrs = {}
        if service_name:
            resource_attrs["service.name"] = service_name
        if deployment_environment:
            resource_attrs["deployment.environment"] = deployment_environment
        resource = Resource.create(resource_attrs) if resource_attrs else Resource.create({})
        self._provider = MeterProvider(metric_readers=[reader], resource=resource)
        self._meter = self._provider.get_meter("splunk_agentic_evals")
        self._eval_score = self._meter.create_histogram(
            name=self.METRIC_NAME,
            description="Quality evaluation score for a GenAI span (0.0-1.0)",
            unit="1",
        )
        self._token_usage = self._meter.create_histogram(
            name="gen_ai.client.token.usage",
            description="Measures number of input and output tokens used",
            unit="{token}",
        )
        self._op_duration = self._meter.create_histogram(
            name="gen_ai.client.operation.duration",
            description="GenAI operation duration",
            unit="s",
        )
        self._agent_duration = self._meter.create_histogram(
            name="gen_ai.agent.duration",
            description="Agent invocation duration",
            unit="s",
        )

    def emit_agent_duration(self, span_attributes: dict, duration_s: float):
        """Emit gen_ai.agent.duration histogram for invoke_agent spans."""
        attrs = {
            "gen_ai.agent.name": span_attributes.get("gen_ai.agent.name", "unknown"),
            "gen_ai.operation.name": "invoke_agent",
        }
        if span_attributes.get("gen_ai.agent.id"):
            attrs["gen_ai.agent.id"] = span_attributes["gen_ai.agent.id"]
        if duration_s > 0:
            self._agent_duration.record(duration_s, attrs)

    def emit_span_metrics(self, span_attributes: dict, input_tokens: int, output_tokens: int, duration_s: float):
        """
        Emit gen_ai.client.token.usage and gen_ai.client.operation.duration histograms
        from span data. These power Splunk's AI monitoring overview/agents pages.
        """
        base_attrs = {
            "gen_ai.operation.name": span_attributes.get("gen_ai.operation.name", "unknown"),
            "gen_ai.system": span_attributes.get("gen_ai.system", "unknown"),
            "gen_ai.request.model": span_attributes.get("gen_ai.request.model", "unknown"),
        }
        if span_attributes.get("gen_ai.agent.name"):
            base_attrs["gen_ai.agent.name"] = span_attributes["gen_ai.agent.name"]

        if input_tokens:
            self._token_usage.record(input_tokens, {**base_attrs, "gen_ai.token.type": "input"})
        if output_tokens:
            self._token_usage.record(output_tokens, {**base_attrs, "gen_ai.token.type": "output"})
        if duration_s > 0:
            self._op_duration.record(duration_s, base_attrs)

    # Labels for passed/failed states per eval type, matching Splunk's conventions
    _PASS_LABELS = {
        "hallucination": "Not hallucinated",
        "relevance": "Relevant",
        "toxicity": "Not toxic",
        "bias": "Not biased",
        "sentiment": "Positive",
        "tool_selection": "Correct tool",
        "goal_completion": "Goal achieved",
        "prompt_injection": "Safe",
    }
    _FAIL_LABELS = {
        "hallucination": "Hallucinated",
        "relevance": "Not relevant",
        "toxicity": "Toxic",
        "bias": "Biased",
        "sentiment": "Negative",
        "tool_selection": "Wrong tool",
        "goal_completion": "Goal not achieved",
        "prompt_injection": "Injection detected",
    }

    def emit(self, results: List[EvalResult], span_attributes: dict):
        """
        Emit each eval result as a gen_ai.evaluation.score data point.
        Attributes follow Splunk's expected schema so scores appear in quality charts.
        """
        # Use gen_ai.system as provider if gen_ai.provider.name is unknown
        provider = span_attributes.get("gen_ai.provider.name", "")
        if not provider or provider == "unknown":
            provider = span_attributes.get("gen_ai.system", "unknown")

        base_attrs = {
            "gen_ai.agent.name": span_attributes.get("gen_ai.agent.name", "unknown"),
            "gen_ai.request.model": span_attributes.get("gen_ai.request.model", "unknown"),
            "gen_ai.provider.name": provider,
        }

        for result in results:
            attrs = {
                **base_attrs,
                "gen_ai.evaluation.name": result.name,
                "gen_ai.evaluation.passed": str(result.passed).lower(),
            }
            self._eval_score.record(result.score, attrs)
            logger.debug(
                "Emitted %s=%s for agent=%s model=%s",
                result.name,
                result.score,
                base_attrs["gen_ai.agent.name"],
                base_attrs["gen_ai.request.model"],
            )
