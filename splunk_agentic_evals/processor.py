import concurrent.futures
import json
import logging
from typing import Optional

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import Event
from opentelemetry.trace import SpanKind

from .evaluator import GenAIEvaluator
from .config import EvalConfig

logger = logging.getLogger(__name__)

# Only evaluate at the agent/workflow level — these spans have gen_ai.agent.name set
# and represent the full interaction, not individual LLM sub-calls
_EVAL_OPERATIONS = {"invoke_agent", "invoke_workflow"}


class AgentContextPropagator(SpanProcessor):
    """
    Propagates gen_ai.agent.name from parent spans to child spans at creation time.

    Strands only sets gen_ai.agent.name on the root invoke_agent span. Child spans
    (chat, execute_tool, etc.) close before the root span and get exported in an
    earlier batch — without gen_ai.agent.name — so they don't appear in Splunk's
    AI Trace Data table.

    Register this BEFORE BatchSpanProcessor so the attribute is set before batching:
        provider.add_span_processor(AgentContextPropagator())
        provider.add_span_processor(BatchSpanProcessor(eval_exporter))
    """

    _PROPAGATE_ATTRS = ("gen_ai.agent.name",)

    def on_start(self, span, parent_context=None):
        if parent_context is None:
            return
        parent_span = trace_api.get_current_span(parent_context)
        if parent_span is None or not parent_span.is_recording():
            return
        attrs = getattr(parent_span, "attributes", None)
        if not attrs:
            return
        for key in self._PROPAGATE_ATTRS:
            value = attrs.get(key)
            if value and not (span.attributes and span.attributes.get(key)):
                span.set_attribute(key, value)

    def on_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


class GenAIEvalSpanProcessor:
    """
    OTel SpanExporter wrapper that intercepts gen_ai spans on export,
    extracts input/output, runs evals, and emits scores — then forwards
    spans to the downstream exporter unchanged.

    Usage:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from splunk_agentic_evals import GenAIEvalSpanProcessor, EvalConfig

        eval_exporter = GenAIEvalSpanProcessor(
            downstream=OTLPSpanExporter(endpoint="http://localhost:4317"),
            config=EvalConfig(),
        )
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(eval_exporter))
    """

    def __init__(
        self,
        downstream: SpanExporter,
        config: Optional[EvalConfig] = None,
    ):
        self._downstream = downstream
        resolved_config = config or EvalConfig()
        self._evaluator = GenAIEvaluator(config=resolved_config)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._sf_environment = resolved_config.deployment_environment or ""

    def export(self, spans) -> SpanExportResult:
        # span_id (hex) -> span, for ancestor lookup during normalization
        span_map = {format(s.context.span_id, "016x"): s for s in spans}
        # span_id (hex) -> List[EvalResult] for sync evals to embed in span attributes
        span_eval_results: dict = {}

        for span in spans:
            operation = self._get_attr(span, "gen_ai.operation.name", "")

            # Emit gen_ai.client.* histogram metrics (powers AI overview + AI agents pages)
            if operation in ("chat", "invoke_agent"):
                self._emit_span_metrics(span)

            if operation not in _EVAL_OPERATIONS:
                continue

            input_text = self._extract_input(span)
            output_text = self._extract_output(span)

            if not input_text or not output_text:
                continue

            attrs = self._extract_attrs(span)

            if self._evaluator.config.async_eval:
                self._executor.submit(self._evaluator.run, input_text, output_text, attrs)
            else:
                results = self._evaluator.run(input_text, output_text, attrs)
                if results:
                    span_id = format(span.context.span_id, "016x")
                    span_eval_results[span_id] = results

        normalized = [self._normalize_span(s, span_eval_results, span_map) for s in spans]
        return self._downstream.export(normalized)

    def shutdown(self):
        self._executor.shutdown(wait=True)
        self._downstream.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._downstream.force_flush(timeout_millis)

    # Events whose content attributes may be JSON-wrapped by Strands
    _NORMALIZABLE_EVENTS = frozenset({
        "gen_ai.user.message",
        "gen_ai.system.message",
        "gen_ai.assistant.message",
        "gen_ai.choice",
        "gen_ai.tool.message",
    })
    # Attribute keys that carry message content (unwrap JSON-wrapped text)
    _CONTENT_KEYS = ("content", "message", "gen_ai.prompt", "gen_ai.completion")

    # Map Strands' gen_ai.system value to the actual provider Splunk recognizes
    _SYSTEM_MAP = {
        "strands-agents": "aws.bedrock",
    }

    def _normalize_span(self, span: ReadableSpan, eval_results: dict = None, span_map: dict = None) -> ReadableSpan:
        """
        Reconstruct span with normalizations for Splunk compatibility:
        1. Unwrap Strands JSON-wrapped content '[{"text":"..."}]' → plain string
        2. Rename gen_ai.choice 'message' attribute → 'content' (Splunk expects 'content')
        3. Map gen_ai.system 'strands-agents' → 'aws.bedrock' so Splunk recognizes the provider
        4. Propagate gen_ai.agent.name from ancestor invoke_agent spans to child chat spans
           so they appear in Splunk's AI Trace Data table (filtered by agent name)
        Returns the original span unchanged if no normalization was needed.
        """
        new_events = []
        events_changed = False

        for event in span.events:
            if event.name not in self._NORMALIZABLE_EVENTS:
                new_events.append(event)
                continue

            attrs = dict(event.attributes) if event.attributes else {}
            event_changed = False

            # 1. Unwrap JSON-wrapped content
            for key in self._CONTENT_KEYS:
                if key in attrs:
                    original = str(attrs[key])
                    unwrapped = self._unwrap_content(original)
                    if unwrapped != original:
                        attrs[key] = unwrapped
                        event_changed = True

            # 2. For gen_ai.choice: rename 'message' → 'content' so Splunk can parse it
            if event.name == "gen_ai.choice" and "message" in attrs and "content" not in attrs:
                attrs["content"] = attrs.pop("message")
                event_changed = True

            if event_changed:
                new_events.append(Event(name=event.name, attributes=attrs, timestamp=event.timestamp))
                events_changed = True
            else:
                new_events.append(event)

        # 3. Fix gen_ai.system so Splunk recognizes the AI provider
        span_attrs = dict(span.attributes) if span.attributes else {}
        current_system = span_attrs.get("gen_ai.system", "")
        mapped_system = self._SYSTEM_MAP.get(current_system)
        attrs_changed = False
        if mapped_system:
            span_attrs["gen_ai.system"] = mapped_system
            attrs_changed = True

        # 4. Propagate gen_ai.agent.name from ancestor spans (e.g. invoke_agent → chat)
        #    Splunk's AI Trace Data table filters by this attribute, so child spans without
        #    it won't appear when filtered by agent name.
        if not span_attrs.get("gen_ai.agent.name") and span_map:
            agent_name = self._find_ancestor_attr(span, "gen_ai.agent.name", span_map)
            if agent_name:
                span_attrs["gen_ai.agent.name"] = agent_name
                attrs_changed = True

        # 5. Promote content from events → span attributes for Splunk AI Trace Data table.
        #    Splunk queries gen_ai.prompt / gen_ai.completion span attributes for the Content column.
        if span_attrs.get("gen_ai.operation.name") and not span_attrs.get("gen_ai.prompt"):
            input_text = self._extract_input(span)
            if input_text:
                span_attrs["gen_ai.prompt"] = input_text[:2048]
                attrs_changed = True
        if span_attrs.get("gen_ai.operation.name") and not span_attrs.get("gen_ai.completion"):
            output_text = self._extract_output(span)
            if output_text:
                span_attrs["gen_ai.completion"] = output_text[:2048]
                attrs_changed = True

        # 6. Add sf_environment and deployment.environment as span attributes for Splunk AI trace search.
        #    The AI trace search filters by the span-level tag sf_environment, but standard
        #    OTLP ingest only sets deployment.environment in resource attributes. APM converts
        #    it, but the AI trace search backend does not — so we must set both explicitly.
        if self._sf_environment:
            if not span_attrs.get("sf_environment"):
                span_attrs["sf_environment"] = self._sf_environment
                attrs_changed = True
            if not span_attrs.get("deployment.environment"):
                span_attrs["deployment.environment"] = self._sf_environment
                attrs_changed = True

        # 7. Set gen_ai.evaluation.sampled = true so Splunk's AI trace indexing pipeline
        #    picks up this span and includes it in the AI Trace Data table.
        #    Splunk uses this attribute to opt spans into AI monitoring evaluation.
        if span_attrs.get("gen_ai.operation.name") and not span_attrs.get("gen_ai.evaluation.sampled"):
            span_attrs["gen_ai.evaluation.sampled"] = True
            attrs_changed = True

        # Embed eval scores into the span attributes so Splunk can read them from traces
        if eval_results:
            span_id = format(span.context.span_id, "016x")
            results = eval_results.get(span_id, [])
            for result in results:
                span_attrs[f"gen_ai.evaluation.{result.name}.score"] = result.score
                span_attrs[f"gen_ai.evaluation.{result.name}.passed"] = str(result.passed).lower()
            if results:
                attrs_changed = True

        if not events_changed and not attrs_changed:
            return span

        return ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=span_attrs if attrs_changed else span.attributes,
            events=new_events,
            links=span.links,
            kind=SpanKind.CLIENT if span_attrs.get("gen_ai.operation.name") else span.kind,
            instrumentation_scope=getattr(span, "instrumentation_scope", None),
            status=span.status,
            start_time=span.start_time,
            end_time=span.end_time,
        )

    @staticmethod
    def _find_ancestor_attr(span: ReadableSpan, key: str, span_map: dict) -> str:
        """Walk up the span tree via parent_id to find the nearest ancestor with the given attribute."""
        current = span
        while current.parent:
            parent_id = format(current.parent.span_id, "016x")
            parent = span_map.get(parent_id)
            if not parent:
                break
            value = parent.attributes.get(key, "") if parent.attributes else ""
            if value:
                return str(value)
            current = parent
        return ""

    def _emit_span_metrics(self, span: ReadableSpan):
        """Emit gen_ai.client.* and gen_ai.agent.duration metrics from a span."""
        try:
            input_tokens = int(self._get_attr(span, "gen_ai.usage.input_tokens", "0") or 0)
            output_tokens = int(self._get_attr(span, "gen_ai.usage.output_tokens", "0") or 0)
            duration_s = 0.0
            if span.end_time and span.start_time:
                duration_s = (span.end_time - span.start_time) / 1e9
            raw_system = self._get_attr(span, "gen_ai.system", "unknown")
            operation = self._get_attr(span, "gen_ai.operation.name", "unknown")
            span_attrs = {
                "gen_ai.operation.name": operation,
                "gen_ai.system": self._SYSTEM_MAP.get(raw_system, raw_system),
                "gen_ai.request.model": self._get_attr(span, "gen_ai.request.model", "unknown"),
                "gen_ai.agent.name": self._get_attr(span, "gen_ai.agent.name", ""),
                "gen_ai.agent.id": format(span.context.span_id, "016x"),
            }
            self._evaluator._emitter.emit_span_metrics(span_attrs, input_tokens, output_tokens, duration_s)
            if operation == "invoke_agent":
                self._evaluator._emitter.emit_agent_duration(span_attrs, duration_s)
        except Exception as e:
            logger.debug("Failed to emit span metrics: %s", e)

    def _extract_input(self, span: ReadableSpan) -> str:
        for event in span.events:
            if event.name in ("gen_ai.user.message", "gen_ai.prompt"):
                raw = (
                    event.attributes.get("gen_ai.prompt")
                    or event.attributes.get("content", "")
                )
                if raw:
                    return self._unwrap_content(str(raw))
        return self._get_attr(span, "gen_ai.prompt", "") or self._get_attr(span, "input.value", "")

    def _extract_output(self, span: ReadableSpan) -> str:
        for event in span.events:
            if event.name in ("gen_ai.choice", "gen_ai.completion"):
                # Strands uses "message" key; standard OTel uses "content" or "gen_ai.completion"
                raw = (
                    event.attributes.get("message")
                    or event.attributes.get("gen_ai.completion")
                    or event.attributes.get("content", "")
                )
                if raw:
                    return self._unwrap_content(str(raw))
        return self._get_attr(span, "gen_ai.completion", "") or self._get_attr(span, "output.value", "")

    @staticmethod
    def _unwrap_content(raw: str) -> str:
        """
        Strands wraps content as JSON: '[{"text": "actual message"}]'
        Extract the plain text value if present, otherwise return raw.
        """
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                texts = [
                    item.get("text", "")
                    for item in parsed
                    if isinstance(item, dict) and "text" in item
                ]
                result = " ".join(t for t in texts if t).strip()
                return result if result else raw
        except (json.JSONDecodeError, AttributeError):
            pass
        return raw

    def _extract_attrs(self, span: ReadableSpan) -> dict:
        keys = [
            "gen_ai.agent.name",
            "gen_ai.request.model",
            "gen_ai.provider.name",
            "gen_ai.operation.name",
            "gen_ai.workflow.name",
            "gen_ai.system",
        ]
        attrs = {k: self._get_attr(span, k, "unknown") for k in keys}
        # Apply system mapping so downstream gets aws.bedrock instead of strands-agents
        raw_system = attrs.get("gen_ai.system", "unknown")
        attrs["gen_ai.system"] = self._SYSTEM_MAP.get(raw_system, raw_system)

        # Extract tool results from gen_ai.tool.message events — used as ground truth for hallucination eval
        tool_results = []
        for e in span.events:
            if e.name == "gen_ai.tool.message":
                raw = e.attributes.get("content", "")
                if raw:
                    tool_results.append(self._unwrap_content(str(raw)))
        if tool_results:
            attrs["tool_results"] = tool_results

        # Extract tool call names — check events first, then gen_ai.agent.tools attribute
        tool_calls = [
            e.attributes.get("gen_ai.tool.name", "")
            for e in span.events
            if e.name == "gen_ai.tool.call"
        ]
        if not tool_calls:
            tools_attr = self._get_attr(span, "gen_ai.agent.tools", "")
            if tools_attr:
                try:
                    tool_calls = json.loads(tools_attr)
                except (json.JSONDecodeError, TypeError):
                    pass
        if tool_calls:
            attrs["tool_calls"] = [t for t in tool_calls if t]

        return attrs

    @staticmethod
    def _get_attr(span: ReadableSpan, key: str, default: str = "") -> str:
        if span.attributes and key in span.attributes:
            return str(span.attributes[key])
        return default
