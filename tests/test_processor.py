import pytest
from unittest.mock import MagicMock, patch
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

from splunk_agentic_evals.processor import GenAIEvalSpanProcessor
from splunk_agentic_evals.config import EvalConfig


def _make_span(attributes: dict = None, events: list = None, span_id: int = 0x1234567890abcdef) -> MagicMock:
    span = MagicMock(spec=ReadableSpan)
    span.attributes = attributes or {}
    span.events = events or []
    span.context.span_id = span_id
    span.parent = None
    span.resource = None
    return span


def _make_event(name: str, attributes: dict = None) -> MagicMock:
    event = MagicMock()
    event.name = name
    event.attributes = attributes or {}
    return event


@pytest.fixture
def processor():
    config = EvalConfig(
        otlp_endpoint="http://localhost:4317",
        enabled_evals=["hallucination"],
        async_eval=False,
        sample_rate=1.0,
        deployment_environment="test-env",
    )
    downstream = MagicMock()
    downstream.export.return_value = SpanExportResult.SUCCESS

    with patch("boto3.client"), patch("splunk_agentic_evals.emitter.OTLPMetricExporter"), \
         patch("splunk_agentic_evals.emitter.MeterProvider"):
        proc = GenAIEvalSpanProcessor(downstream=downstream, config=config)
        proc._evaluator.run = MagicMock(return_value=[])
        return proc


def test_export_forwards_spans_to_downstream(processor):
    spans = [_make_span({"gen_ai.operation.name": "chat"})]
    result = processor.export(spans)
    assert result == SpanExportResult.SUCCESS
    processor._downstream.export.assert_called_once()


def test_export_skips_non_genai_spans(processor):
    spans = [_make_span({"gen_ai.operation.name": "embeddings"})]
    processor.export(spans)
    processor._evaluator.run.assert_not_called()


def test_export_skips_span_with_no_input(processor):
    spans = [_make_span({"gen_ai.operation.name": "chat"})]  # no input/output
    processor.export(spans)
    processor._evaluator.run.assert_not_called()


def test_export_extracts_input_from_event(processor):
    event = _make_event("gen_ai.user.message", {"content": "Hello agent"})
    output_event = _make_event("gen_ai.choice", {"content": "Hello user"})
    span = _make_span(
        {"gen_ai.operation.name": "invoke_agent"},
        events=[event, output_event],
    )
    processor.export([span])
    processor._evaluator.run.assert_called_once()
    call_args = processor._evaluator.run.call_args
    assert call_args[0][0] == "Hello agent"
    assert call_args[0][1] == "Hello user"


def test_export_extracts_input_from_span_attribute(processor):
    span = _make_span({
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.prompt": "Book a flight",
        "gen_ai.completion": "Done, booked.",
    })
    processor.export([span])
    processor._evaluator.run.assert_called_once()
    call_args = processor._evaluator.run.call_args
    assert call_args[0][0] == "Book a flight"
    assert call_args[0][1] == "Done, booked."


def test_export_extracts_tool_calls_from_events(processor):
    tool_event = _make_event("gen_ai.tool.call", {"gen_ai.tool.name": "flight_search"})
    input_event = _make_event("gen_ai.user.message", {"content": "find flights"})
    output_event = _make_event("gen_ai.choice", {"content": "found 3 flights"})
    span = _make_span(
        {"gen_ai.operation.name": "invoke_agent"},
        events=[input_event, output_event, tool_event],
    )
    processor.export([span])
    call_args = processor._evaluator.run.call_args
    attrs = call_args[0][2]
    assert "tool_calls" in attrs
    assert "flight_search" in attrs["tool_calls"]


def test_normalize_adds_sf_environment(processor):
    span = _make_span({"gen_ai.operation.name": "chat", "gen_ai.system": "strands-agents"})
    processor.export([span])
    exported = processor._downstream.export.call_args[0][0]
    assert exported[0].attributes.get("sf_environment") == "test-env"


def test_export_passes_span_attributes(processor):
    span = _make_span({
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "travel-bot",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.provider.name": "openai",
        "gen_ai.prompt": "hello",
        "gen_ai.completion": "hi",
    })
    processor.export([span])
    call_args = processor._evaluator.run.call_args
    attrs = call_args[0][2]
    assert attrs["gen_ai.agent.name"] == "travel-bot"
    assert attrs["gen_ai.request.model"] == "gpt-4o"
