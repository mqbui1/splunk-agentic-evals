"""
Example: integrate splunk-agentic-evals with an existing OTel-instrumented agent.

Two usage patterns:
  1. SpanProcessor (automatic) — wraps your OTLP exporter, intercepts spans on export
  2. Direct API (manual) — call evaluator.run() explicitly after each LLM call
"""

# ── Pattern 1: Automatic via SpanProcessor ────────────────────────────────────
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from splunk_agentic_evals import GenAIEvalSpanProcessor, EvalConfig

config = EvalConfig(
    otlp_endpoint="http://localhost:4317",
    judge_model="claude-haiku-4-5-20251001",
    enabled_evals=["hallucination", "relevance", "toxicity", "tool_selection", "goal_completion", "prompt_injection"],
    sample_rate=1.0,
    async_eval=True,
)

otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
eval_exporter = GenAIEvalSpanProcessor(downstream=otlp_exporter, config=config)

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(eval_exporter))

# Your agent runs normally — evals fire automatically on export


# ── Pattern 2: Direct API ─────────────────────────────────────────────────────
from splunk_agentic_evals import GenAIEvaluator

evaluator = GenAIEvaluator(config=config)

results = evaluator.run(
    input="Book me the cheapest flight from SFO to JFK next Friday",
    output="I found 3 options. The cheapest is United at $289 departing 6am.",
    span_attributes={
        "gen_ai.agent.name": "travel-planner",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.provider.name": "openai",
        "gen_ai.operation.name": "invoke_agent",
        "tool_calls": ["flight_search", "price_compare"],
    },
)

for r in results:
    status = "PASS" if r.passed else "FAIL"
    print(f"[{status}] {r.name}: {r.score:.2f} — {r.reason}")
