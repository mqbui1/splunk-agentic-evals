"""
Travel planner agent using Strands + Claude Haiku on Bedrock.
Instrumented with OTel gen_ai spans + splunk-agentic-evals processor.
"""

import os
import json
import logging

# Clear any OTel env vars injected by the host process (e.g. Claude Code sets
# env=prod in OTEL_RESOURCE_ATTRIBUTES, which causes spans to be indexed under
# the wrong environment in Splunk APM).
os.environ.pop("OTEL_RESOURCE_ATTRIBUTES", None)
os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)

from strands import Agent, tool
from strands.models import BedrockModel
from strands.telemetry import StrandsTelemetry

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from splunk_agentic_evals import GenAIEvalSpanProcessor, AgentContextPropagator, EvalConfig

from .tools import search_flights, search_hotels, get_weather

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── OTel + Eval setup ─────────────────────────────────────────────────────────

SPLUNK_TOKEN = os.getenv("SPLUNK_INGEST_TOKEN", "")
SPLUNK_REALM = os.getenv("SPLUNK_REALM", "us1")
SPLUNK_TRACE_ENDPOINT = f"https://ingest.{SPLUNK_REALM}.signalfx.com/v2/trace/otlp"
SPLUNK_METRIC_ENDPOINT = f"https://ingest.{SPLUNK_REALM}.signalfx.com/v2/datapoint/otlp"
DEPLOYMENT_ENV = os.getenv("DEPLOYMENT_ENV", "splunk-agentic-evals-test")
USE_CONSOLE = os.getenv("USE_CONSOLE_EXPORTER", "false").lower() == "true"
# Set OTEL_COLLECTOR_ENDPOINT to route through a local Splunk OTel Collector
# e.g. OTEL_COLLECTOR_ENDPOINT=http://localhost:4318
# When set, traces bypass the Splunk ingest endpoint and go via the collector instead.
OTEL_COLLECTOR_ENDPOINT = os.getenv("OTEL_COLLECTOR_ENDPOINT", "")

splunk_headers = {"X-SF-Token": SPLUNK_TOKEN} if SPLUNK_TOKEN else {}

# When routing through a local collector, metrics also go there
metric_endpoint = f"{OTEL_COLLECTOR_ENDPOINT}/v1/metrics" if OTEL_COLLECTOR_ENDPOINT else SPLUNK_METRIC_ENDPOINT

config = EvalConfig(
    otlp_endpoint=metric_endpoint,
    judge_model="arn:aws:bedrock:us-west-2:387769110234:application-inference-profile/fky19kpnw2m7",
    enabled_evals=[
        "hallucination",
        "relevance",
        "toxicity",
        "bias",
        "sentiment",
        "tool_selection",
        "goal_completion",
        "prompt_injection",
    ],
    sample_rate=1.0,
    async_eval=False,
    aws_region=os.getenv("AWS_DEFAULT_REGION", "us-west-2"),
    bedrock_profile_arn="arn:aws:bedrock:us-west-2:387769110234:application-inference-profile/fky19kpnw2m7",
    otlp_headers=splunk_headers,
    otlp_insecure=False,
    service_name="travel-planner",
    deployment_environment=DEPLOYMENT_ENV,
)

if USE_CONSOLE:
    downstream = ConsoleSpanExporter()
elif OTEL_COLLECTOR_ENDPOINT:
    downstream = OTLPSpanExporter(endpoint=f"{OTEL_COLLECTOR_ENDPOINT}/v1/traces")
else:
    downstream = OTLPSpanExporter(
        endpoint=SPLUNK_TRACE_ENDPOINT,
        headers=splunk_headers,
    )

eval_exporter = GenAIEvalSpanProcessor(downstream=downstream, config=config)
provider = TracerProvider(resource=Resource.create({
    "service.name": "travel-planner",
    "deployment.environment": DEPLOYMENT_ENV,
}))
provider.add_span_processor(AgentContextPropagator())
provider.add_span_processor(BatchSpanProcessor(eval_exporter))

# Set as global provider BEFORE Strands initializes so it doesn't override ours
trace_api.set_tracer_provider(provider)

# Pass our provider to Strands so its spans flow through our eval processor
StrandsTelemetry(tracer_provider=provider)

# ── Strands tool definitions ──────────────────────────────────────────────────

@tool
def find_flights(origin: str, destination: str, date: str) -> str:
    """Search for available flights. Args: origin (IATA code), destination (IATA code), date (YYYY-MM-DD)."""
    result = search_flights(origin, destination, date)
    return json.dumps(result)


@tool
def find_hotels(city: str, check_in: str, check_out: str) -> str:
    """Search for hotels in a city. Args: city, check_in (YYYY-MM-DD), check_out (YYYY-MM-DD)."""
    result = search_hotels(city, check_in, check_out)
    return json.dumps(result)


@tool
def check_weather(city: str, date: str) -> str:
    """Get weather forecast for a city. Args: city, date (YYYY-MM-DD)."""
    result = get_weather(city, date)
    return json.dumps(result)


# ── Agent ─────────────────────────────────────────────────────────────────────

def build_agent() -> Agent:
    model_id = os.getenv(
        "BEDROCK_MODEL_ID",
        "arn:aws:bedrock:us-west-2:387769110234:application-inference-profile/fky19kpnw2m7",
    )
    model = BedrockModel(
        model_id=model_id,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-west-2"),
    )

    return Agent(
        name="travel-planner",
        model=model,
        tools=[find_flights, find_hotels, check_weather],
        system_prompt=(
            "You are a helpful travel planning assistant. "
            "Use the available tools to find flights, hotels, and weather information. "
            "Always provide specific options with prices and details."
        ),
    )
