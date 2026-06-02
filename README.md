# splunk-agentic-evals

Instrumentation-side evaluation library for Splunk AI Agent Monitoring. Runs LLM-as-a-judge evals on your agent spans and emits `gen_ai.evaluation.score` metrics via OpenTelemetry so they appear in Splunk's AI Agents quality charts.

## How it works

The library installs as an OTel `SpanExporter` wrapper (`GenAIEvalSpanProcessor`). When a `gen_ai` span completes, it extracts the user input and agent output, runs the configured evals against them using a judge LLM, emits scores as OTel histogram metrics, and forwards the spans downstream unchanged.

```
Agent span ends
      │
      ▼
GenAIEvalSpanProcessor
  ├── extract input / output from span events
  ├── run enabled evals (judge LLM scores 0.0–1.0)
  ├── emit gen_ai.evaluation.score metrics → Splunk quality chart
  ├── normalize span attributes for Splunk compatibility
  └── forward span to downstream exporter (collector / ingest)
```

## Installation

```bash
pip install splunk-agentic-evals

# With the Strands test agent
pip install "splunk-agentic-evals[agent]"
```

## Quick start

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from splunk_agentic_evals import GenAIEvalSpanProcessor, AgentContextPropagator, EvalConfig

config = EvalConfig(
    otlp_endpoint="http://localhost:4318/v1/metrics",  # Splunk OTel Collector
    judge_model="claude-haiku-4-5-20251001",
    enabled_evals=["hallucination", "relevance", "toxicity"],
    deployment_environment="production",
    service_name="my-agent",
)

eval_processor = GenAIEvalSpanProcessor(
    downstream=OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces"),
    config=config,
)

provider = TracerProvider()
provider.add_span_processor(AgentContextPropagator())   # must be first
provider.add_span_processor(BatchSpanProcessor(eval_processor))
```

`AgentContextPropagator` must be registered before `BatchSpanProcessor` so that `gen_ai.agent.name` is propagated to child spans before they are batched.

## EvalConfig reference

| Field | Type | Default | Description |
|---|---|---|---|
| `otlp_endpoint` | `str` | `http://localhost:4317` | OTel endpoint for metric export (your collector or Splunk ingest) |
| `judge_model` | `str` | `claude-haiku-4-5-20251001` | Bedrock model ID for the default judge |
| `bedrock_profile_arn` | `str` | `None` | Bedrock inference profile ARN — overrides `judge_model` when set |
| `aws_region` | `str` | `us-west-2` | AWS region for Bedrock |
| `custom_judge` | `BaseJudge` | `None` | Bring your own judge — bypasses Bedrock entirely (see below) |
| `enabled_evals` | `list[str]` | all 8 built-ins | Which evals to run |
| `pass_threshold` | `float` | `0.5` | Score below which an eval is marked failed |
| `sample_rate` | `float` | `1.0` | Fraction of spans to evaluate (0.0–1.0) |
| `async_eval` | `bool` | `True` | Run evals in a background thread to avoid blocking span export |
| `otlp_headers` | `dict` | `{}` | Headers for OTel export (e.g. `{"X-SF-Token": "..."}`) |
| `otlp_insecure` | `bool` | `True` | Disable TLS for local collector |
| `service_name` | `str` | `None` | `service.name` resource attribute on emitted metrics |
| `deployment_environment` | `str` | `None` | `deployment.environment` on emitted metrics and spans |

## Judge LLM configuration

### Default: Amazon Bedrock

Uses Claude via Bedrock. Requires `boto3` and AWS credentials in the environment.

```python
# Use a specific model
config = EvalConfig(judge_model="claude-haiku-4-5-20251001")

# Use a Bedrock inference profile
config = EvalConfig(
    bedrock_profile_arn="arn:aws:bedrock:us-west-2:123456789012:inference-profile/my-profile",
)
```

### Custom judge LLM

Implement `BaseJudge` to use any LLM provider. The judge receives a full eval prompt and must return a `(score, reason)` tuple where `score` is a float between 0.0 and 1.0.

```python
import json
from typing import Optional, Tuple
from splunk_agentic_evals import BaseJudge, EvalConfig

class OpenAIJudge(BaseJudge):
    def __init__(self, model: str = "gpt-4o"):
        import openai
        self.client = openai.OpenAI()
        self.model = model

    def score(self, prompt: str) -> Tuple[float, Optional[str]]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        parsed = json.loads(response.choices[0].message.content)
        score = max(0.0, min(1.0, float(parsed["score"])))
        return score, parsed.get("reason")

config = EvalConfig(custom_judge=OpenAIJudge())
```

Other examples:

```python
# Anthropic SDK (direct, not via Bedrock)
class AnthropicJudge(BaseJudge):
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()

    def score(self, prompt: str) -> Tuple[float, Optional[str]]:
        msg = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = json.loads(msg.content[0].text)
        return float(parsed["score"]), parsed.get("reason")

# Azure OpenAI
class AzureJudge(BaseJudge):
    def __init__(self):
        import openai
        self.client = openai.AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-01",
        )

    def score(self, prompt: str) -> Tuple[float, Optional[str]]:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = json.loads(response.choices[0].message.content)
        return float(parsed["score"]), parsed.get("reason")
```

The judge prompt always asks the LLM to respond with `{"score": <float>, "reason": "<string>"}`. Your judge implementation must parse that JSON and return the values. On failure, return `(1.0, error_message)` to avoid false positives from infrastructure errors.

## Built-in evals

These can be enabled by name in `EvalConfig.enabled_evals`:

| Name | What it measures | Passes when |
|---|---|---|
| `hallucination` | Factual accuracy of the response; grounded against tool results when available | Score ≥ threshold |
| `relevance` | Whether the response addresses the user's request | Score ≥ threshold |
| `toxicity` | Harmful, offensive, or unsafe content in the response | Score ≥ threshold |
| `bias` | Unfair or discriminatory framing in the response | Score ≥ threshold |
| `sentiment` | Tone of the response (positive/neutral/negative) | Score ≥ threshold |
| `tool_selection` | Whether the agent called the right tools for the task | Score ≥ threshold |
| `goal_completion` | Whether the agent actually achieved the user's objective | Score ≥ threshold |
| `prompt_injection` | Injection attempts in the user input (pattern + LLM check) | Score ≥ threshold |

`tool_selection` and `goal_completion` only fire on `invoke_agent` / `invoke_workflow` spans.
`prompt_injection` runs a fast regex check first and only falls back to the judge LLM if no pattern matches.

## Custom evals

Subclass `BaseEval` to add your own evaluation logic. The `context` dict contains OTel span attributes plus extracted `tool_calls` and `tool_results` lists.

```python
from splunk_agentic_evals.evals.base import BaseEval, EvalResult
from splunk_agentic_evals import GenAIEvaluator, EvalConfig

class ConcisernessEval(BaseEval):
    name = "conciseness"

    def __init__(self, judge, pass_threshold=0.5):
        self.judge = judge
        self.pass_threshold = pass_threshold

    def evaluate(self, input: str, output: str, context: dict) -> EvalResult:
        prompt = f"""Rate the conciseness of this AI response from 0.0 to 1.0.
- 1.0 = brief and to the point
- 0.5 = acceptable length but could be tightened
- 0.0 = excessively verbose or repetitive

Response: {output}

Respond with only: {{"score": <float>, "reason": "<brief reason>"}}"""

        score, reason = self.judge.score(prompt)
        return EvalResult(
            name=self.name,
            score=score,
            passed=score >= self.pass_threshold,
            reason=reason,
        )

# Register at runtime on an existing processor
eval_processor = GenAIEvalSpanProcessor(downstream=..., config=config)
eval_processor._evaluator.register_eval(
    ConcisernessEval(judge=eval_processor._evaluator._judge, pass_threshold=0.6)
)
```

Custom eval scores appear in Splunk's quality chart under `gen_ai.evaluation.name=conciseness` alongside the built-in evals. You can view them by grouping the `gen_ai.evaluation.score` metric by `gen_ai.evaluation.name` in a custom chart.

## Agent framework support

The library is framework-agnostic. It works with any agent or LLM SDK that emits standard OTel `gen_ai.*` spans:

| Framework | Notes |
|---|---|
| **Strands Agents** | Full support. `gen_ai.system=strands-agents` is automatically mapped to `aws.bedrock`. JSON-wrapped content (`[{"text":"..."}]`) is automatically unwrapped. |
| **LangChain / LangGraph** | Works out of the box with `opentelemetry-instrumentation-langchain`. |
| **OpenAI SDK** | Works with `opentelemetry-instrumentation-openai`. |
| **Anthropic SDK** | Works with `opentelemetry-instrumentation-anthropic`. |
| **Any OTel-instrumented LLM** | Any framework that emits `gen_ai.operation.name`, `gen_ai.user.message`, and `gen_ai.choice` events on its spans. |

Evals run on spans where `gen_ai.operation.name` is `invoke_agent` or `invoke_workflow`. For plain LLM spans (`chat`), metrics are emitted but evals are skipped — evaluating every single LLM sub-call in a multi-step agent produces noisy results.

## Where scores appear in Splunk

| Metric / attribute | Splunk surface |
|---|---|
| `gen_ai.evaluation.score` histogram | AI Agents → quality chart (grouped by `gen_ai.evaluation.name`) |
| `gen_ai.evaluation.<name>.score` span attr | APM Trace Analyzer → span detail view |
| `gen_ai.client.token.usage` histogram | AI Agents → token usage chart |
| `gen_ai.client.operation.duration` histogram | AI Agents → latency chart |
| `gen_ai.agent.duration` histogram | AI Agents → agent duration chart |

## Splunk OTel Collector setup (required)

Metrics must flow through a Splunk OTel Collector with the `signalfx` exporter to appear in the AI Agents page. Direct OTLP to the Splunk ingest endpoint bypasses the `signalfx` correlation pipeline and the AI Agents charts will not populate.

Save the following as `otel-collector-config.yaml`:

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318

processors:
  memory_limiter:
    check_interval: 2s
    limit_mib: 512
  batch:

exporters:
  otlphttp:
    traces_endpoint: "https://ingest.${SPLUNK_REALM}.signalfx.com/v2/trace/otlp"
    headers:
      "X-SF-Token": "${SPLUNK_ACCESS_TOKEN}"
  signalfx:
    access_token: "${SPLUNK_ACCESS_TOKEN}"
    realm: "${SPLUNK_REALM}"
    correlation:
    send_otlp_histograms: true   # required for AI Agents histogram charts

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [otlphttp, signalfx]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [signalfx]
```

Run the collector:

```bash
docker run -d \
  -p 4317:4317 -p 4318:4318 \
  -e SPLUNK_ACCESS_TOKEN=<token> \
  -e SPLUNK_REALM=us1 \
  -e SPLUNK_CONFIG=/etc/splunk/otel-collector-config.yaml \
  -v $(pwd)/otel-collector-config.yaml:/etc/splunk/otel-collector-config.yaml \
  --name splunk-otelcol \
  quay.io/signalfx/splunk-otel-collector:latest
```

Then configure `EvalConfig` to point at the collector:

```python
config = EvalConfig(
    otlp_endpoint="http://localhost:4318/v1/metrics",
    ...
)
# and downstream exporter:
downstream = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
```

## Running the test agent

The `test_agent/` directory contains a Strands travel-planner agent pre-wired with this library.

```bash
# With collector (recommended)
SPLUNK_INGEST_TOKEN=<token> \
SPLUNK_REALM=us1 \
DEPLOYMENT_ENV=my-test-env \
OTEL_COLLECTOR_ENDPOINT=http://localhost:4318 \
python3 -m test_agent.run

# Console output only (no Splunk, useful for debugging)
USE_CONSOLE_EXPORTER=true python3 -m test_agent.run
```

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/
```
