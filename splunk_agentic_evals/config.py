from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .judge import BaseJudge


@dataclass
class EvalConfig:
    # OTel export endpoint (same one used for your traces/metrics)
    otlp_endpoint: str = "http://localhost:4317"

    # Anthropic Bedrock model for LLM-as-judge
    judge_model: str = "claude-haiku-4-5-20251001"

    # Which evals to run. Built-in: hallucination, relevance, toxicity, bias, sentiment
    # Custom: tool_selection, goal_completion, prompt_injection
    enabled_evals: list[str] = field(default_factory=lambda: [
        "hallucination",
        "relevance",
        "toxicity",
        "bias",
        "sentiment",
        "tool_selection",
        "goal_completion",
        "prompt_injection",
    ])

    # Score threshold below which an eval is considered failed (Splunk default is 0.5)
    pass_threshold: float = 0.5

    # Run evals in a separate thread to avoid blocking the main span export
    async_eval: bool = True

    # Sample rate 0.0-1.0 (1.0 = evaluate every span)
    sample_rate: float = 1.0

    # AWS region for Bedrock
    aws_region: str = "us-west-2"

    # Bedrock inference profile ARN (optional, overrides judge_model if set)
    bedrock_profile_arn: Optional[str] = None

    # OTel export headers (e.g. {"X-SF-Token": "..."} for Splunk ingest)
    otlp_headers: dict = field(default_factory=dict)

    # Use TLS for OTel export (set False for local collector)
    otlp_insecure: bool = True

    # Resource attributes for the metric emitter — should match your service's TracerProvider
    service_name: Optional[str] = None
    deployment_environment: Optional[str] = None

    # Bring your own judge: any object with a score(prompt: str) -> (float, Optional[str]) method.
    # When set, overrides judge_model / bedrock_profile_arn and skips Bedrock entirely.
    # See BaseJudge for the interface and an OpenAI example.
    custom_judge: Optional["BaseJudge"] = None
