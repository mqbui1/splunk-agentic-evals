import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import boto3

logger = logging.getLogger(__name__)


class BaseJudge(ABC):
    """
    Abstract base class for LLM-as-a-judge implementations.

    Implement this to use any LLM provider (OpenAI, Anthropic direct, Azure, etc.)
    instead of the built-in Bedrock judge.

    Example::

        class MyOpenAIJudge(BaseJudge):
            def __init__(self):
                self.client = openai.OpenAI()

            def score(self, prompt: str) -> Tuple[float, Optional[str]]:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                )
                parsed = json.loads(response.choices[0].message.content)
                return float(parsed["score"]), parsed.get("reason")

        config = EvalConfig(custom_judge=MyOpenAIJudge())
    """

    @abstractmethod
    def score(self, prompt: str) -> Tuple[float, Optional[str]]:
        """
        Score the given eval prompt.

        Args:
            prompt: The full eval prompt including input, output, and instructions.

        Returns:
            Tuple of (score, reason) where score is 0.0-1.0 and reason is an
            optional explanation string. Return (1.0, error_message) on failure
            to avoid false positives from infrastructure errors.
        """


class LLMJudge(BaseJudge):
    """
    LLM-as-a-judge using Claude via Amazon Bedrock.
    Parses JSON responses with {"score": float, "reason": str}.
    """

    def __init__(
        self,
        model_id: str = "claude-haiku-4-5-20251001",
        aws_region: str = "us-west-2",
        bedrock_profile_arn: Optional[str] = None,
    ):
        self.model_id = bedrock_profile_arn or model_id
        self.client = boto3.client("bedrock-runtime", region_name=aws_region)

    def score(self, prompt: str) -> Tuple[float, Optional[str]]:
        """
        Send prompt to Claude and parse the score/reason JSON response.
        Returns (score, reason). On failure returns (1.0, error_message) to avoid
        false positives from infrastructure errors.
        """
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 256,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )
            body = json.loads(response["body"].read())
            text = body["content"][0]["text"].strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            parsed = json.loads(text)
            score = float(parsed["score"])
            score = max(0.0, min(1.0, score))
            reason = parsed.get("reason")
            return score, reason

        except Exception as e:
            logger.warning("LLMJudge scoring failed: %s", e)
            return 1.0, f"eval_error: {str(e)}"
