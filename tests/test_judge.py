import json
import pytest
from unittest.mock import MagicMock, patch
from splunk_agentic_evals.judge import LLMJudge


def _mock_bedrock_response(score: float, reason: str):
    body_bytes = json.dumps({
        "content": [{"text": json.dumps({"score": score, "reason": reason})}]
    }).encode()
    mock_body = MagicMock()
    mock_body.read.return_value = body_bytes
    return {"body": mock_body}


@pytest.fixture
def judge():
    with patch("boto3.client"):
        j = LLMJudge(model_id="claude-haiku-4-5-20251001", aws_region="us-west-2")
        return j


def test_score_returns_float_and_reason(judge):
    judge.client.invoke_model.return_value = _mock_bedrock_response(0.9, "Good response")
    score, reason = judge.score("test prompt")
    assert score == 0.9
    assert reason == "Good response"


def test_score_clamps_above_1(judge):
    judge.client.invoke_model.return_value = _mock_bedrock_response(1.5, "Over max")
    score, _ = judge.score("test prompt")
    assert score == 1.0


def test_score_clamps_below_0(judge):
    judge.client.invoke_model.return_value = _mock_bedrock_response(-0.2, "Under min")
    score, _ = judge.score("test prompt")
    assert score == 0.0


def test_score_strips_markdown_code_fences(judge):
    body_bytes = json.dumps({
        "content": [{"text": "```json\n{\"score\": 0.75, \"reason\": \"ok\"}\n```"}]
    }).encode()
    mock_body = MagicMock()
    mock_body.read.return_value = body_bytes
    judge.client.invoke_model.return_value = {"body": mock_body}
    score, reason = judge.score("test prompt")
    assert score == 0.75
    assert reason == "ok"


def test_score_returns_safe_default_on_error(judge):
    judge.client.invoke_model.side_effect = Exception("Bedrock unavailable")
    score, reason = judge.score("test prompt")
    assert score == 1.0
    assert "eval_error" in reason


def test_score_returns_safe_default_on_bad_json(judge):
    body_bytes = json.dumps({
        "content": [{"text": "not valid json"}]
    }).encode()
    mock_body = MagicMock()
    mock_body.read.return_value = body_bytes
    judge.client.invoke_model.return_value = {"body": mock_body}
    score, reason = judge.score("test prompt")
    assert score == 1.0
    assert "eval_error" in reason
