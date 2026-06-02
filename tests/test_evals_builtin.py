import pytest
from unittest.mock import MagicMock
from splunk_agentic_evals.evals.builtin import (
    HallucinationEval,
    RelevanceEval,
    ToxicityEval,
    BiasEval,
    SentimentEval,
)


@pytest.fixture
def mock_judge():
    judge = MagicMock()
    return judge


def _set_judge_score(judge, score: float, reason: str = "test reason"):
    judge.score.return_value = (score, reason)


# ── HallucinationEval ─────────────────────────────────────────────────────────

def test_hallucination_pass(mock_judge):
    _set_judge_score(mock_judge, 0.9)
    result = HallucinationEval(mock_judge).evaluate("What is 2+2?", "4", {})
    assert result.name == "hallucination"
    assert result.score == 0.9
    assert result.passed is True


def test_hallucination_fail(mock_judge):
    _set_judge_score(mock_judge, 0.2, "Contains fabricated facts")
    result = HallucinationEval(mock_judge).evaluate("Who won WW2?", "Mars did", {})
    assert result.passed is False
    assert result.reason == "Contains fabricated facts"


# ── RelevanceEval ─────────────────────────────────────────────────────────────

def test_relevance_pass(mock_judge):
    _set_judge_score(mock_judge, 0.85)
    result = RelevanceEval(mock_judge).evaluate("Book a flight", "Here are flight options...", {})
    assert result.name == "relevance"
    assert result.passed is True


def test_relevance_fail(mock_judge):
    _set_judge_score(mock_judge, 0.1, "Response is off-topic")
    result = RelevanceEval(mock_judge).evaluate("Book a flight", "Here is a recipe...", {})
    assert result.passed is False


# ── ToxicityEval ──────────────────────────────────────────────────────────────

def test_toxicity_pass(mock_judge):
    _set_judge_score(mock_judge, 1.0)
    result = ToxicityEval(mock_judge).evaluate("", "Here is your answer.", {})
    assert result.name == "toxicity"
    assert result.passed is True


def test_toxicity_fail(mock_judge):
    _set_judge_score(mock_judge, 0.0, "Offensive content detected")
    result = ToxicityEval(mock_judge).evaluate("", "offensive response", {})
    assert result.passed is False


# ── BiasEval ──────────────────────────────────────────────────────────────────

def test_bias_pass(mock_judge):
    _set_judge_score(mock_judge, 0.95)
    result = BiasEval(mock_judge).evaluate("", "Balanced response.", {})
    assert result.name == "bias"
    assert result.passed is True


def test_bias_fail(mock_judge):
    _set_judge_score(mock_judge, 0.1, "Shows strong bias")
    result = BiasEval(mock_judge).evaluate("", "biased response", {})
    assert result.passed is False


# ── SentimentEval ─────────────────────────────────────────────────────────────

def test_sentiment_pass(mock_judge):
    _set_judge_score(mock_judge, 0.8)
    result = SentimentEval(mock_judge).evaluate("", "Happy to help!", {})
    assert result.name == "sentiment"
    assert result.passed is True


def test_sentiment_fail(mock_judge):
    _set_judge_score(mock_judge, 0.1, "Hostile tone")
    result = SentimentEval(mock_judge).evaluate("", "I refuse.", {})
    assert result.passed is False


# ── Custom pass_threshold ─────────────────────────────────────────────────────

def test_custom_pass_threshold(mock_judge):
    _set_judge_score(mock_judge, 0.6)
    # threshold 0.8 → 0.6 should fail
    result = HallucinationEval(mock_judge, pass_threshold=0.8).evaluate("q", "a", {})
    assert result.passed is False

    # threshold 0.5 → 0.6 should pass
    result = HallucinationEval(mock_judge, pass_threshold=0.5).evaluate("q", "a", {})
    assert result.passed is True
