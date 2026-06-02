import pytest
from unittest.mock import MagicMock, patch
from splunk_agentic_evals.evaluator import GenAIEvaluator
from splunk_agentic_evals.config import EvalConfig
from splunk_agentic_evals.evals.base import BaseEval, EvalResult


@pytest.fixture
def config():
    return EvalConfig(
        otlp_endpoint="http://localhost:4317",
        enabled_evals=["hallucination", "relevance"],
        sample_rate=1.0,
        async_eval=False,
    )


@pytest.fixture
def evaluator(config):
    with patch("boto3.client"), patch("splunk_agentic_evals.emitter.OTLPMetricExporter"), \
         patch("splunk_agentic_evals.emitter.MeterProvider"):
        ev = GenAIEvaluator(config=config)
        ev._emitter = MagicMock()
        # Mock all evals to return controlled scores
        for e in ev._evals:
            e.evaluate = MagicMock(return_value=EvalResult(
                name=e.name, score=0.8, passed=True, reason="mocked"
            ))
        return ev


def test_run_returns_results(evaluator):
    results = evaluator.run("input", "output", {"gen_ai.operation.name": "chat"})
    assert len(results) == 2
    assert all(r.passed for r in results)


def test_run_emits_metrics(evaluator):
    evaluator.run("input", "output", {})
    evaluator._emitter.emit.assert_called_once()


def test_run_skips_when_sampled_out(config):
    config.sample_rate = 0.0
    with patch("boto3.client"), patch("splunk_agentic_evals.emitter.OTLPMetricExporter"), \
         patch("splunk_agentic_evals.emitter.MeterProvider"):
        ev = GenAIEvaluator(config=config)
        ev._emitter = MagicMock()
        results = ev.run("input", "output", {})
    assert results == []
    ev._emitter.emit.assert_not_called()


def test_run_handles_eval_exception_gracefully(evaluator):
    evaluator._evals[0].evaluate.side_effect = RuntimeError("eval crashed")
    results = evaluator.run("input", "output", {})
    # Second eval still runs
    assert len(results) == 1


def test_register_custom_eval(evaluator):
    class MyEval(BaseEval):
        name = "custom_metric"
        def evaluate(self, input, output, context):
            return EvalResult(name=self.name, score=0.7, passed=True)

    evaluator.register_eval(MyEval())
    results = evaluator.run("input", "output", {})
    names = [r.name for r in results]
    assert "custom_metric" in names


def test_unknown_eval_name_is_skipped(config):
    config.enabled_evals = ["hallucination", "nonexistent_eval"]
    with patch("boto3.client"), patch("splunk_agentic_evals.emitter.OTLPMetricExporter"), \
         patch("splunk_agentic_evals.emitter.MeterProvider"):
        ev = GenAIEvaluator(config=config)
    assert len(ev._evals) == 1
    assert ev._evals[0].name == "hallucination"
