# Product Brief: Instrumentation-Side Evaluations for Splunk AI Agent Monitoring

## Problem

Splunk AI Agent Monitoring currently forces customers into a binary choice:

- **Platform-side evaluations** (Splunk-managed): unlocks AI Trace Data, cost metrics, risk metrics, and the full Observability Cloud feature set — but requires routing all LLM traffic through a Splunk-managed integration and offers no ability to customize eval logic.
- **Instrumentation-side evaluations** (customer-managed): enables custom evals, domain-specific rubrics, and any-framework support — but per current documentation, is incompatible with the Observability Cloud data path, losing AI Trace Data, cost metrics, and risk metrics.

Customers building production AI agents need both: the operational visibility of the Observability Cloud feature set *and* the flexibility to define and run their own quality checks.

## Proof of Concept

We built [`splunk-agentic-evals`](https://github.com/mqbui1/splunk-agentic-evals) — an open-source Python library that demonstrates this is technically feasible today.

**What it does:**
- Wraps any OTel `SpanExporter` to intercept `gen_ai` spans at export time
- Extracts input/output, runs LLM-as-a-judge evals (built-in or custom), and emits `gen_ai.evaluation.score` histograms via OTel
- Routes spans and metrics through the Splunk OTel Collector with `signalfx` exporter

**What works today:**
- AI Agents page populates with token usage, latency, agent duration, and quality scores
- Works with Strands, LangChain, OpenAI SDK, Anthropic SDK — any OTel-instrumented framework
- Users can plug in any judge LLM (Bedrock, OpenAI, Azure, Anthropic direct) via `BaseJudge`
- Custom evals emit as named metric dimensions for custom dashboards

**Confirmed gap:** AI Trace Data page does not populate — this is the primary blocker to a complete customer experience.

## Platform Gaps to Close

| Gap | Root Cause | Proposed Fix |
|---|---|---|
| AI Trace Data empty for instrumentation-side spans | Feature gated to platform-side eval path | Decouple AI Trace Data from eval source — surface any span with `gen_ai.evaluation.sampled=true` |
| `gen_ai.evaluation.score` not in AI Trace Data table | Platform doesn't ingest instrumentation-side eval histograms | Treat `gen_ai.evaluation.score` as a first-class signal regardless of origin |
| No cost metrics from instrumentation path | No defined OTel semantic conventions for LLM cost | Define `gen_ai.cost.*` conventions; customers emit, platform renders |
| Risk metrics require Cisco AI Defense | Tightly coupled to platform-side integration | Expose risk signal as an OTel dimension customers can populate from their own guardrails |

## Proposed Roadmap

**Quick win (docs + minor platform change)**
Document that `gen_ai.evaluation.score` histograms from the instrumentation path are supported. Surface them in the AI Agents quality chart alongside platform-side scores. This alone closes the most common customer request.

**Medium lift**
Decouple AI Trace Data from the platform-side eval pipeline. Any span with `gen_ai.evaluation.sampled=true` and `gen_ai.operation.name` set should appear in AI Trace Data regardless of how evals were run. The PoC already sets this attribute.

**Longer term**
Define official OpenTelemetry semantic conventions for LLM cost and risk so instrumentation-side customers can emit those signals and have them render natively in the platform — eliminating the need for the Cisco AI Defense dependency for customers who manage their own guardrails.

## Ask

1. Review the [proof-of-concept repo](https://github.com/mqbui1/splunk-agentic-evals) and validate the technical approach
2. Prioritize decoupling AI Trace Data from the platform-side eval requirement
3. Engage with the OpenTelemetry GenAI SIG to align on cost/risk semantic conventions

---

*Contact: Minh Bui — proof-of-concept built and validated in production against Splunk Observability Cloud (us1)*
