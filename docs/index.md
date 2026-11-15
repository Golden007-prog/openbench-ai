# OpenBench AI

**Is the difference between two LLMs real, or noise?**

OpenBench AI is statistically sound LLM benchmarking in Python. It runs models on your
datasets, compares them question by question with confidence intervals and exact tests, ranks
several models with uncertainty, measures run-to-run variability, collects blinded human
preferences and checks that datasets and runs can be trusted. The core install has no runtime
dependencies.

```bash
pip install openbench-ai
```

## Start here

New to the statistics? Start with the [glossary](glossary.md).

| If you want to… | Read |
| --- | --- |
| Install it and run a first comparison | [Installation](installation.md), [Usage](usage.md) |
| Know whether model B really beats model A | [Model comparison](model_comparison.md), [Reading a comparison report](reading_a_report.md) |
| Rank several models, or plan how many questions you need | [Ranking and planning](ranking_and_planning.md) |
| Run any model: Ollama, vLLM, LM Studio, llama.cpp or a hosted API | [Providers](providers.md), [Ollama](ollama.md) |
| Measure run-to-run variability | [Repeated runs](repeated_runs.md) |
| Compare with people, or use an LLM judge you can measure | [Human preferences](human_preferences.md), [LLM judge](llm_judge.md) |
| Check a benchmark before trusting it | [Quality audits](quality_audits.md), [Benchmark tools](benchmark_tools.md) |
| Share results with people who don't use Python | [HTML reports and dashboard](dashboard.md) |
| Extend OpenBench with your own metrics, providers or datasets | [Plugins](plugins.md), [Custom providers](custom_provider.md) |
| Understand the statistics | [Statistical methodology](statistical_methodology.md) |
| Fix an error | [Troubleshooting](troubleshooting.md) |

## Principles

- **The tool never claims more than the statistics support.** A non-significant difference is
  never reported as equivalence, and every interval says what it covers.
- **Every number carries its provenance:** input checksums, seeds, parameters and environment.
- **The core install has no runtime dependencies.** Optional features live in extras.

The project is open source (MIT) and welcomes contributions: see the
[repository](https://github.com/lakshaychhaba0001-sys/openbench-ai), the
[good first issues](https://github.com/lakshaychhaba0001-sys/openbench-ai/labels/good%20first%20issue)
and [contributing notes](contributing.md).
