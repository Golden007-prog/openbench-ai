"""v0.5.0 Milestone 1 (Foundations): OpenAI-compatible adapter, rate limits, plugins, cache locking, CLI."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from openbench.adapters import (
    GenerationConfig,
    OpenAICompatibleAdapter,
    create_provider,
    list_registered_providers,
)
from openbench.adapters.openai_compatible import OpenAICompatibleConnectionError, parse_retry_after
from openbench.cache import ResponseCache
from openbench.cli import _compact_runs_table, _exit_quietly_after_closed_stdout, build_parser, main
from openbench.configuration import ProfileLoader
from openbench.dataset import Dataset, DatasetLoader, DatasetRecord
from openbench.evaluator import ResponseItem
from openbench.exceptions import ConfigurationError, PluginError, ProviderRateLimitError
from openbench.execution import ExecutionConfig, ExecutionEngine, RequestRateLimiter, RetryPolicy
from openbench.filelock import FileLock, FileLockTimeoutError

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "datasets" / "reasoning" / "sample.jsonl"


def _record(i: int = 1, prompt: str = "What is 2+2?") -> DatasetRecord:
    return DatasetRecord(id=f"q{i}", prompt=prompt, expected_answer="4", category="math", metadata={"difficulty": "easy"})


def _dataset(n: int) -> Dataset:
    return Dataset(name="tiny", version="1.0", records=[_record(i) for i in range(1, n + 1)], source_path="memory")


# ---------------------------------------------------------------------------------------------
# OpenAI-compatible adapter (#8)
# ---------------------------------------------------------------------------------------------
class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, payload, headers, timeout):
        self.calls.append({"method": method, "url": url, "payload": payload, "headers": dict(headers)})
        return self.responses.pop(0)


def _completion(text="4", usage=True):
    body = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if usage:
        body["usage"] = {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}
    return 200, {}, json.dumps(body)


def test_openai_adapter_builds_request_and_parses_response(monkeypatch):
    monkeypatch.setenv("OB_TEST_KEY", "sk-test-secret-123")
    transport = FakeTransport([_completion("4")])
    adapter = OpenAICompatibleAdapter(
        model_name="m1",
        endpoint="http://localhost:8000/v1",
        api_key_env="OB_TEST_KEY",
        generation_config=GenerationConfig(temperature=0.2, max_tokens=32, seed=5, stop_sequences=["\n"]),
        system_prompt="Answer briefly.",
        transport_fn=transport,
    )
    item = adapter.generate_one(_record())
    assert item.response == "4"
    assert item.token_usage == {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}
    call = transport.calls[0]
    assert call["method"] == "POST" and call["url"] == "http://localhost:8000/v1/chat/completions"
    assert call["payload"]["messages"] == [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    assert call["payload"]["seed"] == 5 and call["payload"]["stop"] == ["\n"]
    assert call["headers"]["Authorization"] == "Bearer sk-test-secret-123"
    assert adapter.seed_forward_count == 1
    # The key never reaches stored metadata.
    stored = json.dumps([adapter.inference_config, adapter.reproducibility_metadata()])
    assert "sk-test-secret-123" not in stored and "OB_TEST_KEY" not in stored


def test_openai_adapter_without_key_sends_no_authorization(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    transport = FakeTransport([_completion("ok", usage=False)])
    adapter = OpenAICompatibleAdapter(model_name="m", transport_fn=transport)
    item = adapter.generate_one(_record())
    assert item.token_usage is None
    assert "Authorization" not in transport.calls[0]["headers"]
    assert "seed" not in transport.calls[0]["payload"]


def test_openai_adapter_rate_limit_and_errors(monkeypatch):
    adapter = OpenAICompatibleAdapter(
        model_name="m",
        api_key_env="OB_MISSING",
        transport_fn=FakeTransport(
            [
                (429, {"Retry-After": "3"}, '{"error": {"message": "slow down"}}'),
                (401, {}, '{"error": {"message": "bad key"}}'),
                (500, {}, "boom"),
                (200, {}, '{"choices": []}'),
            ]
        ),
    )
    with pytest.raises(ProviderRateLimitError) as info:
        adapter.generate_one(_record())
    assert info.value.retry_after_seconds == 3.0
    with pytest.raises(OpenAICompatibleConnectionError, match="OB_MISSING"):
        adapter.generate_one(_record())
    with pytest.raises(OpenAICompatibleConnectionError, match="HTTP 500"):
        adapter.generate_one(_record())
    with pytest.raises(OpenAICompatibleConnectionError, match="no choices"):
        adapter.generate_one(_record())


def test_parse_retry_after_forms():
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("-4") == 0.0
    # HTTP-date 10 seconds after a fixed reference time.
    assert parse_retry_after("Thu, 01 Jan 1970 00:00:10 GMT", now=0.0) == 10.0


def test_openai_adapter_check_connection_and_security():
    transport = FakeTransport([(200, {}, json.dumps({"data": [{"id": "m"}, {"id": "other"}]}))])
    health = OpenAICompatibleAdapter(model_name="m", transport_fn=transport).check_connection()
    assert health.available and health.details["model_listed"] is True
    assert transport.calls[0]["method"] == "GET" and transport.calls[0]["url"].endswith("/v1/models")
    with pytest.raises(ConfigurationError):
        OpenAICompatibleAdapter(model_name="m", endpoint="http://remote.example.com/v1")
    with pytest.raises(ConfigurationError):
        OpenAICompatibleAdapter(model_name="m", endpoint="https://user:pw@api.example.com/v1")
    assert OpenAICompatibleAdapter(model_name="m", endpoint="https://api.example.com/v1").endpoint == "https://api.example.com/v1"


def test_openai_adapter_is_registered_with_aliases():
    names = [p["provider_name"] for p in list_registered_providers()]
    assert "openai_compatible" in names
    for alias in ("openai_compatible", "openai", "openai-compatible"):
        assert isinstance(create_provider(alias, model_name="m"), OpenAICompatibleAdapter)


class _FakeOpenAIServer(BaseHTTPRequestHandler):
    seen_auth: list[str | None] = []

    def log_message(self, *args):  # silence test output
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        _FakeOpenAIServer.seen_auth.append(self.headers.get("Authorization"))
        prompt = body["messages"][-1]["content"]
        answer = {"choices": [{"message": {"content": f"echo:{len(prompt)}"}}],
                  "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
        data = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_cli_run_against_local_openai_compatible_server(tmp_path, monkeypatch, capsys):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("OB_LOCAL_KEY", "sk-local-abc")
        _FakeOpenAIServer.seen_auth.clear()
        code = main([
            "run", "--dataset", str(SAMPLE), "--provider", "openai_compatible", "--model", "fake",
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--api-key-env", "OB_LOCAL_KEY",
            "--seed", "4", "--max-requests-per-minute", "60000",
            "--experiments-dir", str(tmp_path / "exp"), "--cache-dir", str(tmp_path / "cache"),
        ])
    finally:
        server.shutdown()
    assert code == 0
    assert _FakeOpenAIServer.seen_auth and all(a == "Bearer sk-local-abc" for a in _FakeOpenAIServer.seen_auth)
    run_json = next((tmp_path / "exp").rglob("run.json"), None) or next((tmp_path / "exp").rglob("*.json"))
    text = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "exp").rglob("*.json"))
    text += (tmp_path / "cache" / "cache_entries.jsonl").read_text(encoding="utf-8")
    assert "sk-local-abc" not in text and "OB_LOCAL_KEY" not in text
    assert run_json is not None
    capsys.readouterr()
    assert main(["runs", "list", "--experiments-dir", str(tmp_path / "exp"), "--format", "json"]) == 0
    runs = json.loads(capsys.readouterr().out)
    assert runs[0]["lineage"]["seed_status"] == "passed_to_provider"
    assert runs[0]["execution_configuration"]["max_requests_per_minute"] == 60000


# ---------------------------------------------------------------------------------------------
# Rate limits and Retry-After (#32)
# ---------------------------------------------------------------------------------------------
class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_spaces_requests_evenly():
    fc = FakeClock()
    limiter = RequestRateLimiter(30, clock=fc.clock, sleep=fc.sleep)  # one every 2 s
    waits = [limiter.acquire() for _ in range(4)]
    assert waits == [0.0, 2.0, 2.0, 2.0]
    assert limiter.total_wait_seconds == 6.0
    fc.now += 10  # idle time is not banked as burst credit beyond the next slot
    assert limiter.acquire() == 0.0
    with pytest.raises(ConfigurationError):
        RequestRateLimiter(0)


class FlakyProvider:
    """Mock-like adapter that rate-limits the first ``failures`` calls."""

    provider_name = "flaky"
    model_name = "flaky"
    model_version = "1"
    is_mock = True
    inference_config: dict = {}

    def __init__(self, failures: int, retry_after: float | None):
        self.failures = failures
        self.retry_after = retry_after
        self.calls = 0

    def generate_one(self, record):
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderRateLimitError("HTTP 429", retry_after_seconds=self.retry_after)
        return ResponseItem(id=record.id, response="4", latency_ms=1.0)

    def reproducibility_metadata(self):
        return {"provider": "flaky"}


def test_engine_honours_retry_after_with_cap_and_counts_retries():
    fc = FakeClock()
    cfg = ExecutionConfig(retry_policy=RetryPolicy(max_attempts=4), use_cache=False, max_retry_after_seconds=5.0)
    engine = ExecutionEngine(FlakyProvider(2, retry_after=30.0), cfg, sleep_fn=fc.sleep, clock_fn=fc.clock)
    outcome = engine.execute_dataset(_dataset(1))
    assert outcome.completed_record_ids == ["q1"]
    assert fc.sleeps == [5.0, 5.0]  # Retry-After 30 s capped at 5 s
    assert outcome.rate_limited_retries == 2


def test_engine_uses_backoff_without_retry_after_and_limits_rate():
    fc = FakeClock()
    cfg = ExecutionConfig(
        retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0.5),
        use_cache=False,
        max_requests_per_minute=60,
    )
    engine = ExecutionEngine(FlakyProvider(1, retry_after=None), cfg, sleep_fn=fc.sleep, clock_fn=fc.clock)
    outcome = engine.execute_dataset(_dataset(3))
    assert len(outcome.completed_record_ids) == 3
    assert 0.5 in fc.sleeps  # normal backoff after the 429 without Retry-After
    assert outcome.rate_limit_wait_seconds > 0
    assert outcome.rate_limited_retries == 1


def test_execution_config_serialisation_is_unchanged_by_default():
    assert set(ExecutionConfig().to_dict()) == {
        "timeout_seconds", "concurrency", "retry_policy", "use_cache", "estimate_missing_tokens",
        "privacy_mode", "pricing", "request_metadata",
    }
    d = ExecutionConfig(max_requests_per_minute=120, max_retry_after_seconds=10).to_dict()
    assert d["max_requests_per_minute"] == 120 and d["max_retry_after_seconds"] == 10
    with pytest.raises(ConfigurationError):
        ExecutionConfig(max_requests_per_minute=-1)


def test_profile_accepts_rate_limit_keys():
    profile = ProfileLoader.validate_dict(
        {
            "name": "p",
            "dataset": {"path": str(SAMPLE)},
            "metrics": ["exact_match"],
            "provider": {"name": "openai_compatible", "model": "m", "api_key_env": "X", "system_prompt": "Be brief."},
            "execution": {"max_requests_per_minute": 90, "max_retry_after_seconds": 12},
        }
    )
    assert profile.execution_config.max_requests_per_minute == 90
    assert profile.execution_config.max_retry_after_seconds == 12


# ---------------------------------------------------------------------------------------------
# Plugins (#31)
# ---------------------------------------------------------------------------------------------
PLUGIN_MODULE = '''
from openbench.adapters.base import BaseModelAdapter
from openbench.dataset import Dataset, DatasetRecord
from openbench.evaluator import ResponseItem
from openbench.metrics.base import BaseMetric, MetricResult


class ShoutMetric(BaseMetric):
    name = "shout_match"
    description = "1 when the response, upper-cased, equals the expected answer upper-cased."

    def compute_item(self, record, response_text):
        ok = response_text.strip().upper() == (record.expected_answer or "").strip().upper()
        return MetricResult(metric_name=self.name, score=1.0 if ok else 0.0, passed=ok)


class EchoAdapter(BaseModelAdapter):
    provider_name = "echo_plugin"
    description = "Echoes the expected answer."
    is_mock = True

    def generate_one(self, record):
        return ResponseItem(id=record.id, response=record.expected_answer or "", latency_ms=0.0)


def load_numbers(argument):
    n = int(argument or 3)
    return Dataset(
        name="plugin_numbers", version="1.0", source_path="plugin:numbers",
        records=[DatasetRecord(id=f"n{i}", prompt=f"Say {i}", expected_answer=str(i), category="numbers") for i in range(n)],
    )
'''


@pytest.fixture
def plugin_env(tmp_path, monkeypatch):
    from openbench import plugins
    from openbench.adapters.base import PROVIDER_REGISTRY
    from openbench.metrics import METRIC_REGISTRY

    site = tmp_path / "site"
    (site / "ob_test_plugin").mkdir(parents=True)
    (site / "ob_test_plugin" / "__init__.py").write_text(textwrap.dedent(PLUGIN_MODULE), encoding="utf-8")
    dist = site / "ob_test_plugin-0.1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: ob-test-plugin\nVersion: 0.1.0\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text(
        "[openbench.metrics]\nshout_match = ob_test_plugin:ShoutMetric\n\n"
        "[openbench.providers]\necho_plugin = ob_test_plugin:EchoAdapter\n\n"
        "[openbench.datasets]\nnumbers = ob_test_plugin:load_numbers\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.delenv("OPENBENCH_DISABLE_PLUGINS", raising=False)
    plugins.reset_plugins_for_tests()
    yield site
    plugins.reset_plugins_for_tests()
    METRIC_REGISTRY.pop("shout_match", None)
    PROVIDER_REGISTRY.pop("echo_plugin", None)
    sys.modules.pop("ob_test_plugin", None)


def test_plugins_register_metric_provider_and_dataset(plugin_env, capsys):
    from openbench.metrics import create_metric
    from openbench.plugins import load_plugins, plugin_source

    infos = load_plugins()
    assert {(i.kind, i.name) for i in infos} == {("metrics", "shout_match"), ("providers", "echo_plugin"), ("datasets", "numbers")}
    assert all(i.distribution == "ob-test-plugin" and i.version == "0.1.0" for i in infos)

    metric = create_metric("shout_match")
    assert plugin_source(metric)["distribution"] == "ob-test-plugin"

    adapter = create_provider("echo_plugin", model_name="e")
    assert adapter.reproducibility_metadata()["plugin"]["entry_point"] == "ob_test_plugin:EchoAdapter"

    ds = DatasetLoader.load("plugin:numbers:4")
    assert [r.id for r in ds.records] == ["n0", "n1", "n2", "n3"]

    assert main(["plugins", "list"]) == 0
    out = capsys.readouterr().out
    assert "echo_plugin" in out and "ob-test-plugin 0.1.0" in out
    assert main(["providers", "list"]) == 0
    assert "plugin: ob-test-plugin" in capsys.readouterr().out
    assert main(["metrics", "list", "--format", "json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert any(r["name"] == "shout_match" and r["source"]["distribution"] == "ob-test-plugin" for r in rows)


def test_plugin_is_loaded_lazily_by_name(plugin_env):
    from openbench.plugins import _LOADED

    assert "providers" not in _LOADED
    create_provider("echo_plugin", model_name="e")
    assert "providers" in _LOADED


def test_broken_and_disabled_plugins(plugin_env, monkeypatch):
    from openbench import plugins

    (plugin_env / "ob_test_plugin-0.1.0.dist-info" / "entry_points.txt").write_text(
        "[openbench.metrics]\nbroken = ob_test_plugin:DoesNotExist\n", encoding="utf-8"
    )
    plugins.reset_plugins_for_tests()
    with pytest.raises(PluginError, match="broken.*ob-test-plugin"):
        plugins.load_plugins("metrics")
    plugins.reset_plugins_for_tests()
    monkeypatch.setenv("OPENBENCH_DISABLE_PLUGINS", "1")
    assert plugins.load_plugins() == []
    with pytest.raises(PluginError):
        plugins.load_dataset_from_plugin("plugin:numbers")


def test_unknown_dataset_plugin_message(plugin_env):
    with pytest.raises(PluginError, match="Unknown dataset plugin 'nope'.*numbers"):
        DatasetLoader.load("plugin:nope")


# ---------------------------------------------------------------------------------------------
# Cross-process cache locking (#17)
# ---------------------------------------------------------------------------------------------
WRITER = '''
import sys
from openbench.cache import ResponseCache
from openbench.evaluator import ResponseItem
cache = ResponseCache(cache_dir=sys.argv[1])
tag = sys.argv[2]
for i in range(int(sys.argv[3])):
    cache.put("p", "m", "1", "d", "1", f"{tag}-{i}", "prompt " + "x" * 2000, ResponseItem(id=f"{tag}-{i}", response="r" * 3000))
'''


def test_two_processes_share_a_cache_without_losing_entries(tmp_path):
    script = tmp_path / "writer.py"
    script.write_text(WRITER, encoding="utf-8")
    cache_dir = tmp_path / "cache"
    env_path = str(ROOT / "src")
    import os

    env = {**os.environ, "PYTHONPATH": env_path + os.pathsep + os.environ.get("PYTHONPATH", "")}
    procs = [
        subprocess.Popen([sys.executable, str(script), str(cache_dir), tag, "150"], env=env)
        for tag in ("a", "b", "c")
    ]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    lines = (cache_dir / "cache_entries.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 450
    assert all(json.loads(line)["response"] == "r" * 3000 for line in lines)
    assert len(ResponseCache(cache_dir=cache_dir).stats()) and ResponseCache(cache_dir=cache_dir).stats()["stored_entries"] == 450


def test_file_lock_times_out_with_clear_message(tmp_path):
    held = FileLock(tmp_path / "x.lock")
    held.acquire()
    try:
        code = (
            "import sys; sys.path.insert(0, sys.argv[2]);"
            "from openbench.filelock import FileLock, FileLockTimeoutError\n"
            "try:\n    FileLock(sys.argv[1], timeout_seconds=0.2).acquire()\nexcept FileLockTimeoutError as e:\n    print('TIMEOUT', e)\n"
        )
        out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "x.lock"), str(ROOT / "src")],
                             capture_output=True, text=True, timeout=60).stdout
        assert out.startswith("TIMEOUT") and "--cache-dir" in out
    finally:
        held.release()
    with FileLock(tmp_path / "x.lock", timeout_seconds=1):
        with FileLock(tmp_path / "y.lock"):
            pass
    assert FileLockTimeoutError.__mro__[1].__name__ == "OpenBenchError"


# ---------------------------------------------------------------------------------------------
# CLI improvements (#4, #6, #38, #39)
# ---------------------------------------------------------------------------------------------
def test_version_includes_python_and_platform(capsys):
    import platform

    with pytest.raises(SystemExit):
        main(["--version"])
    out = capsys.readouterr().out.strip()
    assert out.startswith("openbench ") and f"Python {platform.python_version()}" in out and "\n" not in out


@pytest.mark.parametrize("argv", [["run"], ["analyze"], ["evaluate"], ["preferences", "create"]])
def test_main_commands_show_examples(argv):
    parser = build_parser()
    sub = parser
    for name in argv:
        sub = next(a for a in sub._actions if a.__class__.__name__ == "_SubParsersAction").choices[name]
    assert "examples:" in sub.format_help()


def test_runs_list_compact_and_wide(tmp_path, capsys):
    exp = tmp_path / "exp"
    assert main(["run", "--dataset", str(SAMPLE), "--provider", "mock", "--model", "a-very-long-model-name-for-testing-width",
                 "--repetitions", "2", "--seed", "1", "--experiments-dir", str(exp), "--no-cache"]) == 0
    capsys.readouterr()
    assert main(["runs", "list", "--experiments-dir", str(exp)]) == 0
    compact = capsys.readouterr().out.splitlines()
    assert compact[1].split() == ["RUN", "ID", "STATUS", "MODEL", "ITEMS", "SCORE", "GROUP"]
    assert max(len(line) for line in compact[:-1]) <= 120
    assert main(["runs", "list", "--experiments-dir", str(exp), "--wide"]) == 0
    wide = capsys.readouterr().out
    assert "origin=provider_executed" in wide and "scores={" in wide


def test_runs_list_exports_mock_run(tmp_path, capsys):
    model = 'long model name for export, with "quotes" and a | pipe'
    common = ["--experiments-dir", str(tmp_path / "exp")]
    assert main([
        "run", "--dataset", str(SAMPLE), "--provider", "mock", "--model", model,
        "--no-cache", *common,
    ]) == 0
    capsys.readouterr()
    args = ["runs", "list", *common, "--provider", "mock"]

    assert main([*args, "--format", "json"]) == 0
    record, = json.loads(capsys.readouterr().out)
    assert record["model_name"] == model

    assert main([*args, "--format", "csv"]) == 0
    header, row = csv.reader(io.StringIO(capsys.readouterr().out))
    assert header == ["RUN ID", "STATUS", "MODEL", "ITEMS", "SCORE", "GROUP"]
    assert row[:4] == [record["run_id"], "completed", f"mock:{model} [mock]",
                       f'{record["completed_items"]}/{record["total_items"]}']
    assert row[5] == "-"

    assert main([*args, "--format", "markdown", "--wide"]) == 0
    header, separator, row = capsys.readouterr().out.splitlines()
    assert header == "| RUN ID | STATUS | ORIGIN | MODEL | DATASET | ITEMS | SCORES | GROUP |"
    assert separator == "| --- | --- | --- | --- | --- | --- | --- | --- |"
    assert record["run_id"] in row
    assert "provider_executed" in row
    assert f"mock:{model} [MOCK]" in row.replace(r"\|", "|")
    assert r"and a \| pipe" in row
    assert str(record["aggregate_scores"]) in row


@pytest.mark.parametrize("fmt, expected", [
    ("csv", '"RUN ID","STATUS","MODEL","ITEMS","SCORE","GROUP"\n'),
    ("markdown", "| RUN ID | STATUS | MODEL | ITEMS | SCORE | GROUP |\n"
                 "| --- | --- | --- | --- | --- | --- |\n"),
])
def test_runs_list_empty_exports_keep_headers(tmp_path, capsys, fmt, expected):
    assert main(["runs", "list", "--experiments-dir", str(tmp_path), "--format", fmt]) == 0
    assert capsys.readouterr().out == expected


def test_compact_table_handles_no_scores():
    from types import SimpleNamespace

    make_run = lambda: SimpleNamespace(  # noqa: E731
        run_id="run_x", status="failed", model_provider="mock", model_name="m", is_mock_provider=True,
        aggregate_scores={}, repeat_group_id=None, repetition_index=0, completed_items=0, total_items=3,
    )

    lines = _compact_runs_table([make_run()])
    assert lines[1].split()[-2:] == ["-", "-"]


def test_quiet_exit_without_real_stdout(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert _exit_quietly_after_closed_stdout() == 1


def test_real_model_check_script_runs_with_mock(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "real_model_check.py"), "--provider", "mock", "--model", "a",
         "--model-b", "b", "--dataset", str(SAMPLE), "--output", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "REAL-MODEL CHECK PASSED" in proc.stdout
