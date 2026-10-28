"""Exercise test orchestration without Docker, model downloads or API keys."""
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/e2e"))
import model_smoke
import ner_integration
import ner_test_support as support
from deployment_topology_gate import Gate
from presidio.tests.test_onnx_artifact import fixture_metadata
from presidio.model_artifact import ModelArtifactError


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))


def test_wait_ready_has_a_total_deadline():
    now = [0]
    probes = []
    def sleep(value):
        now[0] += value
    with pytest.raises(TimeoutError):
        support.wait_ready(lambda: probes.append(True) or False, 2.5, sleep=sleep, clock=lambda: now[0])
    assert now[0] == 2.5
    assert len(probes) == 4


def test_wait_ready_stops_after_first_success():
    probes = iter([False, True, False])
    support.wait_ready(lambda: next(probes), 5, sleep=lambda _: None)
    assert next(probes) is False


@pytest.mark.parametrize("output", ["not-an-image", "sha256:short", ""])
def test_invalid_image_identity_is_not_accepted(monkeypatch, output):
    monkeypatch.setattr(support, "docker", lambda *a, **k: SimpleNamespace(stdout=output))
    with pytest.raises(RuntimeError, match="identity"):
        support.image_id("tag")


@pytest.mark.parametrize("exception", [subprocess.TimeoutExpired("probe", 1), KeyboardInterrupt()])
def test_offline_probe_removes_its_container_on_failure(monkeypatch, exception):
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise exception
    monkeypatch.setattr(support, "docker", fail)
    monkeypatch.setattr(support.subprocess, "run", lambda *a, **k: calls.append(a) or SimpleNamespace(returncode=0))
    with pytest.raises(type(exception)):
        support.offline_probe("image", "python", "check.py")
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[-1][0] == ["docker", "rm", "-f", name]
    assert "--network" in calls[0] and "none" in calls[0]


@pytest.mark.parametrize("mismatch", [False, True])
def test_both_images_must_match_the_checkout_and_each_other(monkeypatch, mismatch):
    _, artifact = fixture_metadata()
    expected = {"model_id": artifact["source_model_id"], "revision": artifact["source_revision"],
                "onnx_sha256": artifact["onnx_sha256"]}
    peer = dict(expected, onnx_sha256="f" * 64) if mismatch else expected
    replies = iter([artifact, peer])
    monkeypatch.setattr(support, "offline_probe", lambda *a, **k: SimpleNamespace(stdout=json.dumps(next(replies))))
    if mismatch:
        with pytest.raises(RuntimeError, match="same trusted"):
            support.identities({"presidio-analyzer": "a", "ner-triton": "t"})
    else:
        assert support.identities({"presidio-analyzer": "a", "ner-triton": "t"}) == expected


def test_source_mismatch_cannot_be_learned_from_the_server(monkeypatch):
    _, artifact = fixture_metadata()
    artifact["source_revision"] = "main"
    monkeypatch.setattr(support, "offline_probe", lambda *a, **k: SimpleNamespace(stdout=json.dumps(artifact)))
    with pytest.raises(ModelArtifactError):
        support.identities({"presidio-analyzer": "a", "ner-triton": "t"})


@pytest.mark.parametrize("internal", [True, False])
def test_network_isolation_is_checked_not_assumed(monkeypatch, internal):
    monkeypatch.setattr(support, "docker", lambda *a, **k: SimpleNamespace(stdout=json.dumps([{"Internal": internal}])))
    if internal:
        support.check_internal_network("test")
    else:
        with pytest.raises(RuntimeError):
            support.check_internal_network("test")


def test_diagnostic_report_does_not_copy_container_secrets(monkeypatch):
    data = [{"Name": "/test", "Image": "sha256:test", "RestartCount": 0,
             "Config": {"Env": ["PASSWORD=not-for-report"]}, "State": {
                 "Status": "running", "ExitCode": 0, "OOMKilled": False,
                 "Health": {"Status": "healthy", "Log": [{"Output": "private payload"}]}}}]
    monkeypatch.setattr(support, "docker", lambda *a, **k: SimpleNamespace(stdout=json.dumps(data)))
    result = support.container_states(["one"])
    assert result[0]["health"] == "healthy"
    assert "not-for-report" not in json.dumps(result)
    assert "private payload" not in json.dumps(result)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tool", [False, True])
@pytest.mark.parametrize("password_only", [False, True])
def test_password_mock_echo_selection_does_not_change_the_request(stream, tool, password_only):
    import mock_openai_upstream as mock

    payload = {"messages": [{"role": "user", "content": "<PASSWORD_9>"},
                             {"role": "user", "content": "STOCK_BRIDGE_ECHO Пар<PERSON_1>: <PASSWORD_1>"}],
               "stream": stream}
    if password_only:
        payload["k07_password_echo"] = True
    if tool:
        payload["tools"] = [{"type": "function", "function": {"name": "save"}}]
    original = json.dumps(payload, sort_keys=True)
    output = {}
    handler = SimpleNamespace(
        _write_json=lambda status, body: output.update(status=status, body=body),
        _write_sse=lambda events: output.update(events=events),
    )
    mock.Handler._write_chat_bridge_echo(handler, payload)
    expected = ["<PASSWORD_1>"] if password_only else ["<PASSWORD_9>", "<PERSON_1>", "<PASSWORD_1>"]
    if stream:
        chunks = [event[1] for event in output["events"] if isinstance(event[1], dict)]
        if tool:
            actual = "".join(call.get("function", {}).get("arguments", "")
                             for event in chunks for call in event["choices"][0]["delta"].get("tool_calls", []))
        else:
            actual = "".join(event["choices"][0]["delta"].get("content", "") for event in chunks)
    else:
        assert output["status"] == 200
        message = output["body"]["choices"][0]["message"]
        actual = message["tool_calls"][0]["function"]["arguments"] if tool else message["content"]
    assert (json.loads(actual)["values"] if tool else actual) == (expected if tool else " / ".join(expected))
    assert json.dumps(payload, sort_keys=True) == original


@pytest.mark.parametrize("exception", [RuntimeError("readiness"), KeyboardInterrupt()])
def test_model_smoke_cleans_up_and_reports_failure(monkeypatch, tmp_path, exception):
    monkeypatch.setattr(model_smoke, "Context", lambda **k: SimpleNamespace(
        model={"services": {"nginx": {"image": "nginx:test"}}}, start_timeout=1))
    monkeypatch.setattr(model_smoke, "image_id", lambda x: x)
    monkeypatch.setattr(model_smoke, "identities", lambda x: {"revision": "test"})
    monkeypatch.setattr(model_smoke, "offline_probe", lambda *a: None)
    monkeypatch.setattr(model_smoke, "check_internal_network", lambda *a: None)
    monkeypatch.setattr(model_smoke, "docker", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(model_smoke, "wait_ready", lambda *a, **k: (_ for _ in ()).throw(exception))
    calls = []
    monkeypatch.setattr(model_smoke.subprocess, "run", lambda args, **k:
                        calls.append(args) or SimpleNamespace(stdout="", returncode=0))
    report = tmp_path / "model.json"
    with pytest.raises(type(exception)):
        model_smoke.run("analyzer", "triton", report)
    assert any(call[1:3] == ["rm", "-f"] for call in calls)
    assert any(call[1:3] == ["network", "rm"] for call in calls)
    result = json.loads(report.read_text())
    assert result["status"] == "failed" and result["cleanup_ok"]
    assert result["phase"] == "readiness"


@pytest.mark.parametrize("suite", ["model", "proxy", "all"])
def test_missing_images_fail_before_a_provider_gate(monkeypatch, tmp_path, suite):
    monkeypatch.setattr(ner_integration, "image_id", lambda _: (_ for _ in ()).throw(RuntimeError("missing image")))
    calls = []
    monkeypatch.setattr(ner_integration, "run_child", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError):
        ner_integration.run(suite, report_dir=tmp_path)
    assert calls == []
    result = json.loads((tmp_path / f"integration-{suite}.json").read_text())
    assert result["status"] == "failed" and result["phase"] == "images"


def test_build_uses_synthetic_env_and_dedicated_tags(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ner_integration, "docker", lambda *a, **k: calls.append((a, k)))
    ner_integration.build_images(tmp_path, ["presidio-analyzer", "ner-triton"], support.IMAGE_TAGS["presidio-analyzer"])
    args, kwargs = calls[0]
    assert args[args.index("--env-file") + 1] == str(ROOT / ".env.example")
    assert kwargs["env"]["CODEX_LB_API_KEY"] == "build-only"
    assert args[-3:] == ("build", "presidio-analyzer", "ner-triton")
    override = json.loads((tmp_path / "build.json").read_text())
    assert all(v["image"].startswith("ru-llm-proxy-ner-tests:") for v in override["services"].values())


@pytest.mark.parametrize("interrupted", [subprocess.TimeoutExpired("child", 1), KeyboardInterrupt()])
def test_child_interruption_gives_the_shell_a_chance_to_clean_up(monkeypatch, interrupted):
    class Child:
        pid = 12345
        calls = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def poll(self):
            return None
        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise interrupted
            return 0
    child = Child()
    kills = []
    monkeypatch.setattr(support.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(support.os, "killpg", lambda *a: kills.append(a))
    with pytest.raises(type(interrupted)):
        support.run_child(["test"])
    assert kills == [(child.pid, signal.SIGTERM)]
    assert child.calls == 2


def test_topology_gate_cannot_pass_without_any_model_images(tmp_path, monkeypatch):
    import deployment_topology_gate as module
    gate = Gate(tmp_path, images=dict(support.IMAGE_TAGS))
    monkeypatch.setattr(gate, "sample_memory", lambda: None)
    monkeypatch.setattr(module, "image_id", lambda _: (_ for _ in ()).throw(RuntimeError("missing image")))
    with pytest.raises(RuntimeError):
        gate.run()
    assert gate.result["status"] == "failed"
    assert gate.result["phase"] == "identity"
    assert gate.result["cases"] == []


def test_cleanup_failure_cannot_leave_a_passed_report(tmp_path, monkeypatch):
    import deployment_topology_gate as module
    gate = Gate(tmp_path, images=dict(support.IMAGE_TAGS))
    gate.result["status"] = "passed"
    gate.last_context = SimpleNamespace(command=["docker", "compose"], environment={}, container_ids=lambda: [])
    monkeypatch.setattr(module, "container_states", lambda _: [])
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        subprocess.CalledProcessError(1, "docker")))
    with pytest.raises(subprocess.CalledProcessError):
        gate.cleanup()
    assert gate.result["status"] == "failed" and gate.result["cleanup_ok"] is False


@pytest.mark.parametrize("stack", ["litellm-presidio", "litellm-presidio-codex-lb"])
@pytest.mark.parametrize("count", [1, 2])
def test_topology_context_is_offline_isolated_and_uses_small_test_resources(tmp_path, stack, count):
    gate = Gate(tmp_path, images=dict(support.IMAGE_TAGS))
    before = gate.secrets()
    ctx = gate.context(stack, "production", "mask", triton_replicas=count)
    overlay = json.loads(gate.overlay.read_text())
    assert overlay["networks"]["ru-llm-proxy"]["internal"]
    assert overlay["services"]["nginx"]["networks"][:2] == ["ru-llm-proxy", "test-ingress"]
    assert "container_name" not in overlay["services"]["ner-triton"]
    assert ctx.counts["ner-triton"] == count
    assert ctx.counts["presidio-analyzer"] == 2
    assert ctx.values["LITELLM_MEMORY"] == "2g"
    assert gate.secrets() == before
    assert str(ROOT / ".env") != str(ctx.env_file)


def test_all_new_analyzer_tests_run_in_baseline_ci():
    workflow = yaml.safe_load((ROOT / ".github/workflows/baseline.yml").read_text())
    steps = workflow["jobs"]["recognizer-api"]["steps"]
    assert any(step.get("run") == "make test-recognizers" for step in steps)
    for module in ("test_triton_resilience.py", "test_onnx_artifact.py", "test_numpy_runtime.py",
                   "test_analyzer_budget.py", "test_ner_value_repair.py", "test_triton_service.py"):
        assert (ROOT / "presidio/tests" / module).is_file()


def test_ci_paths_include_every_shared_runtime_test_helper():
    import fnmatch
    workflow = yaml.safe_load((ROOT / ".github/workflows/ner-model-integration.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    for event in ("push", "pull_request"):
        patterns = triggers[event]["paths"]
        for name in ("model_smoke.py", "ner_integration.py", "ner_test_support.py",
                     "triton_routing_checks.py", "deployment_topology_gate.py",
                     "topology-resources.env", "nginx.ner-proxy.conf", "check_responses_history.py"):
            assert any(fnmatch.fnmatch("tests/e2e/" + name, pattern) for pattern in patterns), (event, name)
    assert workflow["permissions"] == {"contents": "read"}
    assert "pull_request_target" not in triggers


def test_independent_groups_continue_but_any_failure_is_fatal():
    calls, results = [], []
    def fail():
        calls.append("failed")
        raise AssertionError(("known_case", "do not include this request body"))
    with pytest.raises(RuntimeError, match="independent"):
        support.run_checks([("first", fail), ("next", lambda: calls.append("passed"))],
                           results, case_ids=["known_case"])
    assert calls == ["failed", "passed"]
    assert results == [
        {"check": "first", "status": "failed", "error_type": "AssertionError", "case_id": "known_case"},
        {"check": "next", "status": "passed"},
    ]
    assert "request body" not in json.dumps(results)


def test_successful_independent_groups_do_not_report_a_failure():
    results = []
    support.run_checks([("one", lambda: None), ("two", lambda: None)], results)
    assert [row["status"] for row in results] == ["passed", "passed"]


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(143)])
def test_interruption_does_not_start_more_independent_groups(exc):
    results, calls = [], []
    def stop():
        raise exc
    with pytest.raises(type(exc)):
        support.run_checks([("stop", stop), ("never", lambda: calls.append(True))], results)
    assert not calls
    assert results == [{"check": "stop", "status": "interrupted", "error_type": type(exc).__name__}]


def test_arbitrary_exception_text_is_not_a_reported_case_id():
    results = []
    with pytest.raises(RuntimeError):
        support.run_checks([("private", lambda: (_ for _ in ()).throw(
            ValueError(("not-a-corpus-id", "private data"))))], results, case_ids=["known_case"])
    assert "not-a-corpus-id" not in json.dumps(results)
    assert "private data" not in json.dumps(results)


def test_known_case_prefix_and_http_code_are_value_free_diagnostics():
    from urllib.error import HTTPError
    results = []
    checks = [("assertion", lambda: (_ for _ in ()).throw(AssertionError("known_case: private request body"))),
              ("http", lambda: (_ for _ in ()).throw(HTTPError("http://private", 503, "private error", {}, None)))]
    with pytest.raises(RuntimeError):
        support.run_checks(checks, results, case_ids=["known_case"])
    assert results[0]["case_id"] == "known_case"
    assert results[1]["http_status"] == 503
    assert "private" not in json.dumps(results)


def test_proxy_failure_does_not_skip_topologies(monkeypatch, tmp_path):
    import deployment_topology_gate as topology
    calls = []
    class FakeGate:
        def __init__(self, *a, **k):
            self.result = {"status": "running"}
        def run(self):
            calls.append("topologies")
            self.result["status"] = "passed"
    monkeypatch.setattr(topology, "Gate", FakeGate)
    monkeypatch.setattr(ner_integration, "image_id", lambda value: value)
    monkeypatch.setattr(ner_integration, "identities", lambda images: {"revision": "test"})
    monkeypatch.setattr(ner_integration, "build_images", lambda *a, **k: None)
    monkeypatch.setattr(ner_integration, "run_child", lambda *a, **k:
                        (_ for _ in ()).throw(RuntimeError("proxy failure")))
    with pytest.raises(RuntimeError):
        ner_integration.run("proxy", report_dir=tmp_path)
    assert calls == ["topologies"]
    report = json.loads((tmp_path / "integration-proxy.json").read_text())
    assert report["status"] == "failed"
    assert [item["status"] for item in report["checks"]] == ["failed", "passed"]
    assert json.loads((tmp_path / "topologies.json").read_text())["status"] == "passed"


def test_model_only_success_cannot_overwrite_a_failed_full_run(monkeypatch, tmp_path):
    aggregate = tmp_path / "integration-all.json"
    aggregate.write_text('{"status":"failed","suite":"all"}')
    monkeypatch.setattr(ner_integration, "image_id", lambda value: value)
    monkeypatch.setattr(ner_integration, "identities", lambda images: {"revision": "test"})
    monkeypatch.setattr(model_smoke, "run", lambda *a, **k: None)
    ner_integration.run("model", report_dir=tmp_path)
    assert json.loads(aggregate.read_text())["status"] == "failed"
    assert json.loads((tmp_path / "integration-model.json").read_text())["status"] == "passed"


def test_instruction_failure_does_not_skip_other_groups(monkeypatch, tmp_path):
    from tests.e2e import ner_instruction_checks as module
    calls = []
    monkeypatch.setattr(sys, "argv", ["instruction-checks"] + [
        item for name in ("analyzer", "proxy", "block-proxy", "capture")
        for item in ("--" + name + "-url", "http://test")])
    monkeypatch.setenv("NER_TEST_REPORT_DIR", str(tmp_path))
    monkeypatch.setenv("PRESIDIO_ANALYZER_BASE64_POLICY", "content")
    def fail(*args):
        raise AssertionError(("dialogue_role_which", "unchanged check"))
    monkeypatch.setattr(module, "check_dialogue_arguments", fail)
    monkeypatch.setattr(module, "check_instructions", lambda *a: calls.append("instructions"))
    monkeypatch.setattr(module, "check_names", lambda *a, **k: calls.append("names"))
    with pytest.raises(RuntimeError):
        module.main()
    assert calls == ["instructions", "names", "names", "names"]
    report = json.loads((tmp_path / "instructions-content.json").read_text())
    assert report["status"] == "failed"
    assert report["checks"][0]["case_id"] == "dialogue_role_which"
    assert all(item["status"] == "passed" for item in report["checks"][1:])


def test_dkb_failure_does_not_skip_base64_boundary_checks(monkeypatch, tmp_path):
    import dkb_corpus_checks as module
    calls = []
    monkeypatch.setattr(sys, "argv", ["dkb-checks"] + [
        item for name in ("analyzer", "proxy", "block-proxy", "capture")
        for item in ("--" + name + "-url", "http://test")])
    monkeypatch.setenv("NER_TEST_REPORT_DIR", str(tmp_path))
    monkeypatch.setenv("PRESIDIO_ANALYZER_BASE64_POLICY", "logistic")
    monkeypatch.setattr(module, "check", lambda *a: (_ for _ in ()).throw(AssertionError("synthetic failure")))
    monkeypatch.setattr(module, "check_chunk_boundaries", lambda *a: calls.append("boundaries"))
    monkeypatch.setattr(module, "check_base64_policy", lambda *a: calls.append("base64"))
    with pytest.raises(RuntimeError):
        module.main()
    assert calls == ["boundaries", "base64"]
    report = json.loads((tmp_path / "dkb-logistic.json").read_text())
    assert [item["status"] for item in report["checks"]] == ["failed", "passed", "passed"]


def test_logistic_ci_step_is_not_hidden_by_a_proxy_test_failure():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ner-model-integration.yml").read_text())
    steps = workflow["jobs"]["real-model-integration"]["steps"]
    assert any(step.get("id") == "proxy" for step in steps)
    logistic = next(step for step in steps if step.get("env", {}).get("PRESIDIO_ANALYZER_BASE64_POLICY") == "logistic")
    assert "!cancelled()" in logistic["if"]
    assert "steps.proxy.outcome != 'skipped'" in logistic["if"]
    assert not logistic.get("continue-on-error")


@pytest.mark.parametrize("target,suite,build", [
    ("test-hf-model", "model", True), ("test-hf-model-run", "model", False),
    ("test-ner-proxy", "proxy", False), ("test-ner-integration", "all", True),
])
def test_existing_make_commands_dispatch_without_docker(target, suite, build):
    result = subprocess.run(["make", "-n", target, "PYTHON_LOCAL=python3"], cwd=ROOT,
                            env={k: v for k, v in os.environ.items() if k not in {
                                "MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES"}},
                            text=True, capture_output=True, check=True)
    command = next(line for line in result.stdout.splitlines() if "ner_integration.py " in line)
    assert f"ner_integration.py {suite} " in command
    assert ("--build-model" in command) == build
    assert "docker compose" not in result.stdout
