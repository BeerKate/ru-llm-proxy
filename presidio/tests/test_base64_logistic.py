"""Integration of the supplied cascade, declarations, filters and source offsets."""
import base64
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recognizers.base64_data import Base64DataRecognizer, POLICY_ENV, load_policy
from recognizers import base64_logistic as cascade


def slices(text, results):
    return [(r.entity_type, text[r.start:r.end]) for r in results]


def test_default_and_unknown_policy(monkeypatch):
    monkeypatch.delenv(POLICY_ENV, raising=False)
    assert Base64DataRecognizer().policy == "content"
    assert load_policy({POLICY_ENV: " LOGISTIC "}) == "logistic"
    with pytest.raises(ValueError):
        Base64DataRecognizer(policy="unknown")


def test_model_coefficients_and_thresholds_are_frozen():
    assert cascade._W == [-1.02, -6.78, 5.07, -3.22, 7.81, 0.27, -2.03, -5.57]
    assert cascade._B == 7.75
    assert (cascade.SCORE_THRESHOLD, cascade.MIN_LEN, cascade.LONG_LEN, cascade.PREFIX) == (1.5, 16, 64, 256)


@pytest.mark.parametrize("text", [
    "test data function true false flag",
    '{"flag": true, "data": "test"}',
    "getUserProfileById /usr/local/lib/python3.13",
])
def test_plain_text_has_no_new_base64(text):
    assert Base64DataRecognizer(policy="logistic").analyze(text, ["BASE64_DATA"]) == []


@pytest.mark.parametrize("urlsafe,padding", [(False, True), (False, False), (True, True), (True, False)])
def test_plain_encoded_message_is_detected_without_sensitive_content(urlsafe, padding):
    encode = base64.urlsafe_b64encode if urlsafe else base64.b64encode
    value = encode("привет из обычного сообщения".encode()).decode()
    if not padding:
        value = value.rstrip("=")
    text = "Передано " + value + "."
    recognizer = Base64DataRecognizer(policy="logistic")
    assert slices(text, recognizer.analyze(text, ["BASE64_DATA"])) == [("BASE64_DATA", value)]
    assert recognizer.analyze(text, ["PERSON"]) == []


@pytest.mark.parametrize("key,kind", [("Base64", "BASE64_DATA"), ("password_base64", "PASSWORD")])
@pytest.mark.parametrize("value", ["TQ==", "test", "bad@encoding", "A" * 20000])
def test_native_declarations_keep_full_value_and_type(key, kind, value):
    text = f'{key}: "{value}"'
    recognizer = Base64DataRecognizer(policy="logistic")
    assert slices(text, recognizer.analyze(text, [kind])) == [(kind, value)]


def test_logistic_does_not_call_upstream_content_inspection(monkeypatch):
    recognizer = Base64DataRecognizer(policy="logistic")
    def forbidden(*args, **kwargs):
        raise AssertionError("upstream content inspection must not run in logistic mode")
    monkeypatch.setattr(recognizer, "_has_sensitive_content", forbidden)
    value = base64.b64encode("обычное сообщение без секретов".encode()).decode()
    assert recognizer.analyze(value, ["BASE64_DATA"])


def test_json_escaped_base64_keeps_original_source_offsets(monkeypatch):
    from recognizers.json_strings import JsonStringRecognizer
    monkeypatch.setenv(POLICY_ENV, "logistic")
    value = base64.b64encode("строка для проверки смещений".encode()).decode()
    # Escaping a valid symbol requires the native JSON projection, not replacement.
    text = json.dumps({"message": value}).replace(value[0], "\\u%04x" % ord(value[0]), 1)
    results = JsonStringRecognizer().analyze(text, ["BASE64_DATA"])
    assert len(results) == 1
    encoded = text[results[0].start:results[0].end]
    assert json.loads('"' + encoded + '"') == value


def test_fixed_cascade_known_short_and_hex_limits_are_not_hidden():
    # Characterize the original supplied algorithm; native declarations override this.
    assert cascade.find_spans("TQ==") == []
    assert cascade.find_spans(base64.b64encode(bytes(12)).decode()) == []


def test_policy_participates_in_cache_signature_without_loading_a_model():
    import ast
    import os
    import types
    source = Path(__file__).resolve().parents[1] / "analyzer_server.py"
    tree = ast.parse(source.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {"ANALYSIS_SIGNATURE_ENV_NAMES", "ANALYSIS_SIGNATURE_SCHEMA", "ANALYSIS_SIGNATURE_DISTRIBUTIONS"}
        for t in n.targets)]
    selected += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_analysis_signature"]
    namespace = {"Path": Path, "__file__": str(source), "hashlib": hashlib, "json": json, "os": os,
                 "_analysis_source_files": lambda: [source, Path(cascade.__file__)],
                 "_distribution_version": lambda name: "fixed", "MODEL_ID": "fixed", "MODEL_REVISION": "fixed"}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    from unittest.mock import patch
    with patch.dict(os.environ, {POLICY_ENV: "content"}):
        content = namespace["_build_analysis_signature"]()
    with patch.dict(os.environ, {POLICY_ENV: "logistic"}):
        logistic = namespace["_build_analysis_signature"]()
    assert content != logistic


def _deterministic_results(text, policy, monkeypatch):
    from recognizers import ALL_RECOGNIZERS
    from result_merging import merge_results
    monkeypatch.setenv(POLICY_ENV, policy)
    results = []
    for cls in ALL_RECOGNIZERS:
        recognizer = cls()
        results.extend(recognizer.analyze(text, recognizer.supported_entities))
    return merge_results(text, results, score_threshold=0.35).results


@pytest.mark.parametrize("project_json", [False, True])
@pytest.mark.parametrize("file_number", range(1, 19))
def test_native_corpus_ranges_and_types_are_preserved(monkeypatch, project_json, file_number):
    root = Path(__file__).resolve().parents[2]
    text = (root / f"tests/fixtures/dkb/{file_number}.txt").read_bytes().decode("utf-8")
    if project_json:
        text = json.dumps({"source": text}, ensure_ascii=False)
    baseline = _deterministic_results(text, "content", monkeypatch)
    integrated = _deterministic_results(text, "logistic", monkeypatch)
    keys = lambda values: {(r.start, r.end, r.entity_type) for r in values}
    assert keys(baseline) <= keys(integrated)
    assert all(a.end <= b.start for a, b in zip(integrated, integrated[1:]))


def test_overlapping_base64_adds_only_uncovered_native_gaps():
    from presidio_analyzer import RecognizerResult
    from result_merging import merge_results
    recognizer = Base64DataRecognizer(policy="logistic")
    native = RecognizerResult("AUTH_TOKEN", 5, 15, 0.95)
    results = merge_results("x" * 30, [
        native, recognizer._result(0, 10, "base64.logistic-cascade"),
        recognizer._result(8, 25, "base64.logistic-cascade"),
    ]).results
    assert [(r.start, r.end, r.entity_type) for r in results] == [
        (0, 5, "BASE64_DATA"), (5, 15, "AUTH_TOKEN"), (15, 25, "BASE64_DATA")
    ]
    assert results[1] is native
    assert merge_results("x" * 30, results).results == results


@pytest.mark.parametrize("entities,threshold", [(["PERSON"], 0.35), (None, 0.96)])
def test_added_coverage_respects_entity_filter_and_score(entities, threshold):
    from result_merging import merge_results
    finding = Base64DataRecognizer(policy="logistic")._result(0, 10, "base64.logistic-cascade")
    assert merge_results("x" * 10, [finding], requested_entities=entities, score_threshold=threshold).results == ()
