"""Full Analyzer API with every deterministic recognizer and the document-level cascade.

The DKB corpus, its JSON projection and selective entity filters are compared
between the content and logistic policies. The transformer forward is stubbed.
"""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from presidio import analyzer_server
from recognizers import ALL_RECOGNIZERS
from recognizers.base64_data import Base64DataRecognizer

CORPUS = Path(__file__).resolve().parents[2] / "tests/fixtures/dkb"
TYPED = {"PASSWORD", "SECRET_KEY", "LOGIN", "AUTH_TOKEN"}


def client(monkeypatch, policy):
    monkeypatch.setenv("PRESIDIO_ANALYZER_BASE64_POLICY", policy)
    registry = RecognizerRegistry(supported_languages=["ru"])
    for recognizer_cls in ALL_RECOGNIZERS:
        registry.add_recognizer(recognizer_cls())
    engine = AnalyzerEngine(registry=registry, nlp_engine=analyzer_server.nlp_engine, supported_languages=["ru"])
    monkeypatch.setattr(analyzer_server, "analyzer", engine)
    monkeypatch.setattr(analyzer_server.ner_recognizer, "require_ready", lambda: None)
    monkeypatch.setattr(analyzer_server.ner_recognizer, "analyze", lambda *_a, **_k: [])
    return TestClient(analyzer_server.app)


def analyze(api, text, entities=None):
    response = api.post("/api/v1/analyze", json={"text": text, "entities": entities, "score_threshold": 0.35})
    assert response.status_code == 200
    return [(e["entity_type"], e["start"], e["end"]) for e in response.json()["entities"]]


VALUE = "U3ludGhldGljLVBhc3N3b3JkLTIwMjY="  # synthetic "Synthetic-Password-2026"
OTHER = "b3JkaW5hcnkgdGV4dCB3aXRob3V0IHNlY3JldHM="  # synthetic ordinary text
DECLARED_KEYS = ["password_base64", "secret_key_base64", "login_base64", "auth_token_base64", "api_key_base64"]


def declared_documents():
    """Synthetic declared credentials: plain and JSON-escaped keys, nesting, arrays, text."""
    for key in DECLARED_KEYS:
        escaped = key.replace("base64", "b\\u0061se64")
        for spelling in (key, escaped):
            yield f"{key}:{spelling == key}:flat", '{"%s":"%s","note":"%s"}' % (spelling, VALUE, OTHER)
            yield f"{key}:{spelling == key}:nested", '{"a":{"b":[{"%s":"%s"}]},"c":"%s"}' % (spelling, VALUE, OTHER)
        yield f"{key}:text", f"{key}={VALUE}\n{OTHER}"


def corpus():
    for path in sorted(CORPUS.glob("*.txt"), key=lambda p: int(p.stem)):
        text = path.read_bytes().decode("utf-8")
        yield path.stem, text
        yield path.stem + ".json", json.dumps({"source": text}, ensure_ascii=False)
    yield from declared_documents()


@pytest.fixture(scope="module")
def results():
    monkeypatch = pytest.MonkeyPatch()
    try:
        out = {}
        for policy in ("content", "logistic"):
            api = client(monkeypatch, policy)
            for name, text in corpus():
                out[policy, name] = analyze(api, text)
                if policy == "logistic":
                    out["filtered", name] = analyze(api, text, ["BASE64_DATA"])
        return out
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize("name", [name for name, _ in corpus()])
def test_logistic_keeps_every_content_range_and_type(results, name):
    content, logistic = set(results["content", name]), results["logistic", name]
    assert content <= set(logistic)
    assert all(a[2] <= b[1] for a, b in zip(sorted(logistic, key=lambda r: r[1]), sorted(logistic, key=lambda r: r[1])[1:]))


@pytest.mark.parametrize("name,text", list(corpus()))
def test_excluded_declared_types_are_not_reclassified(results, name, text):
    declared = [
        (r.start, r.end)
        for r in Base64DataRecognizer(policy="logistic").analyze_document(text, None)
        if r.entity_type in TYPED
    ]
    for _, start, end in results["filtered", name]:
        assert not any(a < end and start < b for a, b in declared)
    if name.startswith(tuple(DECLARED_KEYS)):
        assert declared, "synthetic documents must contain a declared credential"
        assert any(start <= text.index(OTHER) < end for _, start, end in results["filtered", name])
