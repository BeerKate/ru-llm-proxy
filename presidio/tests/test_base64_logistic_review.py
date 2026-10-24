"""Regressions from the PR #104 review: content parity, linear assembly, MIME and filters.

All values are synthetic. API tests stub the transformer forward.
"""
import base64
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from presidio import analyzer_server
from recognizers.base64_data import Base64DataRecognizer
from recognizers.base64_logistic import find_spans
from recognizers.credential_rules import AuthTokenRecognizer
from recognizers.json_strings import JsonStringRecognizer
from result_merging import merge_results

b64 = lambda raw: base64.b64encode(raw).decode()
SHORT_SENSITIVE = ["a@b.co", "a@b.ru", "x@y.io", "p@q.org", "ab@c.ru"]


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("PRESIDIO_ANALYZER_BASE64_POLICY", "logistic")
    registry = RecognizerRegistry(supported_languages=["ru"])
    for recognizer in (Base64DataRecognizer(), JsonStringRecognizer(), AuthTokenRecognizer()):
        registry.add_recognizer(recognizer)
    engine = AnalyzerEngine(registry=registry, nlp_engine=analyzer_server.nlp_engine, supported_languages=["ru"])
    monkeypatch.setattr(analyzer_server, "analyzer", engine)
    monkeypatch.setattr(analyzer_server.ner_recognizer, "require_ready", lambda: None)
    monkeypatch.setattr(analyzer_server.ner_recognizer, "analyze", lambda *_a, **_k: [])
    return TestClient(analyzer_server.app)


def spans(api, text, entities=("BASE64_DATA",)):
    response = api.post("/api/v1/analyze", json={"text": text, "entities": list(entities), "score_threshold": 0.35})
    assert response.status_code == 200
    return [(e["entity_type"], e["start"], e["end"]) for e in response.json()["entities"]]


# --- 1. The logistic policy keeps every content-policy protection ------------------------

@pytest.mark.parametrize("source", SHORT_SENSITIVE)
def test_short_sensitive_values_are_protected_through_api(api, source):
    value = b64(source.encode()).rstrip("=")
    for text in (value, f"Код {value} из письма.", json.dumps({"value": value})):
        start = text.index(value)
        assert spans(api, text) == [("BASE64_DATA", start, start + len(value))]


def test_ordinary_words_get_no_new_masks_through_api(api):
    text = "config defaults padding headers username getUserProfileById CPU BERT baseline"
    assert spans(api, text) == []


def test_single_chunk_request_runs_document_cascade(api):
    value = b64("обычное сообщение без секретов".encode())
    text = f"Передано {value}."
    start = text.index(value)
    assert spans(api, text) == [("BASE64_DATA", start, start + len(value))]


# --- 2. Many positive candidates: full coverage, no quadratic passes -------------------------

def test_many_values_are_all_covered():
    value = b64(b'password="Synthetic-Pass-2026"')
    text = "\n".join(f'"{value}"' for _ in range(5000))
    expected = [(i * (len(value) + 3) + 1, i * (len(value) + 3) + 1 + len(value)) for i in range(5000)]
    assert find_spans(text) == expected
    results = Base64DataRecognizer(policy="logistic").analyze_document(text, ["BASE64_DATA"])
    assert sorted((r.start, r.end) for r in results) == expected


def test_many_values_across_outer_chunks_through_api(api):
    value = b64(b"synthetic-binary-\x00\x01\x02-payload-0123456789")
    line = f'"{value}"\n'
    text = line * (140_000 // len(line))  # longer than one 128k outer chunk
    found = spans(api, text)
    assert len(found) == text.count(value)
    assert all(text[s:e] == value for _, s, e in found)


# --- 3. MIME blocks do not swallow the following text ---------------------------------------

def mime(raw, width=76, newline="\n"):
    encoded = b64(raw)
    return newline.join(encoded[i:i + width] for i in range(0, len(encoded), width))


@pytest.mark.parametrize("width", [64, 76])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("size", [57 * 3, 57 * 3 + 1, 57 * 3 + 2, 500])
def test_real_mime_block_is_fully_covered(width, newline, size):
    block = mime(bytes((i * 37 + 11) % 256 for i in range(size)), width, newline)
    text = "Вложение:" + newline + block + newline + "Конец письма."
    start = text.index(block)
    assert find_spans(text) == [(start, start + len(block))]


@pytest.mark.parametrize("tail", ["VLBCAnSLGHYhKRKU", "jdDKcNAGuxevYAwv", "yeavRBaLaohUNxziBsdg"])
def test_letters_only_mime_tail_is_covered(tail):
    body = mime(bytes((i * 29 + 3) % 256 for i in range(57 * 4)), 76)
    text = "Вложение:\n" + body + "\n" + tail + "\n\nКонец."
    start = text.index(body)
    assert find_spans(text) == [(start, start + len(body) + 1 + len(tail))]


def test_json_escaped_mime_block_is_fully_covered():
    block = mime(bytes((i * 53 + 7) % 256 for i in range(400)), 76)
    text = json.dumps({"attachment": block})
    start = text.index(json.dumps(block)[1:-1])
    assert find_spans(text) == [(start, start + len(json.dumps(block)) - 2)]


@pytest.mark.parametrize("follow", ["Return the result exactly.", "HelloWorld", "NextStep", "UserName",
                                    "def main():", "", "\nReturn the result exactly."])
def test_text_after_full_width_line_is_not_captured(follow):
    source = ('password="Synthetic-Pass-2026"' + " " * 48)[:48]
    value = b64(source.encode())
    assert len(value) == 64
    text = value + "\n" + follow
    assert find_spans(text) == [(0, 64)]


def test_adjacent_independent_values_are_not_joined():
    first, second = b64(b"first-synthetic-value-0001"), b64(b"second-synthetic-value-02")
    for separator in ("\n", " ", "\n\n", "  "):
        text = first + separator + second
        assert find_spans(text) == [(0, len(first)), (len(first) + len(separator), len(text))]


# --- 4. Declared containers keep span and class under entity filters -------------------------

DECLARED_KEYS = [("password_base64", "PASSWORD"), ("secret_key_base64", "SECRET_KEY")]


@pytest.mark.parametrize("key,kind", DECLARED_KEYS)
def test_excluded_declared_value_is_not_reclassified(api, key, kind):
    declared = b64(b"Synthetic-Pass-2026")
    other = b64(b"ordinary text without secrets")
    text = f"{key}={declared}\n{other}"
    first, second = text.index(declared), text.index(other)
    assert spans(api, text, ["BASE64_DATA"]) == [("BASE64_DATA", second, second + len(other))]
    assert spans(api, text, ["BASE64_DATA", kind]) == [
        (kind, first, first + len(declared)), ("BASE64_DATA", second, second + len(other))
    ]


@pytest.mark.parametrize("key,kind", DECLARED_KEYS)
def test_declared_value_in_json_keeps_class(api, key, kind):
    declared = b64(b"Synthetic-Pass-2026")
    text = json.dumps({key: declared, "note": b64(b"ordinary text without secrets")})
    start = text.index(declared)
    assert (kind, start, start + len(declared)) in spans(api, text, ["BASE64_DATA", kind])
    assert all(not (s < start + len(declared) and start < e) for _, s, e in spans(api, text, ["BASE64_DATA"]))


def test_merge_is_idempotent_with_partial_overlap():
    from presidio_analyzer import RecognizerResult
    text = "x" * 60
    recognizer = Base64DataRecognizer(policy="logistic")
    declared = recognizer._result(10, 30, "base64.declared", "PASSWORD")
    cascade = recognizer._result(0, 50, "base64.logistic-cascade", encoded_container=False)
    once = merge_results(text, [declared, cascade], requested_entities=["BASE64_DATA"]).results
    assert [(r.start, r.end, r.entity_type) for r in once] == [(0, 10, "BASE64_DATA"), (30, 50, "BASE64_DATA")]
    assert merge_results(text, once, requested_entities=["BASE64_DATA"]).results == once
    native = RecognizerResult("AUTH_TOKEN", 5, 15, 0.95)
    both = merge_results(text, [native, cascade]).results
    assert [(r.start, r.end, r.entity_type) for r in both] == [
        (0, 5, "BASE64_DATA"), (5, 15, "AUTH_TOKEN"), (15, 50, "BASE64_DATA")
    ]


# --- Second review: MIME tails versus identifiers with digits --------------------------------

FULL_LINE = b64((('password="Synthetic-Pass-2026"' + " " * 48)[:48]).encode())
IDENTIFIERS_WITH_DIGITS = [
    # word + number
    "Version2", "Default2", "Return42", "Section2", "Figure12", "ResNet50", "Release2026", "Python312",
    "Chapter10", "Build4521", "Item007", "Layer12", "Test123", "Model3", "README1", "Table3",
    # abbreviations and protocol or codec names
    "H264", "IPv6", "A1B2", "HTTP2", "TLS13", "MySQL8", "GPT4o", "UTF16", "OAuth2", "i18n", "l10n", "Web3",
    # lowercase identifiers, hex digests
    "user42", "page2", "utf8", "sha256", "node18", "8827e1b2", "deadbeef12345678",
]


@pytest.mark.parametrize("word", IDENTIFIERS_WITH_DIGITS)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_identifier_with_digits_after_full_line_is_not_captured(api, word, newline):
    for width in (64, 76):
        line = (FULL_LINE * 2)[:width]
        assert spans(api, line + newline + word) == [("BASE64_DATA", 0, width)]


@pytest.mark.parametrize("word", IDENTIFIERS_WITH_DIGITS)
def test_identifier_with_digits_after_full_line_in_json_is_not_captured(api, word):
    text = json.dumps({"attachment": FULL_LINE + "\n" + word})
    start = text.index(FULL_LINE)
    assert spans(api, text) == [("BASE64_DATA", start, start + len(FULL_LINE))]


@pytest.mark.parametrize("tail", ["Qr3yIQbw", "d2Lw88tNdk3H", "ByBRFZoPifLG2srj", "uzESRf1v", "QUJDRA==", "QUI=", "a+b/"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_real_short_tail_is_covered_through_api(api, tail, newline):
    text = "Вложение:\n" + FULL_LINE + newline + tail + "\nКонец."
    start = text.index(FULL_LINE)
    end = start + len(FULL_LINE) + len(newline) + len(tail)
    found = spans(api, text)
    # The block may be split where the content policy already masks the first line.
    assert all(start <= s and e <= end and kind == "BASE64_DATA" for kind, s, e in found)
    assert {i for _, s, e in found for i in range(s, e)} >= set(range(start, start + len(FULL_LINE))) | set(range(end - len(tail), end))


# --- Second review: declared keys written with JSON escapes ----------------------------------

TYPED_KEYS = [("password_base64", "PASSWORD"), ("secret_key_base64", "SECRET_KEY"), ("login_base64", "LOGIN"),
              ("auth_token_base64", "AUTH_TOKEN"), ("api_key_base64", "SECRET_KEY"), ("document_base64", "BASE64_DATA")]
VALUE = b64(b"Synthetic-Password-2026")


def escaped_keys(key):
    """The plain key and spellings with one letter written as a JSON \\uXXXX escape."""
    yield key
    for index in (0, key.index("base64")):
        yield key[:index] + "\\u%04x" % ord(key[index]) + key[index + 1:]


def documents(key, value=VALUE):
    yield '{"%s":"%s"}' % (key, value)
    yield '{"outer":{"%s":"%s"},"note":"x"}' % (key, value)
    yield '[{"other":1},{"%s":"%s"}]' % (key, value)


@pytest.mark.parametrize("key,kind", TYPED_KEYS)
def test_escaped_declared_key_keeps_class_and_filter(api, key, kind):
    for spelling in escaped_keys(key):
        for text in documents(spelling):
            start = text.index(VALUE)
            value = (start, start + len(VALUE))
            assert (kind, *value) in spans(api, text, ["BASE64_DATA", kind])
            excluded = spans(api, text, ["BASE64_DATA"])
            if kind == "BASE64_DATA":
                assert excluded == [("BASE64_DATA", *value)]
            else:
                assert excluded == []


def test_escaped_key_and_escaped_value_keep_offsets(api):
    escaped_value = "\\u%04x" % ord(VALUE[0]) + VALUE[1:]
    text = '{"password_b\\u0061se64":"%s"}' % escaped_value
    start = text.index(escaped_value)
    assert spans(api, text, ["BASE64_DATA", "PASSWORD"]) == [("PASSWORD", start, start + len(escaped_value))]
    assert spans(api, text, ["BASE64_DATA"]) == []
