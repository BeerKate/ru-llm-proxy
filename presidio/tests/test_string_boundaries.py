"""Synthetic regressions for source-preserving masks in JSON and quoted code."""
import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from presidio_analyzer import RecognizerResult
from recognizers import ALL_RECOGNIZERS
from result_merging import (
    merge_results, DETECTION_SOURCE_METADATA_KEY, SOURCE_NER, SOURCE_NATIVE_CREDENTIAL,
)


@pytest.fixture(scope='module')
def recognizers():
    return [cls() for cls in ALL_RECOGNIZERS]


def detections(text, recognizers, entities=None):
    return [r for recognizer in recognizers
            for r in recognizer.analyze(text, entities or recognizer.supported_entities, None)]


def finding(kind, start, end, source=SOURCE_NER):
    return RecognizerResult(kind, start, end, .9, recognition_metadata={
        DETECTION_SOURCE_METADATA_KEY: source,
        RecognizerResult.RECOGNIZER_NAME_KEY: 'synthetic-boundary-probe',
    })


def mask_and_restore(text, results):
    spans = merge_results(text, results, score_threshold=.35).results
    assert all(a.end <= b.start for a, b in zip(spans, spans[1:]))
    masked = text
    saved = {}
    for i, result in reversed(list(enumerate(spans))):
        token = f'<MASK_{i}>'
        saved[token] = text[result.start:result.end]
        masked = masked[:result.start] + token + masked[result.end:]
    restored = masked
    for token, value in saved.items():
        restored = restored.replace(token, value)
    assert restored.encode('utf-8') == text.encode('utf-8')
    return masked, spans


@pytest.mark.parametrize('value', ['"', '\\', '$$', '-$', '\\"', 'a"b', 'a\\b', '\n', '\t', '😀"\\', 'a', 'ab', '0'])
@pytest.mark.parametrize('ascii_only', [False, True])
def test_json_password_escapes_preserve_structure_and_restore(value, ascii_only, recognizers):
    text = json.dumps({'password': value, 'next': 'ordinary'}, ensure_ascii=ascii_only)
    masked, spans = mask_and_restore(text, detections(text, recognizers))
    document = json.loads(masked)
    assert document['next'] == 'ordinary'
    assert document['password'] != value
    assert any(r.entity_type == 'PASSWORD' for r in spans)
    raw_start = text.index('\"', text.index(':')) + 1
    raw_end = text.index(',') - 1
    assert all(any(r.start <= i < r.end for r in spans) for i in range(raw_start, raw_end))


@pytest.mark.parametrize('ascii_only', [False, True])
def test_json_address_with_internal_quotes(ascii_only, recognizers):
    value = 'ул. Ленина, д. 5, кв. "12"'
    text = json.dumps({'address': value, 'next': 7}, ensure_ascii=ascii_only)
    masked, spans = mask_and_restore(text, detections(text, recognizers))
    assert json.loads(masked)['next'] == 7
    assert json.loads(masked)['address'].startswith('<MASK_')
    assert '12' not in json.loads(masked)['address']
    assert all(r.entity_type == 'RU_ADDRESS' for r in spans)


def test_controlled_ner_tail_stops_at_json_value():
    text = '{"password":"Synthetic42","x":1}'
    start = text.index('Synthetic42'); end = start + len('Synthetic42')
    masked, spans = mask_and_restore(text, [
        finding('PASSWORD', start, end, SOURCE_NATIVE_CREDENTIAL),
        finding('PASSWORD', start, end + 2),
    ])
    assert json.loads(masked) == {'password': '<MASK_0>', 'x': 1}
    assert spans[-1].end == end


def test_controlled_name_balance_keeps_outer_quote():
    text = json.dumps({'text': 'ООО "Вектор'}, ensure_ascii=False)
    start = text.index('ООО'); end = len(text) - 2
    masked, _ = mask_and_restore(text, [finding('ORGANIZATION', start, end)])
    assert json.loads(masked) == {'text': '<MASK_0>'}


@pytest.mark.parametrize('escape', [r'\"', r'\\', r'\u041c', r'\ud83d\ude00'])
def test_every_partial_escape_span_is_indivisible(escape):
    text = '{"value":"' + escape + '","x":1}'
    start = text.index(escape)
    for left in range(len(escape)):
        for right in range(left + 1, len(escape) + 1):
            masked, _ = mask_and_restore(text, [finding('PASSWORD', start + left, start + right)])
            assert json.loads(masked)['x'] == 1
            assert json.loads(masked)['value'] == '<MASK_0>'


def test_name_after_literal_backslash_n_keeps_letter_n():
    text = json.dumps({'text': r'\nСервис Плюс'}, ensure_ascii=False)
    start = text.index('nСервис')
    _, spans = mask_and_restore(text, [finding('ORGANIZATION', start, len(text)-2)])
    assert spans[0].start == start


@pytest.mark.parametrize('prefix', ['', 'Запрос:\n', '```json\n'])
def test_json_in_text_and_code_fence_keeps_boundaries(prefix):
    text = prefix + '{"password":"Synthetic42","x":1}' + ('\n```' if prefix.startswith('```') else '')
    start = text.index('Synthetic42'); end = start + len('Synthetic42')
    masked, _ = mask_and_restore(text, [finding('PASSWORD', start, end + 2)])
    assert json.loads(masked[len(prefix):].removesuffix('\n```'))['x'] == 1


@pytest.mark.parametrize('text', [
    r'password = "a\"b"',
    r"password = 'a\'b'",
    r'password = r"a\"b"',
    'password = """a\\"b"""',
])
def test_python_literal_preserves_syntax(text, recognizers):
    ast.parse(text)
    masked, spans = mask_and_restore(text, detections(text, recognizers, ['PASSWORD']))
    ast.parse(masked)
    assert spans
    assert 'b' not in masked.split('=', 1)[1]


def test_nested_and_duplicate_keys_are_all_protected(recognizers):
    text = '{"password":"\\\"","nested":[{"password":"\\\\"}],"password":"$$","x":1}'
    masked, spans = mask_and_restore(text, detections(text, recognizers))
    document = json.loads(masked)
    assert document['x'] == 1
    assert len([r for r in spans if r.entity_type == 'PASSWORD']) == 3


def test_two_names_in_separate_array_values_never_join():
    text = '["Москва" "Гавана"]'  # Python adjacent literals, not valid JSON.
    results = [finding('LOCATION', text.index(v), text.index(v)+len(v)) for v in ('Москва', 'Гавана')]
    _, spans = mask_and_restore(text, results)
    assert len(spans) == 2


def test_noisy_ner_edges_cannot_consume_json_syntax():
    text = '{"password":"Synthetic42","x":1}'
    start = text.index('Synthetic42'); end = start + len('Synthetic42')
    for left in range(start - 2, start + 3):
        for right in range(end - 2, end + 4):
            masked, _ = mask_and_restore(text, [finding('PASSWORD', left, right)])
            assert json.loads(masked)['x'] == 1


def test_generated_short_passwords_preserve_structure(recognizers):
    # All combinations of syntax-sensitive symbols, plus Unicode/control input.
    import itertools
    alphabet = ['"', '\\', "'", '$', '\n', 'Ж', '😀']
    for size in (1, 2, 3):
        for chars in itertools.product(alphabet, repeat=size):
            value = ''.join(chars)
            text = json.dumps({'password': value, 'next': {'x': 1}})
            masked, spans = mask_and_restore(text, detections(text, recognizers, ['PASSWORD']))
            doc = json.loads(masked)
            assert doc['next'] == {'x': 1}
            assert doc['password'] != value, repr(value)
            content_start = text.index('"', text.index(':')) + 1
            content_end = text.index(',') - 1
            assert all(any(r.start <= i < r.end for r in spans)
                       for i in range(content_start, content_end)), repr(value)


def test_raw_python_backslash_n_does_not_hide_name_letter():
    text = r'org = r"\nСервис Плюс"'
    start = text.index('nСервис')
    _, spans = mask_and_restore(text, [finding('ORGANIZATION', start, len(text)-1)])
    assert spans[0].start <= start < spans[0].end


@pytest.mark.parametrize('policy', ['content', 'logistic'])
def test_real_api_json_escaping_and_entity_filters(monkeypatch, recognizers, policy):
    from fastapi.testclient import TestClient
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio import analyzer_server

    monkeypatch.setenv('PRESIDIO_ANALYZER_BASE64_POLICY', policy)
    registry = RecognizerRegistry(supported_languages=['ru'])
    for recognizer in recognizers:
        registry.add_recognizer(recognizer)
    engine = AnalyzerEngine(registry=registry, nlp_engine=analyzer_server.nlp_engine,
                            supported_languages=['ru'])
    monkeypatch.setattr(analyzer_server, 'analyzer', engine)
    monkeypatch.setattr(analyzer_server.ner_recognizer, 'require_ready', lambda: None)
    monkeypatch.setattr(analyzer_server.ner_recognizer, 'analyze', lambda *_a, **_k: [])
    api = TestClient(analyzer_server.app)
    for value in ['"', '\\', '$$', '😀\\"']:
        text = json.dumps({'password': value, 'x': 1})
        response = api.post('/api/v1/analyze', json={'text': text, 'entities': ['PASSWORD'],
                                                    'score_threshold': .35})
        assert response.status_code == 200
        findings = [finding(r['entity_type'], r['start'], r['end'], SOURCE_NATIVE_CREDENTIAL)
                    for r in response.json()['entities']]
        masked, _ = mask_and_restore(text, findings)
        assert json.loads(masked)['password'] != value
        assert json.loads(masked)['x'] == 1
        excluded = api.post('/api/v1/analyze', json={'text': text, 'entities': ['RU_PASSPORT']})
        assert excluded.json()['entities'] == []


@pytest.mark.parametrize('key', [r'passw\u006frd', r'\u0061ddress'])
def test_escaped_json_key_is_bound_to_its_own_string(key, recognizers):
    value = '"' if 'passw' in key else 'ул. Ленина, д. 5, кв. "12"'
    text = '{"%s":%s,"next":"ordinary"}' % (key, json.dumps(value))
    masked, _ = mask_and_restore(text, detections(text, recognizers))
    document = json.loads(masked)
    decoded_key = json.loads('"' + key + '"')
    assert document[decoded_key] != value
    assert document['next'] == 'ordinary'


def test_json_above_projection_limit_still_has_safe_mask_boundaries(recognizers):
    # Lexical boundary safety is independent of the bounded decoded recognizer.
    text = json.dumps({'padding': 'a' * 66000, 'password': '"', 'next': 'ordinary'})
    masked, _ = mask_and_restore(text, detections(text, recognizers, ['PASSWORD']))
    document = json.loads(masked)
    assert document['padding'] == 'a' * 66000
    assert document['next'] == 'ordinary'
    assert document['password'] != '"'


@pytest.mark.parametrize('projected', [False, True])
def test_later_address_can_cover_a_trimmed_ner_remainder_inside_json(projected):
    from result_merging import SOURCE_STRUCTURAL

    value = 'password=Alpha42&tail ул. Ленина, д. 5'
    text = json.dumps({'source': value, 'next': 'ordinary'}, ensure_ascii=False)
    value_start = text.index('password=')
    value_end = text.index('", "next"')
    native_start = text.index('Alpha42')
    native = finding('PASSWORD', native_start, native_start + len('Alpha42'), SOURCE_NATIVE_CREDENTIAL)
    wide = finding('PASSWORD', value_start, value_end)
    additive = finding('RU_ADDRESS', native_start, value_end, SOURCE_STRUCTURAL)
    additive.recognition_metadata[RecognizerResult.RECOGNIZER_NAME_KEY] = (
        'JsonStringRecognizer' if projected else 'RuAddressChainRecognizer'
    )
    if projected:
        additive.recognition_metadata['projected_recognizer_name'] = 'RuAddressChainRecognizer'
    import itertools
    for order in itertools.permutations([native, wide, additive]):
        masked, spans = mask_and_restore(text, order)
        assert json.loads(masked)['next'] == 'ordinary'
        assert 'Ленина' not in json.loads(masked)['source']
        assert all(any(r.start <= i < r.end for r in spans)
                   for i in range(text.index('ул.'), value_end))
