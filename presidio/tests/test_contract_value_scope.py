"""Synthetic contract-key binding and natural-language context regressions."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from result_merging import (
    contract_context_evidence, merge_results, CONTRACT_CONTEXT_SCORE,
    DETECTION_SOURCE_METADATA_KEY, SOURCE_NER,
)
from presidio_analyzer import RecognizerResult


def values(text):
    return [text[s:e] for s, e in contract_context_evidence(text)]


@pytest.mark.parametrize('empty', ['null', '""', '{}', '[]', 'false'])
@pytest.mark.parametrize('other', ['"BUILD-2025/99"', '120000', '"12.10.2026"', '"2.4.1"'])
def test_empty_or_container_contract_never_confirms_neighbor(empty, other):
    text = '{"contract_id":%s,"other":%s}' % (empty, other)
    assert values(text) == []


@pytest.mark.parametrize('key', ['contract_id', 'contractNumber', 'client_contract_no',
    'agreement_ref', 'номер_договора', 'nomer_dogovora', 'dogovor_id'])
def test_contract_key_confirms_only_its_own_string(key):
    text = json.dumps({key: 'DG-2026-0042', 'build_id': 'BUILD-2025/99'}, ensure_ascii=False)
    assert values(text) == ['DG-2026-0042']


@pytest.mark.parametrize('text', [
    'contract_id="DG-2026-0042"; build_id="BUILD-2025/99"',
    'contract_id=DG-2026-0042\nbuild_id=BUILD-2025/99',
    'contract_id=DG-2026-0042 build_id=BUILD-2025/99',
    'fn(contract_id="DG-2026-0042", build_id="BUILD-2025/99")',
    'contractNumber: DG-2026-0042\nother: BUILD-2025/99',
    '{"contract_id":"DG-2026-0042","nested":{"build_id":"BUILD-2025/99"}}',
    '{"contract_id":null,"items":["BUILD-2025/99",120000]}',
])
def test_assignment_and_container_boundaries(text):
    assert values(text) == (['DG-2026-0042'] if 'DG-' in text else [])


@pytest.mark.parametrize('text', [
    '{"note":"Договор подписан","build_id":"BUILD-2025/99"}',
    '{"note":"Договор подписан","timeout_ms":120000}',
    '["Договор подписан","BUILD-2025/99"]',
    'Договор подписан. build_id=BUILD-2025/99',
    'Договор подписан, build_id=BUILD-2025/99',
    'contract_id = other_field; timeout_ms = 120000',
    '{"contract_id":{"build_id":"BUILD-2025/99"}}',
    '{"contract_id":["BUILD-2025/99"]}',
])
def test_context_never_leaks_from_other_values_or_keys(text):
    assert values(text) == []


@pytest.mark.parametrize('ascii_only', [False, True])
def test_natural_contract_context_inside_json_value_is_preserved(ascii_only):
    value = 'Договор № ДГ-2026/117 подписан.'
    text = json.dumps({'note': value, 'build_id': 'BUILD-2025/99'}, ensure_ascii=ascii_only)
    evidence = contract_context_evidence(text)
    assert len(evidence) == 1
    raw = text[evidence[0][0]:evidence[0][1]]
    assert json.loads('"' + raw + '"') == 'ДГ-2026/117'


def test_escaped_contract_key_and_value_use_original_coordinates():
    text = '{"contract_\\u0069d":"\\u0414\\u0413-2026/117","other":"BUILD-2025/99"}'
    found = values(text)
    assert found == [r'\u0414\u0413-2026/117']


def test_duplicate_contract_keys_keep_every_own_value():
    text = '{"contract_id":"AA-2026/17","other":"BUILD-2025/99","contract_id":"BB-2026/18"}'
    assert values(text) == ['AA-2026/17', 'BB-2026/18']


@pytest.mark.parametrize('value', ['12.10.2026', '2.4.1', '123'])
def test_dates_versions_short_numbers_are_not_contract_identifiers(value):
    text = json.dumps({'contract_id': value, 'other': 'BUILD-2025/99'})
    assert values(text) == []


def test_json_mask_preserves_neighbors_and_restores_source():
    text = '{"contract_id":"AB-2024/17","timeout_ms":120000,"build_id":"BUILD-2025/99"}'
    spans = merge_results(text, [], score_threshold=.35).results
    assert len(spans) == 1 and spans[0].entity_type == 'CONTRACT_NUMBER'
    r = spans[0]
    masked = text[:r.start] + '<CONTRACT_NUMBER_1>' + text[r.end:]
    assert json.loads(masked) == {'contract_id':'<CONTRACT_NUMBER_1>', 'timeout_ms':120000,
                                 'build_id':'BUILD-2025/99'}
    assert masked.replace('<CONTRACT_NUMBER_1>', text[r.start:r.end]).encode() == text.encode()


def test_ner_on_neighbor_is_not_confirmed_by_contract_key():
    text = '{"contract_id":"AB-2024/17","build_id":"BUILD-2025/99"}'
    start = text.index('BUILD-')
    wrong = RecognizerResult('CONTRACT_NUMBER', start, start+len('BUILD-2025/99'), .99,
        recognition_metadata={DETECTION_SOURCE_METADATA_KEY:SOURCE_NER})
    result = merge_results(text, [wrong]).results
    assert [text[r.start:r.end] for r in result] == ['AB-2024/17']


def test_binding_respects_filters_and_threshold():
    text = '{"contract_id":"AB-2024/17","other":120000}'
    assert merge_results(text, [], requested_entities=['PERSON']).results == ()
    assert merge_results(text, [], score_threshold=CONTRACT_CONTEXT_SCORE+.01).results == ()


@pytest.mark.parametrize('text', [
    'contract_id=\nbuild_id=BUILD-2025/99',
    'contract_id:\nother: BUILD-2025/99',
    'contract_id = make_id(120000); timeout_ms=240000',
    'contract_id == 120000',
    '{"Договор 123456"' + ' '*100 + ':null,"timeout_ms":120000}',
    '{"contract":"","timeout_ms":120000}',
    '{"договор":null,"build_id":"BUILD-2025/99"}',
])
def test_second_pass_missing_values_and_natural_key_names(text):
    assert values(text) == []


@pytest.mark.parametrize('text,value', [
    ('Договоры AA-101/26, BB-202/26 связаны.', ['AA-101/26', 'BB-202/26']),
    ('Номер договора из PDF: ДГ-2026/117.', ['ДГ-2026/117']),
    ('Номер договора:\nDG-2026-0042', ['DG-2026-0042']),
    ('contract_id: "DG-2026-0042"', ['DG-2026-0042']),
    ('{"contract_id":12345678,"timeout_ms":120000}', ['12345678']),
    ('{"contract_id":"2026-10-22","timeout_ms":120000}', ['2026-10-22']),
    ('note="contract_id=DG-2026-0042"; other="BUILD-2025/99"', ['DG-2026-0042']),
])
def test_second_pass_preserves_positive_forms_and_numeric_contract(text, value):
    assert values(text) == value


def test_generated_neighbor_fields_are_invariant_to_order_and_nesting():
    import itertools
    fields = [('contract_id', 'DG-2026-0042'), ('build_id', 'BUILD-2025/99'),
              ('timeout_ms', 120000), ('version', '2.4.1')]
    for order in itertools.permutations(fields):
        for obj in (dict(order), {'nested': [dict(order)]}):
            text = json.dumps(obj, ensure_ascii=False)
            assert values(text) == ['DG-2026-0042']


@pytest.mark.parametrize('key', ['номер договора', 'номер контракта', 'contract number', 'agreement no.'])
def test_natural_field_labels_keep_their_own_values(key):
    text = json.dumps({key:'AB-2026/17', 'other':'BUILD-2025/99'}, ensure_ascii=False)
    assert values(text) == ['AB-2026/17']


@pytest.mark.parametrize('label', ['Договоры', 'Договор', 'Contracts'])
def test_prose_label_keeps_a_list_without_confirming_other_assignments(label):
    text = label + ': AA-101/26 и BB-202/26; build_id=BUILD-2025/99'
    assert values(text) == ['AA-101/26', 'BB-202/26']


def test_generated_identifier_key_spellings_keep_their_own_value():
    import itertools
    for noun, plural, sep, suffix, prefix in itertools.product(
        ['contract','agreement'], ['', 's'], ['','_','-','.'],
        ['id','no','nr','num','number','ref'], ['', 'client_', 'client_'+'x_'*150],
    ):
        key = prefix + noun + plural + sep + suffix
        text = json.dumps({key:'DG-2026-0042', 'build_id':'BUILD-2025/99'})
        assert values(text) == ['DG-2026-0042'], key


@pytest.mark.parametrize('policy', ['content','logistic'])
def test_api_binding_does_not_confirm_neighbor_even_with_wrong_ner(monkeypatch, policy):
    from fastapi.testclient import TestClient
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio import analyzer_server
    from recognizers import ALL_RECOGNIZERS

    monkeypatch.setenv('PRESIDIO_ANALYZER_BASE64_POLICY', policy)
    registry = RecognizerRegistry(supported_languages=['ru'])
    for cls in ALL_RECOGNIZERS:
        registry.add_recognizer(cls())
    engine = AnalyzerEngine(registry=registry, nlp_engine=analyzer_server.nlp_engine,
                            supported_languages=['ru'])
    monkeypatch.setattr(analyzer_server, 'analyzer', engine)
    monkeypatch.setattr(analyzer_server.ner_recognizer, 'require_ready', lambda: None)

    def wrong_ner(text, **_kwargs):
        if 'BUILD-2025/99' not in text:
            return []
        start = text.index('BUILD-2025/99')
        return [RecognizerResult('CONTRACT_NUMBER', start, start+len('BUILD-2025/99'), .99,
            recognition_metadata={DETECTION_SOURCE_METADATA_KEY:SOURCE_NER})]

    monkeypatch.setattr(analyzer_server.ner_recognizer, 'analyze', wrong_ner)
    api = TestClient(analyzer_server.app)
    for contract in ['AB-2024/17', None, '', {}, []]:
        text = json.dumps({'contract_id':contract, 'build_id':'BUILD-2025/99', 'timeout_ms':120000})
        response = api.post('/api/v1/analyze', json={'text':text, 'entities':['CONTRACT_NUMBER'],
                                                    'score_threshold':.35})
        assert response.status_code == 200
        entities = response.json()['entities']
        assert [text[r['start']:r['end']] for r in entities] == (
            ['AB-2024/17'] if isinstance(contract,str) and contract else []
        )
        masked = text
        for r in reversed(entities):
            masked = masked[:r['start']]+'<CONTRACT_NUMBER_1>'+masked[r['end']:]
        obj = json.loads(masked)
        assert obj['build_id']=='BUILD-2025/99' and obj['timeout_ms']==120000
        assert api.post('/api/v1/analyze', json={'text':text, 'entities':['CONTRACT_NUMBER'],
            'score_threshold':.9}).json()['entities'] == []
