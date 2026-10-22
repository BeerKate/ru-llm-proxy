"""Controlled NER boundaries: only proven code symbols may be suppressed."""
import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from presidio_analyzer import RecognizerResult
from result_merging import merge_results, SOURCE_NER, SOURCE_NATIVE_CREDENTIAL, DETECTION_SOURCE_METADATA_KEY


def org(text, part):
    start = text.index(part)
    return RecognizerResult('ORGANIZATION', start, start+len(part), .9,
        recognition_metadata={DETECTION_SOURCE_METADATA_KEY:SOURCE_NER})


@pytest.mark.parametrize('text,part', [
    ('Компания OpenAI получила письмо.', 'Open'),
    ('Договор с компанией TechCorp подписан.', 'Tech'),
    ('NorthWind оказывает услуги.', 'North'),
    ('Компания Studio42 получила запрос.', 'Studio'),
    ('Письмо направлено в user_service.', 'user_serv'),
    ('name: Name for the TensorFlow ops.', 'TensorF'),
    ('"Qwen2ForCausalLM",', 'alLM'),
    ('OpenAI', 'Open'),
    ('`OpenAI`', 'Open'),
    ('Компания: OpenAI', 'Open'),
    ('# Компания OpenAI получила письмо.\nservice = factory()', 'Open'),
    ('"""Компания OpenAI получила письмо."""\nservice = factory()', 'Open'),
    ('company = "OpenAI"', 'Open'),
    ('{"company":"TechCorp"}', 'Tech'),
    ('{"organization":"OpenAI","code":"factory()"}', 'Open'),
    ('token = OpaqueToken42', 'Opaque'),
    ('Компания OpenAI получила письмо.\n```python\nservice = factory()\n```', 'Open'),
])
def test_ambiguous_or_data_organization_fragment_is_retained(text, part):
    result = org(text, part)
    outcome = merge_results(text, [result])
    covered = {i for r in outcome.results for i in range(r.start, r.end)}
    assert set(range(result.start,result.end)) <= covered
    assert not any(d.reason=='ner_identifier_fragment_suppressed' for d in outcome.decisions)


@pytest.mark.parametrize('text,part', [
    ('__all__ = ["UniSpeechSatForCTC"]', 'UniSpeechSatForC'),
    ('service = user_service_factory()', 'service_fact'),
    ('from transformers import Qwen2ForCausalLM', 'alLM'),
    ('import TensorFlow as tf', 'TensorF'),
    ('class TechCorpClient:\n    pass', 'TechCorp'),
    ('def user_service_factory():\n    pass', 'service_fact'),
    ('TechCorpClient = 1', 'TechCorp'),
    ('client = OpenAI.Client()', 'Open'),
    ('Образец:\n```python\nservice = user_service_factory()\n```', 'service_fact'),
    ('сообщение = "обычное"; service = user_service_factory()', 'service_fact'),
])
def test_partial_organization_is_suppressed_only_at_a_code_symbol(text, part):
    outcome = merge_results(text, [org(text,part)])
    assert outcome.results == ()
    assert [d.reason for d in outcome.decisions] == ['ner_identifier_fragment_suppressed']


@pytest.mark.parametrize('kind', ['PASSWORD','SECRET_KEY','AUTH_TOKEN'])
def test_other_detection_preserves_the_organization_piece_of_a_secret(kind):
    text = 'client = OpaqueToken42()'
    model_org = org(text,'Opaque')
    start = text.index('OpaqueToken42')
    native = RecognizerResult(kind,start,start+len('OpaqueToken42'),.95,
        recognition_metadata={DETECTION_SOURCE_METADATA_KEY:SOURCE_NATIVE_CREDENTIAL})
    for order in itertools.permutations([model_org,native]):
        result = merge_results(text,order).results
        assert len(result)==1 and result[0].entity_type==kind
        assert (result[0].start,result[0].end)==(start,start+len('OpaqueToken42'))


def test_whole_organization_in_code_is_not_a_partial_identifier():
    text = 'client = OpenAI()'
    assert merge_results(text,[org(text,'OpenAI')]).results


def test_json_company_keeps_valid_structure_and_exact_restore():
    text = json.dumps({'company':'OpenAI','next':'ordinary'})
    spans = merge_results(text,[org(text,'Open')]).results
    assert spans
    r=spans[0];token='<ORGANIZATION_1>'
    masked=text[:r.start]+token+text[r.end:]
    assert json.loads(masked)['next']=='ordinary'
    assert masked.replace(token,text[r.start:r.end]).encode()==text.encode()


@pytest.mark.parametrize('text', [
    'result = factory(secret=OpaqueToken42).run()',
    'result = factory(OpaqueToken42)()',
    '"""Пример:\n```python\nOpenAI()\n```\n"""',
    'company: OpenAI',
    'class IncompleteTechCorp:',
])
def test_second_pass_uncertain_data_and_incomplete_code_are_retained(text):
    part = 'Opaque' if 'Opaque' in text else ('TechCorp' if 'TechCorp' in text else 'Open')
    result = org(text,part)
    spans = merge_results(text,[result]).results
    assert all(any(r.start <= i < r.end for r in spans)
               for i in range(result.start,result.end))


def test_typed_store_and_async_function_identifiers_are_proven_symbols():
    for text,part in [('TechCorpClient: int = 1','TechCorp'),
                      ('async def TechCorpClient():\n    pass','TechCorp')]:
        assert merge_results(text,[org(text,part)]).results == ()


def test_code_evidence_never_propagates_to_the_same_name_in_a_data_literal():
    text = 'client = OpenAI()\ncompany = "OpenAI"'
    first = org(text,'Open')
    start = text.rindex('Open')
    second = RecognizerResult('ORGANIZATION',start,start+len('Open'),.9,
        recognition_metadata={DETECTION_SOURCE_METADATA_KEY:SOURCE_NER})
    for order in itertools.permutations([first,second]):
        spans = merge_results(text,order).results
        assert len(spans)==1 and (spans[0].start,spans[0].end)==(second.start,second.end)


def test_generated_company_shapes_are_data_and_callable_names_are_symbols():
    cases=[('OpenAI','Open'),('TechCorp','Tech'),('Studio42','Studio'),
           ('NorthWind','North'),('user_service','user_serv')]
    for name,part in cases:
        for text in [f'Компания {name} получила письмо.', json.dumps({'company':name}),
                     f'company = "{name}"', f'# {name}\nclient = factory()',
                     f'def send():\n    return {name}']:
            result=org(text,part)
            spans=merge_results(text,[result]).results
            assert all(any(r.start<=i<r.end for r in spans)
                       for i in range(result.start,result.end)), text
        code=f'client = {name}()'
        assert merge_results(code,[org(code,part)]).results == ()


@pytest.mark.parametrize('source', [SOURCE_NER,SOURCE_NATIVE_CREDENTIAL])
def test_secret_on_another_piece_of_the_same_callable_preserves_partial_org(source):
    text='client = OpaqueToken42()'
    model_org=org(text,'Opaque')
    start=text.index('Token42')
    secret=RecognizerResult('SECRET_KEY',start,start+len('Token42'),.95,
        recognition_metadata={DETECTION_SOURCE_METADATA_KEY:source})
    for order in itertools.permutations([model_org,secret]):
        spans=merge_results(text,order).results
        assert all(any(r.start<=i<r.end for r in spans)
                   for i in range(text.index('Opaque'),start+len('Token42')))


@pytest.mark.parametrize('newline', ['\n','\r\n'])
def test_utf8_ast_columns_after_emoji_map_to_source_characters(newline):
    text='сообщение = "😀"; client = OpenAI()'+newline
    assert merge_results(text,[org(text,'Open')]).results == ()


@pytest.mark.parametrize('policy', ['content','logistic'])
def test_api_keeps_data_fragment_but_suppresses_same_callable_name(monkeypatch,policy):
    from fastapi.testclient import TestClient
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio import analyzer_server
    from recognizers import ALL_RECOGNIZERS

    monkeypatch.setenv('PRESIDIO_ANALYZER_BASE64_POLICY',policy)
    registry=RecognizerRegistry(supported_languages=['ru'])
    for cls in ALL_RECOGNIZERS:
        registry.add_recognizer(cls())
    engine=AnalyzerEngine(registry=registry,nlp_engine=analyzer_server.nlp_engine,
                          supported_languages=['ru'])
    monkeypatch.setattr(analyzer_server,'analyzer',engine)
    monkeypatch.setattr(analyzer_server.ner_recognizer,'require_ready',lambda:None)
    text='client = OpenAI()\ncompany = "OpenAI"'
    first=org(text,'Open');start=text.rindex('Open')
    second=RecognizerResult('ORGANIZATION',start,start+4,.9,
        recognition_metadata={DETECTION_SOURCE_METADATA_KEY:SOURCE_NER})
    monkeypatch.setattr(analyzer_server.ner_recognizer,'analyze',lambda *_a,**_k:[first,second])
    response=TestClient(analyzer_server.app).post('/api/v1/analyze',json={
        'text':text,'entities':['ORGANIZATION'],'score_threshold':.35})
    assert response.status_code==200
    assert [(r['start'],r['end']) for r in response.json()['entities']]==[(start,start+4)]


def test_unpaired_surrogate_cannot_turn_a_read_only_heuristic_into_an_error():
    text='Компания OpenAI получила письмо. \ud800'
    result=org(text,'Open')
    assert merge_results(text,[result]).results


def test_non_line_separator_in_a_unicode_python_literal_preserves_offsets():
    text='сообщение = "a\v"; client = OpenAI()'
    assert merge_results(text,[org(text,'Open')]).results == ()


def test_bare_cr_source_lines_have_correct_symbol_positions():
    text='сообщение = "😀"\rclient = OpenAI()\r'
    assert merge_results(text,[org(text,'Open')]).results == ()


@pytest.mark.parametrize('text,part', [('OpenAI: поставщик','Open'),('TechCorp: vendor','Tech')])
def test_prose_name_label_is_not_proven_code_despite_valid_annotation_syntax(text,part):
    # Python accepts bare annotations, but these may be ordinary glossary labels.
    assert merge_results(text,[org(text,part)]).results
