"""Synthetic physical-address evidence and configuration counterexamples."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recognizers.ru_address_chain import RuAddressChainRecognizer
from recognizers import ALL_RECOGNIZERS
from result_merging import merge_results


def direct(text):
    return RuAddressChainRecognizer().analyze(text,['RU_ADDRESS'])


@pytest.mark.parametrize('key',['region','address','city','street','district'])
@pytest.mark.parametrize('value',['Sidebar','Auto','Default','MainPanel','Buffer42','Рабочая область'])
def test_arbitrary_configuration_value_is_not_an_address(key,value):
    assert direct(json.dumps({key:value},ensure_ascii=False)) == []


@pytest.mark.parametrize('text',[
    'layout = {"region":"Sidebar"}', 'region = "Auto"',
    'address = "HttpClient"', '{"address":"0.0.0.0","port":8080}',
    '{"address":"::1"}', '{"address":"https://example.org"}',
    '{"address":"mail@example.org"}', '{"region":"GB","city":"US"}',
    '{"city":"Sidebar","house":"5"}',
    '{"street":"Советская","neighbor":{"house":"5"}}',
    '[{"street":"Советская"},{"house":"5"}]',
    '{"note":"house: 5","street":"Советская","city":"Sidebar"}',
])
def test_no_network_address_or_cross_container_physical_context(text):
    assert direct(text) == []


@pytest.mark.parametrize('value',[
    'г. Тверь, ул. Советская, д. 5, кв. 12',
    'Тверь, Советская 5', 'Ленина 5-12',
    'ул. Ленина, д. 5, кв. "12"',
    'г. тверь, ул. советская, д. 5',
    'Tver, Lenina st., 17, apt. 12',
])
@pytest.mark.parametrize('ascii_only',[False,True])
def test_real_full_address_key_is_covered_and_json_restores(value,ascii_only):
    text=json.dumps({'address':value,'next':'ordinary'},ensure_ascii=ascii_only)
    findings=[r for cls in ALL_RECOGNIZERS for r in cls().analyze(text,None,None)]
    spans=merge_results(text,findings,score_threshold=.35).results
    start=text.index('"',text.index(':'))+1
    end=text.index(', "next"')-1
    covered={i for r in spans for i in range(r.start,r.end)}
    assert all(i in covered for i in range(start,end))
    masked=text;saved={}
    for i,r in reversed(list(enumerate(spans))):
        token=f'<MASK_{i}>';saved[token]=text[r.start:r.end]
        masked=masked[:r.start]+token+masked[r.end:]
    assert json.loads(masked)['next']=='ordinary'
    for token,raw in saved.items():masked=masked.replace(token,raw)
    assert masked.encode()==text.encode()


@pytest.mark.parametrize('obj',[
    {'city':'Тверь','street':'ул. Советская','house':'5','flat':'12'},
    {'region':'Тверская область','city':'Тверь','street':'Советская','building':'5, корп. 2','apartment':'12'},
    {'город':'Тверь','улица':'Советская','дом':'5','квартира':'12'},
])
@pytest.mark.parametrize('ascii_only',[False,True])
def test_split_address_fields_in_the_same_object_are_preserved(obj,ascii_only):
    text=json.dumps(obj,ensure_ascii=ascii_only)
    spans=direct(text)
    covered={i for r in spans for i in range(r.start,r.end)}
    for value in obj.values():
        raw=json.dumps(value,ensure_ascii=ascii_only)[1:-1]
        start=text.index(raw)
        assert all(i in covered for i in range(start,start+len(raw))), value


def test_physical_context_does_not_promote_an_automatic_region_choice():
    text=json.dumps({'city':'Тверь','street':'ул. Ленина','house':'5','region':'Auto'},ensure_ascii=False)
    spans=direct(text);start=text.index('Auto')
    assert not any(r.start<=start<r.end for r in spans)


@pytest.mark.parametrize('text',[
    '{"address":null,"name":"Релиз 5"}',
    '{"address":"Sidebar","name":"Меню 5"}',
    'address=null; name="Релиз 5"',
    'address=null\nname=Релиз 5',
    'address=null Релиз 5',
])
def test_informal_context_cannot_leak_from_address_key(text):
    assert direct(text)==[]


def test_python_dictionary_components_are_kept_in_one_object():
    text="customer = {'city':'ТВЕРЬ','street':'СОВЕТСКАЯ','house':'5','flat':'12'}"
    spans=direct(text)
    for value in ['ТВЕРЬ','СОВЕТСКАЯ','5','12']:
        start=text.index(value)
        assert all(any(r.start<=i<r.end for r in spans) for i in range(start,start+len(value)))


def test_numeric_house_can_confirm_string_components_without_new_scalar_masking():
    text='{"city":"Тверь","street":"Советская","house":5,"region":"Auto"}'
    spans=direct(text)
    for value in ['Тверь','Советская']:
        start=text.index(value)
        assert any(r.start<=start<r.end for r in spans)
    assert not any(r.start<=text.index('Auto')<r.end for r in spans)
    assert not any(r.start<=text.index('5')<r.end for r in spans)


@pytest.mark.parametrize('text',[
    "assert normalize_address('Мичурина 31-43') is not None",
    'logger.info("delivery addr=%s", "Мичурина 31-43")',
    'logger.info("address=%s, id=%s", "Мичурина 31-43", 5)',
])
def test_explicit_address_argument_context_is_preserved(text):
    start=text.index('Мичурина 31-43');end=start+len('Мичурина 31-43')
    spans=direct(text)
    assert all(any(r.start<=i<r.end for r in spans) for i in range(start,end))


@pytest.mark.parametrize('text',[
    '["address=%s","Релиз 5"]',
    '{"message":"address=%s","name":"Релиз 5"}',
    'logger.info("id=%d; address=%s", "Релиз 5", "Sidebar")',
    'logger.info("address=%%s", "Релиз 5")',
])
def test_format_context_needs_actual_first_address_slot(text):
    assert direct(text)==[]


@pytest.mark.parametrize('value',[
    'микрорайон Сосновый, 155, г.Заречье',
    '123456 Заречье Академика Иванова пр. дом 83к1, корп. 1',
    'г Заречье ш. Сосновая д 96 к2',
    'г.Заречье наб. Сосновая д.154Б',
    'ул. Д\'Артаньяна, дом 5',
])
def test_declared_address_keeps_physical_components_not_joined_by_chain(value):
    text=json.dumps({'address':value},ensure_ascii=False)
    raw=json.dumps(value,ensure_ascii=False)[1:-1];start=text.index(raw)
    assert any(r.start<=start and r.end>=start+len(raw) for r in direct(text))


@pytest.mark.parametrize('text',[
    "INSERT INTO clients (fio, address) VALUES ('Тестовый Автор', 'Сосновая 96-276');",
    "INSERT INTO clients (city, street, house) VALUES ('Заречье','Сосновая','5');",
])
def test_sql_literals_are_bound_to_their_own_address_columns(text):
    spans=direct(text)
    assert spans
    assert not any(r.start<=text.find('Тестовый Автор')<r.end for r in spans)
    if '96-276' in text:
        start=text.index('Сосновая 96-276')
        assert any(r.start<=start and r.end>=start+len('Сосновая 96-276') for r in spans)


@pytest.mark.parametrize('text',[
    "INSERT INTO clients (address, note) VALUES (NULL, 'Релиз 5');",
    "INSERT INTO clients (street, house) VALUES ('Сосновая',NULL),(NULL,'5');",
    "INSERT INTO clients (address, note) VALUES (get_addr(), 'Релиз 5');",
])
def test_sql_context_never_crosses_a_column_row_or_expression(text):
    assert direct(text)==[]


@pytest.mark.parametrize('empty',['null','None','true','false'])
def test_placeholder_string_cannot_create_physical_address_group(empty):
    assert direct(json.dumps({'street':empty,'house':'5','city':'Sidebar'}))==[]


def test_street_and_apartment_strings_form_one_physical_group():
    text='{"street":"Сосновая","flat":"12"}'
    spans=direct(text)
    assert any(text[r.start:r.end]=='Сосновая' for r in spans)
    assert any(text[r.start:r.end]=='12' for r in spans)


@pytest.mark.parametrize('ascii_only',[False,True])
def test_quoted_apartment_in_split_fields_keeps_escape_and_restore(ascii_only):
    text=json.dumps({'street':'Сосновая','house':'5','flat':'"12"','next':'ordinary'},ensure_ascii=ascii_only)
    spans=direct(text);raw=r'\"12\"';start=text.index(raw)
    assert any(r.start<=start and r.end>=start+len(raw) for r in spans)
    masked=text
    for i,r in reversed(list(enumerate(spans))):
        masked=masked[:r.start]+f'<MASK_{i}>'+masked[r.end:]
    assert json.loads(masked)['next']=='ordinary'


def test_explicit_settlement_and_house_fields_form_physical_group():
    text='{"city":"г. Заречье","house":"5"}'
    assert {text[r.start:r.end] for r in direct(text)}=={'г. Заречье','5'}


def test_generated_field_order_and_nested_scopes_do_not_change_policy():
    import itertools
    fields=[('city','Тверь'),('street','Сосновая'),('house','5'),('region','Auto')]
    for order in itertools.permutations(fields):
        for obj in (dict(order), {'nested':[dict(order)]}):
            text=json.dumps(obj,ensure_ascii=False);spans=direct(text)
            for value in ['Тверь','Сосновая','5']:
                start=text.index(value)
                assert all(any(r.start<=i<r.end for r in spans) for i in range(start,start+len(value)))
            assert not any(r.start<=text.index('Auto')<r.end for r in spans)


@pytest.mark.parametrize('policy',['content','logistic'])
def test_api_address_keys_need_physical_evidence(monkeypatch,policy):
    from fastapi.testclient import TestClient
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio import analyzer_server

    monkeypatch.setenv('PRESIDIO_ANALYZER_BASE64_POLICY',policy)
    registry=RecognizerRegistry(supported_languages=['ru'])
    for cls in ALL_RECOGNIZERS:registry.add_recognizer(cls())
    engine=AnalyzerEngine(registry=registry,nlp_engine=analyzer_server.nlp_engine,supported_languages=['ru'])
    monkeypatch.setattr(analyzer_server,'analyzer',engine)
    monkeypatch.setattr(analyzer_server.ner_recognizer,'require_ready',lambda:None)
    monkeypatch.setattr(analyzer_server.ner_recognizer,'analyze',lambda *_a,**_k:[])
    api=TestClient(analyzer_server.app)
    for obj in [{'region':'Sidebar'},{'region':'Auto'},{'address':'Buffer42'},
                {'address':'Тверь, Советская 5','next':'ordinary'},
                {'city':'Тверь','street':'Сосновая','house':'5','region':'Auto'}]:
        text=json.dumps(obj,ensure_ascii=False)
        response=api.post('/api/v1/analyze',json={'text':text,'entities':['RU_ADDRESS'],'score_threshold':.35})
        assert response.status_code==200
        entities=response.json()['entities']
        if obj.get('region') in {'Sidebar','Auto'} and 'street' not in obj:
            assert entities==[]
        elif obj.get('address')=='Buffer42':assert entities==[]
        else:assert entities
        masked=text;saved={}
        for i,r in reversed(list(enumerate(entities))):
            token=f'<RU_ADDRESS_{i}>';saved[token]=text[r['start']:r['end']]
            masked=masked[:r['start']]+token+masked[r['end']:]
        decoded=json.loads(masked)
        if 'region' in obj:assert decoded['region']==obj['region']
        for token,raw in saved.items():masked=masked.replace(token,raw)
        assert masked.encode()==text.encode()
        assert api.post('/api/v1/analyze',json={'text':text,'entities':['RU_PASSPORT']}).json()['entities']==[]


@pytest.mark.parametrize('key,value',[('city','Тверь'),('город','НИЖНИЙ НОВГОРОД'),
    ('region','Тверская область'),('регион','Республика Карелия')])
@pytest.mark.parametrize('ascii_only',[False,True])
def test_explicit_geographic_components_keep_original_json_spelling(key,value,ascii_only):
    text=json.dumps({key:value,'next':'ordinary'},ensure_ascii=ascii_only)
    raw=json.dumps(value,ensure_ascii=ascii_only)[1:-1];start=text.index(raw)
    assert any(r.start<=start and r.end>=start+len(raw) for r in direct(text))


def test_decoded_city_field_keeps_the_existing_length_bound():
    assert direct(json.dumps({'city':'Т'+'в'*200},ensure_ascii=True)) == []
