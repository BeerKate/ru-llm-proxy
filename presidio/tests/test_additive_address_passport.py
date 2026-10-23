"""Additive address chains and passport shapes on fictitious values."""

import pytest

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from presidio.recognizers import ALL_RECOGNIZERS


@pytest.fixture(scope="module")
def analyzer():
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "ru", "model_name": "ru_core_news_sm"}],
        }
    ).create_engine()
    engine = AnalyzerEngine(nlp_engine=nlp_engine)
    for recognizer_cls in ALL_RECOGNIZERS:
        engine.registry.add_recognizer(recognizer_cls())
    return engine


def _covered(analyzer, text, value, entity_type):
    results = analyzer.analyze(text=text, language="ru", score_threshold=0.35)
    covered = bytearray(len(text))
    for result in results:
        if result.entity_type == entity_type:
            covered[result.start : result.end] = b"\x01" * (result.end - result.start)
    start = text.index(value)
    return all(text[i].isspace() or covered[i] for i in range(start, start + len(value)))


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("серия 4510 номер 123456", "4510 номер 123456"),
        ("Паспорт: 45-10 № 123456, выдан ОВД района.", "45-10 № 123456"),
        ("Клиент предъявил паспорт 4510-123456.", "4510-123456"),
        ('passport = "4510123456"', "4510123456"),
        ("Загранпаспорт № 75 № 1234567", "75 № 1234567"),
    ],
)
def test_passport_shapes_are_covered(analyzer, text, value):
    assert _covered(analyzer, text, value, "RU_PASSPORT")


@pytest.mark.parametrize(
    "text",
    [
        "Build 4510-123456 failed.",
        "Range 10-20, offset 4510, limit 123456.",
        "Заказ 2025-123456 оформлен 12.10.2025.",
    ],
)
def test_passport_look_alikes_stay_visible(analyzer, text):
    results = analyzer.analyze(text=text, language="ru", score_threshold=0.35)
    assert [r for r in results if r.entity_type == "RU_PASSPORT"] == []


@pytest.mark.parametrize(
    ("text", "value"),
    [
        (
            "Доставка: г. Тверь, ул. Советская, д. 5, кв. 12",
            "г. Тверь, ул. Советская, д. 5, кв. 12",
        ),
        ("Офис на ул. Академика Королева, 12, корп. 2", "ул. Академика Королева, 12, корп. 2"),
        ("Адрес: Ленина 5-12", "Ленина 5-12"),
        ('{"address": "Тверь, Советская 5"}', "Тверь, Советская 5"),
        ("Москва, Tverskaya st. 17", "Tverskaya st. 17"),
    ],
)
def test_address_chains_are_covered(analyzer, text, value):
    assert _covered(analyzer, text, value, "RU_ADDRESS")


@pytest.mark.parametrize(
    "text",
    [
        "Смотри стр. 5 документа и кв. 3 отчёта.",
        "Раздел 5, пункт 3 — версия 2.",
        'config = {"address": "0.0.0.0", "port": 8080}',
        'deploy:\n  region: "GB"\n  zone: "US"',
        "Адрес вашего сервера укажите в настройках.",
    ],
)
def test_address_look_alikes_stay_visible(analyzer, text):
    results = analyzer.analyze(text=text, language="ru", score_threshold=0.35)
    names = {
        (r.recognition_metadata or {}).get("recognizer_name") for r in results
        if r.entity_type == "RU_ADDRESS"
    }
    assert "RuAddressChainRecognizer" not in names


@pytest.mark.parametrize('ascii_only', [False, True])
@pytest.mark.parametrize('value', ['4510-123456', '45-10 № 123456', '45 10 123 456', '75-1234567'])
def test_decoded_passport_preserves_explicit_context(analyzer, ascii_only, value):
    import json

    text = json.dumps({'text': 'Клиент предъявил паспорт ' + value + '.', 'next': 'ordinary'}, ensure_ascii=ascii_only)
    raw = json.dumps(value, ensure_ascii=ascii_only)[1:-1]
    assert _covered(analyzer, text, raw, 'RU_PASSPORT')


@pytest.mark.parametrize('ascii_only', [False, True])
@pytest.mark.parametrize('text', ['Build 4510-123456 failed.', 'Заказ 4510-123456 оформлен.', 'Version 45 10 123 456'])
def test_decoded_passport_needs_its_own_context(analyzer, ascii_only, text):
    import json

    payload = json.dumps({'note': 'Паспорт клиента', 'text': text}, ensure_ascii=ascii_only)
    findings = analyzer.analyze(text=payload, language='ru', entities=['RU_PASSPORT'], score_threshold=.35)
    assert findings == []


@pytest.mark.parametrize('ascii_only', [False, True])
def test_passport_context_does_not_override_request_threshold(analyzer, ascii_only):
    import json

    payload = json.dumps({'text': 'Паспорт 4510-123456.'}, ensure_ascii=ascii_only)
    findings = analyzer.analyze(text=payload, language='ru', entities=['RU_PASSPORT'], score_threshold=.9)
    assert findings == []


@pytest.mark.parametrize('ascii_only', [False, True])
@pytest.mark.parametrize('key', ['паспорт', 'passport'])
def test_own_passport_key_confirms_only_its_scalar(analyzer, ascii_only, key):
    import json

    payload = json.dumps({key: '4510-123456', 'text': 'Build 4511-654321 failed.'}, ensure_ascii=ascii_only)
    findings = analyzer.analyze(text=payload, language='ru', entities=['RU_PASSPORT'], score_threshold=.35)
    assert findings
    assert all(payload[r.start:r.end] == '4510-123456' for r in findings)


@pytest.mark.parametrize('ascii_only', [None, False, True])
@pytest.mark.parametrize('text', ['В серии 4510-123456 опечатка.', 'Сверьте серию 4510-123456.'])
def test_passport_context_preserves_russian_case_forms(analyzer, ascii_only, text):
    import json

    payload = text if ascii_only is None else json.dumps({'text': text}, ensure_ascii=ascii_only)
    assert _covered(analyzer, payload, '4510-123456', 'RU_PASSPORT')
