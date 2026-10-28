"""Unit tests for reversible dictionary substitution policy."""

import json
from pathlib import Path

import pytest

from litellm_guardrails.dictionary_policy import (
    DictionaryPolicyAmbiguousRequestError,
    DictionaryPolicyConfigError,
    DictionarySubstitutionPolicy,
)


def _policy(*rules):
    return DictionarySubstitutionPolicy.from_config({"substitutions": list(rules)})


def test_case_insensitive_russian_phrase_matching_and_restore_mapping():
    policy = _policy(
        {
            "id": "tbank_to_zetta",
            "source": "Т-Банк",
            "replacement": "Зетта Групп",
            "match": {"case_sensitive": False, "whole_phrase": True},
            "restore": True,
        }
    )

    result = policy.apply("Проверь т-банк по договору")

    assert result.text == "Проверь Зетта Групп по договору"
    assert result.mapping == {"Зетта Групп": "т-банк"}
    assert result.rule_counts == {"tbank_to_zetta": 1}


def test_longest_match_first_keeps_overlapping_rules_deterministic():
    policy = _policy(
        {
            "id": "short",
            "source": "Т-Банк",
            "replacement": "Зетта Групп",
        },
        {
            "id": "long",
            "source": "Т-Банк Инвестиции",
            "replacement": "Зетта Инвест",
        },
    )

    result = policy.apply("Открой Т-Банк Инвестиции")

    assert result.text == "Открой Зетта Инвест"
    assert result.mapping == {"Зетта Инвест": "Т-Банк Инвестиции"}
    assert result.rule_counts == {"long": 1}


def test_duplicate_ids_are_rejected():
    with pytest.raises(DictionaryPolicyConfigError, match="duplicate"):
        _policy(
            {"id": "same", "source": "Сбербанк", "replacement": "Северный Траст"},
            {"id": "same", "source": "ВТБ", "replacement": "Восточный Контур"},
        )


def test_duplicate_restore_replacements_are_rejected():
    with pytest.raises(DictionaryPolicyConfigError, match="duplicate restore"):
        _policy(
            {"id": "one", "source": "Сбербанк", "replacement": "Северный Траст"},
            {"id": "two", "source": "ВТБ", "replacement": "Северный Траст"},
        )


def test_original_replacement_text_in_request_is_rejected_as_ambiguous():
    policy = _policy(
        {
            "id": "tbank_to_zetta",
            "source": "Т-Банк",
            "replacement": "Зетта Групп",
        }
    )

    with pytest.raises(DictionaryPolicyAmbiguousRequestError):
        policy.apply("Сравни Т-Банк и Зетта Групп")


def test_default_dictionary_contains_bank_rules_and_reported_code_variants():
    config_path = (
        Path(__file__).resolve().parents[1] / "dictionary-substitutions.default.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rules = config["substitutions"]

    assert len(rules) == 17
    assert all(rule["enabled"] is True for rule in rules)
    assert all(rule["restore"] is True for rule in rules)
    assert {rule["source"] for rule in rules} == {
        "Сбербанк",
        "ВТБ",
        "Газпромбанк",
        "Альфа-Банк",
        "ru.alfabank.android",
        "@alfa-bank/fms-types",
        "Alfa-Bank",
        "alfabank",
        "AlfaBank",
        "ALFABANK",
        "alfaBank",
        "ПСБ",
        "Россельхозбанк",
        "Т-Банк",
        "Московский кредитный банк",
        "Банк Дом.РФ",
        "Совкомбанк",
    }
    assert len({rule["replacement"].casefold() for rule in rules}) == 17

    policy = DictionarySubstitutionPolicy.from_config(config)
    assert len(policy.rules) == 17


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ru.alfabank.mobile.android.R", "ru.ladogafinance1.mobile.android.R"),
        ("ru/alfabank/mobile/Foo", "ru/ladogafinance1/mobile/Foo"),
        ("https://invest.alfabank.ru/path", "https://invest.ladogafinance1.ru/path"),
        ("alfabank://invest/open", "ladogafinance1://invest/open"),
        ("ALFAINVEST_ALFABANK_RU", "ALFAINVEST_LADOGAFINANCE3_RU"),
        ("skipAuthByAlfaBankApp", "skipAuthByLadogaFinance2App"),
        ("isAlfaBankTransfer", "isLadogaFinance2Transfer"),
        ("alfaBankApp", "ladogaFinance4App"),
        ("ru.alfabank.android\n", "ru.ladogafinance.android\n"),
        (
            "import ru.alfabank.android.core.Foo",
            "import ru.ladogafinance.android.core.Foo",
        ),
        (
            '{"devDependencies":{"@alfa-bank/fms-types":"^7.19.0"}}',
            '{"devDependencies":{"@ladoga-finance/fms-types":"^7.19.0"}}',
        ),
        (
            "// Copyright © 2016 Alfa-Bank. All rights reserved.",
            "// Copyright © 2016 Ladoga Finance. All rights reserved.",
        ),
    ],
)
def test_reported_bank_variants_are_masked_and_restore_exactly(text, expected):
    policy = DictionarySubstitutionPolicy.from_file(
        Path(__file__).resolve().parents[1] / "dictionary-substitutions.default.json"
    )
    result = policy.apply(text)
    assert result.text == expected
    assert result.substitution_count == 1
    restored = result.text
    for replacement, original in result.mapping.items():
        restored = restored.replace(replacement, original)
    assert restored == text
    if text.startswith("{"):
        assert json.loads(result.text) == {
            "devDependencies": {"@ladoga-finance/fms-types": "^7.19.0"}
        }


def test_reported_bank_variants_have_distinct_restore_targets_in_one_text():
    policy = DictionarySubstitutionPolicy.from_file(
        Path(__file__).resolve().parents[1] / "dictionary-substitutions.default.json"
    )
    text = "Альфа-Банк; Alfa-Bank; @alfa-bank/fms-types; ru.alfabank.android.core.Foo"
    result = policy.apply(text)
    assert result.substitution_count == 4
    assert len(result.mapping) == 4
    restored = result.text
    for replacement, original in result.mapping.items():
        restored = restored.replace(replacement, original)
    assert restored == text


@pytest.mark.parametrize(
    "text",
    [
        "ru.alfa.android ru.purpur.android mobile.common",
        "alphabet bank Alfa Bank ALFA_BANK",
        "@alfa-bank/other-package",
        "NotAlfa-Bank Alfa-Banker",
    ],
)
def test_reported_bank_rules_do_not_expand_to_unrelated_identifiers(text):
    policy = DictionarySubstitutionPolicy.from_file(
        Path(__file__).resolve().parents[1] / "dictionary-substitutions.default.json"
    )
    result = policy.apply(text)
    assert result.text == text
    assert result.mapping == {}
