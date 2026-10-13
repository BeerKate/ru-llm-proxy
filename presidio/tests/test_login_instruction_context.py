"""Only unambiguously instructional NER login spans may be suppressed."""

import itertools

import pytest
from presidio_analyzer import RecognizerResult

from result_merging import (
    DETECTION_SOURCE_METADATA_KEY, SOURCE_NATIVE_CREDENTIAL, SOURCE_NER,
    SOURCE_STRUCTURAL, merge_results,
)


def entity(text, value, entity_type="LOGIN", source=SOURCE_NER):
    start = text.index(value)
    return RecognizerResult(
        entity_type, start, start + len(value), 0.8,
        recognition_metadata={DETECTION_SOURCE_METADATA_KEY: source},
    )


@pytest.mark.parametrize("command", ["Reply", "Respond", "Answer", "Return", "Print", "Output", "Copy"])
@pytest.mark.parametrize("prefix", ["", "Contact recorded. ", "Contact recorded! ", "Contact recorded? ", "Contact recorded; ", "Contact recorded.\n", "\n", "\r\n", "👩‍💻 Contact recorded. "])
@pytest.mark.parametrize("requested", [None, ["LOGIN"], ["LOGIN", "EMAIL_ADDRESS"]])
def test_instruction_context_suppresses_only_exact_ner_command(command, prefix, requested):
    text = prefix + command + " only with that email."
    outcome = merge_results(text, [entity(text, command)], requested_entities=requested)
    assert not outcome.results
    assert [d.reason for d in outcome.decisions] == ["login_instruction_suppressed"]


@pytest.mark.parametrize("command", ["reply", "REPLY", "Reply"])
@pytest.mark.parametrize("spaces", [" ", "\t", "\u00a0"])
def test_polite_case_and_horizontal_spacing_variants(command, spaces):
    text = "Please" + spaces + command + spaces + "only with the email address."
    assert not merge_results(text, [entity(text, command)]).results


@pytest.mark.parametrize("text", [
    "login=Reply only with that email.",
    "username: Reply only with that email.",
    "username:\nReply only with that email.",
    "login= \r\nReply only with that email.",
    "Логин:\nReply only with that email.",
    'login="Reply only with that email"',
    '{"username": "Reply only with that email"}',
    "The username is Reply only with that email.",
    "User Reply only with that email.",
    "account.Reply only with that email.",
    "Reply",
    "Reply is an account name.",
    "Reply to the customer.",
    "Reply with that email.",
    "Reply only with username.",
    "Reply_01 only with that email.",
    "Reply only with " + " " * 17 + "that email.",
    "Reply only\nwith that email.",
    '"Reply only with that email."',
])
def test_ambiguous_or_credential_context_is_unchanged(text):
    result = entity(text, "Reply")
    assert merge_results(text, [result]).results == (result,)


@pytest.mark.parametrize("source", [SOURCE_NATIVE_CREDENTIAL, SOURCE_STRUCTURAL])
@pytest.mark.parametrize("bounds", ["exact", "wider", "partial"])
def test_independent_detection_vetoes_suppression_regardless_of_order(source, bounds):
    text = "Reply only with that email."
    ner = entity(text, "Reply")
    value = {"exact": "Reply", "wider": "Reply only", "partial": "ply"}[bounds]
    independent = entity(text, value, source=source)
    for results in itertools.permutations([ner, independent]):
        outcome = merge_results(text, results)
        assert outcome.results
        assert "login_instruction_suppressed" not in [d.reason for d in outcome.decisions]


@pytest.mark.parametrize("value,entity_type", [("Reply only", "LOGIN"), ("Rep", "LOGIN"), ("Reply", "PASSWORD"), ("Reply", "PERSON")])
def test_wider_partial_or_other_type_ner_is_unchanged(value, entity_type):
    text = "Reply only with that email."
    result = entity(text, value, entity_type)
    assert merge_results(text, [result]).results == (result,)


def test_long_context_and_repeated_values_keep_real_login_and_email():
    text = "ordinary context\n" * 8000 + "login=Reply. Email: sample@example.test. Reply only with that email."
    login = entity(text, "Reply", source=SOURCE_NATIVE_CREDENTIAL)
    email = entity(text, "sample@example.test", "EMAIL_ADDRESS", SOURCE_STRUCTURAL)
    start = text.rindex("Reply")
    command = RecognizerResult("LOGIN", start, start + 5, 0.8,
                               recognition_metadata={DETECTION_SOURCE_METADATA_KEY: SOURCE_NER})
    for results in itertools.permutations([login, email, command]):
        outcome = merge_results(text, results)
        assert [(r.entity_type, text[r.start:r.end]) for r in outcome.results] == [
            ("LOGIN", "Reply"), ("EMAIL_ADDRESS", "sample@example.test")
        ]


@pytest.mark.parametrize("boundary,retained", [
    (". ", ""), (".\n", "."), ("! ", ""), ("; ", ""), ("\r\n", ""),
])
@pytest.mark.parametrize("requested", [None, ["LOGIN"]])
def test_wider_ner_span_retains_sensitive_prefix(boundary, retained, requested):
    value = "sample@example.test"
    text = value + boundary + "reply only with that email."
    wide = entity(text, value + boundary + "reply")
    outcome = merge_results(text, [wide], requested_entities=requested)
    assert [(r.entity_type, text[r.start:r.end], r.score) for r in outcome.results] == [
        ("LOGIN", value + retained, 0.8)
    ]
    assert outcome.results[0].recognition_metadata == wide.recognition_metadata
    assert wide.end == text.index("reply") + len("reply")
    assert [d.reason for d in outcome.decisions] == ["login_instruction_tail"]


def test_independently_confirmed_tail_does_not_trim_wider_ner():
    text = "sample@example.test. Reply only with that email."
    wide = entity(text, "sample@example.test. Reply")
    native = entity(text, "Reply", source=SOURCE_NATIVE_CREDENTIAL)
    outcome = merge_results(text, [wide, native])
    assert "login_instruction_tail" not in [d.reason for d in outcome.decisions]
    assert any(r.start == native.start and r.end == native.end for r in outcome.results)


@pytest.mark.parametrize("value", ["runner.", "runner!", "runner?", "runner;"])
@pytest.mark.parametrize("boundary,retained", [
    (". ", ""), (".\n", "."), ("\r\n", ""),
])
def test_trim_keeps_punctuation_inside_sensitive_prefix(value, boundary, retained):
    text = value + boundary + "Reply only with that email."
    wide = entity(text, value + boundary + "Reply")
    outcome = merge_results(text, [wide])
    assert [(r.entity_type, text[r.start:r.end]) for r in outcome.results] == [
        ("LOGIN", value + retained)
    ]
