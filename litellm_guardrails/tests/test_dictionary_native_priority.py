"""Dictionary substitution must not suppress complete native findings."""

import copy
import json
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
import pytest_asyncio

from litellm_guardrails.pii_guardrail import (
    ProxyException,
    RuPIIGuardrail,
    close_guardrail_dependency_clients,
)
from litellm_guardrails.tests.test_pii_guardrail import _MemoryRedis
from litellm_guardrails.streaming_restoration import _StreamingPlaceholderReplacer
from litellm_guardrails.tests.test_responses_history import complete, harness


@pytest_asyncio.fixture(autouse=True)
async def close_clients():
    yield
    await close_guardrail_dependency_clients()


def make_guardrail(mode="mask"):
    rules = [
        ("alfabank", "ladogafinance1"),
        ("AlfaBank", "LadogaFinance2"),
        ("ALFABANK", "LADOGAFINANCE3"),
        ("alfaBank", "ladogaFinance4"),
        ("ru.alfabank.android", "ru.ladogafinance.android"),
    ]
    g = RuPIIGuardrail(
        pii_mode=mode,
        dictionary_substitutions_enabled=True,
        dictionary_substitutions_json=json.dumps(
            {"substitutions": [
                {"id": f"rule_{i}", "source": source, "replacement": replacement,
                 "match": {"case_sensitive": True, "whole_phrase": False}}
                for i, (source, replacement) in enumerate(rules)
            ]}
        ),
    )
    g._redis = AsyncMock()
    g._redis.get.return_value = None
    return g


def finding(text, value, kind):
    start = text.index(value)
    return {"start": start, "end": start + len(value), "entity_type": kind, "score": 1.0}


def request(text, shape):
    if shape in ("system", "user", "tool"):
        message = {"role": shape, "content": text}
        if shape == "tool":
            message["tool_call_id"] = "read-1"
        return {"messages": [message]}, lambda d: d["messages"][0]["content"]
    if shape == "instructions":
        return {"instructions": text, "input": "ok"}, lambda d: d["instructions"]
    if shape == "input":
        return {"input": text}, lambda d: d["input"]
    if shape == "function-output":
        return {"input": [{"type": "function_call_output", "call_id": "read-1",
                           "output": text}]}, lambda d: d["input"][0]["output"]
    return {"messages": [{"role": "assistant", "tool_calls": [
        {"id": "write-1", "type": "function", "function": {
            "name": "write_file", "arguments": json.dumps({"text": text})}}
    ]}]}, lambda d: json.loads(d["messages"][0]["tool_calls"][0]["function"]["arguments"])["text"]


NATIVE_CASES = [
    ("PASSWORD", "xalfabankPassword7!"),
    ("EMAIL_ADDRESS", "alice@alfabank.ru"),
    ("INTERNAL_HOST", "internal.alfabank.ru"),
    ("AUTH_TOKEN", "token_alfabank_A19Z7"),
    ("LOGIN", "AlfaBank"),
    ("PERSON", "alfaBank"),
    ("ORGANIZATION", "Example ALFABANK Company"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["system", "user", "tool", "instructions", "input", "function-output", "arguments"])
@pytest.mark.parametrize("kind,value", NATIVE_CASES)
async def test_native_findings_win_and_response_restores_original(shape, kind, value):
    g = make_guardrail()
    text = "Value: " + value
    provider_value = g.dictionary_substitution_policy.apply(value).text

    async def analyze(content):
        return [finding(content, provider_value, kind)] if provider_value in content else []

    data, get = request(text, shape)
    with patch.object(g, "_analyze_text", side_effect=analyze):
        result = await g.async_pre_call_hook(MagicMock(), MagicMock(), data)
    masked = get(result)
    assert masked == f"Value: <{kind}_1>"
    mapping = json.loads(g._redis.setex.call_args[0][2])
    assert mapping[f"<{kind}_1>"] == value
    g._redis.get.return_value = json.dumps(mapping)
    response = litellm.ModelResponse(choices=[litellm.Choices(
        index=0, message=litellm.Message(role="assistant", content=masked), finish_reason="stop"
    )])
    await g.async_post_call_success_hook(result, MagicMock(), response)
    assert response.choices[0].message.content == text


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["user", "instructions", "function-output", "arguments"])
@pytest.mark.parametrize("kind,value", NATIVE_CASES)
async def test_block_mode_rejects_native_findings_without_mutating(shape, kind, value):
    g = make_guardrail("block")
    text = "Value: " + value
    provider_value = g.dictionary_substitution_policy.apply(value).text
    data, _ = request(text, shape)
    before = copy.deepcopy(data)

    async def analyze(content):
        return [finding(content, provider_value, kind)] if provider_value in content else []

    with patch.object(g, "_analyze_text", side_effect=analyze):
        with pytest.raises(litellm.UnprocessableEntityError):
            await g.async_pre_call_hook(MagicMock(), MagicMock(), data)
    assert data == before
    g._redis.setex.assert_not_awaited()


def test_partial_dictionary_overlap_expands_and_merges_to_protect_every_tail():
    g = make_guardrail()
    text = "alfaBank012345"
    result = g.dictionary_substitution_policy.apply(text)
    entities = [
        finding(result.text, "ladoga", "PASSWORD"),
        finding(result.text, "Finance4012345", "AUTH_TOKEN"),
    ]
    masked, mapping = g._mask_text(result.text, entities, exclusion_spans=result.replacement_spans)
    assert masked == "<AUTH_TOKEN_1>"
    assert mapping["<AUTH_TOKEN_1>"] == result.text


def test_adjacent_native_entities_remain_separate():
    g = make_guardrail()
    r = g.dictionary_substitution_policy.apply("alfaBank next")
    entities = [finding(r.text, "ladogaFinance4", "LOGIN"), finding(r.text, "next", "PASSWORD")]
    masked, mapping = g._mask_text(r.text, entities, exclusion_spans=r.replacement_spans)
    assert masked == "<LOGIN_1> <PASSWORD_1>"
    assert len(mapping) == 2


@pytest.mark.asyncio
async def test_alias_organization_is_preserved_but_native_password_is_not_exempt():
    g = make_guardrail()
    text = "Bank alfabank; password xAlfaBank7!"
    result = g.dictionary_substitution_policy.apply(text)
    entities = [finding(result.text, "ladogafinance1", "ORGANIZATION"),
                finding(result.text, "xLadogaFinance27!", "PASSWORD")]
    data = {"messages": [{"role": "user", "content": text}]}
    with patch.object(g, "_analyze_text", return_value=entities):
        output = await g.async_pre_call_hook(MagicMock(), MagicMock(), data)
    assert output["messages"][0]["content"] == "Bank ladogafinance1; password <PASSWORD_1>"
    mapping = json.loads(g._redis.setex.call_args[0][2])
    assert g._replace_placeholders(output["messages"][0]["content"], mapping) == text


def test_restoration_is_atomic_and_matches_stream_for_originals_containing_aliases():
    mapping = {"<PASSWORD_1>": "secret ladogafinance1", "ladogafinance1": "alfabank"}
    text = "<PASSWORD_1> / ladogafinance1"
    expected = "secret ladogafinance1 / alfabank"
    assert RuPIIGuardrail._replace_placeholders(text, mapping) == expected
    for step in (1, 2, 7, len(text)):
        replacer = _StreamingPlaceholderReplacer(mapping)
        restored = "".join(replacer.push((0, "content"), text[i:i+step])
                           for i in range(0, len(text), step))
        restored += replacer.flush((0, "content"))
        assert restored == expected


@pytest.mark.parametrize("mapping,text", [
    ({"aba": "AlfaBank"}, "aba"),
    ({"aba": "AlfaBank"}, "abaaba"),
    ({"Ladoga": "AlfaBank", "ad": "Something"}, "Ladoga"),
    ({"ab": "short", "abc": "long"}, "abc ab abc"),
])
def test_stream_pending_suffix_never_cuts_a_complete_alias(mapping, text):
    expected = RuPIIGuardrail._replace_placeholders(text, mapping)
    for step in range(1, len(text) + 1):
        replacer = _StreamingPlaceholderReplacer(mapping)
        restored = "".join(replacer.push((0, "content"), text[i:i+step])
                           for i in range(0, len(text), step))
        assert restored + replacer.flush((0, "content")) == expected


@pytest.mark.asyncio
async def test_mixed_history_does_not_reinterpret_original_password_in_stream():
    g = make_guardrail()
    text = "secret ladogafinance1"
    data = {"messages": [{"role": "user", "content": text},
                         {"role": "tool", "tool_call_id": "read-1", "content": "alfabank"}]}

    async def analyze(content):
        return [finding(content, text, "PASSWORD")] if content == text else []

    with patch.object(g, "_analyze_text", side_effect=analyze):
        result = await g.async_pre_call_hook(MagicMock(), MagicMock(), data)
    mapping = json.loads(g._redis.setex.call_args[0][2])
    output = "<PASSWORD_1> / ladogafinance1"
    expected = text + " / alfabank"
    assert g._replace_placeholders(output, mapping) == expected
    g._redis.get.return_value = json.dumps(mapping)

    async def chunks():
        for character in output:
            yield litellm.ModelResponseStream(choices=[litellm.StreamingChoices(
                index=0, delta={"content": character}
            )])
        yield litellm.ModelResponseStream(choices=[litellm.StreamingChoices(
            index=0, finish_reason="stop"
        )])

    stream = g.async_post_call_streaming_iterator_hook(MagicMock(), chunks(), result)
    restored = ""
    async for chunk in stream:
        restored += chunk.choices[0].delta.content or ""
    assert restored == expected


@pytest.mark.asyncio
async def test_cache_hit_preserves_native_mask_and_original_request_mapping():
    g = make_guardrail()
    g._analysis_cache_secret = b"unit-test-cache-secret"
    g._redis = _MemoryRedis()
    source = "alice@alfabank.ru"
    auth = MagicMock()
    auth.token = "hashed-virtual-key"

    async def analyze(content):
        return [finding(content, "alice@ladogafinance1.ru", "EMAIL_ADDRESS")]

    with patch.object(g, "_get_analyzer_signature", return_value="a" * 64), patch.object(
        g, "_analyze_text", side_effect=analyze
    ) as detect:
        ids = []
        for _ in range(2):
            result = await g.async_pre_call_hook(auth, MagicMock(), {
                "model": "mock-chat", "messages": [{"role": "user", "content": source}]
            })
            assert result["messages"][0]["content"] == "<EMAIL_ADDRESS_1>"
            request_id = result["metadata"]["pii_request_id"]
            ids.append(request_id)
            mapping = json.loads(g._redis.values[f"pii_mapping:{request_id}"])
            assert mapping["<EMAIL_ADDRESS_1>"] == source
    assert len(set(ids)) == 2
    assert detect.await_count == 1


@pytest.mark.asyncio
async def test_mapping_save_failure_rolls_back_dictionary_and_native_changes():
    g = make_guardrail()
    text = "alice@alfabank.ru"
    data = {"messages": [{"role": "user", "content": text}]}
    with patch.object(g, "_analyze_text", return_value=[finding(
        "alice@ladogafinance1.ru", "alice@ladogafinance1.ru", "EMAIL_ADDRESS"
    )]), patch.object(g, "_save_mapping", side_effect=RuntimeError("redis down")):
        with pytest.raises(ProxyException) as error:
            await g.async_pre_call_hook(MagicMock(), MagicMock(), data)
        assert error.value.status_code == 503
    assert data["messages"][0]["content"] == text


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_stateful_response_continuation_keeps_original_email_mappings(harness, stream):
    g, store, _temporary, _auth = harness

    async def analyze(text, *_args):
        values = [value for value in ("alice@ladogafinance1.ru", "bob@ladogafinance1.ru")
                  if value in text]
        return [finding(text, value, "EMAIL_ADDRESS") for value in values]

    g._analyze_text_with_cache = AsyncMock(side_effect=analyze)
    captured, restored, _ = await complete(
        harness, "alice@alfabank.ru", "bank-parent", "<EMAIL_ADDRESS_1>", stream=stream
    )
    assert captured == "<EMAIL_ADDRESS_1>"
    assert restored == "alice@alfabank.ru"
    parent = copy.deepcopy(store.snapshots)
    captured, restored, _ = await complete(
        harness, "bob@alfabank.ru", "bank-child", "<EMAIL_ADDRESS_1> <EMAIL_ADDRESS_2>",
        previous="bank-parent", stream=stream
    )
    assert captured == "<EMAIL_ADDRESS_2>"
    assert restored == "alice@alfabank.ru bob@alfabank.ru"
    for key, value in parent.items():
        assert store.snapshots[key] == value
