# Imported from user-supplied b64_bench.zip, b64_masker.py.
# Source SHA256: a61ce83859f952c2b518db3ac5061aa1dc7b309ee3707eee69228dda432bb8ab
# Algorithm/coefficients are preserved; reported upstream benchmark claims are unverified here.

"""
Маскирование base64-подстрок перед NER (BERT) в прокси перед vLLM.

Идея: регэксп находит кандидатов, затем каскад дешёвых проверок решает,
base64 это или обычный токен (слово, идентификатор, путь, хэш, UUID).

    from b64_masker import mask_base64
    masked, mapping = mask_base64(text)   # работает и на тексте сообщений, и на сыром JSON-теле

Каскад для токена-кандидата:
  0. Структурные «точно да»: data:...;base64,..., JWT, PEM-блоки.
  1. Синтаксис: алфавит одного вида (std "+/" или url "-_"), корректный паддинг,
     len(core) % 4 != 1, не hex (в т.ч. 0x...), не UUID, не только цифры,
     не путь (src/main/java), не kebab/snake-имя (Llama-2-7b-chat-hf).
  2. Короткая зона 8-15 символов: маскируем только base64 от ASCII/кириллического
     текста (для 8-11 — ещё и с паддингом "="). Так ловятся YWRtaW4= (admin),
     UEBzc3cwcmQh (P@ssw0rd!), 0JjQstCw0L0= (Иван), но не "test", "Installs", "KEY_LEFT".
  3. От 16 символов: декодируется в текст -> base64; много нулевых байтов -> base64.
  4. Иначе — логрег по 8 признакам статистики символов (randomness_score), порог зависит от длины.
  5. Длинные (>= LONG_LEN) std-токены маскируем при слабом признаке случайности (score > -3).
  Декодирование и признаки считаются по первым PREFIX символам — O(1) на любой размер блоба.

Качество (код vLLM/openai-agents/express/cobra/petclinic, траектории SWE-agent, BFCL function
calling, k8s/Terraform, логи loghub, markdown, русские тексты):
  Инъекции (5460 шт.): recall 16+ символов — 100%, 12-15 — 94%, короче 12 — 60%; одинаково на
  тексте сообщений и на сыром JSON. Случайный бинарный base64 16-23 символа — 83-93% (зона риска).
  FP на полном скане 67 млн символов: 33 вхождения (в основном идентификаторы квантования vLLM),
  т.е. ~0.5 на 1 млн символов; наивный regex [A-Za-z0-9+/]{16,} — ~1800 на 1 млн символов.
"""
from __future__ import annotations

import base64
import binascii
import math
import re

MIN_LEN = 16       # минимальная длина core без паддинга для "голого" кандидата
LONG_LEN = 64      # длиннее — маскируем без эвристик (кроме hex/путей)
PREFIX = 256       # декодирование и признаки считаем только по началу токена:
                   # решение от этого не меняется, а стоимость становится O(1), а не O(размер блоба)

# --- структурные паттерны (маскируем всегда) ---------------------------------
DATA_URI = re.compile(r"data:[\w.+-]+/[\w.+-]+(?:;[\w-]+=[\w.-]+)*;base64,[A-Za-z0-9+/_-]{4,}={0,2}")
# JSON-экранирование (\n, \t, \", \\, \uXXXX) в сыром теле запроса: "n" из "\n" входит в алфавит base64
# и прилипает к блобу. Заменяем на пробелы той же длины — смещения не меняются.
JSON_ESC = re.compile(r'\\(?:[nrtbf"\\]|u[0-9a-fA-F]{4})')
JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*")
PEM = re.compile(r"-----BEGIN [A-Z0-9 ]+-----[\s\S]+?-----END [A-Z0-9 ]+-----")

# --- кандидаты ----------------------------------------------------------------
# Граница слева: не символ алфавита (но "=" допустим — token=QUJD...),
# справа: не символ алфавита и не "=".
CANDIDATE = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{6,}={0,2}(?![A-Za-z0-9+/_=-])")

HEX = re.compile(r"(?:0[xX])?[0-9a-fA-F]+")
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
PATHLIKE = re.compile(r"^/?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]*$")
LETTER_RUN = re.compile(r"[A-Za-z]+")
VOWELS = set("aeiouyAEIOUY")


PLAIN_TEXT = re.compile(r"[\x20-\x7e\u0400-\u04ff\u2116«»—–\n\r\t]+")


def _decode(core: str, urlsafe: bool) -> bytes | None:
    s = core + "=" * (-len(core) % 4)
    try:
        if urlsafe:
            return base64.urlsafe_b64decode(s)
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        return None


def _is_meaningful_text(b: bytes, truncated: bool = False) -> bool:
    if len(b) < 4:
        return False
    try:
        t = b.decode("utf-8")
    except UnicodeDecodeError as e:
        # префикс мог разрезать многобайтовый символ (кириллица) на самом конце
        if not truncated or e.start < len(b) - 3:
            return False
        t = b[: e.start].decode("utf-8", errors="ignore")
        if not t:
            return False
    if not PLAIN_TEXT.fullmatch(t):
        return False
    # в декодированном тексте должны быть "буквенные" символы
    letters = sum(ch.isalpha() for ch in t)
    return letters / len(t) >= 0.4




def _is_plain_text(b: bytes) -> bool:
    """Строже, чем _is_meaningful_text: для коротких токенов. Случайные байты часто
    оказываются валидным UTF-8, но тогда это иврит/арабица/CJK, а не ASCII/кириллица."""
    try:
        t = b.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if len(t) < 4 or not PLAIN_TEXT.fullmatch(t):
        return False
    return sum(ch.isalpha() for ch in t) / len(t) >= 0.5


SEGMENT = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")  # разбиение camelCase/PascalCase
SCORE_THRESHOLD = 1.5          # для 16-23 символов: здесь живёт основная масса идентификаторов


def _threshold(L: int, pad: int) -> float:
    """Порог зависит от длины: идентификаторы длиннее 24-32 символов без разделителей редки,
    а у base64 скор с длиной растёт. Паддинг "=" — сам по себе сильный признак -> порог ниже."""
    thr = SCORE_THRESHOLD if L < 24 else 0.0 if L < 32 else -1.5
    return thr - 3.0 if pad else thr


PRONOUNCEABLE = re.compile(r"(?=.*[aeiouy])(?!.*[bcdfghjklmnpqrstvwxz]{4})(?!.*[aeiouy]{3})[a-z]+")


def _features(core: str) -> list[float]:
    L = len(core)
    letters = [c for c in core if c.isalpha()]
    nl = max(len(letters), 1)
    segs = SEGMENT.findall(core)                 # разбиение camelCase: kInt4W4A8Static -> k Int 4 W 4 A 8 Static
    alpha = [x for x in segs if not x.isdigit()]
    return [
        sum(c in VOWELS for c in letters) / nl,                      # доля гласных (base64 ~0.23, слова ~0.40)
        sum(c.isdigit() for c in core) / L,                          # доля цифр
        sum(c.isupper() for c in letters) / nl,                      # доля заглавных (base64 ~0.5)
        sum(map(len, segs)) / max(len(segs), 1),                     # средняя длина сегмента (base64 ~1.5-2)
        min(L, 64) / 64,                                             # длина
        min(max((len(x) for x in alpha), default=0), 12) / 12,       # самое длинное "слово"
        sum(len(x) for x in alpha if len(x) >= 4) / L,               # доля символов в "словах" от 4 букв
        sum(len(x) for x in alpha if len(x) >= 3 and PRONOUNCEABLE.fullmatch(x.lower())) / L,  # произносимые куски
    ]


# LogisticRegression на 8 признаках. Негативы — токены, дошедшие до скора, из stdlib Python и реальных
# корпусов (код vLLM/openai-agents/express/cobra/petclinic, траектории SWE-agent, BFCL, k8s/Terraform,
# логи loghub, markdown); позитивы — случайный base64 и base64 от текста. Обучена на половине токенов
# (split по хэшу), на второй половине при пороге 1.5: TPR ~0.97, FP ~13 вхождений из 33 тыс.
_W = [-1.02, -6.78, 5.07, -3.22, 7.81, 0.27, -2.03, -5.57]
_B = 7.75


def randomness_score(core: str) -> float:
    """> SCORE_THRESHOLD — похоже на base64, ниже — слово/идентификатор."""
    return sum(a * b for a, b in zip(_W, _features(core))) + _B


WORDY = re.compile(r"[a-z][a-z0-9]*|[A-Z][a-z0-9]*|[A-Z0-9]+|[0-9a-fA-F]+")


def _wordy(seg: str) -> bool:
    """Сегмент пути/kebab-имени похож на слово: usr, Llama, 7b, README, ScaledBlas, хэш."""
    if WORDY.fullmatch(seg):
        return True
    parts = SEGMENT.findall(seg)
    return len(seg) >= 6 and sum(map(len, parts)) / max(len(parts), 1) >= 3.5


def _mostly_wordy(segs: list[str]) -> bool:
    return len(segs) >= 2 and sum(map(_wordy, segs)) >= 0.75 * len(segs)


def _looks_like_path(core: str) -> bool:
    """/usr/local/lib, api/v1/users, github-пути с хэшем коммита и т.п."""
    if "/" not in core or "+" in core:
        return False
    return _mostly_wordy([x for x in core.split("/") if x])


def _looks_like_kebab(core: str) -> bool:
    """Llama-2-7b-chat-hf, max_model_len, leet_speak_flag. В случайном base64url "-" и "_"
    встречаются с частотой 1/32, поэтому много разделителей + словоподобные куски = идентификатор."""
    segs = [x for x in re.split(r"[-_]", core) if x]
    if len(segs) >= 2 and all(map(_wordy, segs)):
        return True                       # diff-31e81597..., gpu_memory_utilization
    n_sep = core.count("-") + core.count("_")
    if n_sep < 2 or n_sep * 8 < len(core):
        return False
    return n_sep * 5 >= len(core) or _mostly_wordy([x for x in re.split(r"[-_]", core) if x])


def is_base64(tok: str) -> bool:
    core = tok.rstrip("=")
    pad = len(tok) - len(core)
    L = len(core)
    has_std = "+" in core or "/" in core
    has_url = "-" in core or "_" in core
    if has_std and has_url:
        return False
    if L % 4 == 1 or (pad and (L + pad) % 4):
        return False
    if core.isdigit() or HEX.fullmatch(core) or UUID.fullmatch(core):
        return False
    if L + pad < 8:
        return False

    head = core[:PREFIX]  # PREFIX кратен 4, поэтому префикс тоже валидный base64
    decoded = _decode(head, urlsafe=has_url)
    if decoded is None:
        return False
    if L >= MIN_LEN and decoded.count(0) / len(decoded) >= 0.25:
        return True                       # бинарь с нулями ("AAAA..."); проверяем ДО правила путей:
                                          # байты 0xFF дают много "/", и такой блоб похож на путь
    if has_url and _looks_like_kebab(core):
        return False
    if has_std and _looks_like_path(core):
        return False

    if L < MIN_LEN:
        # Короткая зона (8-15 символов): здесь слова и base64 неотличимы по форме,
        # поэтому берём только base64 от ASCII/кириллического текста, а для 8-11 — ещё и с паддингом.
        if _is_plain_text(decoded) and (pad or L >= 12):
            return True                   # YWRtaW4= (admin), UEBzc3cwcmQh (P@ssw0rd!), 0JjQstCw0L0= (Иван)
        if not pad:
            return False
    elif _is_meaningful_text(decoded, truncated=L > PREFIX):
        return True                       # base64 от текста (в т.ч. русского) — почти без FP
    if L >= LONG_LEN and not has_url and randomness_score(head) > -3.0:
        return True                       # длинный std-токен: достаточно слабого признака случайности
    if not (any(c.islower() for c in core) and any(c.isupper() for c in core)):
        return False                      # UPPER_SNAKE, даты, kebab-case, "----" — у случайного base64 так не бывает
    # паддинг "=" — сам по себе сильный признак base64, поэтому порог ниже
    return randomness_score(head) > _threshold(L, pad)


TAIL = re.compile(r"[ \t]*(?:\r?\n|  )[ \t]*([A-Za-z0-9+/]{1,76}={0,2})(?![A-Za-z0-9+/=])")


def _is_mime_tail(t: str) -> bool:
    return t.endswith("=") or "+" in t or "/" in t or (
        len(t) >= 4 and not t.isdigit() and t != t.lower() and t != t.upper())


def find_spans(text: str) -> list[tuple[int, int]]:
    scan = JSON_ESC.sub(lambda m: " " * len(m.group()), text) if "\\" in text else text
    spans: list[tuple[int, int]] = []
    for rx in (DATA_URI, JWT, PEM):
        spans += [m.span() for m in rx.finditer(scan)]

    def covered(s, e):
        return any(a <= s and e <= b for a, b in spans)

    for m in CANDIDATE.finditer(scan):
        if not covered(*m.span()) and is_base64(m.group()):
            spans.append(m.span())

    # MIME-перенос (строки по 64/76): склеиваем соседние куски, разделённые только пробелами
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and (s <= merged[-1][1] or scan[merged[-1][1]:s].strip() == ""):
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    # короткая последняя строка MIME-блока: берём, только если предыдущая строка полной ширины
    for sp in merged:
        last_line = scan[sp[0]:sp[1]].split()[-1] if scan[sp[0]:sp[1]].split() else ""
        if sp[1] - sp[0] >= 60 and len(last_line) in (64, 76):
            m = TAIL.match(scan, sp[1])
            if m and _is_mime_tail(m.group(1)):
                sp[1] = m.end()
    return [tuple(x) for x in merged]


def mask_base64(text: str, placeholder: str = "<BASE64_{i}>") -> tuple[str, dict[str, str]]:
    """Возвращает замаскированный текст и маппинг плейсхолдер -> оригинал
    (для обратной подстановки в ответ модели, если нужно)."""
    out, mapping, last = [], {}, 0
    for i, (s, e) in enumerate(find_spans(text), 1):
        key = placeholder.format(i=i)
        mapping[key] = text[s:e]
        out += [text[last:s], key]
        last = e
    out.append(text[last:])
    return "".join(out), mapping


if __name__ == "__main__":
    demo = (
        "Привет, вот test data и getUserProfileById, файл /usr/local/lib/python3.13. "
        "Токен: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U "
        "паспорт в base64: 0JjQstCw0L3QvtCyINCY0LLQsNC9IDQ1MTAgMTIzNDU2 "
        "sha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 "
        "картинка data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    masked, mp = mask_base64(demo)
    print(masked)
    for k, v in mp.items():
        print(k, "->", v[:60] + ("..." if len(v) > 60 else ""))
