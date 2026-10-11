# Supplied Claude cascade, imported 2026-10-12 from b64_bench (1).zip.
# Algorithm and coefficients are unchanged; integration uses find_spans.
# Standalone mask_base64 is not the Gateway masking implementation.
# Source SHA256: 59e2b297abfa23c11afe1076201271e0d0adbebf6625bd34e65c59bf5fa4be38

"""
Маскирование base64-подстрок перед NER (BERT) в прокси перед vLLM.

Идея: регэксп находит кандидатов, затем каскад дешёвых проверок решает,
base64 это или обычный токен (слово, идентификатор, путь, хэш, UUID).

    from b64_masker import mask_base64
    masked, mapping = mask_base64(text)   # работает и на тексте сообщений, и на сыром JSON-теле
    # одинаковые строки получают один плейсхолдер; mapping: плейсхолдер -> оригинал

Каскад для токена-кандидата:
  0. Структурные «точно да»: data:...;base64,..., JWT, PEM-блоки.
  1. Синтаксис: алфавит одного вида (std "+/" или url "-_"), корректный паддинг, не hex (в т.ч. 0x...),
     не UUID, не только цифры, не ID с префиксом (toolu_…, msg_…, chatcmpl-…, call_…),
     не путь (src/main/java), не kebab/snake-имя (Llama-2-7b-chat-hf), не "//////".
     Невалидная длина (4k+1, лишний/потерянный символ при копировании) — только от 33 символов,
     проверяется без последнего символа.
  2. Короткая зона 8-15 символов: маскируем только base64 от ASCII/кириллического
     текста, буквы или цифры (для 8-11 — ещё и с паддингом "="). Так ловятся YWRtaW4= (admin),
     UEBzc3cwcmQh (P@ssw0rd!), 0JjQstCw0L0= (Иван), NDUxMCAxMjM0NTY (паспорт), но не "test", "KEY_LEFT".
  3. От 16 символов: декодируется в текст -> base64; много нулевых байтов -> base64.
  3a. Явный контекст: рядом с пометкой base64 ({"encoding": "base64", "value": "…"}, "Base64: …",
     "*_base64": "…", base64.b64decode(…)) берём и короткие литералы в кавычках, если они декодируются
     в текст (NTA0NDU2 -> 504456); сразу после пометки — любую валидную строку с паддингом (YQ==).
  4. Иначе — логрег по 8 признакам статистики символов (randomness_score), порог зависит от длины.
  5. Длинные (>= LONG_LEN) std-токены маскируем при слабом признаке случайности (score > -3).
  Декодирование и признаки считаются по первым PREFIX символам — O(1) на любой размер блоба.

Качество (код vLLM/openai-agents/express/cobra/petclinic, траектории SWE-agent, BFCL function
calling, k8s/Terraform, логи loghub, markdown, русские тексты):
  Инъекции (5460 шт.): recall 16+ символов — 100%, 12-15 — 94%, короче 12 — 60%; одинаково на
  тексте сообщений и на сыром JSON. Случайный бинарный base64 16-23 символа — 83-93% (зона риска).
  FP на полном скане 67 млн символов: 33 вхождения (в основном идентификаторы квантования vLLM),
  т.е. ~0.5 на 1 млн символов; наивный regex [A-Za-z0-9+/]{16,} — ~1800 на 1 млн символов.
  Тест-кейсы ИБ (19 кейсов, 17 base64-строк): 17/17, ложных масок 0 — и на тексте, и на JSON-теле.
  Датасеты Codex (http-456 и др.): 430/433 base64, вне разметки — только неразмеченный настоящий base64.
"""
from __future__ import annotations

import base64
import binascii
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

# ID агентских API: известный префикс, разделитель, случайная часть без разделителей.
# Это идентификаторы, а не закодированные данные; маскировать их вредно — рвётся связь
# tool_call ↔ tool_result. Список ЯВНЫЙ: общий шаблон «префикс_случайное» пропускал бы и секреты
# того же вида (ghp_…, glpat-…, sk_live_…, npm_…). Неизвестный префикс маскируется как обычно.
# Свои форматы ID (например, ID обращений из CRM) добавляйте в ID_PREFIXES.
ID_PREFIXES = (
    "call", "toolu", "srvtoolu", "msg", "req", "chatcmpl", "cmpl",           # OpenAI / Anthropic / vLLM
    "run", "thread", "asst", "step", "file", "batch", "resp", "fc", "ftjob",  # OpenAI Assistants / Responses
)
PREFIXED_ID = re.compile(r"(?:%s)[_-](?:[a-z]{2,12}[_-])?[A-Za-z0-9]{12,}" % "|".join(ID_PREFIXES))
MIN_TRIM_LEN = 33

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
    # в декодированном тексте должны быть буквы или цифры (цифры — ради числовых ПДн и паролей)
    return sum(ch.isalnum() for ch in t) / len(t) >= 0.4




def _is_plain_text(b: bytes, min_len: int = 4, min_alnum: float = 0.5) -> bool:
    """Строже, чем _is_meaningful_text: для коротких токенов. Случайные байты часто
    оказываются валидным UTF-8, но тогда это иврит/арабица/CJK, а не ASCII/кириллица.
    Цифры считаются наравне с буквами: base64 от ПИН-кода, даты, номера паспорта."""
    try:
        t = b.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if len(t) < min_len or not PLAIN_TEXT.fullmatch(t):
        return False
    return sum(ch.isalnum() for ch in t) / len(t) >= min_alnum


# --- явный контекст base64 ---------------------------------------------------
# Короткий base64 (до 15 символов без паддинга) по форме неотличим от слова, но часто стоит рядом с
# явной пометкой: "Base64: …", {"encoding": "base64", "value": "…"}, "document_base64": "…",
# blob = "…"; base64.b64decode(blob). Рядом с пометкой берём только строковые литералы в кавычках
# или значение сразу после пометки ("Base64: X"), если они декодируются в текст; сразу после пометки —
# ещё и любую валидную строку с паддингом. Голые слова рядом с пометкой (encode, by) не трогаем.
B64_MARKER = re.compile(r"(?i)base64|b64")
IMMEDIATE_MARKER = re.compile(r"(?i)(?:base64|b64)[A-Za-z_]*[\"']?[ \t]*[:=][ \t]*[\"']?$")   # в пределах строки
WORDLIKE = re.compile(r"[a-z]+|[A-Z]+|[a-z_-]+")   # 'encode', 'RGB', 'big_endian' — у base64 6+ символов так почти не бывает
SHORT_CAND = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{2,}={0,2}(?![A-Za-z0-9+/_=-])")
CONTEXT_WINDOW = 60


def _context_base64(tok: str, immediate: bool) -> bool:
    """Токен рядом с явной пометкой base64 (проверяется вызывающим кодом).
    immediate — пометка стоит прямо перед токеном ("Base64: X", "x_base64": "X")."""
    core = tok.rstrip("=")
    pad = len(tok) - len(core)
    L = len(core)
    if L < 2 or L % 4 == 1 or (pad and (L + pad) % 4) or B64_MARKER.search(tok):
        return False
    has_url = "-" in core or "_" in core
    if has_url and ("+" in core or "/" in core):
        return False
    if core.isdigit() or PREFIXED_ID.fullmatch(core) or (not immediate and WORDLIKE.fullmatch(core)):
        return False
    decoded = _decode(core, urlsafe=has_url)
    if decoded is None:
        return False
    if immediate:                         # "Base64: YQ==" -> a, "Base64: -__v-g==" (бинарь с паддингом)
        return _is_plain_text(decoded, min_len=1) or bool(pad)
    # литерал в кавычках рядом с пометкой: "dGI5MjAz" -> tb9203, "NTA0NDU2" -> 504456, "MDU3MA==" -> 0570.
    # Не короче 6 символов и 4 декодированных: короткие слова в коде ('fmt', 'big5') случайно
    # декодируются в 1-3 печатных символа.
    return L >= 6 and _is_plain_text(decoded, min_len=4, min_alnum=0.3)   # "MiArIDIgPSA0" -> "2 + 2 = 4"


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
    if L % 4 == 1:
        # Невалидная длина: при копировании добавился или потерялся символ (кейс ИБ: строка из 45 символов).
        # Строгий декодер такое не примет, нестрогий примет. Проверяем без последнего символа, но только
        # длинные токены без паддинга: на длинах до 32 это давало ложные срабатывания (call_-ID, имена vLLM).
        return not pad and L >= MIN_TRIM_LEN and is_base64(core[:-1])
    if pad and (L + pad) % 4:
        return False
    if core.isdigit() or HEX.fullmatch(core) or UUID.fullmatch(core):
        return False
    if PREFIXED_ID.fullmatch(core):
        return False                      # toolu_01…, msg_…, chatcmpl-…, call_…, req_… — ID агентских API
    if L + pad < 8:
        return False

    head = core[:PREFIX]  # PREFIX кратен 4, поэтому префикс тоже валидный base64
    decoded = _decode(head, urlsafe=has_url)
    if decoded is None:
        return False
    if L >= MIN_LEN and decoded.count(0) / len(decoded) >= 0.25:
        return True                       # бинарь с нулями ("AAAA..."); проверяем ДО правила путей:
                                          # байты 0xFF дают много "/", и такой блоб похож на путь
    if len(set(head)) <= 3:
        return False                      # "//////", "+-+-+-" — разделители, а не данные
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

    # контекстный проход: только окна вокруг пометок base64 (иначе на коде, где base64 упоминается часто,
    # пришлось бы проверять каждое слово документа)
    regions: list[list[int]] = []
    for mk in B64_MARKER.finditer(scan):
        a, b = max(0, mk.start() - CONTEXT_WINDOW - 100), mk.end() + CONTEXT_WINDOW + 100
        if regions and a <= regions[-1][1]:
            regions[-1][1] = b
        else:
            regions.append([a, b])
    for ra, rb in regions:
        for m in SHORT_CAND.finditer(scan, ra, min(rb, len(scan))):
            s, e = m.span()
            if any(a < e and s < b for a, b in spans):
                continue
            # кавычки проверяем по исходному тексту: в сыром JSON они экранированы (\"), а в scan уже пробелы
            quoted = s > 0 and text[s - 1] in "\"'" and text[e:e + 1] in ("\"", "'", "\\")
            before = scan[max(0, s - CONTEXT_WINDOW):s]
            # дешёвая предпроверка перед регуляркой: пометка «сразу перед» кончается на : = или кавычку
            immediate = before.rstrip(" \t")[-1:] in (":", "=", '"', "'") and IMMEDIATE_MARKER.search(before) is not None
            if not (quoted or immediate):
                continue
            if (immediate or B64_MARKER.search(before) or B64_MARKER.search(scan[e:e + CONTEXT_WINDOW])) \
                    and _context_base64(m.group(), immediate):
                spans.append((s, e))

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
    out, mapping, seen, last = [], {}, {}, 0
    for s, e in find_spans(text):
        val = text[s:e]
        if val not in seen:   # одинаковые строки получают один плейсхолдер: модель видит, что это одно и то же
            seen[val] = placeholder.format(i=len(seen) + 1)
            mapping[seen[val]] = val
        out += [text[last:s], seen[val]]
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
