"""Russian address chains exposed as RU_ADDRESS, additive to RuAddressRecognizer.

Finds address components (index, country, region, district, settlement,
street, house, building, flat; plus a Latin variant) and joins neighbours
separated only by spaces or one comma. A chain is emitted only when it holds a
strong combination, so an isolated "стр. 5" or "кв. 3" never fires alone.
An informal "улица дом" value is accepted only right after an address word,
and quoted values of address-like keys are covered in code and JSON.
"""

import re
from dataclasses import dataclass
from string_spans import StringSpans

from presidio_analyzer import AnalysisExplanation, EntityRecognizer, RecognizerResult

NAME = r'(?-i:[А-ЯЁ])[а-яё]+(?:-(?:на-)?(?-i:[А-ЯЁа-яё])[а-яё]+)*'
NAMES = rf'{NAME}(?:[ \t]+{NAME}){{0,2}}'
NUMBERED = r'\d{1,3}(?:-?(?:я|й|е|го|ая|ый|ой|ий|летия))?'
STREET_NAME = (rf'(?:{NUMBERED}[ \t]+(?:лет[ \t]+|летия[ \t]+)?)?(?:(?:Академика|Маршала|Генерала|Героев|Красных|Юных)[ \t]+)?'
               rf'{NAMES}(?:[ \t]+(?:линия|Ямская))?')
NUM = r'\d{1,4}(?:(?-i:[А-ЯЁа-яёA-Za-z])(?![А-Яа-яЁёA-Za-z]))?(?:[/-]\d{1,4}(?-i:[А-Яа-я])?)?(?:[ \t]?к\.?[ \t]?\d{1,2})?'
L = r'(?<![0-9A-Za-zА-Яа-яЁё])'
R = r'(?![0-9A-Za-zА-Яа-яЁё])'
STREET_PREFIX = (r'(?:улица|ул\.|ул|проспект|просп\.|пр-т|пр\.|переулок|пер\.|пер|бульвар|б-р|бул\.|шоссе|ш\.|ш|'
                 r'набережная|наб\.|наб|площадь|пл\.|пл|проезд|пр-д|тупик|туп\.|туп|аллея|ал\.|квартал|кв-л|'
                 r'микрорайон|мкр\.|мкр-н|мкр|линия)(?![А-Яа-яЁё])')
STREET_POSTFIX = (r'(?:улица|ул\.?|проспект|просп\.|пр-т|переулок|пер\.?|бульвар|б-р|шоссе|ш\.?|набережная|наб\.?|'
                  r'площадь|пл\.?|проезд|пр-д|тупик|туп\.?|аллея|ал\.|квартал|кв-л|мкр\.?|мкр-н|микрорайон)')
SETTLEMENT_PREFIX = (r'(?:г\.|гор\.|город|г(?=[ \t])|пос\.|п\.|пгт\.?|с\.|село|дер\.|д\.(?=[ \t]*(?-i:[А-ЯЁ]))|деревня|'
                     r'посёлок|поселок|ст-ца|станица|рп\.?|снт)')
COMPONENTS = {
    'index': rf'{L}[1-6]\d{{5}}{R}',
    'country': rf'{L}(?:Российская Федерация|Россия|РФ|Russia|Russian Federation){R}',
    'region': (rf'{L}(?:{NAME}[ \t]+(?:обл\.?|область|край|АО|автономный округ)|(?:Респ\.?|Республика)[ \t]+{NAMES}'
               rf'(?:[ \t]*\({NAME}\))?|{NAME}[ \t]+(?:Республика|Респ\.)){R}'),
    'district': (rf'{L}(?:{NAME}[ \t]+(?:р-н|район)|(?:р-н|район)[ \t]+{NAMES}|(?:мкр\.?|мкр-н|микрорайон)[ \t]*'
                 rf'(?:{NAMES}|\d{{1,3}}(?-i:[А-Я])?)){R}'),
    'settlement': rf'{L}{SETTLEMENT_PREFIX}[ \t]*{NAME}(?:[ \t]+{NAME}(?![ \t]+{STREET_POSTFIX}(?![А-Яа-яЁё]))){{0,2}}{R}',
    'street': (rf'{L}(?:{STREET_PREFIX}[ \t]*{STREET_NAME}|{STREET_NAME}[ \t]+{STREET_POSTFIX}'
               rf'(?![ \t]*(?:\d|(?-i:[А-ЯЁ])))){R}\.?'),
    'house': rf'{L}(?:д\.|дом|д|вл\.|влад\.)[ \t]*{NUM}{R}',
    'building': rf'{L}(?:корп\.?|корпус|к\.|стр\.|строение|лит\.?|литера?)[ \t]*\d{{1,3}}(?-i:[А-Яа-я])?{R}',
    'flat': rf"{L}(?:кв\.?|квартира|оф\.?|офис|пом\.?|помещение|комн\.?|комната)[ \t]*(?:\d{{1,4}}(?-i:[А-Яа-я])?|\"\d{{1,4}}[А-Яа-я]?\"|'\d{{1,4}}[А-Яа-я]?'|«\d{{1,4}}[А-Яа-я]?»){R}",
    'lat_street': (r'(?<![A-Za-z])(?:(?:ul\.|ulitsa|pr\.|prospekt|per\.)[ \t]*[A-Z][a-z]+(?:[ \t][A-Z][a-z]+)?|'
                   r'(?:\d{1,4}[ \t]+)?[A-Z][a-z]+(?:[ \t][A-Z][a-z]+)?[ \t]+(?:st\.|str\.|street|avenue|ave\.|'
                   r'prospekt|lane|blvd\.?|embankment|square|sq\.))(?![A-Za-z])'),
    'lat_house': r'(?<![A-Za-z0-9])(?:d\.|dom|bld\.|building|house|h\.)[ \t]*\d{1,4}[A-Za-z]?(?![A-Za-z0-9])',
    'lat_flat': r'(?<![A-Za-z0-9])(?:apt\.?|apartment|kv\.|flat|office|of\.|suite)[ \t]*\d{1,4}[A-Za-z]?(?![A-Za-z0-9])',
}
COMPILED = {k: re.compile(v, re.I) for k, v in COMPONENTS.items()}
SEP = re.compile(r'[ \t]*,?[ \t]*\Z')
BARE_HOUSE = re.compile(rf'[ \t]*,?[ \t]*({NUM}){R}(?![ \t]*(?:%|руб|лет|год|раз|км|м\b|шт))')
BARE_NAME = re.compile(rf'[ \t]*,[ \t]*({NAMES}|[A-Z][a-z]+(?:[ \t][A-Z][a-z]+)?){R}')
BARE_NAME_LEFT = re.compile(rf'({NAMES}|[A-Z][a-z]+(?:[ \t][A-Z][a-z]+)?)[ \t]*,?[ \t]*\Z')
STRONG_PAIRS = [{'street', 'house'}, {'street', 'settlement'}, {'street', 'flat'}, {'settlement', 'house'},
                {'index', 'settlement'}, {'index', 'street'}, {'region', 'settlement'}, {'index', 'region'},
                {'lat_street', 'lat_house'}, {'lat_street', 'lat_flat'}, {'lat_street', 'bare_house'},
                {'street', 'bare_house'}, {'lat_street', 'country'}, {'lat_street', 'city'},
                {'lat_street_num'}]
KEY_PATTERN = (
    r'(?:registration_|postal_|home_|delivery_|legal_|fact_|actual_)?'
    r'(?:address|addr|city|street|house|flat|apartment|building|region|district|settlement|zip|postcode|'
    r'адрес|город|улица|дом|квартира|регион|район|индекс)'
)
KEY_RULE = re.compile(
    rf'(?i)(?<![0-9A-Za-zА-Яа-яЁё_])["\']?(?P<key>{KEY_PATTERN})["\']?'
    r'[ \t]*(?P<op>[:=])[ \t]*(?P<q>["\'`])'
    r'(?P<value>(?:\\[^\n]|(?!(?P=q))[^\\\n]){1,200})(?P=q)')
NUMERIC_FIELD = re.compile(
    rf'(?i)(?<![0-9A-Za-zА-Яа-яЁё_])["\']?(?P<key>{KEY_PATTERN})["\']?'
    r'[ \t]*(?P<op>[:=])[ \t]*(?P<value>\d{1,6})(?![\w.])')
KEY_NAME = re.compile(rf'(?i){KEY_PATTERN}\Z')
SCOPE_TOKENS = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|`(?:[^`\\]|\\.)*`|[{}]', re.S)
BARE_PLACE = re.compile(rf'(?:{NAMES}|[A-Z][a-z]+(?:[- ][A-Z][a-z]+){{0,3}})\Z')
NUMBER_PARTS = re.compile(
    rf'(?i){NUM}(?:[ \t]*,?[ \t]*(?:корп\.?|корпус|к\.|стр\.?|строение)'
    r'[ \t]*\d{1,3}[А-Яа-я]?)*\Z')

ADDRESS_CONTEXT = re.compile(r'(?i)(?:адрес\w*|address|addr|проживает|прописан\w*|живу|живёт|зарегистрирован\w*|'
                             r'доставк\w*|delivery|shipping)[^\n]{0,12}?(?=[ \t:="\'`(]|$)')
NOT_STREET = (r'(?!(?:Раздел|Глава|Пункт|Статья|Приложение|Таблица|Рисунок|Рис|Версия|Шаг|Этап|Пример|Страница|'
              r'Порт|Номер|Задача|Вопрос|Ответ|Часть|Том|Лист|Форма|Сервер|Строка|Колонка|Модуль|Адрес|Address)(?![а-яё]))')
INFORMAL = re.compile(rf'[ \t:="\'`(]*(?P<v>{NOT_STREET}(?:{NAMES}[ \t]*,[ \t]*)?{STREET_NAME}(?:[ \t]*,[ \t]*|[ \t]+)\d{{1,4}}(?-i:[А-Яа-я])?'
                      rf'(?:[ \t]*[-/][ \t]*\d{{1,4}}|[ \t]*,?[ \t]*(?:кв\.?|квартира|оф\.?|офис)[ \t]*\d{{1,4}})?){R}')


def components(text):
    found = []
    for kind, pattern in COMPILED.items():
        for m in pattern.finditer(text):
            label = 'lat_street_num' if kind == 'lat_street' and m.group()[0].isdigit() else kind
            found.append((m.start(), m.end(), label))
    found.sort(key=lambda x: (x[0], -x[1]))
    merged = []
    for start, end, kind in found:  # on overlap, keep the earlier one; trim a later one if it still parses
        if merged and start < merged[-1][1]:
            if end <= merged[-1][1]:
                continue
            trimmed = merged[-1][1]
            while trimmed < end and text[trimmed] in ' \t,':
                trimmed += 1
            pattern = COMPILED.get('lat_street' if kind == 'lat_street_num' else kind)
            if trimmed >= end or not pattern.fullmatch(text, trimmed, end):
                continue
            start = trimmed
        merged.append((start, end, kind))
    return merged


def chains(text):
    comps = components(text)
    out, i = [], 0
    while i < len(comps):
        start, end, kinds = comps[i][0], comps[i][1], {comps[i][2]}
        j = i + 1
        while True:
            if j < len(comps) and SEP.match(text[end:comps[j][0]]) and comps[j][0] - end <= 3:
                end, j = comps[j][1], j + 1
                kinds.add(comps[j - 1][2])
                continue
            if kinds & {'street', 'lat_street', 'lat_street_num'} and not kinds & {'house', 'lat_house', 'bare_house'}:
                m = BARE_HOUSE.match(text, end)
                if m and m.end() > end:
                    end = m.end(1)
                    kinds.add('bare_house')
                    continue
            break
        if any(pair <= kinds for pair in STRONG_PAIRS):
            m = BARE_NAME.match(text, end)
            if m and not kinds & {'settlement'} and (j >= len(comps) or comps[j][0] >= m.end()):
                end = m.end(1)
                kinds.add('city')
                if (m2 := BARE_NAME.match(text, end)) and re.fullmatch(r'Russia|Россия|РФ', m2.group(1)):
                    end = m2.end(1)
            left = BARE_NAME_LEFT.search(text, max(0, start - 60), start)
            if left and not kinds & {'settlement'} and left.end() == start:
                start = left.start(1)
            out.append((start, end, kinds))
        i = j
    return out


@dataclass
class _AddressField:
    kind: str
    value: str
    start: int
    end: int
    object_member: bool
    emit: bool = True
    scope: int | None = None


def _key_kind(key):
    if not KEY_NAME.fullmatch(key):
        return None
    key = re.sub(r'^(?:registration_|postal_|home_|delivery_|legal_|fact_|actual_)', '', key.lower())
    return {'addr':'address', 'адрес':'address', 'город':'city', 'улица':'street',
            'дом':'house', 'квартира':'flat', 'apartment':'flat', 'регион':'region',
            'район':'district', 'postcode':'zip', 'индекс':'zip'}.get(key, key)


def _key_fields(text, strings):
    import ast
    import json
    from recognizers.json_strings import iter_json_keyed_strings

    fields = {}
    for pattern, emit in ((KEY_RULE, True), (NUMERIC_FIELD, False)):
        for match in pattern.finditer(text):
            owner = strings.at(match.start('key'))
            if owner and (owner.content_start, owner.content_end) != match.span('key'):
                continue  # A mention inside data is not a sibling object member.
            value = match.group('value')
            if emit and match.group('q') != '`':
                try:
                    wrapped = match.group('q') + value + match.group('q')
                    value = json.loads(wrapped) if match.group('q') == '"' else ast.literal_eval(wrapped)
                except (ValueError, SyntaxError):
                    continue
            start, end = match.span('value')
            fields[start, end] = _AddressField(_key_kind(match.group('key')), value.strip(),
                                              start, end, match.group('op') == ':', emit)
    for key, decoded, offsets in iter_json_keyed_strings(text):
        kind = _key_kind(key)
        if kind:
            fields[offsets[0], offsets[-1]] = _AddressField(kind, decoded.strip(),
                                                          offsets[0], offsets[-1], True)
    ordered = sorted(fields.values(), key=lambda field: field.start)
    stack = []
    index = 0
    for token in SCOPE_TOKENS.finditer(text):
        while index < len(ordered) and ordered[index].start < token.end():
            field = ordered[index]
            if field.object_member:
                field.scope = stack[-1] if stack else None
            index += 1
        if token.group() == '{':
            stack.append(token.start())
        elif token.group() == '}' and stack:
            stack.pop()
    # The credential recognizer already has a bounded, expression-free SQL
    # column/literal parser. Reuse its row scopes instead of borrowing context
    # from another VALUES position.
    from recognizers.sql_credentials import iter_insert_literals
    for key, literal, scope in iter_insert_literals(text):
        kind = _key_kind(key)
        if not kind:
            continue
        raw = literal.group()
        quoted = raw.startswith("'")
        if not quoted and not re.fullmatch(r"\d{1,6}", raw):
            continue
        value = raw[1:-1].replace("''", "'") if quoted else raw
        start, end = (literal.start() + 1, literal.end() - 1) if quoted else literal.span()
        ordered.append(_AddressField(kind, value.strip(), start, end, True, quoted, -scope - 1))
    return ordered


def _full_address(value):
    if len(value) > 200:
        return False
    # The declared address value needs a physical core, not just capitals.
    # The whole bounded field remains private, including components the chain
    # did not join (e.g. a bare city between region and street).
    for candidate in (value, value.title()):
        kinds = {kind for _, _, kind in components(candidate)}
        pairs = [*STRONG_PAIRS, {'index', 'house'}, {'district', 'settlement'}]
        street_marker = re.search(STREET_PREFIX, candidate, re.I)
        if (any(pair <= kinds for pair in pairs) or chains(candidate)
                or INFORMAL.fullmatch(candidate)
                or (street_marker and kinds & {'house', 'lat_house'})):
            return True
    return False


def _format_address_argument(text, context, candidate, cache):
    """Bind a literal to an actual call's first %s address slot, never a sibling field."""
    import ast

    if (context is None or candidate is None
            or re.fullmatch(r'\s*,\s*', text[context.end:candidate.start]) is None):
        return False
    if 'pairs' in cache:
        return (context.start, context.end, candidate.start, candidate.end) in cache['pairs']
    pairs = cache['pairs'] = set()
    try:
        tree = ast.parse(text)
    except (ValueError, SyntaxError, RecursionError):
        return False
    lines = re.findall(r'[^\r\n]*(?:\r\n|\r|\n|$)', text)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    columns = {}

    def bounds(node):
        def offset(row, column):
            line = lines[row - 1]
            if not line.isascii():
                if row not in columns:
                    mapping = {0: 0}
                    byte = 0
                    for char_index, char in enumerate(line, 1):
                        byte += len(char.encode())
                        mapping[byte] = char_index
                    columns[row] = mapping
                column = columns[row][column]
            return starts[row - 1] + column
        return (offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        first, second = node.args[:2]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)
                and isinstance(second, ast.Constant) and isinstance(second.value, str)):
            continue
        marker = re.search(r'(?i)(?:address|addr|адрес\w*)[ \t]*[:=]?[ \t]*(?<!%)%s', first.value)
        if marker and '%' not in first.value[:marker.start()]:
            pairs.add((*bounds(first), *bounds(second)))
    return (context.start, context.end, candidate.start, candidate.end) in pairs


def _street_value(value):
    if value.casefold() in {"null", "none", "true", "false"}:
        return False
    candidate = value.title()
    return bool(COMPILED['street'].fullmatch(candidate)
                or COMPILED['lat_street'].fullmatch(candidate)
                or re.fullmatch(rf'{NOT_STREET}{STREET_NAME}', candidate)
                or re.fullmatch(r'[A-Z][a-z]+(?:[- ][A-Z][a-z]+){0,3}', candidate))


def _number_value(value):
    if len(value) >= 2 and {'"':'"', "'":"'", '«':'»'}.get(value[0]) == value[-1]:
        value = value[1:-1].strip()
    return bool(NUMBER_PARTS.fullmatch(value) or COMPILED['house'].fullmatch(value)
                or COMPILED['building'].fullmatch(value) or COMPILED['flat'].fullmatch(value))


def _administrative_region(value):
    candidate = value.title()
    if not COMPILED['region'].fullmatch(candidate):
        return False
    if re.search(r'(?i)респ', value):
        return True
    return re.search(
        r'(?i)(?:ская|ский|цкая|цкий|кая|кий)[ \t]+(?:обл\.?|область|край|АО|автономный округ)\b',
        value,
    ) is not None


def _accepted_fields(text, strings):
    fields = _key_fields(text, strings)
    kinds_by_scope = {}
    for field in fields:
        if field.scope is None:
            continue
        kind = None
        if field.kind == 'street' and _street_value(field.value):
            kind = 'street'
        elif field.kind in {'house','building','flat'} and _number_value(field.value):
            kind = 'flat' if field.kind == 'flat' else 'house'
        elif field.kind in {'city','settlement'} and COMPILED['settlement'].fullmatch(field.value.title()):
            kind = 'settlement'
        elif field.kind in {'region','district'} and COMPILED[field.kind].fullmatch(field.value.title()):
            kind = field.kind
        elif field.kind == 'zip' and re.fullmatch(r'\d{6}', field.value):
            kind = 'index'
        if kind:
            kinds_by_scope.setdefault(field.scope, set()).add(kind)
    strong = {scope for scope, kinds in kinds_by_scope.items()
              if any(pair <= kinds for pair in STRONG_PAIRS)}
    for field in fields:
        value = field.value
        if (not field.emit or not value or len(value) > 200 or value.casefold() in {'null','none','true','false'}
                or re.fullmatch(r'[A-Z]{2,3}', value)):
            continue
        grouped = field.scope is not None and field.scope in strong
        accepted = (field.kind == 'address' and _full_address(value))
        if field.kind in {'city','settlement'}:
            # Explicit geographic fields retain Russian proper names; arbitrary
            # Latin UI words still need a multi-component postal context.
            name = value.title() if value.isupper() else value
            accepted = re.fullmatch(NAMES, name) is not None or COMPILED['settlement'].fullmatch(value) is not None
        elif field.kind == 'region':
            accepted = _administrative_region(value)
        if field.kind == 'zip':
            accepted = re.fullmatch(r'\d{6}', value) is not None
        elif grouped:
            if field.kind in {'city','settlement'}:
                accepted = bool(BARE_PLACE.fullmatch(value.title()) or COMPILED['settlement'].fullmatch(value))
            elif field.kind == 'street':
                accepted = _street_value(value)
            elif field.kind in {'region','district'}:
                accepted = accepted or COMPILED[field.kind].fullmatch(value) is not None
            elif field.kind in {'house','building','flat'}:
                accepted = _number_value(value)
        if accepted:
            yield field


class RuAddressChainRecognizer(EntityRecognizer):
    """Recognize multi-component and informal Russian addresses."""

    def __init__(self, supported_language: str = 'ru'):
        super().__init__(supported_entities=['RU_ADDRESS'], name='RuAddressChainRecognizer',
                         supported_language=supported_language, version='1.0.0')

    def load(self):
        """No external assets are required."""

    def analyze(self, text, entities, nlp_artifacts=None):
        if entities and 'RU_ADDRESS' not in entities:
            return []
        strings = StringSpans(text)
        results = [RecognizerResult('RU_ADDRESS', s, e, 0.75, AnalysisExplanation(self.name, 0.75),
                                    recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                                                          'kinds': '+'.join(sorted(k))})
                   for s, e, k in chains(text)]
        format_cache = {}
        for c in ADDRESS_CONTEXT.finditer(text):
            m = INFORMAL.search(text, c.end(), min(len(text), c.end() + 140))
            context = strings.at(c.start())
            candidate = strings.at(m.start('v')) if m else None
            gap = text[c.end():m.start('v')] if m else ''
            same_value = m and context is candidate
            if m and context is None and candidate is not None:
                same_value = re.fullmatch(r'[ \t\r\n:=(\"\'`]*', gap) is not None
            formatted = m and not same_value and _format_address_argument(text, context, candidate, format_cache)
            same_value = same_value or formatted
            crossed_field = re.search(r'[;{}]|\b\w+[ \t]*=|[,\r\n][ \t]*\w+[ \t]*:', gap)
            empty_assignment = re.search(r'(?i)[:=][ \t]*(?:null|none|true|false)\b', gap)
            if (m and same_value and (formatted or not crossed_field) and not empty_assignment
                    and m.start('v') - c.end() <= 30
                    and not COMPILED['street'].search(m.group('v'))):
                results.append(RecognizerResult('RU_ADDRESS', m.start('v'), m.end('v'), 0.6,
                                                AnalysisExplanation(self.name, 0.6),
                                                recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name, 'kinds': 'informal'}))
        for field in _accepted_fields(text, strings):
            results.append(RecognizerResult('RU_ADDRESS', field.start, field.end,
                                            0.6, AnalysisExplanation(self.name, 0.6), recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name, 'kinds': 'key'}))
        return results
