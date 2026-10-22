"""Russian address chains exposed as RU_ADDRESS, additive to RuAddressRecognizer.

Finds address components (index, country, region, district, settlement,
street, house, building, flat; plus a Latin variant) and joins neighbours
separated only by spaces or one comma. A chain is emitted only when it holds a
strong combination, so an isolated "стр. 5" or "кв. 3" never fires alone.
An informal "улица дом" value is accepted only right after an address word,
and quoted values of address-like keys are covered in code and JSON.
"""

import re

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
    'flat': rf'{L}(?:кв\.?|квартира|оф\.?|офис|пом\.?|помещение|комн\.?|комната)[ \t]*\d{{1,4}}(?-i:[А-Яа-я])?{R}',
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
KEY_RULE = re.compile(
    r'(?i)(?<![A-Za-z_])["\']?(?P<key>(?:registration_|postal_|home_|delivery_|legal_|fact_|actual_)?'
    r'(?:address|addr|city|street|house|flat|apartment|building|region|district|settlement|zip|postcode|'
    r'адрес|город|улица|дом|квартира|регион|район|индекс))["\']?[ \t]*[:=][ \t]*'
    r'(?P<q>["\'`])(?P<value>(?:\\[^\n]|(?!(?P=q))[^\\\n]){1,200})(?P=q)')
ADDRESS_CONTEXT = re.compile(r'(?i)(?:адрес\w*|address|addr|проживает|прописан\w*|живу|живёт|зарегистрирован\w*|'
                             r'доставк\w*|delivery|shipping)[^\n]{0,12}?(?=[ \t:="\'`(]|$)')
NOT_STREET = (r'(?!(?:Раздел|Глава|Пункт|Статья|Приложение|Таблица|Рисунок|Рис|Версия|Шаг|Этап|Пример|Страница|'
              r'Порт|Номер|Задача|Вопрос|Ответ|Часть|Том|Лист|Форма|Сервер|Строка|Колонка|Модуль|Адрес|Address)(?![а-яё]))')
INFORMAL = re.compile(rf'[ \t:="\'`(]*(?P<v>{NOT_STREET}(?:{NAMES}[ \t]*,[ \t]*)?{STREET_NAME}(?:[ \t]*,[ \t]*|[ \t]+)\d{{1,4}}(?-i:[А-Яа-я])?'
                      rf'(?:[ \t]*[-/][ \t]*\d{{1,4}}|[ \t]*,?[ \t]*(?:кв\.?|квартира|оф\.?|офис)[ \t]*\d{{1,4}})?){R}')
LOWER_IDENT = re.compile(r'[a-z0-9_.:/-]+\Z')


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
        results = [RecognizerResult('RU_ADDRESS', s, e, 0.75, AnalysisExplanation(self.name, 0.75),
                                    recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                                                          'kinds': '+'.join(sorted(k))})
                   for s, e, k in chains(text)]
        for c in ADDRESS_CONTEXT.finditer(text):
            m = INFORMAL.search(text, c.end(), min(len(text), c.end() + 140))
            if m and m.start('v') - c.end() <= 30 and not COMPILED['street'].search(m.group('v')):
                results.append(RecognizerResult('RU_ADDRESS', m.start('v'), m.end('v'), 0.6,
                                                AnalysisExplanation(self.name, 0.6),
                                                recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name, 'kinds': 'informal'}))
        for m in KEY_RULE.finditer(text):
            value = m.group('value').strip()
            key = m.group('key').lower()
            if not value or (LOWER_IDENT.match(value) and re.search(r'[a-z]', value)):
                continue
            if re.fullmatch(r'[A-Z]{2,3}', value):  # ISO country/region codes, not an address
                continue
            if key in {'house', 'flat', 'apartment', 'building', 'дом', 'квартира'} and not re.search(r'\d', value):
                continue
            if re.fullmatch(r'[\d.:]+', value) and value.count('.') == 3:  # bind/IP addresses belong to IP rules
                continue
            if not re.search(r'[\dA-ZА-ЯЁ]', value):  # lowercase placeholder prose, e.g. "адрес вашего ..."
                continue
            if key in {'zip', 'postcode', 'индекс'} and not re.fullmatch(r'\d{6}', value):
                continue
            results.append(RecognizerResult('RU_ADDRESS', m.start('value'), m.start('value') + len(m.group('value')),
                                            0.6, AnalysisExplanation(self.name, 0.6), recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name, 'kinds': 'key'}))
        return results
