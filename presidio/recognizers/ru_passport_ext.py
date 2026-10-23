"""Russian passport shapes exposed as RU_PASSPORT, additive to RuPassportRecognizer.

The native patterns need plain whitespace between series and number. Real text
often has words or punctuation between them ("серия 4510 номер 123456",
"45-10 № 123456", "4510-123456") or a glued 10-digit value next to an
explicit passport word. Ambiguous digit-only shapes are emitted with a low
score and pass the 0.35 threshold only through nearby passport context in
the same string value, including decoded JSON values.
"""

import re

from presidio_analyzer import AnalysisExplanation, EntityRecognizer, RecognizerResult
from string_spans import StringSpans

L = r'(?<![0-9A-Za-zА-Яа-яЁё])'
R = r'(?![0-9])'
SERIES = r'\d{2}[ \t-]?\d{2}'
LINK = r'[ \t]*(?:[,;:]|-|–)?[ \t]*(?:(?:номер|ном\.|№|N|No\.?|#)[ \t]*[:.]?[ \t]*)?'
PATTERNS = [
    # серия 45 10 номер 123456 / серия: 4510, № 123456 (labelled series and number)
    ('labelled', re.compile(rf'(?i){L}серия[ \t]*[:.]?[ \t]*(?P<v>{SERIES}{LINK}\d{{6}}){R}'), 0.75),
    # 4510 № 123456 / 45-10 123456 / 4510-123456 / 45 10, 123456
    ('separated', re.compile(rf'{L}(?P<v>\d{{2}}[ \t-]?\d{{2}}[ \t]*[-–,]?[ \t]*(?:(?:№|N|No\.?)[ \t]*)?\d{{6}}){R}'), 0.3),
    # Foreign passport with separators: 75 № 1234567 / 75-1234567
    ('foreign', re.compile(rf'{L}(?P<v>\d{{2}}(?:[ \t]*(?:№|N)[ \t]*|[ \t]*[-–][ \t]*)\d{{7}}){R}'), 0.25),
    # Glued 10 or 9 digits: only with a passport word close on the left.
    ('glued', re.compile(rf'(?i)(?:паспорт\w*|загранпаспорт\w*|загран\w*|passport|паспортные данные|серия и номер|серия/номер)'
                         rf'[^\n\d]{{0,25}}?{L}(?P<v>\d{{10}}|\d{{9}}){R}'), 0.75),
    # Spaced six-digit number groups: 45 10 123 456
    ('grouped', re.compile(rf'{L}(?P<v>\d{{2}}[ \t]\d{{2}}[ \t]\d{{3}}[ \t]\d{{3}}){R}'), 0.3),
]


class RuPassportExtRecognizer(EntityRecognizer):
    """Recognize separated, labelled and glued passport series and numbers."""

    _CONTEXT_RE = re.compile(
        r'(?i)(?<!\w)(?:паспорт[а-яё]*|загранпаспорт[а-яё]*|загран|серия|'
        r'документ[а-яё]*|passport|удостоверени[а-яё]*|выдан[а-яё]*)(?!\w)'
    )

    def __init__(self, supported_language: str = 'ru'):
        super().__init__(supported_entities=['RU_PASSPORT'], name='RuPassportExtRecognizer',
                         supported_language=supported_language, context=[], version='1.0.0')

    def load(self):
        """No external assets are required."""

    def analyze(self, text, entities, nlp_artifacts=None):
        if entities and 'RU_PASSPORT' not in entities:
            return []
        out = []
        strings = None
        for name, pattern, score in PATTERNS:
            for m in pattern.finditer(text):
                digits = re.sub(r'\D', '', m.group('v'))
                if name in {'separated', 'grouped', 'labelled'} and len(digits) == 10 and not 1 <= int(digits[:2]) <= 99:
                    continue
                confidence = score
                if score < 0.35:
                    if strings is None:
                        strings = StringSpans(text)
                    literal = strings.at(m.start('v'))
                    left = max(0, m.start('v') - 100)
                    right = min(len(text), m.end('v') + 50)
                    if literal:
                        left = max(left, literal.content_start)
                        right = min(right, literal.content_end)
                    context = text[left:m.start('v')].rsplit('\n', 1)[-1]
                    context += ' ' + text[m.end('v'):right].split('\n', 1)[0]
                    if self._CONTEXT_RE.search(context):
                        confidence = min(1.0, score + 0.35)
                out.append(RecognizerResult('RU_PASSPORT', m.start('v'), m.end('v'), confidence, AnalysisExplanation(self.name, confidence),
                                            recognition_metadata={RecognizerResult.RECOGNIZER_NAME_KEY: self.name, 'pattern': name}))
        return out
