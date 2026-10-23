"""Lexical string boundaries in original source; never evaluates or rewrites code.

The scanner supports JSON and quoted assignment/call/container literals. Its
purpose is to bound mask edits, not to decide whether a value is sensitive or
whether a document is valid. Unknown/unclosed literals do not suppress findings.
"""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import re

_PREFIX = re.compile(r'(?i)(?<!\w)(?:r|u|b|f|br|rb|fr|rf)$')
_HEX = frozenset('0123456789abcdefABCDEF')


def _reject_json_constant(value):
    raise ValueError(f'Invalid JSON constant: {value}')


@dataclass(frozen=True)
class StringSpan:
    start: int
    end: int
    content_start: int
    content_end: int
    boundaries: tuple[int, ...]
    controls: tuple[tuple[int, int], ...]

    def align(self, start, end, *, name=False):
        start = max(start, self.content_start)
        end = min(end, self.content_end)
        if name:
            # A model beginning on the n in an actual newline escape must not
            # leave a bare slash. A literal backslash followed by n is different.
            index = bisect_right(self.controls, (start, self.content_end)) - 1
            if index >= 0:
                left, right = self.controls[index]
                if left < start < right:
                    start = right
        if start >= end:
            return None
        start = self.boundaries[bisect_right(self.boundaries, start) - 1]
        end = self.boundaries[bisect_left(self.boundaries, end)]
        return start, end


def _escape_end(text, index, limit, raw):
    end = min(index + 2, limit)
    if not raw and text[index + 1:index + 2] == 'u':
        digits = text[index + 2:index + 6]
        if len(digits) == 4 and set(digits) <= _HEX:
            end = index + 6
            if 0xD800 <= int(digits, 16) <= 0xDBFF and text[end:end + 2] == '\\u':
                low = text[end + 2:end + 6]
                if len(low) == 4 and set(low) <= _HEX and 0xDC00 <= int(low, 16) <= 0xDFFF:
                    end += 6
    elif not raw and text[index + 1:index + 2] in ('x', 'U'):
        length = 2 if text[index + 1] == 'x' else 8
        digits = text[index + 2:index + 2 + length]
        if len(digits) == length and set(digits) <= _HEX:
            end = index + 2 + length
    return min(end, limit)


class StringSpans:
    """Lexical string map with validated JSON recovery in prose."""

    def __init__(self, text):
        self.text = text
        spans = []
        python_source = None
        containers = []

        def embedded_container(content_start, closing):
            # An introductory quote can appear to close on a JSON key's quote.
            # Recover only a complete object/array, not a brace mentioned in data.
            nonlocal python_source
            candidate = closing - 1
            while candidate >= content_start and text[candidate].isspace():
                candidate -= 1
            if candidate < content_start or text[candidate] not in '{[':
                return None
            if python_source is None:
                import ast
                try:
                    ast.parse(text)
                except SyntaxError:
                    python_source = False
                except (ValueError, RecursionError):
                    # Uncertain source is not permission to reinterpret literals.
                    python_source = True
                else:
                    python_source = True
            if python_source:
                return None
            import json
            try:
                value, end = json.JSONDecoder(parse_constant=_reject_json_constant).raw_decode(text, candidate)
            except (ValueError, RecursionError):
                return None
            return candidate if isinstance(value, (dict, list)) and end > closing else None

        index = 0
        while index < len(text):
            if text[index] not in '\"\'':
                character = text[index]
                if character in '{[':
                    containers.append(character)
                elif character in '}]' and containers:
                    if containers[-1] == {'}': '{', ']': '['}[character]:
                        containers.pop()
                index += 1
                continue
            opening = index
            prefix = _PREFIX.search(text, max(0, opening - 3), opening)
            before = (prefix.start() if prefix else opening) - 1
            while before >= 0 and text[before].isspace():
                before -= 1
            structured = before < 0 or text[before] in ':=([{,'
            if (not structured and text[opening] == "'" and opening > 0
                    and opening + 1 < len(text) and text[opening - 1].isalnum()
                    and text[opening + 1].isalnum()):
                index += 1  # An apostrophe inside a prose word is not an opening.
                continue
            raw = bool(prefix and 'r' in prefix.group().lower())
            quote = text[index] * (3 if text.startswith(text[index] * 3, index) else 1)
            index += len(quote)
            content_start = index
            boundaries = [index]
            controls = []
            while index < len(text) and not text.startswith(quote, index):
                if len(quote) == 1 and text[index] in '\r\n':
                    break
                if text[index] == '\\' and index + 1 < len(text):
                    end = _escape_end(text, index, len(text), raw)
                    if not raw and text[index + 1] in 'nrtbf':
                        controls.append((index, end))
                    index = end
                else:
                    index += 1
                boundaries.append(index)
            if text.startswith(quote, index):
                # Inside a container, a value such as "{" is genuine string
                # data, even if later values accidentally complete valid JSON.
                recovery = (embedded_container(content_start, index)
                            if len(quote) == 1 and not containers else None)
                if recovery is not None:
                    index = recovery
                    continue
                if structured:
                    spans.append(StringSpan(opening, index + len(quote), content_start,
                                            index, tuple(boundaries), tuple(controls)))
                index += len(quote)
            else:
                # An unclosed introduction must not swallow later valid values.
                index = opening + len(quote)
        self.spans = tuple(spans)
        self.starts = tuple(span.start for span in spans)

    def at(self, position):
        index = bisect_right(self.starts, position) - 1
        if index >= 0 and position < self.spans[index].end:
            return self.spans[index]
        return None

    def align(self, start, end, *, name=False, infer=False):
        span = self.at(start)
        if span is None and infer:
            index = bisect_left(self.starts, start)
            if index < len(self.spans):
                candidate = self.spans[index]
                prefix = self.text[start:candidate.start]
                if candidate.content_start < end and all(c in ' \t\r\n:=([{,' for c in prefix):
                    span = candidate
        return span.align(start, end, name=name) if span else (start, end)

    def same_value(self, left, right):
        return self.at(left) is self.at(right)
