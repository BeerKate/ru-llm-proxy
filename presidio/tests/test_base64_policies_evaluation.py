"""Category labels of the reproducible Base64 policy comparison."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from base64_policies import _category, canonical_decode


@pytest.mark.parametrize("value,expected", [
    ("YQ==", "decodes to text"),            # correct padding
    ("YWI=", "decodes to text"),
    ("YWJj", "decodes to text"),            # no padding needed
    ("YQ", "decodes to text"),              # padding omitted, still canonical
    ("AAE=", "binary"),
    ("AAEC\nAwQF", "binary"),               # MIME line break ignored
    ("-_8", "binary"),                      # URL-safe alphabet
    ("YR", "not canonical Base64"),         # non-zero trailing bits
    ("YR==", "not canonical Base64"),
    ("YQ=", "not canonical Base64"),        # wrong padding length
    ("YQ===", "not canonical Base64"),
    ("YWJjZ", "not canonical Base64"),      # length 4k+1
    ("ab+_", "not canonical Base64"),       # mixed alphabets
    ("data:image/png;base64,", "not canonical Base64"),
    ("", "not canonical Base64"),
])
def test_category(value, expected):
    assert _category(value) == expected


def test_canonical_decode_returns_bytes():
    assert canonical_decode("YQ==") == canonical_decode("YQ") == b"a"
    assert canonical_decode("YR") is None
