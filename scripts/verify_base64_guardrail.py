#!/usr/bin/env python3
"""Check actual Gateway masking/restoration on saved detector coordinates.

Uses local corpora only, without Redis/network/model. Does not establish that
missed Base64 values were protected; quality is evaluated by benchmark_base64.
Run in the guardrail test environment with LiteLLM installed.
"""
import argparse
import ast
from collections import Counter
import json
import os
from pathlib import Path
import sys

from benchmark_base64 import load_common


def syntax_valid(text, syntax):
    try:
        if syntax == "json":
            json.loads(text)
        elif syntax == "python":
            ast.parse(text)
        else:
            return False
        return True
    except (ValueError, SyntaxError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    sys.path.insert(0, str(args.repo))
    from litellm_guardrails.pii_guardrail import RuPIIGuardrail
    guardrail = RuPIIGuardrail()
    source = {
        "fresh": {row["id"]: row for row in map(json.loads, args.controls.read_text().splitlines())},
        "common": {row["id"]: row for row in load_common(args.corpus)},
        "dkb": {p.name: {"text": p.read_bytes().decode("utf-8")} for p in (args.corpus / "dkb-18").glob("*.txt")},
    }
    details = json.loads(args.coordinates.read_text())
    counts = Counter()
    failures = []
    for name, rows in details.items():
        for row in rows:
            original = source[name][row["id"]]["text"]
            masked, mapping = guardrail._mask_text(original, row["entities"])
            counts["cases"] += 1
            restored = guardrail._replace_placeholders(masked, mapping)
            counts["exact_restore"] += restored == original
            counts["masked_cases"] += bool(mapping)
            if restored != original:
                failures.append({"dataset": name, "id": row["id"], "reason": "restore"})
            expected_characters = sum(r["end"] - r["start"] for r in row["entities"])
            kept_characters = sum(map(len, mapping.values()))
            counts["all_entity_characters_retained"] += expected_characters == kept_characters
            if expected_characters != kept_characters:
                failures.append({"dataset": name, "id": row["id"], "reason": "span_discarded"})
            syntax = source[name][row["id"]].get("syntax")
            if syntax_valid(original, syntax):
                counts[syntax + "_original_valid"] += 1
                valid = syntax_valid(masked, syntax)
                counts[syntax + "_masked_valid"] += valid
                if not valid:
                    failures.append({"dataset": name, "id": row["id"], "reason": syntax + "_structure"})
    result = {"counts": dict(counts), "failures": failures,
              "scope": "Real Gateway _mask_text and _replace_placeholders, saved recognizer results. No live Redis, provider, HTTP or NER; unmasked cases can also restore exactly."}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"counts": result["counts"], "failure_reasons": dict(Counter(r["reason"] for r in failures))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
