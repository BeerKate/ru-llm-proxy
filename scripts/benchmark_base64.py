#!/usr/bin/env python3
"""Frozen-corpus comparison, sequential workers; no model or network access.

Input and per-case coordinates may be private. Keep them outside Git. The
aggregate output contains no source values. This does not measure production
latency, model inference, Redis, HTTP transport or concurrent RPS.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import time


def controls():
    """Known intent and source bytes, generated before any detector evaluation."""
    rng = random.Random(20261012)
    rows = []
    for size in (1, 2, 3, 4, 6, 8, 12, 16, 24, 48, 96, 512):
        for sample in range(4):
            raw = bytes(rng.randrange(256) for _ in range(size))
            for urlsafe in (False, True):
                value = (base64.urlsafe_b64encode if urlsafe else base64.b64encode)(raw).decode()
                for padding in (False, True):
                    token = value if padding else value.rstrip("=")
                    for context, template in (
                        ("bare", "Получено {value}."),
                        ("declared", 'Base64: "{value}"'),
                        ("json", '{{"message": "{value}"}}'),
                        ("python", 'value = "{value}"'),
                        ("url", "https://example.org/data?payload={value}"),
                    ):
                        text = template.format(value=token)
                        start = text.index(token)
                        rows.append({"id": f"binary-{size}-{sample}-{urlsafe}-{padding}-{context}",
                                     "group": f"binary-{context}", "core_length": len(token.rstrip("=")),
                                     "syntax": context if context in {"json", "python"} else None,
                                     "text": text, "positive": True,
                                     "targets": [{"start": start, "end": start + len(token)}]})
    for sample, message in enumerate(("M", "Hi", "test", "hello", "привет", "Иван Петров", "обычная строка без секретов", "ordinary text without secrets")):
        for padding in (False, True):
            value = base64.b64encode(message.encode()).decode()
            if not padding:
                value = value.rstrip("=")
            for context, template in (("bare", "Получено {value}."), ("json", '{{"message":"{value}"}}')):
                text = template.format(value=value); start = text.index(value)
                rows.append({"id": f"text-{sample}-{padding}-{context}", "group": "encoded-text",
                             "core_length": len(value.rstrip("=")), "syntax": context if context == "json" else None,
                             "text": text, "positive": True, "targets": [{"start": start, "end": start + len(value)}]})
    for size in (1, 3, 12, 48, 512):
        value = base64.b64encode(bytes(size)).decode()
        rows.append({"id": f"zero-{size}", "group": "zero-bytes", "core_length": len(value.rstrip("=")),
                     "text": value, "positive": True, "targets": [{"start": 0, "end": len(value)}]})
    ordinary = [
        "test data function true false flag", "Hello world this is an ordinary message.",
        "Please explain how this function handles the configuration file.",
        "getUserProfileById setUserConfiguration updateAccountStatus", "normal_identifier another_variable",
        "/usr/local/lib/python3.12 /opt/application/config", "https://example.org/docs/api?view=overview",
        "550e8400-e29b-41d4-a716-446655440000", "2026 12345678 3.141592 1e10",
        "Краткое обычное сообщение без конфиденциальных данных.",
        "server response request context database configuration", "async await import class return default",
        "checksum=deadbeef", "sha256=" + "e3b0c44298fc1c14" * 4,
        "This is a reproducible benchmark for an ordinary software service.",
    ]
    for i, value in enumerate(ordinary):
        for context in ("plain", "json", "python"):
            text = value if context == "plain" else json.dumps({"message": value}) if context == "json" else "message = " + repr(value)
            rows.append({"id": f"negative-{i}-{context}", "group": "ordinary-" + context, "text": text,
                         "positive": False, "targets": [], "syntax": context if context != "plain" else None})
    return rows


def load_common(root):
    rows = []
    for source in (root / "base64-http-456/cases.jsonl").read_text().splitlines():
        row = json.loads(source)
        # Old semantic negatives often are real encodings: use span annotations.
        targets = [t for t in row["targets"] if t.get("kind") == "base64_literal" and t.get("encoding_valid") is not False
                   or row["group"].startswith("fresh-")]
        if targets or row["group"] == "legacy-negative_controls":
            rows.append({**row, "targets": targets, "positive": bool(targets)})
    return rows


def covered(start, end, spans):
    cursor = start
    for a, b in sorted(spans):
        if a > cursor:
            break
        if b > cursor:
            cursor = b
        if cursor >= end:
            return True
    return False


def quality(rows, analyze):
    counts = Counter()
    groups = {}
    details = []
    for row in rows:
        results = analyze(row["text"])
        spans = [(r.start, r.end) for r in results]
        b64 = [(r.start, r.end) for r in results if r.entity_type == "BASE64_DATA"]
        complete = all(covered(t["start"], t["end"], spans) for t in row["targets"])
        any_hit = any(a < t["end"] and b > t["start"] for t in row["targets"] for a, b in spans)
        key = ("tp" if complete else "fn") if row["positive"] else ("fp" if b64 else "tn")
        group = row["group"]
        counts[key] += 1
        groups.setdefault(group, Counter())[key] += 1
        if row["positive"] and not complete:
            counts["partial" if any_hit else "no_hit"] += 1
        if row["positive"]:
            counts["targets"] += len(row["targets"])
            counts["full_targets"] += sum(covered(t["start"], t["end"], spans) for t in row["targets"])
        details.append({"id": row["id"], "classification": key,
                        "entities": [{"start": r.start, "end": r.end, "entity_type": r.entity_type} for r in results],
                        "syntax": row.get("syntax")})
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"counts": dict(counts), "precision": precision, "recall": recall, "f1": f1,
            "groups": {k: dict(v) for k, v in groups.items()}}, details


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def timed(texts, analyze, repeat):
    for text in texts:
        analyze(text)
    walls = []
    cpu_start = time.process_time()
    for _ in range(repeat):
        for text in texts:
            started = time.perf_counter()
            analyze(text)
            walls.append((time.perf_counter() - started) * 1000)
    return {"calls": len(walls), "p50_ms": statistics.median(walls), "p95_ms": percentile(walls, .95),
            "p99_ms": percentile(walls, .99), "cpu_seconds": time.process_time() - cpu_start,
            "total_wall_seconds": sum(walls) / 1000, "characters_per_pass": sum(map(len, texts))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--mode", choices=("content", "format", "logistic"))
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze-controls", action="store_true")
    args = parser.parse_args()
    if args.freeze_controls:
        args.controls.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in controls()))
        print(json.dumps({"controls": len(controls()), "sha256": hashlib.sha256(args.controls.read_bytes()).hexdigest()}))
        return
    os.environ["PRESIDIO_ANALYZER_BASE64_POLICY"] = args.mode
    sys.path.insert(0, str(args.repo / "presidio"))
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    from recognizers import ALL_RECOGNIZERS
    from recognizers.base64_data import Base64DataRecognizer
    from result_merging import merge_results
    recognizers = [cls() for cls in ALL_RECOGNIZERS]
    detector = next(r for r in recognizers if isinstance(r, Base64DataRecognizer))
    initialization_seconds = time.perf_counter() - started
    def analyze(text):
        results = []
        for recognizer in recognizers:
            results.extend(recognizer.analyze(text, recognizer.supported_entities))
        return merge_results(text, results, score_threshold=.35).results
    def base64_only(text):
        return detector.analyze(text, detector.supported_entities)
    fresh = [json.loads(line) for line in args.controls.read_text().splitlines()]
    common = load_common(args.corpus)
    output = {"environment": {"python": platform.python_version(), "platform": platform.platform(),
                               "machine": platform.machine(), "cpu_count": os.cpu_count()},
              "mode": args.mode, "initialization_seconds": initialization_seconds,
              "controls_sha256": hashlib.sha256(args.controls.read_bytes()).hexdigest(),
              "common_sha256": hashlib.sha256((args.corpus / "base64-http-456/cases.jsonl").read_bytes()).hexdigest(),
              "metric_definition": "Positive TP requires 100% union coverage of all labeled Base64 targets. FP is a negative case with any BASE64_DATA. Extra detections in incompletely annotated positive cases are not labeled FP.",
              "datasets": {}, "timing": {}, "native": {}}
    detail = {}
    for name, rows in (("fresh", fresh), ("common", common)):
        output["datasets"][name], detail[name] = quality(rows, analyze)
    # Separate short ambiguity from the lengths advertised for the classifier.
    output["datasets"]["fresh_length_groups"] = {}
    classifications = {d["id"]: d["classification"] for d in detail["fresh"]}
    for label, lower, upper in (("under8", 0, 8), ("8to15", 8, 16), ("16plus", 16, 100000)):
        selected = [r for r in fresh if r["positive"] and lower <= r.get("core_length", 0) < upper]
        counts = Counter(classifications[r["id"]] for r in selected)
        output["datasets"]["fresh_length_groups"][label] = {"counts": dict(counts), "recall": counts["tp"] / len(selected)}
    # DKB is separate; all deterministic classes and projected offsets are checked.
    corpus = json.loads((args.corpus / "dkb-18/manifest.json").read_text())
    dkb = []
    native_counts = Counter()
    for case in corpus:
        # Annotation offsets refer to original bytes, including CRLF characters.
        text = (args.corpus / "dkb-18" / case["file"]).read_bytes().decode("utf-8")
        results = analyze(text)
        for target in case["expected"]:
            if target["deterministic"]:
                native_counts["targets"] += 1
                native_counts["covered"] += int(covered(target["start"], target["end"], [(r.start, r.end) for r in results]))
                native_counts["covered_with_type"] += int(any(r.start <= target["start"] and r.end >= target["end"] and r.entity_type in target["entities"] for r in results))
        dkb.append({"id": case["file"], "entities": [{"start": r.start, "end": r.end, "entity_type": r.entity_type} for r in results]})
    output["native"]["dkb"] = dict(native_counts); detail["dkb"] = dkb
    output["timing"]["common_base64_only"] = timed([r["text"] for r in common], base64_only, 5)
    output["timing"]["common_all_deterministic"] = timed([r["text"] for r in common], analyze, 3)
    value = base64.b64encode("обычная строка проверки производительности".encode()).decode()
    unit = "Обычное сообщение о рабочей задаче. test data function true false.\n"
    for size in (32000, 100000, 200000):
        text = (unit * (size // len(unit) + 1))[:size] + "\n" + value
        output["timing"][f"history_{size}_base64_only"] = timed([text], base64_only, 15)
    for count in (100, 500, 1000):
        text = (value + "; ") * count
        output["timing"][f"candidates_{count}_base64_only"] = timed([text], base64_only, 7)
    # Peak allocation is measured separately, so tracing does not affect timings.
    import tracemalloc
    tracemalloc.start()
    base64_only((value + "; ") * 1000)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    output["memory"] = {"process_peak_rss_bytes": rss_peak * scale,
                        "peak_rss_before_import_bytes": rss_before * scale,
                        "peak_rss_growth_bytes": (rss_peak - rss_before) * scale,
                        "python_peak_allocation_1000_candidates_bytes": peak}
    output["limitations"] = ["Local serial detector measurements; no NER/model, service HTTP, Redis or production concurrency.",
                             "Common corpus is previously opened and overlaps other supplied sets.",
                             "Frozen synthetic controls are first evaluation here, not representative independent production traffic.",
                             "Document full-coverage metrics are stricter than any-hit classifier metrics; positive extras remain unclassified.",
                             "RSS includes Python/Presidio import and native recognizers; not classifier-only memory."]
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".coordinates.json").write_text(json.dumps(detail, ensure_ascii=False) + "\n")
    print(json.dumps({"mode": args.mode, "fresh": output["datasets"]["fresh"]["counts"], "common": output["datasets"]["common"]["counts"], "dkb": output["native"]}))


if __name__ == "__main__":
    main()
