"""Compare PR108 rules against a pinned local baseline, using real CPU BERT.

Runs each checkout in a separate process, sequentially. No network, git writes,
services, or raw payload logs. Reports coverage, not an overall quality score.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

CORPORA = ('ner_migration', 'person_organization_regressions',
           'name_context_regressions', 'name_field_holdout',
           'name_field_header_control', 'instruction_regressions')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def verify_model(checkout, model):
    manifest = json.loads((checkout / 'presidio/model_manifest.json').read_text())
    checks = {}
    for entry in manifest['files']:
        path = model / entry['path']
        assert path.stat().st_size == entry['size'], path.name + ' size mismatch'
        with path.open('rb') as stream:
            checks[entry['path']] = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert checks[entry['path']] == entry['sha256'], path.name + ' digest mismatch'
    return checks


def worker(args):
    import importlib.metadata
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from fastapi.testclient import TestClient
    from presidio import analyzer_server as server
    from presidio.evaluation.corpus import load_corpus
    from ner import HuggingFaceNERRecognizer

    torch.set_num_threads(2)
    server.ner_recognizer = HuggingFaceNERRecognizer(
        tokenizer=AutoTokenizer.from_pretrained(args.model, local_files_only=True),
        model=AutoModelForTokenClassification.from_pretrained(args.model, local_files_only=True),
    )
    ner_call = server.ner_recognizer.analyze
    ner_spans = []

    def capture(*a, **kw):
        findings = ner_call(*a, **kw)
        ner_spans.append(sorted((r.entity_type, r.start, r.end, r.score) for r in findings))
        return findings

    server.ner_recognizer.analyze = capture
    api = TestClient(server.app)  # backend injected; no startup download
    rows = []
    sources = {}
    for name in CORPORA:
        path = args.checkout / 'presidio/evaluation/data' / (name + '.jsonl')
        sources[name] = digest(path.read_bytes())
        for required in (None, frozenset({'PERSON', 'ORGANIZATION'}), frozenset({'PERSON'})):
            try:
                cases = load_corpus(path, **({} if required is None else {'required_entity_types': required}))
                break
            except ValueError:
                if required == frozenset({'PERSON'}):
                    raise
        for case in cases:
            start = time.perf_counter()
            response = api.post('/api/v1/analyze', json={
                'text': case.text, 'language': case.language,
                'score_threshold': case.score_threshold,
            })
            assert response.status_code == 200, (name, case.case_id, response.status_code)
            elapsed = time.perf_counter() - start
            findings = response.json()['entities']
            covered = {i for r in findings for i in range(r['start'], r['end'])}
            full = [all(case.text[i].isspace() or i in covered for i in range(s.start, s.end))
                    for s in case.expected]
            rows.append(dict(corpus=name, id=case.case_id, full=full,
                             covered=sorted(covered), seconds=elapsed,
                             spans=sorted((r['entity_type'], r['start'], r['end']) for r in findings)))
    args.output.write_text(json.dumps(dict(rows=rows, sources=sources,
        ner_digest=digest(json.dumps(ner_spans, sort_keys=True).encode()),
        python=sys.version.split()[0], threads=torch.get_num_threads(),
        versions={name: importlib.metadata.version(name) for name in
                  ('torch', 'transformers', 'presidio-analyzer', 'spacy')})))


def main(args):
    checkout = args.checkout.resolve()
    baseline = subprocess.check_output(['git', 'rev-parse', args.baseline], cwd=checkout, text=True).strip()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout, text=True):
        raise RuntimeError('Product checkout must be clean')
    model_checks = verify_model(checkout, args.model)
    with tempfile.TemporaryDirectory(prefix='pr108-acceptance-') as temporary:
        temp = Path(temporary)
        old = temp / 'baseline'
        old.mkdir()
        archive = subprocess.check_output(['git', 'archive', baseline], cwd=checkout)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(old, filter='data')
        runs = []
        for label, folder in (('before', old), ('after', checkout)):
            output = temp / (label + '.json')
            env = dict(os.environ, PYTHONPATH=str(folder / 'presidio') + os.pathsep + str(folder),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                       PRESIDIO_ANALYZER_BASE64_POLICY='content')
            with (temp / (label + '.log')).open('w') as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    '--worker', '--checkout', str(folder), '--model', str(args.model.resolve()),
                    '--output', str(output)], cwd=folder, env=env, stdout=log, stderr=log)
            if result.returncode:
                print((temp / (label + '.log')).read_text()[-5000:], file=sys.stderr)
                raise RuntimeError(label + ' failed')
            runs.append(json.loads(output.read_text()))
            print(label, len(runs[-1]['rows']), 'rows complete', flush=True)
        before, after = runs
        assert before['sources'] == after['sources']
        assert before['ner_digest'] == after['ner_digest'], 'NER outputs differ'
        stats = dict(rows=0, values=0, before_full=0, after_full=0,
                     lost_values=0, rows_with_removed_coverage=0, changed_rows=0,
                     negative_rows=0, negative_with_findings_before=0, negative_with_findings_after=0)
        timing = {'before_seconds': 0., 'after_seconds': 0.}
        by_corpus = {}
        for a, b in zip(before['rows'], after['rows'], strict=True):
            assert (a['corpus'], a['id']) == (b['corpus'], b['id'])
            row = dict(rows=1, values=len(a['full']), before_full=sum(a['full']), after_full=sum(b['full']),
                       lost_values=sum(x and not y for x, y in zip(a['full'], b['full'], strict=True)),
                       rows_with_removed_coverage=int(bool(set(a['covered']) - set(b['covered']))),
                       changed_rows=int(a['covered'] != b['covered']),
                       negative_rows=int(not a['full']),
                       negative_with_findings_before=int(not a['full'] and bool(a['spans'])),
                       negative_with_findings_after=int(not b['full'] and bool(b['spans'])))
            aggregate = by_corpus.setdefault(a['corpus'], {k: 0 for k in stats})
            for key, value in row.items():
                stats[key] += value
                aggregate[key] += value
            timing['before_seconds'] += a['seconds']
            timing['after_seconds'] += b['seconds']
        report = dict(baseline=baseline, head=head, metrics=stats, by_corpus=by_corpus,
                      corpus_sha256=after['sources'], ner_digest=after['ner_digest'],
                      python=after['python'], versions=after['versions'], threads=2,
                      model_files_sha256=model_checks,
                      timing_single_pass=timing,
                      timing_limit='Sequential single pass, no warm-up/control of host noise; not a benchmark')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(stats), flush=True)
        assert stats['lost_values'] == 0, 'Previously covered annotations were lost'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--baseline', default='c3151d8')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else main(arguments)
