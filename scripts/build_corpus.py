"""Stream MS MARCO QnA v2.1, retain evaluation evidence, trim and audit.

python -m scripts.build_corpus --n 100000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from app.artifacts import checkpoint, read_json, read_jsonl, sha256, write_json, write_jsonl
from app.config import load_config, project_path


def normalize(text: str) -> str:
    return ' '.join(text.split())


def pid_of(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(normalize(text).encode('utf-8'), digest_size=8).digest(), 'big') >> 1


def domain(url: str) -> str:
    value = (urlparse(url if '://' in url else '//' + url).hostname or 'unknown').lower()
    return value.removeprefix('www.')


class TokenLimiter:
    def __init__(self, tokenizer, max_tokens=510):
        if not 1 <= max_tokens <= 510:
            raise ValueError('max_tokens must be between 1 and 510 (plus two special tokens)')
        self.tokenizer, self.max_tokens = tokenizer, max_tokens
        tokenizer.no_truncation()
        tokenizer.no_padding()

    def __call__(self, text):
        text = normalize(text)
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        original = len(encoded.ids)
        if original > self.max_tokens:
            text = text[:encoded.offsets[self.max_tokens - 1][1]].rstrip()
        count = len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if count > self.max_tokens:
            raise ValueError('Tokenizer boundary trimming failed')
        return text, count, original > self.max_tokens


def passage_rows(row):
    passages = row['passages']
    if isinstance(passages, list):
        return passages
    keys = ('passage_text', 'url', 'is_selected')
    if len({len(passages[k]) for k in keys}) != 1:
        raise ValueError('Mismatched MS MARCO passage fields')
    return [dict(zip(keys, values)) for values in zip(*(passages[k] for k in keys))]


class CorpusBuilder:
    """SQLite bounds memory and detects the (unlikely) 63-bit hash collision."""

    def __init__(self, path: Path, limiter):
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE passages (pid INTEGER PRIMARY KEY, text TEXT NOT NULL, metadata TEXT NOT NULL)')
        self.limit = limiter
        self.count = 0
        self.stats = Counter()

    def prepare(self, row):
        category = normalize(row.get('query_type') or 'unknown').lower()
        out = []
        for p in passage_rows(row):
            text, tokens, trimmed = self.limit(p['passage_text'])
            self.stats['raw_passages'] += 1
            if not text or not tokens:
                self.stats['empty_dropped'] += 1
                continue
            self.stats['trimmed_occurrences'] += int(trimmed)
            url = p.get('url') or ''
            out.append({'pid': pid_of(text), 'text': text, 'source': domain(url), 'category': category,
                        'sources': [domain(url)], 'categories': [category], 'url': url,
                        'token_count': tokens, 'trimmed': trimmed, 'selected': p['is_selected'] == 1})
        return out

    def add(self, passage):
        self.stats['accepted_occurrences'] += 1
        p = {k: v for k, v in passage.items() if k != 'selected'}
        old = self.db.execute('SELECT text, metadata FROM passages WHERE pid=?', (p['pid'],)).fetchone()
        if old:
            if old[0] != p['text']:
                raise RuntimeError('Passage hash collision: cannot merge different text')
            self.stats['exact_duplicates'] += 1
            meta = json.loads(old[1])
            for field in ('sources', 'categories'):
                meta[field] = sorted(set(meta[field]) | set(p[field]))
            self.db.execute('UPDATE passages SET metadata=? WHERE pid=?', (json.dumps(meta), p['pid']))
        else:
            self.db.execute('INSERT INTO passages VALUES (?, ?, ?)', (p['pid'], p['text'], json.dumps(p)))
            self.count += 1

    def export(self, path):
        import pyarrow as pa
        import pyarrow.parquet as pq
        schema = pa.schema([('pid', pa.int64()), ('text', pa.string()), ('source', pa.string()),
                            ('category', pa.string()), ('sources', pa.list_(pa.string())),
                            ('categories', pa.list_(pa.string())), ('url', pa.string()),
                            ('token_count', pa.int32()), ('trimmed', pa.bool_())])
        with pq.ParquetWriter(path, schema, compression='zstd') as writer:
            cursor = self.db.execute('SELECT metadata FROM passages ORDER BY rowid')
            while rows := cursor.fetchmany(4096):
                writer.write_table(pa.Table.from_pylist([json.loads(r[0]) for r in rows], schema=schema))


def build_from_streams(train, validation, output: Path, n: int, n_eval: int, n_tune: int, limiter, provenance):
    if n < 1 or n_eval < 1 or n_tune < 0:
        raise ValueError('Invalid corpus or query counts')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise FileExistsError('A completed corpus already exists. Use a new --output directory to rebuild.')
    db_path = output / 'corpus.building.sqlite'
    if db_path.exists():
        raise FileExistsError('Interrupted build found. Use --restart to restart the streamed build.')
    started = time.perf_counter()
    builder = CorpusBuilder(db_path, limiter)
    queries, seen_qids, seen_questions = [], set(), set()
    try:
        for row in validation:
            qid, query = str(row['query_id']), normalize(row['query'])
            if not query or qid in seen_qids or query.casefold() in seen_questions:
                continue
            answers = [normalize(a) for a in row.get('answers', [])
                       if normalize(a) and normalize(a).casefold() != 'no answer present.']
            if not answers:
                continue
            passages = builder.prepare(row)
            relevant = [p for p in passages if p['selected']]
            # Do not give a truncated positive passage a full-passage relevance label.
            if not relevant or any(p['trimmed'] for p in relevant):
                continue
            for p in passages:
                builder.add(p)
            if builder.count > n:
                raise ValueError('Corpus cap cannot hold all evaluation/tuning evidence; increase --n')
            queries.append({'qid': qid, 'query': query, 'reference': answers[0],
                            'category': (row.get('query_type') or 'unknown').lower(),
                            'relevant_pids': sorted({p['pid'] for p in relevant})})
            seen_qids.add(qid)
            seen_questions.add(query.casefold())
            if len(queries) == n_eval + n_tune:
                break
        if len(queries) != n_eval + n_tune:
            raise ValueError('Validation split exhausted before enough answerable, labelled queries were found')
        reserved = builder.count
        builder.db.commit()
        checkpoint(output / 'checks', '01_evaluation_selection', {
            'eval_queries': n_eval, 'tuning_queries': n_tune, 'reserved_passages': reserved,
            'unique_query_ids': len(seen_qids), 'selected_positive_passages': sum(len(q['relevant_pids']) for q in queries)})
        if builder.count < n:
            for row in train:
                if normalize(row['query']).casefold() in seen_questions:
                    builder.stats['heldout_query_rows_excluded'] += 1
                    continue
                for p in builder.prepare(row):
                    builder.add(p)
                    if builder.count == n:
                        break
                if builder.count == n:
                    break
                if builder.count // 5000 != (builder.count - len(passage_rows(row))) // 5000:
                    builder.db.commit()
                    print(f'Prepared {builder.count:,}/{n:,} unique passages', flush=True)
        builder.db.commit()
        recount = builder.db.execute('SELECT COUNT(*) FROM passages').fetchone()[0]
        if recount != n or recount != builder.count:
            raise ValueError(f'Expected exactly {n} unique passages, found {recount}')
        checkpoint(output / 'checks', '02_cleaning', {'unique_passages': recount, **dict(builder.stats)})
        temp = output / 'corpus.parquet.tmp'
        builder.export(temp)
        os.replace(temp, output / 'corpus.parquet')
        write_jsonl(output / 'eval_queries.jsonl', queries[:n_eval])
        write_jsonl(output / 'tuning_queries.jsonl', queries[n_eval:])
        write_json(output / 'qrels.json', {q['qid']: {str(p): 1 for p in q['relevant_pids']} for q in queries})
        audit = audit_corpus(output, expected=n, limiter=limiter)
        manifest = {**provenance, 'passages': n, 'eval_queries': n_eval, 'tuning_queries': n_tune,
                    'evaluation_reserved_passages': reserved, 'build_seconds': time.perf_counter() - started,
                    'files': {name: sha256(output / name) for name in
                              ['corpus.parquet', 'eval_queries.jsonl', 'tuning_queries.jsonl', 'qrels.json']},
                    'recalculation': audit, 'statistics': dict(builder.stats)}
        write_json(output / 'manifest.json', manifest)
        return manifest
    finally:
        builder.db.close()
        if (output / 'manifest.json').exists():
            db_path.unlink(missing_ok=True)


def audit_corpus(output: Path, expected=None, limiter=None):
    import pyarrow.parquet as pq
    pids, categories, sources = set(), Counter(), Counter()
    n, max_tokens, trimmed = 0, 0, 0
    for batch in pq.ParquetFile(output / 'corpus.parquet').iter_batches(batch_size=4096):
        for p in batch.to_pylist():
            if p['pid'] in pids or p['pid'] != pid_of(p['text']) or not p['text']:
                raise ValueError('Duplicate ID, empty text, or corrupt passage hash')
            if not 1 <= p['token_count'] <= 510:
                raise ValueError('Invalid stored passage token count')
            if limiter and limiter(p['text'])[:2] != (p['text'], p['token_count']):
                raise ValueError('Token count does not match a fresh calculation')
            pids.add(p['pid'])
            categories.update(p['categories'])
            sources.update(p['sources'])
            n += 1
            max_tokens = max(max_tokens, p['token_count'])
            trimmed += int(p['trimmed'])
    queries = read_jsonl(output / 'eval_queries.jsonl') + read_jsonl(output / 'tuning_queries.jsonl')
    qrels = read_json(output / 'qrels.json')
    if len({q['qid'] for q in queries}) != len(queries) or len({q['query'].casefold() for q in queries}) != len(queries):
        raise ValueError('Tuning/evaluation query overlap')
    for q in queries:
        if not q['relevant_pids'] or not set(q['relevant_pids']) <= pids:
            raise ValueError(f"Missing ground-truth passage for query {q['qid']}")
        if qrels[q['qid']] != {str(p): 1 for p in q['relevant_pids']}:
            raise ValueError('Qrels and query labels disagree')
    if expected is not None and n != expected:
        raise ValueError(f'Expected {expected} passages; recalculated {n}')
    return checkpoint(output / 'checks', '03_corpus_recalculation', {
        'passages': n, 'unique_ids': len(pids), 'qrel_coverage': 1.0, 'queries': len(queries),
        'max_tokens': max_tokens, 'trimmed_unique_passages': trimmed,
        'token_counts_recomputed': limiter is not None,
        'categories': dict(categories), 'top_sources': dict(sources.most_common(20)),
        'meets_100k_minimum': n >= 100000, 'parquet_bytes': (output / 'corpus.parquet').stat().st_size})


def main():
    cfg = load_config()
    ds = cfg['dataset']
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=ds['passages'])
    ap.add_argument('--n-eval', '--n_eval', type=int, default=ds['eval_queries'])
    ap.add_argument('--n-tune', type=int, default=ds['tuning_queries'])
    ap.add_argument('--output', default=cfg['data_dir'])
    ap.add_argument('--config', default=ds['config'], choices=['v1.1', 'v2.1'])
    ap.add_argument('--smoke', action='store_true', help='Allow a non-qualifying corpus below 100K')
    ap.add_argument('--restart', action='store_true', help='Remove only this build database after interruption')
    args = ap.parse_args()
    if args.n < 100000 and not args.smoke:
        ap.error('The problem statement requires >=100000 passages; small tests require --smoke')
    output = project_path(args.output)
    if args.restart and not (output / 'manifest.json').exists():
        (output / 'corpus.building.sqlite').unlink(missing_ok=True)
        (output / 'corpus.building.sqlite-journal').unlink(missing_ok=True)
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from tokenizers import Tokenizer
    cache = project_path('data/hf_cache')
    revision = HfApi().dataset_info(ds['name'], revision=ds['revision']).sha
    tokenizer_revision = HfApi().model_info(cfg['models']['dense'], revision=ds.get('tokenizer_revision', 'main')).sha
    tokenizer_file = hf_hub_download(cfg['models']['dense'], 'tokenizer.json',
                                     revision=tokenizer_revision, cache_dir=str(cache))
    limiter = TokenLimiter(Tokenizer.from_file(tokenizer_file), ds['max_tokens'])

    def stream(split):
        return load_dataset(ds['name'], args.config, split=split, streaming=True,
                            revision=revision, cache_dir=str(cache)).shuffle(
                                seed=ds['seed'], buffer_size=ds['shuffle_buffer'])

    provenance = {'dataset': ds['name'], 'dataset_config': args.config, 'revision': revision,
                  'tokenizer': cfg['models']['dense'], 'tokenizer_revision': tokenizer_revision,
                  'seed': ds['seed'], 'shuffle_buffer': ds['shuffle_buffer'], 'max_tokens': ds['max_tokens'],
                  'sampling': 'seeded bounded-buffer shuffle; validation evidence first, then train distractors',
                  'deduplication': 'exact normalized text, not semantic near-duplicate removal'}
    manifest = build_from_streams(stream('train'), stream('validation'), output, args.n,
                                  args.n_eval, args.n_tune, limiter, provenance)
    print(json.dumps({'passages': manifest['passages'], 'seconds': manifest['build_seconds'], 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
