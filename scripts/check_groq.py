"""Verify the configured Groq model and optionally one real RAGAS judgment."""
import argparse
import asyncio
import os

from app.artifacts import checkpoint, read_jsonl
from app.config import load_config, project_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ragas-smoke', action='store_true')
    args = ap.parse_args()
    cfg = load_config()
    if not os.getenv('GROQ_API_KEY'):
        ap.error('GROQ_API_KEY is missing from .env')
    from groq import Groq
    client = Groq(api_key=os.environ['GROQ_API_KEY'], max_retries=1, timeout=60)
    available = {m.id for m in client.models.list().data}
    for role in ['model', 'judge_model']:
        if cfg['llm'][role] not in available:
            raise ValueError(f"Configured {role} is unavailable: {cfg['llm'][role]}")
    response = client.chat.completions.create(model=cfg['llm']['model'], temperature=0,
        max_completion_tokens=128, messages=[{'role': 'user', 'content': 'Reply with READY only.'}])
    if not response.choices[0].message.content:
        raise RuntimeError('Groq returned an empty response')
    result = {'authentication': 'passed', 'model': cfg['llm']['model'], 'completion': 'passed'}
    if args.ragas_smoke:
        import pyarrow.parquet as pq
        from eval.ragas_eval import score_samples
        data = project_path(cfg['data_dir'])
        q = read_jsonl(data / 'eval_queries.jsonl')[0]
        relevant = set(q['relevant_pids'])
        passages = []
        for batch in pq.ParquetFile(data / 'corpus.parquet').iter_batches(columns=['pid', 'text']):
            passages.extend(p for p in batch.to_pylist() if p['pid'] in relevant)
        if len(passages) != len(relevant):
            raise ValueError('Cannot find all selected ground-truth passages')
        scores = asyncio.run(score_samples([{**q, 'hits': passages}], cfg,
                              project_path(cfg['reports_dir']) / 'ragas_smoke_cache'))
        result['ragas'] = {'status': 'passed', 'samples': len(scores),
                           'purpose': 'Judge integration test on known relevant evidence, NOT a retrieval benchmark'}
    checkpoint(project_path(cfg['reports_dir']) / 'checks', 'groq_connection', result)
    print(result)


if __name__ == '__main__':
    main()
