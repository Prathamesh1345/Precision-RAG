from app.artifacts import read_jsonl, write_jsonl
from eval.metrics import ir_metrics, mean_metrics


def run(service, queries, mode, fusion, output, filtered=False, overrides=None):
    rows = []
    for q in queries:
        result = service.search(q['query'], mode=mode, fusion=fusion, top_k=10,
                                category=q['category'] if filtered else None,
                                use_cache=False, overrides=overrides)
        rows.append({'qid': q['qid'], 'query': q['query'], 'reference': q['reference'],
                     'relevant_pids': q['relevant_pids'], **result})
    write_jsonl(output, rows)
    # Recalculation is intentionally from the serialized run, not running totals.
    saved = read_jsonl(output)
    scores = [ir_metrics([h['pid'] for h in r['hits']], r['relevant_pids']) for r in saved]
    return mean_metrics(scores)
