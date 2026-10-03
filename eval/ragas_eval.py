"""Groq-backed RAGAS 0.2.15, resumable per sample AND per metric.

Several free-tier keys can be listed in GROQ_API_KEYS (comma-separated). When one key hits its
daily token limit the judge moves to the next key and retries the same query, so the judge model,
prompts and cache stay identical across keys.
"""
import asyncio
import math
import os

from app.artifacts import fingerprint, read_json, read_jsonl, write_json


def groq_keys():
    """All configured keys in order: GROQ_API_KEYS first, then GROQ_API_KEY as a fallback."""
    keys = [k.strip() for k in os.getenv('GROQ_API_KEYS', '').split(',') if k.strip()]
    single = os.getenv('GROQ_API_KEY', '').strip()
    if single and single not in keys:
        keys.append(single)
    return keys


def _is_rate_limit(exc):
    from groq import RateLimitError
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, RateLimitError) or 'rate_limit_exceeded' in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _build_metrics(cfg, api_key):
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.run_config import RunConfig

    run_config = RunConfig(max_workers=1, max_retries=3, max_wait=30, timeout=180)
    llm = LangchainLLMWrapper(ChatGroq(model=cfg['llm']['judge_model'], temperature=0,
                                      api_key=api_key, max_retries=2), run_config=run_config)
    metrics = {'context_precision': LLMContextPrecisionWithReference(llm=llm),
               'context_recall': LLMContextRecall(llm=llm)}
    for metric in metrics.values():
        metric.init(run_config)
    return metrics


async def score_samples(rows, cfg, cache_dir):
    from ragas import SingleTurnSample

    keys = groq_keys()
    if not keys:
        raise ValueError('Set GROQ_API_KEY or GROQ_API_KEYS in .env')
    key_index = 0
    metrics = _build_metrics(cfg, keys[key_index])
    print(f'RAGAS judge {cfg["llm"]["judge_model"]} with {len(keys)} Groq key(s)', flush=True)
    out = []
    for row in rows:
        sample_data = {'user_input': row['query'], 'reference': row['reference'],
                       'retrieved_contexts': [h['text'] for h in row['hits'][:5]]}
        result = {'qid': row['qid'], **sample_data}
        for name in list(metrics):
            key = fingerprint({'sample': sample_data, 'metric': name, 'ragas': '0.2.15',
                               'judge': cfg['llm']['judge_model'], 'temperature': 0, 'prompt_version': 1})
            cache_path = cache_dir / f'{key}.json'
            if cache_path.exists():
                value = read_json(cache_path)['score']
            else:
                while True:
                    try:
                        value = float(await metrics[name].single_turn_ascore(SingleTurnSample(**sample_data)))
                        break
                    except Exception as exc:
                        if not _is_rate_limit(exc):
                            raise
                        key_index += 1
                        if key_index >= len(keys):
                            raise RuntimeError(
                                f'All {len(keys)} Groq keys are rate-limited. Add another key to GROQ_API_KEYS '
                                f'or wait for the daily limit to reset; completed judgments are cached.') from exc
                        print(f'Groq key {key_index} rate-limited; switching to key {key_index + 1}/{len(keys)}',
                              flush=True)
                        metrics = _build_metrics(cfg, keys[key_index])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'RAGAS {name} failed for query {row["qid"]}; refusing a partial mean')
                write_json(cache_path, {'score': value, 'metric': name, 'judge': cfg['llm']['judge_model']})
                await asyncio.sleep(cfg['evaluation']['judge_pause_seconds'])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Corrupt RAGAS cache')
            result[name] = value
        out.append(result)
        print(f'RAGAS {len(out)}/{len(rows)} queries completed', flush=True)
    return out


def run(rankings_path, cfg, output, cache_dir):
    if not groq_keys():
        raise ValueError('GROQ_API_KEY or GROQ_API_KEYS is required for mandatory RAGAS. '
                         'Use --skip-ragas only for an incomplete development report.')
    n = cfg['evaluation']['ragas_queries']
    rows = read_jsonl(rankings_path)[:n]
    if n < 20 or len(rows) != n:
        raise ValueError('RAGAS requires >=20 frozen queries, with the same count for every mode')
    results = asyncio.run(score_samples(rows, cfg, cache_dir))
    write_json(output, {'judge_model': cfg['llm']['judge_model'], 'ragas_version': '0.2.15', 'rows': results})
    saved = read_json(output)['rows']
    return {key: sum(row[key] for row in saved) / len(saved) for key in ['context_precision', 'context_recall']}
