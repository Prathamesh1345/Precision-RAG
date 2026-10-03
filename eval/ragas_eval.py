"""Groq-backed RAGAS 0.2.15, resumable per sample AND per metric."""
import asyncio
import math
import os

from app.artifacts import fingerprint, read_json, read_jsonl, write_json


async def score_samples(rows, cfg, cache_dir):
    from langchain_groq import ChatGroq
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.run_config import RunConfig

    run_config = RunConfig(max_workers=1, max_retries=8, max_wait=60, timeout=180)
    llm = LangchainLLMWrapper(ChatGroq(model=cfg['llm']['judge_model'], temperature=0,
                                      api_key=os.environ['GROQ_API_KEY'], max_retries=5), run_config=run_config)
    metrics = {'context_precision': LLMContextPrecisionWithReference(llm=llm),
               'context_recall': LLMContextRecall(llm=llm)}
    for metric in metrics.values():
        metric.init(run_config)
    out = []
    for row in rows:
        sample_data = {'user_input': row['query'], 'reference': row['reference'],
                       'retrieved_contexts': [h['text'] for h in row['hits'][:5]]}
        result = {'qid': row['qid'], **sample_data}
        for name, metric in metrics.items():
            key = fingerprint({'sample': sample_data, 'metric': name, 'ragas': '0.2.15',
                               'judge': cfg['llm']['judge_model'], 'temperature': 0, 'prompt_version': 1})
            cache_path = cache_dir / f'{key}.json'
            if cache_path.exists():
                value = read_json(cache_path)['score']
            else:
                value = float(await metric.single_turn_ascore(SingleTurnSample(**sample_data)))
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
    if not os.getenv('GROQ_API_KEY'):
        raise ValueError('GROQ_API_KEY is required for mandatory RAGAS. Use --skip-ragas only for an incomplete development report.')
    n = cfg['evaluation']['ragas_queries']
    rows = read_jsonl(rankings_path)[:n]
    if n < 20 or len(rows) != n:
        raise ValueError('RAGAS requires >=20 frozen queries, with the same count for every mode')
    results = asyncio.run(score_samples(rows, cfg, cache_dir))
    write_json(output, {'judge_model': cfg['llm']['judge_model'], 'ragas_version': '0.2.15', 'rows': results})
    saved = read_json(output)['rows']
    return {key: sum(row[key] for row in saved) / len(saved) for key in ['context_precision', 'context_recall']}
