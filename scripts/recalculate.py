"""Independently recompute stored corpus and benchmark outputs."""
import argparse

from app.artifacts import read_json
from app.config import load_config, project_path
from eval.run_all import make_report, recalculate
from scripts.build_corpus import audit_corpus, TokenLimiter
from scripts.ingest import verify_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', help='A reports/runs/<run-id> directory to recalculate')
    args = ap.parse_args()
    cfg = load_config()
    data = project_path(cfg['data_dir'])
    manifest = verify_manifest(data)
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    tokenizer_file = hf_hub_download(manifest['tokenizer'], 'tokenizer.json',
                                     revision=manifest['tokenizer_revision'],
                                     cache_dir=str(project_path('data/hf_cache')), local_files_only=True)
    limiter = TokenLimiter(Tokenizer.from_file(tokenizer_file), manifest['max_tokens'])
    print(audit_corpus(data, expected=manifest['passages'], limiter=limiter))
    if args.run:
        directory = project_path(args.run)
        summary = read_json(directory / 'summary.json')
        rows = recalculate(directory, [r['mode'] for r in summary])
        make_report(directory, rows, read_json(directory / 'run_context.json'))
        print(directory / 'benchmark_report.md')


if __name__ == '__main__':
    main()
