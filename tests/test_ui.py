import shutil
from pathlib import Path

from app.artifacts import write_json
from app.config import ROOT


def test_existing_ui_handles_missing_ragas_and_large_ids(tmp_path):
    from streamlit.testing.v1 import AppTest
    ui = tmp_path / 'Work Done till now' / 'ui'
    shutil.copytree(ROOT / 'Work Done till now' / 'ui', ui, ignore=shutil.ignore_patterns('__pycache__'))
    write_json(tmp_path / 'reports' / 'summary.json', [
        {'mode': 'dense', 'context_precision': None, 'context_recall': None,
         'mrr@10': .2, 'recall@5': .3, 'p95': 301},
        {'mode': 'hybrid_rrf', 'context_precision': None, 'context_recall': None,
         'mrr@10': .4, 'recall@5': .5, 'p95': 250}])
    app = AppTest.from_file(str(ui / 'streamlit_app.py'), default_timeout=20).run()
    assert not app.exception
    passage_id = next(t for t in app.text_input if t.label == 'Passage id')
    passage_id.set_value('9223372036854775807').run()
    assert not app.exception
    assert any('RAGAS is not measured' in w.value for w in app.warning)
