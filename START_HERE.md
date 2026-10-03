# PrecisionRAG project archive

This package contains the project source, original problem/proposal PDFs, existing team work, tests, CPU/GPU dependency files, the frozen **100,000-passage MS MARCO dataset**, the downloaded model files, and available verification/evaluation artifacts.

It is a point-in-time snapshot. Read `PACKAGE_MANIFEST.json` for the exact included files and hashes. A missing final RAGAS/benchmark report means that evaluation had not finished at packaging time. Do not present incomplete results as final scores.

## Run after extracting

Use Python **3.12** and Docker Desktop with Linux containers. Open PowerShell in the extracted `PrecisionRAG` directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-eval.txt
Copy-Item .env.example .env
```

Add your own `GROQ_API_KEY` to the new `.env` for answer generation and RAGAS. No private API key is included. Retrieval can run without a key.

```powershell
docker compose up -d qdrant
# Dataset and model files are included; the Docker index must be rebuilt locally.
.\.venv\Scripts\python.exe -m scripts.ingest
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --workers 1
```

In a second terminal in the same directory:

```powershell
.\.venv\Scripts\python.exe -m streamlit run "Work Done till now/ui/streamlit_app.py"
```

Open http://127.0.0.1:8501. Confirm the live connection and turn off **Use sample data** for real retrieval. API docs: http://127.0.0.1:8000/docs.

The corpus is already prepared. Do not run `scripts.build_corpus` again in the same data directory. The original workspace ingestion checkpoint is stored as `reports/packaged_source_ingest_checkpoint.json` for reference, not as active resume state; a fresh machine has no corresponding database yet. Existing ingestion reports describe the source workspace and will be replaced by your own measured build.

## GPU and evaluation

See `docs/gpu.md` for the separately installed CUDA environment. Do not install CPU and GPU ONNX/FastEmbed packages into the same environment. The current reference run uses CPU dense embeddings and CUDA reranking. Use matching dense devices for indexing and querying; a fully CUDA-built experiment should use a separate collection.

After indexing:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m eval.run_all --phase core --ragas-queries 20
.\.venv\Scripts\python.exe -m scripts.recalculate
```

The core run compares dense retrieval with hybrid plus reranking, using 100 distinct latency queries and the first 20 frozen RAGAS queries per mode. `python -m eval.run_all` adds fusion/filter ablations with 50 RAGAS queries by default. Groq access and sufficient free-tier quota are required for judging.

`README.md` contains the architecture and complete pipeline. `reports/invigilation_walkthrough.md` provides presentation steps. Paths and runtime URLs in historical reports refer to the original machine; they do not prove services are running on a newly extracted copy.

## Excluded from the ZIP

- Private `.env`, credentials and secret configuration.
- Installed `.venv`/`.venv-gpu` environments, which must be recreated per machine.
- Live Docker database storage. Rebuild it from the included frozen dataset and models.
- Installers, logs, temporary files and Python bytecode.
