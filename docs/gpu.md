# RTX 3060 execution

The tested machine has a Ryzen 7 6800HS, 16 GB system RAM, and RTX 3060 Laptop GPU with 6 GB VRAM. Keep the laptop connected to power for sustained work. Smaller inference batches reduce padding allocations; document batches are capped at 32 and reranker batches at 16.

The existing CPU environment is `.venv`; the isolated CUDA environment is `.venv-gpu`. Both reuse the same frozen model files in `data/models`. CUDA/cuDNN libraries are installed within the GPU environment and preloaded through ONNX Runtime. No system CUDA Toolkit installation is required for this setup. The NVIDIA driver must already be available.

```powershell
.\.venv\Scripts\python.exe -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
$env:HF_HUB_OFFLINE='1' # only after the model cache has been downloaded
.\.venv-gpu\Scripts\python.exe -m scripts.benchmark_devices
```

Do not install `requirements.txt` or `requirements-eval.txt` directly into this environment: they pull in CPU FastEmbed/ONNX Runtime, which share import names with the GPU packages. For evaluation, install the additional packages explicitly:

```powershell
.\.venv-gpu\Scripts\python.exe -m pip install 'ragas==0.2.15' 'langchain-groq==0.3.2' 'langchain>=0.3.21,<0.4' 'langchain-community>=0.3.20,<0.4' 'langchain-openai>=0.3,<0.4' 'matplotlib>=3.9,<4'
```

Choose devices separately with `PRAG_DENSE_DEVICE` and `PRAG_RERANK_DEVICE` (`cpu` or `cuda`, default `cpu`). A requested CUDA session fails explicitly if it cannot initialize. Small shape/control operations may still execute on CPU; the profiling report records actual kernel placements. Each CUDA model's ONNX arena is capped at 2 GiB; this is not a cap on the whole process or all VRAM allocations.

For the current CPU-built index, the presentation uses CPU dense queries plus CUDA reranking:

```powershell
.\scripts\start_demo.ps1
```

This starts a read-only presentation with logs in `tmp`. Use `-Cpu` for both models on CPU. It refuses to start if either server port is already occupied.

The same quantized BGE artifact runs on CUDA, but the probe found small numerical differences from CPU (maximum absolute coordinate difference about 0.000925). Therefore finish the current CPU index consistently. For a fully CUDA-built performance experiment, use a **new collection and checkpoint**, then use CUDA dense encoding for both indexing and queries. Do not silently mix CPU and CUDA embeddings in an unfinished measured build. Record environment/device selection and validate complete retrieval quality again.

The local component probe confirmed CUDA kernel execution and measured about 64x faster dense batch encoding and 13x faster reranking on that sample. It ran while CPU ingestion was active, so it is not a controlled whole-pipeline comparison. See `reports/checks/device_benchmark.json` for raw repetitions and numerical checks. Final acceptance requires the separate frozen 100-query benchmark.

BM25 preprocessing and the current Qdrant container run on CPU. Answer generation remains remote on Groq. Local GPU acceleration applies to embedding/reranking, not download speed or Groq generation.

Sources: [FastEmbed GPU support](https://qdrant.github.io/fastembed/examples/FastEmbed_GPU/) and [ONNX Runtime CUDA dependencies and DLL preloading](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
