# Frontend (Streamlit)

## Run
```bash
pip install -r requirements-ui.txt
streamlit run ui/streamlit_app.py
```
Open http://localhost:8501. Every time you save a file, the page reloads by itself (`.streamlit/config.toml` → `runOnSave`).

## Mock vs live
- If the API at `http://localhost:8000` is down, the UI uses `ui/mock_backend.py` and shows a **MOCK DATA** badge.
- Start the backend (`uvicorn app.api:app --port 8000`) and click **Recheck** in the sidebar to switch to live.
- Use a different API URL with `PRAG_API_URL=http://host:port streamlit run ui/streamlit_app.py`.

## Files
- `ui/streamlit_app.py`: the 4 tabs (Search, Compare, Live updates, Evaluation)
- `ui/styles.py`: fonts, colours, card styles
- `ui/api_client.py`: calls to the real API
- `ui/mock_backend.py`: fake data that uses the same API contract
- The Evaluation tab reads `reports/summary.json` and `reports/latency_*.json` once the eval scripts have produced them.
