# Frontend (Streamlit)

## Run

The complete project now runs from the parent repository root:

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --workers 1
python -m streamlit run "Work Done till now/ui/streamlit_app.py"
```

The commands below still work for standalone frontend development from this folder:
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
- The Evaluation tab follows the root `reports/latest_run.json` to a coherent measured run. Without measured results it shows a waiting message, never placeholder benchmark scores.
