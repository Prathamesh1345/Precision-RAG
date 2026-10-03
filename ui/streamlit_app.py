"""PrecisionRAG frontend.

Run from the project root:
    streamlit run ui/streamlit_app.py

Uses the real API at $PRAG_API_URL (default http://localhost:8000) when it is up,
otherwise a built-in mock backend so the UI can be developed before the backend exists.
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from ui.api_client import API_URL, ApiBackend, api_alive
from ui.mock_backend import MockBackend
from ui.styles import CSS

st.set_page_config(page_title="PrecisionRAG", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

MODES = {
    "dense": "Dense only",
    "hybrid": "Hybrid",
    "hybrid_rerank": "Hybrid + rerank",
}
MODE_NOTE = {
    "dense": "Phase 1 baseline. Cosine search over BGE embeddings.",
    "hybrid": "Dense and BM25 combined using the configured fusion method.",
    "hybrid_rerank": "Hybrid candidates rescored by a cross-encoder.",
}
CATEGORIES = ["description", "numeric", "entity", "location", "person"]
LATENCY_SLA_MS = 300
REPORTS = ROOT / "reports"
latest_run = REPORTS / "latest_run.json"
if latest_run.exists():
    # eval.run_all writes an absolute path; fall back to the run id so a copied repo still works.
    pointer = json.loads(latest_run.read_text(encoding="utf-8"))
    for candidate in (Path(pointer.get("path", "")), REPORTS / "runs" / pointer.get("run_id", "")):
        if candidate.is_dir():
            REPORTS = candidate
            break


# ---------------------------------------------------------------- backend
def get_backend():
    st.session_state.setdefault("force_mock", False)
    if not st.session_state.force_mock and api_alive():
        return ApiBackend()
    if "mock" not in st.session_state:          # keep mock state (upserts) across reruns
        st.session_state.mock = MockBackend()
    return st.session_state.mock


backend = get_backend()


# ---------------------------------------------------------------- helpers
def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:  # show backend errors in the UI instead of crashing
        st.error(f"The backend returned an error: {e}")
        return None


def run_chips(r: dict):
    t = r.get("timings_ms", {})
    total = t.get("total")
    chips = [f'<span class="chip">{esc(MODES.get(r["mode"], r["mode"]))}</span>']
    if r.get("fusion"):
        chips.append(f'<span class="chip">{esc(r["fusion"]).upper() if r["fusion"] in ("rrf", "dbsf") else esc(r["fusion"])} fusion</span>')
    flt = {k: v for k, v in (r.get("filter") or {}).items() if v}
    for k, v in flt.items():
        chips.append(f'<span class="chip">{esc(k)}: {esc(v)}</span>')
    for k in ("encode", "vector_db", "rerank"):
        if k in t:
            chips.append(f'<span class="chip">{esc(k.replace("_", " "))} {t[k]:.0f} ms</span>')
    if total is not None:
        cls = "slow" if total > LATENCY_SLA_MS else "time"
        chips.append(f'<span class="chip {cls}">{total:.0f} ms total</span>')
    if "cache_hit" in t:
        chips.append('<span class="chip time">served from cache</span>')
    st.markdown(f'<div class="run">{"".join(chips)}</div>', unsafe_allow_html=True)


def card(i: int, h: dict, trim: int | None = None) -> str:
    text = h.get("text", "")
    if trim and len(text) > trim:
        text = text[:trim].rstrip() + "…"
    return f"""
<div class="card {'top' if i == 1 else ''}">
  <div class="head"><span class="rank">{i}</span><span class="score">{float(h.get('score', 0)):.3f}</span></div>
  <div class="text">{esc(text)}</div>
  <div class="foot"><span><b>{esc(h.get('source', ''))}</b></span><span>{esc(h.get('category', ''))}</span><span>id {esc(h.get('pid'))}</span></div>
</div>"""


def masonry(hits: list[dict]):
    st.markdown('<div class="masonry">' + "".join(card(i, h) for i, h in enumerate(hits, 1)) + "</div>",
                unsafe_allow_html=True)


def stack(hits: list[dict], trim=200):
    st.markdown('<div class="stack">' + "".join(card(i, h, trim) for i, h in enumerate(hits, 1)) + "</div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Backend")
    stats = call(backend.stats) or {}
    if backend.is_mock:
        st.markdown('<div class="status"><span class="dot mock"></span>Sample data</div>', unsafe_allow_html=True)
        st.caption(f"No API at {API_URL}. Start it with `uvicorn app.api:app --port 8000` and recheck.")
    else:
        st.markdown('<div class="status"><span class="dot live"></span>Connected</div>', unsafe_allow_html=True)
        st.caption(API_URL)
    st.markdown(f'<div class="stat"><div class="num">{int(stats.get("points", 0)):,}</div>'
                f'<div class="lbl">passages indexed</div></div>', unsafe_allow_html=True)
    if st.button("Recheck connection"):
        st.rerun()
    st.session_state.force_mock = st.toggle("Use sample data", value=st.session_state.force_mock)

    st.markdown("### Results")
    top_k = st.slider("Passages to return", 1, 10, 5)
    st.caption("Fusion method and weights are set in config.yaml on the backend.")

# ---------------------------------------------------------------- page
st.markdown('<div class="lead">Find the passage, not just a similar one.</div>'
            '<div class="lead-sub">Hybrid dense and BM25 retrieval over MS MARCO, reranked by a cross-encoder, on Qdrant.</div>',
            unsafe_allow_html=True)

if backend.is_mock:
    st.markdown('<div class="note">You are looking at 16 sample passages with made-up scores. '
                'Start the backend to search the real index.</div>', unsafe_allow_html=True)

tab_search, tab_compare, tab_update, tab_eval = st.tabs(["Search", "Compare", "Update index", "Results"])

# ---------------------------------------------------------------- Search
with tab_search:
    q = st.text_input("Question", "what is the normal body temperature of a dog",
                      label_visibility="collapsed", placeholder="Ask a question")
    mode = st.pills("Retrieval", list(MODES), format_func=MODES.get, default="hybrid_rerank") or "hybrid_rerank"
    st.caption(MODE_NOTE[mode])
    cat = st.pills("Only this category", CATEGORIES)
    c3, c4, c5 = st.columns([1.2, 1, 1.2])
    src = c3.text_input("Only this source", placeholder="petmd.com")
    gen = c4.checkbox("Write an answer with citations")
    go = c5.button("Search", type="primary", width="stretch")

    if go and q.strip():
        with st.spinner("Searching"):
            r = call(backend.search, q.strip(), mode=mode, category=cat, source=src.strip() or None,
                     top_k=top_k, generate=gen)
        if r:
            run_chips(r)
            if r.get("generation_error"):
                st.warning(r["generation_error"])
            if r.get("generation_ms") is not None:
                st.caption(f"Answer generation: {r['generation_ms']:.0f} ms (separate from retrieval)")
            if gen and r.get("answer"):
                st.markdown(f'<div class="answer"><small>Answer, grounded in the passages below</small>{esc(r["answer"])}</div>',
                            unsafe_allow_html=True)
            if r.get("hits"):
                masonry(r["hits"])
            else:
                st.info("Nothing matched those filters. Clear the category or source and try again.")

# ---------------------------------------------------------------- Compare
with tab_compare:
    st.caption("One question, three retrieval modes. Pick a query with an exact name or number to see where dense search slips.")
    q2 = st.text_input("Question to compare", "what is form 1099-r used for", label_visibility="collapsed",
                       placeholder="Ask a question", key="q_cmp")
    if st.button("Compare", type="primary") and q2.strip():
        cols = st.columns(3)
        for col, m in zip(cols, MODES):
            with col:
                r = call(backend.search, q2.strip(), mode=m, top_k=top_k)
                if r:
                    total = r.get("timings_ms", {}).get("total", 0)
                    st.markdown(f'<div class="colhead">{MODES[m]}</div><div class="colsub">{total:.0f} ms</div>',
                                unsafe_allow_html=True)
                    stack(r.get("hits", []))

# ---------------------------------------------------------------- Update index
with tab_update:
    st.caption("Add, replace or remove one passage. The rest of the index is untouched.")
    c1, c2 = st.columns([1, 2])
    with c1:
        pid_text = st.text_input("Passage id", "999000001", help="Integer ID; text input preserves all 63 bits.")
        try:
            pid = int(pid_text)
            valid_pid = 0 <= pid <= 2**63 - 1
        except ValueError:
            pid, valid_pid = 0, False
        if not valid_pid:
            st.warning("Enter an integer from 0 to 9223372036854775807.")
        u_src = st.text_input("Source", "custom")
        u_cat = st.selectbox("Category", CATEGORIES + ["custom"], index=5)
    with c2:
        text = st.text_area("Passage text", height=150,
                            value="ADROSONIC BUILD is a 24-hour student hackathon held at BIT Mesra "
                                  "from 2nd to 4th October 2026.")
    b1, b2, b3, _ = st.columns([1.3, 1.3, 1.7, 2])
    if stats.get('read_only'):
        st.info('The live presentation preserves the frozen benchmark corpus. Index updates are disabled in this session.')
    if b1.button("Save passage", type="primary", disabled=not valid_pid or stats.get('read_only', False)):
        r = call(backend.upsert, pid, text, u_src, u_cat)
        if r:
            st.success(f"Saved passage {r.get('pid')}.")
    if b2.button("Delete passage", disabled=not valid_pid or stats.get('read_only', False)):
        r = call(backend.delete, pid)
        if r:
            if r.get("status") == "deleted":
                st.success(f"Deleted passage {r.get('pid')}.")
            else:
                st.warning(f"Passage {r.get('pid')} was not in the index.")
    if b3.button("Check with a search"):
        r = call(backend.search, "when is ADROSONIC BUILD hackathon", mode="hybrid_rerank", top_k=3)
        if r:
            found = any(int(h.get("pid", -1)) == int(pid) for h in r.get("hits", []))
            if found:
                st.success(f"Passage {pid} is in the top 3 for “when is ADROSONIC BUILD hackathon”.")
            else:
                st.warning(f"Passage {pid} is not in the top 3.")
            stack(r.get("hits", []))

# ---------------------------------------------------------------- Results
with tab_eval:
    summary_path = REPORTS / "summary.json"
    if summary_path.exists():
        df = pd.read_json(summary_path)
    else:
        st.info("No measured benchmark yet. Run python -m eval.run_all from the project root.")
        st.stop()

    if df.empty:
        st.info("This evaluation has no completed modes yet.")
        st.stop()
    if df["context_precision"].isna().any():
        st.warning("RAGAS is not measured for one or more modes. This is an incomplete development report.")

    selection = st.selectbox("Measured mode", df["mode"].tolist())
    best = df[df["mode"] == selection].iloc[0]
    stats_row = [("Context precision", best.get("context_precision"), 0.75, ">", "{:.2f}"),
                 ("Context recall", best.get("context_recall"), 0.70, ">", "{:.2f}"),
                 ("p95 latency", best.get("p95"), LATENCY_SLA_MS, "<", "{:.0f} ms"),
                 ("Passages indexed", None if backend.is_mock else stats.get("points"), 100_000, ">=", "{:,.0f}")]
    cols = st.columns(4)
    for col, (lbl, val, tgt, op, fmt) in zip(cols, stats_row):
        if val is None or pd.isna(val):
            cls, shown = "", "—"
        else:
            ok = (float(val) >= tgt if op == ">=" else float(val) > tgt) if op in (">", ">=") else float(val) < tgt
            cls, shown = ("ok" if ok else "miss"), fmt.format(float(val))
        tgt_txt = f"target {'at least' if op == '>=' else 'above' if op == '>' else 'under'} {tgt:,}" + (" ms" if "ms" in fmt else "")
        col.markdown(f'<div class="stat"><div class="num {cls}">{shown}</div>'
                     f'<div class="lbl">{lbl}</div><div class="tgt">{tgt_txt}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec">What each stage adds</div>', unsafe_allow_html=True)
    metric_cols = [c for c in ["context_precision", "context_recall", "mrr@10", "recall@5"] if c in df.columns]
    long = df.melt(id_vars="mode", value_vars=metric_cols, var_name="metric", value_name="value")
    long["metric"] = long["metric"].str.replace("_", " ")
    pretty = {"dense": "Dense", "hybrid_rrf": "Hybrid", "hybrid_dbsf": "Hybrid (DBSF)", "hybrid_weighted": "Hybrid (weighted)",
              "hybrid_rerank_rrf": "Rerank", "hybrid_rerank": "Rerank", "hybrid": "Hybrid"}
    long["mode"] = long["mode"].map(lambda m: pretty.get(m, m))
    order = [pretty.get(m, m) for m in df["mode"]]
    palette = ["#c9c9c9", "#8fbfae", "#0e6b4a", "#202020", "#8b6f47", "#497e9c"][:len(order)]
    chart = alt.Chart(long).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("mode:N", title=None, sort=order, axis=alt.Axis(labelAngle=0, labelFont="Figtree", labelFontSize=12, ticks=False, domain=False)),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(grid=True, gridColor="#efefef", labelFont="Figtree", ticks=False, domain=False)),
        color=alt.Color("mode:N", scale=alt.Scale(domain=order, range=palette), legend=None),
        column=alt.Column("metric:N", title=None, header=alt.Header(labelFont="Figtree", labelFontSize=13, labelFontWeight=600)),
        tooltip=["mode", "metric", alt.Tooltip("value:Q", format=".3f")],
    ).properties(width=190, height=220).configure_view(strokeWidth=0).configure_facet(spacing=24).configure(background="#ffffff")
    st.altair_chart(chart, theme=None)
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown('<div class="sec">Latency across 100 queries</div>', unsafe_allow_html=True)
    lat_files = sorted(REPORTS.glob("latency_*.json")) if REPORTS.exists() else []
    if lat_files:
        pick = st.selectbox("Run", lat_files, format_func=lambda p: p.stem.replace("latency_", ""))
        d = json.loads(Path(pick).read_text())
        raw = pd.DataFrame({"ms": d["raw_ms"]})
        s = d["summary"]
        hist = alt.Chart(raw).mark_bar(color="#0e6b4a", cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("ms:Q", bin=alt.Bin(maxbins=30), title="milliseconds"), y=alt.Y("count()", title="queries"))
        sla = alt.Chart(pd.DataFrame({"x": [LATENCY_SLA_MS]})).mark_rule(color="#9b2217", strokeDash=[5, 4]).encode(x="x:Q")
        p95 = alt.Chart(pd.DataFrame({"x": [s["p95"]]})).mark_rule(color="#202020", size=2).encode(x="x:Q")
        st.altair_chart((hist + sla + p95).properties(height=240).configure_view(stroke=None).configure(background="#ffffff"),
                        theme=None, width="stretch")
        st.caption(f"p50 {s['p50']:.0f} ms, p95 {s['p95']:.0f} ms, p99 {s['p99']:.0f} ms. "
                   f"Dashed line is the 300 ms limit, solid line is p95.")
    else:
        st.caption("Run python -m eval.latency_bench to see the latency distribution here.")
