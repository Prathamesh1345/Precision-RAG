CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;800&display=swap');

:root {
  --paper:#ffffff; --mist:#efefef; --mist-2:#e2e2e2; --ink:#202020; --slate:#767676;
  --moss:#0e6b4a; --moss-tint:#e4f1ea; --honey:#fff3d6; --honey-ink:#7a4e00;
  --r-pill:999px; --r-card:16px;
}

html, body, .stApp, [data-testid="stAppViewContainer"] { background: var(--paper); }
html, body, [class*="css"], .stMarkdown, .stTextInput, .stButton, .stSelectbox,
.stTextArea, .stNumberInput, .stCheckbox, .stSlider, .stPills, .stCaption, p, label, span, div {
  font-family: 'Figtree', 'Segoe UI', system-ui, sans-serif;
}
h1, h2, h3, h4 { font-family: 'Figtree', 'Segoe UI', system-ui, sans-serif !important; color: var(--ink); }

.block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1080px; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- opening line ---------- */
.lead { font-size: 34px; font-weight: 800; letter-spacing: -0.6px; line-height: 1.1; color: var(--ink); margin: 6px 0 4px; }
.lead-sub { font-size: 16px; color: var(--slate); margin: 0 0 22px; }

/* ---------- inputs as pills ---------- */
[data-testid="stTextInput"] input {
  background: var(--mist); border: 2px solid transparent; border-radius: var(--r-pill);
  padding: 14px 22px; font-size: 17px; color: var(--ink); height: auto;
}
[data-testid="stTextInput"] input:focus { border-color: var(--moss); box-shadow: none; background: #fff; }
[data-testid="stTextInput"] > div > div { border: none; background: transparent; }
[data-testid="stTextInput"] label, [data-testid="stSelectbox"] label, [data-testid="stTextArea"] label,
[data-testid="stNumberInput"] label, [data-testid="stPills"] label, [data-testid="stSlider"] label {
  font-size: 13.5px; font-weight: 600; color: var(--ink);
}
[data-testid="stTextArea"] textarea { background: var(--mist); border: 2px solid transparent; border-radius: var(--r-card); font-size: 15px; }
[data-testid="stTextArea"] textarea:focus { border-color: var(--moss); background: #fff; box-shadow: none; }
[data-testid="stTextArea"] > div { border: none; }
[data-testid="stNumberInput"] > div > div, [data-testid="stSelectbox"] > div > div {
  background: var(--mist); border: none; border-radius: 12px;
}

/* pills (st.pills) */
button[data-variant="pills"] {
  border-radius: var(--r-pill) !important; background: var(--mist) !important; border: 2px solid transparent !important;
  color: var(--ink) !important; font-weight: 600 !important; padding: 6px 16px !important; margin: 0 6px 6px 0 !important;
}
button[data-variant="pills"] p { color: inherit !important; font-weight: 600 !important; }
button[data-variant="pills"]:hover { background: var(--mist-2) !important; }
button[data-variant="pills"][data-selected="true"] { background: var(--ink) !important; color: #fff !important; }
button[data-variant="pills"][data-selected="true"] p { color: #fff !important; }
button[data-variant="pills"]:focus-visible { border-color: var(--moss) !important; box-shadow: none !important; }

/* buttons */
.stButton > button { border-radius: var(--r-pill); font-weight: 600; padding: 10px 22px; border: none; background: var(--mist); color: var(--ink); }
.stButton > button:hover { background: var(--mist-2); color: var(--ink); border: none; }
.stButton > button[kind="primary"] { background: var(--moss); color: #fff; }
.stButton > button[kind="primary"]:hover { background: #0b5a3e; color: #fff; }
.stButton > button:focus-visible { outline: 3px solid var(--moss-tint); }

/* tabs: quiet underline */
[data-testid="stTabs"] [role="tablist"] { gap: 22px; border-bottom: 1px solid var(--mist-2); }
[data-testid="stTabs"] button[role="tab"] { font-size: 15px; font-weight: 600; color: var(--slate); padding: 10px 0; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--ink); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--ink); height: 3px; border-radius: 3px; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }

/* sidebar */
[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--mist); }
[data-testid="stSidebar"] h3 { font-size: 15px; font-weight: 800; margin-bottom: 4px; }

/* ---------- status line ---------- */
.status { display: flex; align-items: center; gap: 8px; font-size: 14px; color: var(--slate); margin: 0 0 6px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.dot.live { background: var(--moss); } .dot.mock { background: #e0a400; }

.note { background: var(--honey); color: var(--honey-ink); border-radius: 12px; padding: 10px 16px; font-size: 14px; margin: 4px 0 16px; }

/* ---------- result meta ---------- */
.run { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 18px 0 14px; }
.run .chip { background: var(--mist); border-radius: var(--r-pill); padding: 5px 12px; font-size: 13px; font-weight: 600; color: var(--ink); }
.run .chip.time { background: var(--moss-tint); color: var(--moss); }
.run .chip.slow { background: #fde3e0; color: #9b2217; }

/* ---------- masonry cards ---------- */
.masonry { column-count: 2; column-gap: 16px; }
@media (max-width: 760px) { .masonry { column-count: 1; } }
.card { break-inside: avoid; background: var(--mist); border-radius: var(--r-card); padding: 18px 20px 16px; margin: 0 0 16px; }
.card.top { background: var(--moss-tint); }
.card .head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 8px; }
.card .rank { font-size: 30px; font-weight: 800; letter-spacing: -1px; color: var(--ink); line-height: 1; }
.card.top .rank { color: var(--moss); }
.card .score { font-size: 14px; font-weight: 600; color: var(--slate); font-variant-numeric: tabular-nums; }
.card .text { font-size: 15px; line-height: 1.5; color: var(--ink); }
.card .foot { margin-top: 10px; font-size: 13px; color: var(--slate); display: flex; gap: 14px; }
.card .foot b { color: var(--ink); font-weight: 600; }

.stack .card { margin-bottom: 12px; padding: 14px 16px 12px; }
.stack .card .rank { font-size: 22px; }
.stack .card .text { font-size: 14px; }

/* answer */
.answer { background: var(--moss-tint); border-radius: var(--r-card); padding: 18px 22px; margin: 4px 0 18px; font-size: 16px; line-height: 1.55; color: var(--ink); }
.answer small { display: block; color: var(--moss); font-weight: 600; margin-bottom: 6px; font-size: 13px; }

/* column headings in compare */
.colhead { font-size: 15px; font-weight: 800; margin: 6px 0 2px; color: var(--ink); }
.colsub { font-size: 13px; color: var(--slate); margin-bottom: 10px; }

/* ---------- evaluation ---------- */
.stat { padding: 6px 0 14px; }
.stat .num { font-size: 40px; font-weight: 800; letter-spacing: -1.5px; line-height: 1; color: var(--ink); font-variant-numeric: tabular-nums; }
.stat .num.ok { color: var(--moss); } .stat .num.miss { color: #9b2217; }
.stat .lbl { font-size: 14px; color: var(--ink); font-weight: 600; margin-top: 6px; }
.stat .tgt { font-size: 13px; color: var(--slate); }
.sec { font-size: 20px; font-weight: 800; letter-spacing: -0.3px; margin: 22px 0 8px; color: var(--ink); }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
