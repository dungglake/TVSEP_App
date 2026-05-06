"""
Shared utilities used across all pages.

Centralizes:
- Loading model artifacts (cached)
- Loading raw dataset (cached)
- Class metadata (names, colors)
- Common helper functions & styled HTML components
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "TVSEP2024_HHQ_V3VN.dta"

# ---------------------------------------------------------------------------
# CLASS METADATA
# After REVERSE encoding (y = 5 - stata_code): 0 = worst, 4 = best.
# ---------------------------------------------------------------------------
CLASS_INFO = [
    {"id": 0, "name": "Much worse off",  "color": "#A32D2D", "bg": "#A32D2D22",
     "description": "Substantial decline vs last year"},
    {"id": 1, "name": "Worse off",       "color": "#D85A30", "bg": "#D85A3022",
     "description": "Some decline vs last year"},
    {"id": 2, "name": "Same",            "color": "#888780", "bg": "#88878022",
     "description": "About the same as last year"},
    {"id": 3, "name": "Better off",      "color": "#639922", "bg": "#63992222",
     "description": "Some improvement vs last year"},
    {"id": 4, "name": "Much better off", "color": "#0F6E56", "bg": "#0F6E5622",
     "description": "Substantial improvement vs last year"},
]

CLASS_NAMES   = [c["name"]  for c in CLASS_INFO]
CLASS_COLORS  = [c["color"] for c in CLASS_INFO]
CLASS_BGS     = [c["bg"]    for c in CLASS_INFO]
TARGET_COLUMN = "v31313a"

# Global accent palette (used in charts and UI accents)
ACCENT_PRIMARY   = "#378ADD"
ACCENT_SECONDARY = "#0F6E56"
ACCENT_WARN      = "#D85A30"

# ---------------------------------------------------------------------------
# GLOBAL CSS INJECTION
# Call inject_global_css() once per page (at the top after set_page_config).
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
/* ── Sidebar gradient ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
}
[data-testid="stSidebar"] * { color: #e8edf0 !important; }
[data-testid="stSidebar"] hr { border-color: #ffffff33; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a2a3a 0%, #0d1b2a 100%);
    border-radius: 12px;
    padding: 16px !important;
    border: 1px solid #2a4060;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}
[data-testid="stMetricLabel"]  { color: #8aafc8 !important; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"]  { color: #e8f4fd !important; font-weight: 700; }
[data-testid="stMetricDelta"]  { font-size: 12px; }

/* ── Expander header ── */
[data-testid="stExpander"] summary {
    background: linear-gradient(90deg, #1a3a5c 0%, #1a2a3a 100%);
    border-radius: 8px;
    padding: 10px 16px;
    color: #cce4f7 !important;
    font-weight: 600;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600;
    color: #7aabcc;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #378ADD;
    border-bottom: 3px solid #378ADD;
}

/* ── Page title ── */
h1 { 
    background: linear-gradient(90deg, #378ADD, #0F6E56);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
h2, h3 { color: #cce4f7; }

/* ── Info / warning / error boxes ── */
[data-testid="stAlert"] { border-radius: 10px; }

/* ── Buttons ── */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #378ADD 0%, #0F6E56 100%);
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: opacity 0.2s;
}
[data-testid="stButton"] button[kind="primary"]:hover { opacity: 0.88; }
</style>
"""

def inject_global_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ARTIFACT LOADING
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "tvsep_model_5class.pkl")


@st.cache_resource
def load_top_features():
    return joblib.load(BASE_DIR / "tvsep_top_features_5class.pkl")


@st.cache_data
def load_raw_data():
    df = pd.read_stata(DATA_PATH, convert_categoricals=False)
    with pd.io.stata.StataReader(DATA_PATH) as reader:
        var_labels   = reader.variable_labels()
        value_labels = reader.value_labels()
    return df, var_labels, value_labels


@st.cache_data
def load_feature_importance():
    return pd.read_csv(BASE_DIR / "tvsep_feature_importance_5class.csv")


@st.cache_data
def load_xgb_importance():
    return pd.read_csv(BASE_DIR / "tvsep_xgb_importance_5class.csv")


# ---------------------------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------------------------
def render_class_badge(class_id: int, confidence: float) -> str:
    info = CLASS_INFO[class_id]
    return f"""
    <div style="background: linear-gradient(135deg, {info['bg']}, {info['color']}15);
                border-left: 6px solid {info['color']};
                padding: 20px 24px; border-radius: 12px; margin-bottom: 20px;
                box-shadow: 0 4px 16px {info['color']}33;">
        <div style="font-size: 11px; color: #8aafc8; text-transform: uppercase;
                    letter-spacing: 1.5px; margin-bottom: 4px;">✦ Predicted class</div>
        <div style="font-size: 30px; font-weight: 700; color: {info['color']};
                    margin: 4px 0; letter-spacing: -0.5px;">{info['name']}</div>
        <div style="font-size: 13px; color: #94aec0; margin-top: 4px;">
            {info['description']}
            &nbsp;·&nbsp;
            <span style="color:{info['color']}; font-weight:600;">{confidence*100:.0f}% confidence</span>
        </div>
    </div>
    """


def render_probability_bars(probabilities) -> str:
    pred = int(probabilities.argmax())
    rows = []
    for i, p in enumerate(probabilities):
        info  = CLASS_INFO[i]
        pct   = p * 100
        is_pred = (i == pred)
        bold  = "font-weight:700;" if is_pred else "opacity:0.80;"
        outline = f"outline: 2px solid {info['color']}; outline-offset:1px;" if is_pred else ""
        rows.append(f"""
        <div style="display:flex; align-items:center; gap:10px; margin:8px 0;">
          <div style="width:140px; font-size:13px; color:#1a3a5c; {bold}">{info['name']}</div>
          <div style="flex:1; background:#dde8f0; height:16px; border-radius:4px;
                      overflow:hidden; {outline}">
            <div style="background: linear-gradient(90deg, {info['color']}cc, {info['color']});
                        height:16px; width:{pct}%; border-radius:4px;"></div>
          </div>
          <div style="width:50px; text-align:right; font-size:13px; color:#1a3a5c; {bold}">{pct:.1f}%</div>
        </div>
        """)
    return "\n".join(rows)


def check_artifacts() -> bool:
    required = [
        BASE_DIR / "tvsep_model_5class.pkl",
        BASE_DIR / "tvsep_top_features_5class.pkl",
        DATA_PATH,
    ]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        st.error(
            "**Missing required files:** "
            + ", ".join(f"`{f.name}`" for f in missing)
            + "\n\nRun `Train5class.ipynb` first."
        )
        st.stop()
    return True
