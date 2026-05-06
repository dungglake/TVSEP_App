"""
==============================================================================
TVSEP Household Well-being Classifier  (5-class)
==============================================================================
Entry point — sidebar navigation + landing page.

Run:  streamlit run app.py
==============================================================================
"""

import streamlit as st
from pathlib import Path

# ── MUST be first Streamlit call ─────────────────────────────────────────────
st.set_page_config(
    page_title="TVSEP Well-being Classifier",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import inject_global_css, CLASS_INFO, CLASS_COLORS, ACCENT_PRIMARY, ACCENT_SECONDARY

inject_global_css()

# ── HERO SECTION ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2a3a 60%, #0F2027 100%);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    border: 1px solid #2a4060;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
">
    <div style="font-size: 42px; font-weight: 800; 
                background: linear-gradient(90deg, #378ADD, #0F6E56);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text; margin-bottom: 8px;">
        🏡 TVSEP Well-being Classifier
    </div>
    <div style="font-size: 16px; color: #8aafc8; margin-bottom: 24px;">
        5-class subjective well-being prediction &nbsp;·&nbsp; Thailand-Vietnam Socio-Economic Panel 2024
    </div>
    <div style="display: flex; gap: 12px; flex-wrap: wrap;">
        <span style="background:#378ADD22; border:1px solid #378ADD55; color:#7ac4f7;
                     padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600;">
            XGBoost + SMOTE
        </span>
        <span style="background:#0F6E5622; border:1px solid #0F6E5655; color:#4ecfa0;
                     padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600;">
            50 RF-selected features
        </span>
        <span style="background:#D85A3022; border:1px solid #D85A3055; color:#f5a07a;
                     padding:6px 14px; border-radius:20px; font-size:13px; font-weight:600;">
            Random seed 42
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── CLASS TABLE ───────────────────────────────────────────────────────────────
st.markdown("### About this application")

cols = st.columns(5)
for col, info in zip(cols, CLASS_INFO):
    with col:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {info['color']}22, {info['color']}08);
                    border: 1px solid {info['color']}44; border-top: 4px solid {info['color']};
                    border-radius: 10px; padding: 14px 12px; text-align: center;">
            <div style="font-size: 22px; font-weight: 800; color: {info['color']};">{info['id']}</div>
            <div style="font-size: 12px; font-weight: 700; color: {info['color']}; margin: 4px 0;">{info['name']}</div>
            <div style="font-size: 10px; color: #8aafc8;">{info['description']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
The model is **XGBoost + SMOTE** trained on **50 numerical features** selected
by Random Forest importance (after removing data-leakage variables).
""")

# ── NAVIGATION CARDS ─────────────────────────────────────────────────────────
st.markdown("### Navigate using the sidebar")

nav_items = [
    ("🏠", "Overview",          "Dataset summary, class distribution, data quality",           "#378ADD"),
    ("🎯", "Predict Single",    "Input one household → get a 5-class probability vector",      "#0F6E56"),
    ("📊", "Predict Batch",     "Upload CSV/DTA → predictions for many households",            "#639922"),
    ("📈", "Performance",       "Accuracy, F1, QWK, confusion matrix, ROC curves",             "#D85A30"),
    ("🔍", "Feature Analysis",  "Feature importance, distributions, correlation matrix",       "#888780"),
]

cols = st.columns(len(nav_items))
for col, (icon, title, desc, color) in zip(cols, nav_items):
    with col:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}18, {color}08);
                    border: 1px solid {color}44; border-radius: 12px;
                    padding: 18px 14px; text-align: center; height: 130px;">
            <div style="font-size: 28px;">{icon}</div>
            <div style="font-size: 13px; font-weight: 700; color: {color}; margin: 6px 0;">{title}</div>
            <div style="font-size: 11px; color: #8aafc8; line-height: 1.4;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── CAVEATS ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.info("""
**📌 Important caveats for thesis interpretation**

- Subjective well-being is **inherently hard to predict** from objective economic features.  
  Expected accuracy is ~52–55% (only ~2 pp above the majority-class baseline).
- Use **Quadratic Weighted Kappa (QWK)** as the primary metric — it is ordinal-aware
  (penalizes "much wrong" predictions more than "slightly wrong" ones).
- The two extreme classes (*Much worse* with 23 hh, *Much better* with 37 hh) are
  essentially impossible to learn well due to extreme class imbalance.
""")

# ── ARTIFACT STATUS ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
REQUIRED_FILES = [
    BASE_DIR / "tvsep_model_5class.pkl",
    BASE_DIR / "tvsep_top_features_5class.pkl",
    BASE_DIR / "tvsep_class_names_5class.pkl",
    BASE_DIR / "tvsep_feature_importance_5class.csv",
    BASE_DIR / "tvsep_xgb_importance_5class.csv",
    BASE_DIR / "data" / "TVSEP2024_HHQ_V3VN.dta",
]
missing = [f for f in REQUIRED_FILES if not f.exists()]

st.markdown("---")
st.subheader("System status")
if missing:
    st.error(
        "**Some required files are missing:**\n\n"
        + "\n".join(f"- `{f.name}`" for f in missing)
        + "\n\nRun `Train5class.ipynb` first."
    )
else:
    st.success("✅ All artifacts loaded — use the sidebar to navigate.")

# ── SIDEBAR FOOTER ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px; color:#8aafc8; line-height: 1.8;">
        <b style="color:#cce4f7;">Model</b> · XGBoost + SMOTE<br>
        <b style="color:#cce4f7;">Features</b> · 50 (RF-selected)<br>
        <b style="color:#cce4f7;">Train rows</b> · 1,758<br>
        <b style="color:#cce4f7;">Test rows</b> · 440<br>
        <b style="color:#cce4f7;">Random seed</b> · 42
    </div>
    """, unsafe_allow_html=True)
