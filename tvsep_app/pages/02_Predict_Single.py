"""Page 2 — Predict Single Household"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    load_model, load_top_features, load_raw_data,
    CLASS_INFO, TARGET_COLUMN, check_artifacts, inject_global_css,
    render_class_badge, render_probability_bars,
)

st.set_page_config(
    page_title="Predict Single · TVSEP",
    page_icon="🎯",
    layout="wide",
)
inject_global_css()
check_artifacts()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#1a3a5c,#0d1b2a);
            border-left:6px solid #0F6E56; border-radius:10px;
            padding:18px 24px; margin-bottom:24px;">
    <span style="font-size:28px; font-weight:800; color:#4ecfa0;">🎯 Predict Single Household</span>
    <div style="font-size:13px; color:#8aafc8; margin-top:4px;">
        Enter household features → get probabilities for each of 5 well-being classes
    </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD RESOURCES ────────────────────────────────────────────────────────────
model        = load_model()
top_features = load_top_features()
df, var_labels, _ = load_raw_data()
df_no_target = df.drop(columns=[TARGET_COLUMN], errors="ignore")

# ── FEATURE GROUPING ──────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "💰 Household Expenditure (v81*)":
        [f for f in top_features if f.startswith("v81")],
    "🎯 Aspirations (v3141*, v31328)":
        [f for f in top_features if f.startswith("v3141") or f == "v31328"],
    "🌾 Agriculture (v42*)":
        [f for f in top_features if f.startswith("v42")],
    "🏠 Household Characteristics (v92*, v82*)":
        [f for f in top_features if f.startswith("v92") or f.startswith("v82")],
    "🧠 Personality / Risk (v100*, v31*)":
        [f for f in top_features if f.startswith("v100") or
         (f.startswith("v31") and not f.startswith("v3141") and f != "v31328")],
}
already_grouped = set().union(*FEATURE_GROUPS.values())
FEATURE_GROUPS["📦 Other"] = [f for f in top_features if f not in already_grouped]

# ── QUICK FILL ────────────────────────────────────────────────────────────────
top_row = st.columns([1, 1, 3])
with top_row[0]:
    if st.button("🎲 Random sample", help="Fill with a real household's values"):
        candidates = df_no_target.dropna(subset=top_features)
        if len(candidates) > 0:
            sample = candidates.sample(1).iloc[0]
            for f in top_features:
                st.session_state[f"input_{f}"] = float(sample[f])
            st.rerun()
with top_row[1]:
    if st.button("🔄 Reset to median", help="Fill with median of each feature"):
        for f in top_features:
            st.session_state[f"input_{f}"] = float(df_no_target[f].median())
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ── 2-COLUMN LAYOUT ───────────────────────────────────────────────────────────
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    st.markdown(f"#### Input features  *({len(top_features)} total, grouped by topic)*")

    inputs = {}
    for group_name, group_features in FEATURE_GROUPS.items():
        if not group_features:
            continue
        is_open = group_name.startswith(("💰", "🎯"))
        with st.expander(f"{group_name}  ({len(group_features)} features)", expanded=is_open):
            for f in group_features:
                label = var_labels.get(f, "")[:80]
                default_value = (
                    float(df_no_target[f].median())
                    if f in df_no_target.columns and not df_no_target[f].isna().all()
                    else 0.0
                )
                value = st.number_input(
                    f"{f} — {label}",
                    value=st.session_state.get(f"input_{f}", default_value),
                    key=f"input_{f}",
                    format="%.2f",
                )
                inputs[f] = value

    predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=True)

# ── RESULT COLUMN ─────────────────────────────────────────────────────────────
with col_result:
    st.markdown("#### Prediction")

    if not predict_clicked:
        st.markdown("""
        <div style="background:#1a2a3a; border:1px dashed #378ADD55;
                    border-radius:12px; padding:32px; text-align:center; color:#8aafc8;">
            <div style="font-size:48px; margin-bottom:12px;">🔮</div>
            <div style="font-size:14px;">Fill in the household features<br>
            (or click <b style="color:#7ac4f7;">Random sample</b>),<br>
            then press <b style="color:#7ac4f7;">Predict</b>.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        X_input      = pd.DataFrame([inputs])[top_features]
        probabilities = model.predict_proba(X_input)[0]
        predicted_class = int(probabilities.argmax())
        confidence   = float(probabilities[predicted_class])

        st.markdown(render_class_badge(predicted_class, confidence), unsafe_allow_html=True)

        st.markdown("##### Probability for each class")
        st.markdown(render_probability_bars(probabilities), unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("##### Top features by global importance")
        st.caption(
            "Features the model relies on most overall · value shown is what you entered."
        )
        xgb_clf = model.named_steps["model"]
        contribution_df = pd.DataFrame({
            "Feature":     top_features,
            "Importance":  xgb_clf.feature_importances_,
            "Your value":  [inputs[f] for f in top_features],
            "Description": [var_labels.get(f, "")[:50] for f in top_features],
        }).sort_values("Importance", ascending=False).head(10).reset_index(drop=True)
        st.dataframe(contribution_df, use_container_width=True, hide_index=True)
