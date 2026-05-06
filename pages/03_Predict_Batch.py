"""Page 3 — Batch Prediction"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    load_model, load_top_features,
    CLASS_INFO, CLASS_NAMES, CLASS_COLORS, check_artifacts, inject_global_css,
)

st.set_page_config(
    page_title="Batch Prediction · TVSEP",
    page_icon="📊",
    layout="wide",
)
inject_global_css()
check_artifacts()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#1a3a5c,#0d1b2a);
            border-left:6px solid #639922; border-radius:10px;
            padding:18px 24px; margin-bottom:24px;">
    <span style="font-size:28px; font-weight:800; color:#a8d85a;">📊 Batch Prediction</span>
    <div style="font-size:13px; color:#8aafc8; margin-top:4px;">
        Upload a file with multiple households to predict them all at once
    </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD RESOURCES ────────────────────────────────────────────────────────────
model        = load_model()
top_features = load_top_features()

# ── STEP 1 — FILE UPLOAD ─────────────────────────────────────────────────────
st.markdown("""
<div style="background:#1a3a5c22; border:1px solid #378ADD44;
            border-radius:10px; padding:16px 20px; margin-bottom:16px;">
    <b style="color:#7ac4f7; font-size:15px;">Step 1 — Upload file</b>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop a .csv or .dta file",
    type=["csv", "dta"],
    help=f"File must contain these {len(top_features)} columns: "
         + ", ".join(top_features[:5]) + ", ...",
)

with st.expander(f"📋 Required column names  ({len(top_features)})"):
    st.code(", ".join(top_features), language="text")

if uploaded_file is None:
    st.info("👆 Upload a file to begin.")
    st.stop()

# ── STEP 2 — PARSE ────────────────────────────────────────────────────────────
try:
    if uploaded_file.name.endswith(".csv"):
        df_input = pd.read_csv(uploaded_file)
    else:
        df_input = pd.read_stata(uploaded_file, convert_categoricals=False)
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()

st.success(f"✅ Loaded **{len(df_input):,}** households with **{df_input.shape[1]}** columns")

# ── STEP 2 — COLUMN VALIDATION ────────────────────────────────────────────────
st.markdown("""
<div style="background:#1a3a5c22; border:1px solid #378ADD44;
            border-radius:10px; padding:16px 20px; margin: 16px 0;">
    <b style="color:#7ac4f7; font-size:15px;">Step 2 — Column validation</b>
</div>
""", unsafe_allow_html=True)

missing_cols = sorted(set(top_features) - set(df_input.columns))
extra_cols   = sorted(set(df_input.columns) - set(top_features))

c1, c2 = st.columns(2)
c1.metric("Required columns present",
          f"{len(top_features) - len(missing_cols)} / {len(top_features)}")
c2.metric("Extra columns (will be ignored)", len(extra_cols))

if missing_cols:
    st.error(
        f"❌ Missing **{len(missing_cols)}** required columns:\n\n"
        + ", ".join(f"`{c}`" for c in missing_cols[:30])
        + ("..." if len(missing_cols) > 30 else "")
    )
    st.stop()

st.success("✅ All required columns are present")

# ── STEP 3 — PREDICT ─────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#1a3a5c22; border:1px solid #378ADD44;
            border-radius:10px; padding:16px 20px; margin: 16px 0;">
    <b style="color:#7ac4f7; font-size:15px;">Step 3 — Run predictions</b>
</div>
""", unsafe_allow_html=True)

if not st.button("🚀 Predict all", type="primary"):
    st.stop()

X = df_input[top_features]
with st.spinner(f"Predicting {len(X):,} households..."):
    predictions   = model.predict(X)
    probabilities = model.predict_proba(X)

df_output = df_input.copy()
df_output["predicted_class_id"]    = predictions
df_output["predicted_class_label"] = [CLASS_NAMES[p] for p in predictions]
df_output["confidence"]            = probabilities.max(axis=1).round(3)
for i, name in enumerate(CLASS_NAMES):
    df_output[f"prob_{name.replace(' ','_')}"] = probabilities[:, i].round(3)

# ── RESULTS ───────────────────────────────────────────────────────────────────
st.markdown("### Results")

c1, c2, c3 = st.columns(3)
c1.metric("Predictions made",    len(predictions))
c2.metric("Average confidence",  f"{probabilities.max(axis=1).mean()*100:.1f}%")
c3.metric("Most predicted class",
          CLASS_NAMES[pd.Series(predictions).mode()[0]])

# Distribution chart
pred_counts    = pd.Series(predictions).value_counts().sort_index()
display_counts = [pred_counts.get(i, 0) for i in range(5)]

fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#0d1b2a")
ax.set_facecolor("#0d1b2a")
bars = ax.bar(CLASS_NAMES, display_counts, color=CLASS_COLORS,
              edgecolor="#0d1b2a", linewidth=1.5, width=0.6)
for bar, count in zip(bars, display_counts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, str(count),
                ha="center", fontsize=9, color="#cce4f7", fontweight="600")
ax.set_ylabel("Predicted households", color="#8aafc8")
ax.set_title("Distribution of predicted classes", color="#cce4f7", fontsize=13)
ax.tick_params(colors="#8aafc8")
for spine in ax.spines.values():
    spine.set_edgecolor("#2a4060")
ax.yaxis.grid(True, color="#2a4060", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# Results table
preview_cols = ["predicted_class_label", "confidence"] + \
               [f"prob_{n.replace(' ','_')}" for n in CLASS_NAMES]
st.dataframe(df_output[preview_cols], use_container_width=True)

csv_data = df_output.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download full results as CSV",
    csv_data,
    file_name=f"tvsep_predictions_{len(predictions)}households.csv",
    mime="text/csv",
)
