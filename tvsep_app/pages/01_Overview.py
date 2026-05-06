"""Page 1 — Dataset Overview"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    load_raw_data, load_top_features,
    CLASS_INFO, CLASS_COLORS, CLASS_NAMES,
    TARGET_COLUMN, check_artifacts, inject_global_css,
    ACCENT_PRIMARY,
)

st.set_page_config(
    page_title="Overview · TVSEP",
    page_icon="🏠",
    layout="wide",
)
inject_global_css()
check_artifacts()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#1a3a5c,#0d1b2a);
            border-left:6px solid #378ADD; border-radius:10px;
            padding:18px 24px; margin-bottom:24px;">
    <span style="font-size:28px; font-weight:800; color:#7ac4f7;">🏠 Dataset Overview</span>
    <div style="font-size:13px; color:#8aafc8; margin-top:4px;">
        TVSEP 2024 — Vietnam Household Questionnaire
    </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df, var_labels, value_labels = load_raw_data()
top_features = load_top_features()
df_clean = df.dropna(subset=[TARGET_COLUMN])

# ── KEY METRICS ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total households",    f"{len(df):,}")
c2.metric("With valid target",   f"{len(df_clean):,}",
          delta=f"-{len(df)-len(df_clean)}",
          help="Households dropped due to missing v31313a")
c3.metric("Features used",       len(top_features),
          help="After removing leakage, ID, and high-NaN variables")
c4.metric("Target classes",      "5  (ordinal)")

st.markdown("<br>", unsafe_allow_html=True)

# ── TARGET DISTRIBUTION ───────────────────────────────────────────────────────
st.markdown("### Target distribution: `v31313a`")
st.caption(f"Question: {var_labels.get(TARGET_COLUMN, '')}")

target_counts   = df_clean[TARGET_COLUMN].value_counts().sort_index(ascending=False)
counts_ordered  = [int(target_counts.get(5 - i, 0)) for i in range(5)]

fig, ax = plt.subplots(figsize=(11, 4.5), facecolor="#0d1b2a")
ax.set_facecolor("#0d1b2a")

bars = ax.bar(CLASS_NAMES, counts_ordered, color=CLASS_COLORS,
              edgecolor="#0d1b2a", linewidth=1.5, width=0.6)

for bar, count in zip(bars, counts_ordered):
    pct = count / sum(counts_ordered) * 100
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts_ordered) * 0.02,
            f"{int(count)}\n({pct:.1f}%)",
            ha="center", fontsize=9, color="#cce4f7", fontweight="600")

ax.set_ylabel("Number of households", color="#8aafc8")
ax.set_title("HH well-being vs last year  (worst → best)", color="#cce4f7", fontsize=13, pad=12)
ax.set_ylim(0, max(counts_ordered) * 1.18)
ax.tick_params(colors="#8aafc8")
for spine in ax.spines.values():
    spine.set_edgecolor("#2a4060")
ax.yaxis.grid(True, color="#2a4060", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)

# ── IMBALANCE WARNING ─────────────────────────────────────────────────────────
ratio = max(counts_ordered) / min(counts_ordered)
st.warning(
    f"**Class imbalance ratio: {ratio:.0f}×** — the majority class "
    f"('Same', {max(counts_ordered)} hh) is {ratio:.0f}× larger than "
    f"the minority class ('Much worse off', {min(counts_ordered)} hh). "
    "This makes the two extreme classes very hard to learn even with SMOTE."
)

st.markdown("---")

# ── DATA CLEANING TRANSPARENCY ────────────────────────────────────────────────
st.markdown("### Data cleaning summary")
st.markdown(
    "The original dataset has **969 columns**. After cleaning we keep **50 features**. "
    "Below is what was removed and why — critical for thesis methodology."
)

col_left, col_right = st.columns([3, 2])

with col_left:
    with st.expander("🚨 11 leakage variables removed  *(most critical)*", expanded=True):
        st.markdown(
            "These variables are **conceptually identical** to the target and would "
            "artificially inflate accuracy if included."
        )
        leakage_vars = [
            ("v31313b", "Better off than last year? (person) — near-copy of target"),
            ("v31314a", "Better off than 5 years ago? (HH)"),
            ("v31314b", "Better off than 5 years ago? (person)"),
            ("v31317",  "Best year in last 5 years"),
            ("v31318",  "Worst year in last 5 years"),
            ("v31319a", "Will HH be better off next year?"),
            ("v31319b", "Will person be better off next year?"),
            ("v31320a", "Will HH be better off in 5 years?"),
            ("v31320b", "Will person be better off in 5 years?"),
            ("v91005",  "Self-comparison with other HH (village)"),
            ("v91006",  "Self-comparison with other HH (country)"),
        ]
        for var, desc in leakage_vars:
            st.markdown(f"- **`{var}`** — {desc}")

    with st.expander("🟡 13 subjective satisfaction variables removed  *(debatable)*"):
        st.markdown(
            "Variables `v93001a-l` and `v93002` measure satisfaction with various life "
            "domains. Highly correlated with subjective well-being but different constructs. "
            "Excluded to be conservative; one could argue to keep them for a different framing."
        )

with col_right:
    with st.expander("⚫ ~37 ID variables removed"):
        st.markdown(
            "`QID`, `HID`, `v10001–v10009`, `v100xxx` — administrative IDs with no "
            "predictive value. Can cause overfitting (model memorising village IDs)."
        )

    with st.expander("🟠 ~310 high-missingness variables removed"):
        st.markdown(
            "Variables with **>50% missing values** are too unreliable. "
            "Median imputation would mostly produce the same value for half the dataset, "
            "biasing the model."
        )

    # Mini class-count color legend
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Class sample sizes**")
    for info, cnt in zip(CLASS_INFO, counts_ordered):
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:8px; margin:4px 0;'>"
            f"<div style='width:14px; height:14px; border-radius:3px; background:{info['color']};'></div>"
            f"<span style='font-size:12px; color:#cce4f7;'>{info['name']}</span>"
            f"<span style='margin-left:auto; font-size:12px; color:#8aafc8;'>{cnt:,}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
