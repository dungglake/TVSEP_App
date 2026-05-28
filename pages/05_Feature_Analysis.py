"""Page 5 — Feature Analysis"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    load_top_features, load_xgb_importance, load_raw_data,
    CLASS_INFO, CLASS_NAMES, CLASS_COLORS, TARGET_COLUMN,
    check_artifacts, inject_global_css, ACCENT_PRIMARY, ACCENT_SECONDARY,
)

st.set_page_config(
    page_title="Feature Analysis · TVSEP",
    page_icon="🔍",
    layout="wide",
)
inject_global_css()
check_artifacts()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#1a3a5c,#0d1b2a);
            border-left:6px solid #888780; border-radius:10px;
            padding:18px 24px; margin-bottom:24px;">
    <span style="font-size:28px; font-weight:800; color:#cce4f7;">🔍 Feature Analysis</span>
    <div style="font-size:13px; color:#8aafc8; margin-top:4px;">
        Understand which features the model relies on and how they relate to well-being
    </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD RESOURCES ────────────────────────────────────────────────────────────
top_features = load_top_features()
xgb_imp_df   = load_xgb_importance()
df, var_labels, _ = load_raw_data()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Importance ranking",
    "📦 Distribution by class",
    "🔗 Correlation matrix",
    "🔎 Feature search",
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — FEATURE IMPORTANCE BAR CHART
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Feature importance (final XGBoost model)")

    n_show = st.slider(
        "How many top features to display?",
        min_value=10, max_value=min(50, len(xgb_imp_df)),
        value=20, step=5,
    )

    top_n = xgb_imp_df.head(n_show).copy()
    top_n["DisplayName"] = top_n.apply(
        lambda r: f"{r['Feature']}  —  {str(r['Label'])[:40]}", axis=1
    )

    fig, ax = plt.subplots(figsize=(9, max(4, n_show * 0.35)), facecolor="#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    colors_bar = [
        ACCENT_SECONDARY if i < 3 else ACCENT_PRIMARY if i < 10 else "#5a7fa0"
        for i in range(n_show)
    ]
    bars = ax.barh(range(n_show), top_n["Importance"].values[::-1],
                   color=colors_bar[::-1], edgecolor="#0d1b2a", height=0.7)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(top_n["DisplayName"].values[::-1], fontsize=8, color="#cce4f7")
    ax.set_xlabel("XGBoost feature importance", color="#8aafc8")
    ax.set_title(f"Top {n_show} features by importance", color="#cce4f7", fontsize=13)
    ax.tick_params(colors="#8aafc8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a4060")
    ax.xaxis.grid(True, color="#2a4060", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("""
    <div style="background:#1a2a3a; border:1px solid #2a4060; border-radius:8px; padding:12px 16px; font-size:12px; color:#8aafc8;">
        🟢 <b style="color:#4ecfa0;">Dark green</b> = top 3 features · 
        🔵 <b style="color:#7ac4f7;">Blue</b> = top 4–10 · 
        ⚫ <b style="color:#8aafc8;">Gray-blue</b> = features 11–50
    </div>
    """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — DISTRIBUTION BY CLASS
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Feature distribution by well-being class")
    st.caption("Pick a feature to see how its distribution differs across the 5 classes")

    selected_feature = st.selectbox(
        "Choose a feature",
        options=top_features,
        format_func=lambda f: f"{f}  —  {var_labels.get(f, '')[:60]}",
    )

    df_clean = df.dropna(subset=[TARGET_COLUMN, selected_feature]).copy()
    df_clean["_class_id_"]    = 5 - df_clean[TARGET_COLUMN].astype(int)
    df_clean["_class_label_"] = df_clean["_class_id_"].map(
        {i: CLASS_NAMES[i] for i in range(5)}
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), facecolor="#0d1b2a")
    for ax in axes:
        ax.set_facecolor("#0d1b2a")

    # Boxplot
    sns.boxplot(
        data=df_clean, x="_class_label_", y=selected_feature,
        order=CLASS_NAMES, hue="_class_label_", palette=CLASS_COLORS,
        legend=False, ax=axes[0],
        boxprops=dict(edgecolor="#2a4060"),
        medianprops=dict(color="#ffffff", linewidth=2),
        whiskerprops=dict(color="#8aafc8"),
        capprops=dict(color="#8aafc8"),
        flierprops=dict(marker="o", markerfacecolor="#8aafc8", markersize=3, alpha=0.5),
    )
    axes[0].set_title(f"Boxplot of {selected_feature} by class", color="#cce4f7", fontsize=12)
    axes[0].set_xlabel("", color="#8aafc8")
    axes[0].set_ylabel(selected_feature, color="#8aafc8")
    axes[0].tick_params(colors="#8aafc8", axis="x", rotation=20)
    axes[0].tick_params(colors="#8aafc8", axis="y")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#2a4060")
    axes[0].yaxis.grid(True, color="#2a4060", linestyle="--", alpha=0.5)

    # Mean line plot
    class_means = df_clean.groupby("_class_id_")[selected_feature].mean().reindex(range(5))
    axes[1].fill_between(range(5), class_means.values, alpha=0.15, color=ACCENT_PRIMARY)
    axes[1].plot(range(5), class_means.values, marker="o", color=ACCENT_PRIMARY,
                 lw=2.5, markersize=8, markerfacecolor="#0d1b2a",
                 markeredgecolor=ACCENT_PRIMARY, markeredgewidth=2.5)
    for i, (x, y) in enumerate(zip(range(5), class_means.values)):
        if not np.isnan(y):
            axes[1].annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                             xytext=(0, 10), ha="center", fontsize=9,
                             color=CLASS_COLORS[i], fontweight="600")
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(CLASS_NAMES, rotation=20, color="#8aafc8", fontsize=9)
    axes[1].set_title(f"Mean of {selected_feature} per class",
                      color="#cce4f7", fontsize=12)
    axes[1].set_ylabel(selected_feature, color="#8aafc8")
    axes[1].tick_params(colors="#8aafc8")
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#2a4060")
    axes[1].yaxis.grid(True, color="#2a4060", linestyle="--", alpha=0.5)
    axes[1].set_axisbelow(True)

    fig.patch.set_facecolor("#0d1b2a")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.caption(
        f"📖 **{selected_feature}**: {var_labels.get(selected_feature, 'N/A')}. "
        "If means trend monotonically across classes, the feature carries ordinal signal."
    )

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — CORRELATION HEATMAP
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Correlation matrix — top 15 features + target")

    n_corr  = 15
    df_work = df.dropna(subset=[TARGET_COLUMN]).copy()
    for c in top_features[:n_corr]:
        df_work[c] = pd.to_numeric(df_work[c], errors="coerce")
    df_work["target"] = 5 - df_work[TARGET_COLUMN].astype(int)

    corr_data   = df_work[top_features[:n_corr] + ["target"]]
    corr_matrix = corr_data.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 9), facecolor="#0d1b2a")
    ax.set_facecolor("#0d1b2a")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, mask=mask,
        linewidths=0.5, linecolor="#0d1b2a",
        cbar_kws={"shrink": 0.7}, ax=ax,
        annot_kws={"fontsize": 8, "color": "#000000"},
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="#ffffff")
    cbar.outline.set_edgecolor("#ffffff")
    ax.set_title(f"Correlation: top {n_corr} features + target",
                 color="#cce4f7", fontsize=13)
    ax.tick_params(colors="#8aafc8", labelsize=8)
    fig.patch.set_facecolor("#0d1b2a")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # High-correlation warning
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.7:
                high_corr_pairs.append({
                    "Feature A": corr_matrix.columns[i],
                    "Feature B": corr_matrix.columns[j],
                    "r": round(r, 3),
                })

    if high_corr_pairs:
        st.warning(
            f"⚠️ Found **{len(high_corr_pairs)}** highly-correlated pairs (|r| > 0.7) — "
            "these may indicate multicollinearity. "
            "XGBoost handles this better than linear models, but worth noting in the thesis."
        )
        st.dataframe(pd.DataFrame(high_corr_pairs),
                     use_container_width=True, hide_index=True)
    else:
        st.success("✓ No feature pairs with |r| > 0.7 — multicollinearity looks acceptable.")

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — SEARCHABLE FEATURE TABLE
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### All 50 features — searchable")

    table = xgb_imp_df.copy()
    table["Importance"] = table["Importance"].round(4)
    table["Rank"] = range(1, len(table) + 1)

    search_query = st.text_input(
        "🔎 Search by feature name or description",
        placeholder="e.g. 'fertilizer'  or  'v81045'",
    )

    if search_query:
        mask = (
            table["Feature"].str.contains(search_query, case=False, na=False)
            | table["Label"].str.contains(search_query, case=False, na=False)
        )
        filtered = table[mask]
        st.write(f"Found **{len(filtered)}** matches")
    else:
        filtered = table

    st.dataframe(filtered[["Rank", "Feature", "Importance", "Label"]],
                 use_container_width=True, hide_index=True)
