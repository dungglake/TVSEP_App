"""Page 4 — Model Performance"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    load_model, load_top_features, load_raw_data,
    CLASS_NAMES, CLASS_COLORS, TARGET_COLUMN, check_artifacts, inject_global_css,
)

st.set_page_config(
    page_title="Performance · TVSEP",
    page_icon="📈",
    layout="wide",
)
inject_global_css()
check_artifacts()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#1a3a5c,#0d1b2a);
            border-left:6px solid #D85A30; border-radius:10px;
            padding:18px 24px; margin-bottom:24px;">
    <span style="font-size:28px; font-weight:800; color:#f5a07a;">📈 Model Performance</span>
    <div style="font-size:13px; color:#8aafc8; margin-top:4px;">
        Evaluation on a held-out 20% test set  (n = 440)
    </div>
</div>
""", unsafe_allow_html=True)

# ── EVALUATION ────────────────────────────────────────────────────────────────
@st.cache_data
def evaluate_on_test():
    model        = load_model()
    top_features = load_top_features()
    df, var_labels, _ = load_raw_data()

    LEAKAGE = ["v31313b","v31314a","v31314b","v31317","v31318",
               "v31319a","v31319b","v31320a","v31320b","v91005","v91006"]
    SUBJECTIVE = [c for c in df.columns if c.startswith("v93001") or c == "v93002"]
    ID_VARS = []
    for c in df.columns:
        if c in {"QID", "HID"}:
            ID_VARS.append(c)
        elif c.startswith("v1000") and len(c) <= 6:
            ID_VARS.append(c)
        elif c.startswith("v100") and c[4:].rstrip("ab").isdigit() \
                and 0 < int(c[4:].rstrip("ab")) < 25:
            ID_VARS.append(c)

    drop_vars = list(set(LEAKAGE + SUBJECTIVE + ID_VARS + [TARGET_COLUMN]))
    drop_vars = [c for c in drop_vars if c in df.columns]

    df_clean = df.dropna(subset=[TARGET_COLUMN]).copy()
    df_work  = df_clean.copy()
    for col in df_work.columns:
        if col != TARGET_COLUMN:
            df_work[col] = pd.to_numeric(df_work[col], errors="coerce")

    X_full = df_work.drop(columns=drop_vars)
    high_nan = X_full.isna().mean()[X_full.isna().mean() > 0.5].index
    X_full   = X_full.drop(columns=high_nan)
    constant = X_full.nunique()[X_full.nunique() <= 1].index
    X_full   = X_full.drop(columns=constant)

    y = 5 - df_work[TARGET_COLUMN].astype(int).values

    _, X_test_full, _, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test = X_test_full[top_features]

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return y_test, y_pred, y_proba

with st.spinner("Evaluating on test set…"):
    y_test, y_pred, y_proba = evaluate_on_test()

# ── HEADLINE METRICS ──────────────────────────────────────────────────────────
acc      = accuracy_score(y_test, y_pred)
f1_mac   = f1_score(y_test, y_pred, average="macro")
qwk      = cohen_kappa_score(y_test, y_pred, weights="quadratic")
baseline = pd.Series(y_test).value_counts(normalize=True).max()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy",   f"{acc:.3f}",
          delta=f"{(acc-baseline)*100:+.1f}pp vs baseline")
c2.metric("F1-macro",   f"{f1_mac:.3f}",
          help="Average F1 across all 5 classes")
c3.metric("QWK ⭐",      f"{qwk:.3f}",
          help="Quadratic Weighted Kappa — ordinal-aware. 1=perfect, 0=random.")
c4.metric("Baseline",   f"{baseline:.3f}",
          help="Accuracy from always predicting the majority class")

st.warning(f"""
**How to read these numbers:**

- Accuracy **{acc:.1%}** vs baseline **{baseline:.1%}** → model is only **{(acc-baseline)*100:.1f} pp** 
  better than always guessing 'Same'.
- **QWK = {qwk:.2f}** indicates *{'weak' if qwk < 0.2 else 'fair' if qwk < 0.4 else 'moderate'}* 
  ordinal agreement  (< 0.2 = poor · 0.2–0.4 = fair · 0.4–0.6 = moderate · > 0.6 = good).
- This is realistic — subjective well-being is hard to predict from objective economic 
  features alone. Report this honestly in the thesis as a finding, not a failure.
""")

st.markdown("---")

# ── CONFUSION MATRIX ──────────────────────────────────────────────────────────
st.markdown("### Confusion matrix")

cm      = confusion_matrix(y_test, y_pred, labels=list(range(5)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1b2a")
for ax in axes:
    ax.set_facecolor("#0d1b2a")

cmap_custom = sns.color_palette("Blues", as_cmap=True)
cmap_custom.set_bad("#1a2a3a")

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, linecolor="#1a2a3a",
            annot_kws={"color": "black", "fontsize": 11},
            ax=axes[0])
axes[0].set_title("Counts", color="black", fontsize=13)
axes[0].set_xlabel("Predicted", color="black")
axes[0].set_ylabel("Actual",    color="black")
axes[0].tick_params(colors="black", rotation=20)

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, linecolor="#1a2a3a",
            annot_kws={"color": "black", "fontsize": 11},
            ax=axes[1])
axes[1].set_title("Normalized (recall per class)", color="black", fontsize=13)
axes[1].set_xlabel("Predicted", color="black")
axes[1].set_ylabel("Actual",    color="black")
axes[1].tick_params(colors="black", rotation=20)

fig.patch.set_facecolor("#9dc9f8")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)

st.caption(
    "💡 **Read the normalized matrix row-by-row**: each row sums to 1.0 and shows "
    "what the model predicts for households of that *true* class. "
    "Mass near the diagonal = model captures ordinal structure."
)

st.markdown("---")

# ── PER-CLASS METRICS ─────────────────────────────────────────────────────────
st.markdown("### Per-class metrics")

report = classification_report(
    y_test, y_pred,
    labels=list(range(5)),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report).T.round(3)
report_df = report_df.loc[CLASS_NAMES + ["macro avg", "weighted avg"]]
st.dataframe(report_df, use_container_width=True)

st.caption(
    "**Precision** = of all predictions for this class, what fraction were correct? · "
    "**Recall** = of all actual examples of this class, what fraction did we catch?"
)

st.markdown("---")

# ── ROC CURVES ────────────────────────────────────────────────────────────────
st.markdown("### ROC curves  (one-vs-rest)")

y_test_bin = label_binarize(y_test, classes=list(range(5)))

fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d1b2a")
ax.set_facecolor("#0d1b2a")

for i, color, name in zip(range(5), CLASS_COLORS, CLASS_NAMES):
    if y_test_bin[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    auc_i = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC = {auc_i:.2f})")

ax.plot([0, 1], [0, 1], color="#ffffff33", lw=1, linestyle="--", label="Random")
ax.set_xlabel("False Positive Rate", color="#8aafc8")
ax.set_ylabel("True Positive Rate",  color="#8aafc8")
ax.set_title("ROC curves per class", color="#cce4f7", fontsize=13)
ax.legend(loc="lower right", fontsize=9, facecolor="#1a2a3a",
          edgecolor="#2a4060", labelcolor="#cce4f7")
ax.tick_params(colors="#8aafc8")
for spine in ax.spines.values():
    spine.set_edgecolor("#2a4060")
ax.grid(alpha=0.2, color="#8aafc8")
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
