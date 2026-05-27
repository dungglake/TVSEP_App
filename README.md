    # TVSEP Household Well-being Classifier (5-class)

A Streamlit web application for predicting Vietnamese household subjective
well-being using the TVSEP 2024 dataset.

## Folder structure

```
tvsep_app/
├── app.py                          # Entry point (sidebar navigation + landing)
├── utils.py                        # Shared utilities (data loading, constants, CSS)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── Train5class.ipynb               # Training notebook (run first!)
├── data/
│   └── TVSEP2024_HHQ_V3VN.dta     # Raw dataset
├── .streamlit/
│   └── config.toml                 # Hide topbar
└── pages/
    ├── 01_Overview.py              # Dataset summary
    ├── 02_Predict_Single.py        # Single household prediction form
    ├── 03_Predict_Batch.py         # Batch prediction from file upload
    ├── 04_Performance.py           # Model evaluation metrics
    └── 05_Feature_Analysis.py      # Feature importance + EDA
```

After running the training notebook, these artifacts are created in the root:

```
tvsep_model_5class.pkl              # Pickled imblearn Pipeline
tvsep_top_features_5class.pkl       # List of 50 feature names
tvsep_class_names_5class.pkl        # Human-readable class names
tvsep_feature_importance_5class.csv # RF importance ranking
tvsep_xgb_importance_5class.csv     # Final XGBoost importance ranking
tvsep_k_sweep_5class.csv            # K-sweep results (new in Train5class.ipynb)
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place TVSEP2024_HHQ_V3VN.dta in the data/ sub-folder

# 3. Train the model
jupyter nbconvert --to notebook --execute Train5class.ipynb \
    --output Train5class_executed.ipynb (for Linux)

jupyter nbconvert --to notebook --execute Train5class.ipynb --output Train5class_executed.ipynb (for Windows)

# 4. Launch the app
streamlit run app.py
```

The app opens at http://localhost:8501.

## Design decisions

### Icons via set_page_config (not file names)
Each page sets its icon in code with `st.set_page_config(page_icon="🎯")`.
This avoids filename encoding issues on Windows/Mac that corrupt emoji characters.

### Dark theme with color accents
The global CSS in `utils.py` applies a dark background and gradient accents
consistently across all pages without requiring a Streamlit theme config file.

### Multi-page architecture
Streamlit auto-discovers files inside `pages/` and renders them in the sidebar.
Pages are numbered (`01_`, `02_`, …) so the sidebar order is deterministic.

### Caching strategy
- `@st.cache_resource` — mutable objects (sklearn Pipeline, feature list)
- `@st.cache_data` — immutable values (DataFrames, dicts)

### Class encoding (worst → best)
Stata code 1 = "Much better off", 5 = "Much worse off".
We apply `y = 5 - stata_code` so `y=0` = worst, `y=4` = best throughout.

## Reading the metrics

| Metric    | Value | Interpretation                                   |
|-----------|-------|--------------------------------------------------|
| Accuracy  | ~0.52 | ~2 pp above majority-class baseline (0.50)       |
| F1-macro  | ~0.25 | Poor due to extreme class imbalance              |
| QWK ⭐    | ~0.23 | Weak ordinal agreement (0=random, 1=perfect)     |

These are realistic numbers for predicting **subjective** well-being from
**objective** economic features. Report them honestly in the thesis.
