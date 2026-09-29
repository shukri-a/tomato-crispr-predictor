# CRISPR Editing Efficiency Predictor

Food Systems Collective — Research Prototype

Local Streamlit application using the supplied four-model voting ensemble. The model artifact, 72-feature engineering logic and grouped SHAP method are unchanged. No training dataset or Colab notebook is needed to run inference.

## Run locally

Use the pinned dependencies in `requirements.txt`:

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or activate on macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install -r requirements.txt
python verify.py
python verify_shap.py
python verify_comparison.py
python verify_uat.py
python -m streamlit run app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

Open `http://localhost:8501` in your browser. Stop with Ctrl+C.

## App layout

The interface has two tabs: **Predict / UAT** and **Compare Guides**.

### Predict / UAT

Enter a 20-base DNA guide plus genomic region, ATAC status, leaf expression and
T0 expression. Each input has a short help explanation. Alternatively, select
an assigned UAT Test ID and click **Load UAT example** to fill these same five
editable fields. Click **Predict**, then **Explain prediction** for grouped SHAP.
Loading a case replaces the fields; manual edits make it a modified input.
Experimental outcomes remain hidden. Optional software demo examples are in an expander.

The **Final Model Performance** table uses all 84 rows of the original test split:
MAE 19.7692 percentage points, RMSE 25.5450 percentage points, R² 0.2343.
These were recomputed offline using the unchanged saved ensemble and existing
prediction function, because supplied final-ensemble tables report cross-validation
rather than this test-set evaluation. The split matches all four saved scalers.
They are fixed dataset-level scores, never calculated from the entered guide.
These rows were held out from final fitting, but prior tuning/cross-validation
used the full dataset; this is not an untouched or unseen-gene validation.

`model_performance.json` stores only aggregate metrics and source/model hashes.
`build_performance.py` reproduces them offline from the original workbook:
`python build_performance.py /path/to/tomato_crispr_editing_data.xlsx`.
The workbook and per-row experimental outcomes are not bundled or read by the app.

### Compare Guides

Choose two or three guides, enter each guide's context separately and click **Compare**. The app shows predictions side-by-side, flags context differences and allows SHAP inspection of one selected guide. A higher model estimate is a model ranking only, not proof of experimental superiority.

## Included files

- `app.py` — two-tab Streamlit interface.
- `predictor.py` — validation, exact 72-feature extraction and inference.
- `final_ensemble.joblib` — fitted equal-weight Lasso/CatBoost/Gradient Boosting/Random Forest ensemble.
- `model_reference.json` — feature schema and five software-reference examples.
- `explanations.py` — exact grouped SHAP for the full ensemble.
- `shap_background.json` — 32 training-only input rows and their 72 engineered features.
- `build_background.py` — reproducible SHAP-background preparation.
- `comparison.py` — manual two/three-guide comparison interface.
- `uat_cases.py` and `uat_test_cases.csv` — blinded UAT workflow and 14 input-only UAT cases.
- `verify.py`, `verify_shap.py`, `verify_comparison.py`, `verify_uat.py` — model/UI/UAT verification scripts.
- `requirements.txt` — pinned runtime dependencies.

Only `uat_test_cases.csv` is bundled for UAT. Experimental outcomes and outcome-bearing benchmark files must remain outside the deployed app.

## SHAP method and limits

Explanations use the final VotingRegressor prediction function, including all four fitted preprocessing/model pipelines. They report five groups: all 68 sequence-derived features together, genomic region, chromatin accessibility, leaf expression and T0 expression.

The SHAP masker swaps a complete 20-base sequence and regenerates its dependent sequence features together, avoiding incoherent independent trinucleotide masking. With five coherent groups, `ExactExplainer` evaluates all 32 coalitions. The baseline is the ensemble mean prediction on the 32 selected training-background inputs. Contributions are percentage points and describe model behaviour, not biological causation.

The background was reconstructed from the original row-order-preserving `train_test_split(test_size=0.20, random_state=42)`, followed by `DataFrame.sample(n=32, random_state=42)` from the 336 training rows. No editing-outcome column is used to construct the SHAP background.

These checks establish implementation consistency, not independent predictive performance or external biological validation. The project uses a random 80/20 train/test split, not a complete held-out-gene benchmark. No model retraining or model changes are included in this UI update.
