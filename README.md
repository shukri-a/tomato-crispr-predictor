# CRISPR Editing Efficiency Predictor

Food Systems Collective — Research Prototype

Minimal, local Streamlit application. The supplied four-model ensemble and
reference JSON are included unchanged. No training dataset or Colab is needed.

## Run locally

Use **Python 3.12** (verified on **3.12.14**, Linux). Open a terminal in this folder:

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
python -m streamlit run app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

Open http://localhost:8501 in your browser. Stop with Ctrl+C.

## For FSC scientists

Enter a 20-base DNA-formatted guide and select all four biological context
values, then click **Predict**. Alternatively, select a reference example,
click **Try an Example**, then **Predict**. Supply known context; do not guess
missing values. Results are estimated editing-efficiency percentages, not
guarantees of experimental success. Out-of-range predictions are shown with
an explicit warning and are never clamped. After predicting, click **Explain
prediction** to see a five-group SHAP waterfall and a summary of positive and
negative contributions. Changing any input clears the previous result and
explanation. An explanation error leaves the prediction available.

For multiple candidates, expand **Compare Guides**, choose **2** or **3**, and
enter each guide's sequence and context separately. Click **Compare**. Select
a result and click **Explain selected guide** for its existing grouped SHAP.
All included guides must be valid; choose two to omit Guide C. Editing any
comparison input clears its results and explanation. Differing contexts are
flagged, since prediction differences cannot then be attributed to sequence alone.

Quick demo: choose **3**, click **Load comparison examples**, then **Compare**.
Guides A/B/C produce approximately **35.78% / 32.54% / 39.63%**, respectively.
These examples have different biological contexts. Select Guide C and click
**Explain selected guide** to inspect its contribution plot.

## Included files and verification

- `app.py`: single-page interface with cached model loading.
- `predictor.py`: validation, exact 72-feature extraction and inference.
- `final_ensemble.joblib`: original fitted Lasso/CatBoost/Gradient Boosting/Random Forest ensemble with preprocessing and equal weights.
- `model_reference.json`: original schema and five reference cases.
- `requirements.txt`: pinned inference dependencies and Streamlit.
- `verify.py`: feature, prediction, validation, UI and local-server checks.
- `verification_results.txt`: successful test output from this build.
- `explanations.py`: on-demand, cached-engine SHAP of the complete ensemble.
- `shap_background.json`: 32 training-only input rows, their 72 features and provenance.
- `build_background.py`: reproducible background preparation from the original workbook.
- `verify_shap.py`: grouped SHAP correctness, timing and UI checks.
- `comparison.py`: minimal two/three-guide interface using the existing functions.
- `verify_comparison.py`: comparison validation, prediction parity and selected-guide SHAP checks.

All five reference predictions passed at absolute tolerance 1e-6.
Feature values, order and dtypes matched exactly. See verification_results.txt
for measured prediction errors, SHAP reconstruction errors and timing.
The exported model was saved under Python 3.13.15; this app was successfully
verified under Python 3.12.14 with every supplied model-library version preserved.
Do not upgrade scikit-learn away from 1.6.1 without repeating verification.

## SHAP method and limits

Explanations use the final VotingRegressor.predict function, including all four
fitted preprocessing/model pipelines. They report five groups: all 68 sequence
features together, genomic region, chromatin accessibility, leaf expression
and T0 expression. The 72 original model columns, types and predictions stay
unchanged. The waterfall uses readable labels; internal model names are retained.

Ordinary independent masking can make trinucleotide counts fail to sum to 18
and conflict with GC content or end bases. Our masker swaps a complete 20-base
sequence and regenerates its entire feature block together. Hidden context
inputs come from the same background row. Between-group combinations can still
be unobserved biologically. These are interventional model comparisons, not
conditional or causal biological explanations, and not recommended interventions.

PermutationExplainer was considered and benchmarked. With only five coherent
groups, ExactExplainer evaluates all 32 coalitions. This gives exact grouped
Shapley values relative to the selected background, without permutation
sampling error. It does not provide separate attributions to individual
trinucleotides. The baseline is the ensemble's mean prediction on the 32
background inputs; contributions are percentage points, not relative percentages.
The background is small, so explanations depend on this reference sample.

Background selection reconstructs the notebook's original row-order-preserving
train_test_split(test_size=0.20, random_state=42), then selects 32 of the 336
training rows using DataFrame.sample(n=32, random_state=42). No outcome column
is read or used. The training means, variances and sample counts match all four
saved scalers. Every context category is covered; all 84 test rows, including
the five reference examples, are excluded. Source hashes, row indices and
category counts are embedded in shap_background.json.

The original workbook is unnecessary for running the app. To reproduce the
bundled background only:

```bash
python build_background.py /path/to/tomato_crispr_editing_data.xlsx
```

Method references: [SHAP ExactExplainer](https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html),
[custom maskers](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/maskers/custom.html).

These checks establish implementation consistency, not independent predictive
performance or a held-out-gene benchmark. No model was retrained or changed.
Predictions do not establish experimental superiority. Hosting remains outside
this release. Nothing was deployed.
