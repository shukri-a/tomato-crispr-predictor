"""Recreate training-only SHAP background; never fit a model or read outcomes.

Usage: python build_background.py /path/to/tomato_crispr_editing_data.xlsx
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from predictor import ROOT, build_features, load_model, reference


def build(source):
    source = Path(source)
    columns = ["Guide sequence", "feature", "within_atac_peak", "leaf_exp", "t0_exp"]
    raw = pd.read_excel(source, usecols=columns)[columns]
    assert len(raw) == 420 and raw["Guide sequence"].nunique() == 420
    assert not raw.isna().any().any()
    inputs = [dict(sequence=r[0], region=r[1], chromatin=bool(r[2]), leaf=r[3], t0=r[4])
              for r in raw.itertuples(index=False, name=None)]
    features = pd.concat([build_features(**row) for row in inputs], ignore_index=True)
    # Matches notebook cell 153; row order is retained from the source workbook.
    training, test = train_test_split(features, test_size=0.20, random_state=42)
    model = load_model()
    for pipeline in model.estimators_:
        prep = pipeline.named_steps["preprocessor"]
        scaler = prep.named_transformers_["numerical"]
        numeric = prep.transformers_[1][2]
        assert scaler.n_samples_seen_ == len(training) == 336
        np.testing.assert_allclose(training[numeric].mean(), scaler.mean_, atol=1e-12, rtol=0)
        np.testing.assert_allclose(training[numeric].var(ddof=0), scaler.var_, atol=1e-12, rtol=0)
    # No target values are loaded or used in selection. No seed search.
    selected = training.sample(n=32, random_state=42)
    assert not set(selected.index) & set(test.index)
    reference_sequences = set(reference()["guide_sequences"])
    assert not reference_sequences & {inputs[i]["sequence"] for i in selected.index}
    distributions = {}
    for column in ("feature", "leaf_exp", "t0_exp", "within_atac_peak"):
        assert set(selected[column]) == set(training[column]), f"Background lacks a category: {column}"
        distributions[column] = {
            "training": {str(k): int(v) for k, v in training[column].value_counts().items()},
            "background": {str(k): int(v) for k, v in selected[column].value_counts().items()},
        }
    payload = {
        "metadata": {
            "source_filename": source.name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256((ROOT / "final_ensemble.joblib").read_bytes()).hexdigest(),
            "split": "sklearn train_test_split(test_size=0.20, random_state=42), original workbook row order",
            "selection": "32 training rows: pandas DataFrame.sample(n=32, random_state=42)",
            "training_rows": 336, "test_rows": 84,
            "source_row_indices_zero_based": selected.index.tolist(),
            "training_scaler_statistics_verified": True,
            "outcomes_loaded_or_used": False,
            "category_counts": distributions,
        },
        "inputs": [inputs[i] for i in selected.index],
        "feature_names": reference()["feature_names"],
        "feature_dtypes": reference()["feature_dtypes"],
        "features": selected.to_dict(orient="records"),
    }
    target = ROOT / "shap_background.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PASS: reconstructed 336-row training split matches all four fitted scalers.")
    print("PASS: 32 deterministic training-only rows; all context categories covered; no reference/test rows or outcomes.")
    print("Saved", target.name)


if __name__ == "__main__":
    build(sys.argv[1])
