"""Offline only: evaluate the existing model on the original 84-row test split.
Usage: python build_performance.py /path/to/tomato_crispr_editing_data.xlsx
Saves aggregates/provenance only. Does not fit models or expose row outcomes.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from predictor import ROOT, build_features, load_model, predict, reference


def build(source):
    source = Path(source)
    background = json.loads((ROOT / 'shap_background.json').read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == background['metadata']['source_sha256']
    raw = pd.read_excel(source)
    assert len(raw) == 420 and raw['Guide sequence'].nunique() == 420
    inputs = [dict(sequence=r['Guide sequence'], region=r['feature'],
                   chromatin=bool(r['within_atac_peak']), leaf=r['leaf_exp'], t0=r['t0_exp'])
              for _, r in raw.iterrows()]
    features = pd.concat([build_features(**row) for row in inputs], ignore_index=True)
    training, test = train_test_split(features, test_size=0.20, random_state=42)
    model = load_model()
    for pipeline in model.estimators_:
        prep = pipeline.named_steps['preprocessor']
        scaler = prep.named_transformers_['numerical']
        numeric = prep.transformers_[1][2]
        assert scaler.n_samples_seen_ == len(training) == 336
        np.testing.assert_allclose(training[numeric].mean(), scaler.mean_, atol=1e-12, rtol=0)
        np.testing.assert_allclose(training[numeric].var(ddof=0), scaler.var_, atol=1e-12, rtol=0)
    assert len(test) == 84 and not set(test.index) & set(training.index)
    assert not set(test.index) & set(background['metadata']['source_row_indices_zero_based'])
    predictions = np.array([predict(model, **inputs[i]) for i in test.index])
    np.testing.assert_allclose(predictions, model.predict(test), atol=1e-10, rtol=0)
    ref = reference()
    for seq, expected in zip(ref['guide_sequences'], ref['expected_predictions']):
        idx = next(i for i in test.index if inputs[i]['sequence'] == seq)
        np.testing.assert_allclose(predict(model, **inputs[idx]), expected, atol=1e-6, rtol=0)
    actual = raw.loc[test.index, 'editing%'].to_numpy(dtype=float)
    assert np.isfinite(actual).all() and np.isfinite(predictions).all()
    metrics = {'MAE': float(mean_absolute_error(actual, predictions)),
               'RMSE': float(np.sqrt(mean_squared_error(actual, predictions))),
               'R2': float(r2_score(actual, predictions))}
    result = {'metrics': metrics, 'test_rows': 84, 'training_rows': 336,
              'split': 'train_test_split(test_size=0.20, random_state=42); original workbook row order',
              'source_sha256': source_hash,
              'model_sha256': hashlib.sha256((ROOT / 'final_ensemble.joblib').read_bytes()).hexdigest(),
              'derivation': 'Recomputed offline from the unchanged saved ensemble via predictor.predict; not copied from cross-validation scores.',
              'limitation': 'Held out from final fitting; earlier tuning/cross-validation used the full dataset. Not untouched external or unseen-gene validation.'}
    (ROOT / 'model_performance.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: original split verified against all four fitted scalers; references match; pipeline parity verified.')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    build(sys.argv[1])
