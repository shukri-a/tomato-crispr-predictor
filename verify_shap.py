"""Focused SHAP checks: python verify_shap.py. Run verify.py as well."""
from io import BytesIO
from itertools import product
from math import factorial
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image
import shap
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

from explanations import EnsembleExplainer, GROUP_NAMES, INPUT_KEYS, SEQUENCE_COLUMNS, waterfall_png
from predictor import ROOT, build_features, example_inputs, load_model, predict, reference


def verify():
    print("SHAP version:", shap.__version__)
    payload = json.loads((ROOT / "shap_background.json").read_text())
    train, test = train_test_split(np.arange(420), test_size=0.2, random_state=42)
    selected = pd.Series(train).sample(n=32, random_state=42).tolist()
    assert selected == payload["metadata"]["source_row_indices_zero_based"]
    assert set(selected) <= set(train) and not set(selected) & set(test)
    assert all(set(row) == set(INPUT_KEYS) for row in payload["inputs"])
    assert all(set(row) == set(reference()["feature_names"]) for row in payload["features"])
    assert not set(reference()["guide_sequences"]) & {r["sequence"] for r in payload["inputs"]}
    print("PASS: deterministic 32-row training selection, disjoint test/reference cases, input-only background.")

    model = load_model()
    before = hashlib.sha256((ROOT / "final_ensemble.joblib").read_bytes()).hexdigest()
    engine = EnsembleExplainer(model)
    times, errors = [], []
    for i in range(5):
        inputs = example_inputs(i)
        explanation, seconds = engine.explain(**inputs)
        prediction = predict(model, **inputs)
        np.testing.assert_allclose(prediction, reference()["expected_predictions"][i], atol=1e-6, rtol=0)
        assert np.isfinite(explanation.values).all() and np.isfinite(explanation.base_values)
        assert explanation.feature_names == list(GROUP_NAMES)
        reconstructed = float(explanation.base_values + explanation.values.sum())
        error = abs(reconstructed - prediction)
        np.testing.assert_allclose(reconstructed, prediction, atol=1e-6, rtol=0)
        frame = build_features(**inputs)
        component_mean = np.mean([pipeline.predict(frame)[0] for pipeline in model.estimators_])
        np.testing.assert_allclose(prediction, component_mean, atol=1e-10, rtol=0)
        times.append(seconds)
        errors.append(error)
        print(f"PASS: example {i+1}: prediction={prediction:.12f}, SHAP reconstruction error={error:.3e}, calculation={seconds:.3f}s")
    print(f"Background ensemble mean: {engine.expected_value:.12f}%")

    # Check every coalition is built from a complete real sequence feature block.
    x = np.asarray([example_inputs(0)[k] for k in INPUT_KEYS], dtype=object)
    masks = np.asarray(list(product((False, True), repeat=5)))
    frames = []
    target = build_features(**example_inputs(0))
    background = pd.DataFrame(payload["features"]).astype(reference()["feature_dtypes"])
    for mask in masks:
        raw = engine.masker(mask, x)[0]
        frame = engine.feature_frame(raw)
        assert (frame.filter(like="trinuc_").sum(axis=1) == 18).all()
        expected = pd.concat([target] * 32, ignore_index=True) if mask[0] else background
        pd.testing.assert_frame_equal(frame[list(SEQUENCE_COLUMNS)], expected[list(SEQUENCE_COLUMNS)], check_exact=True)
        frames.append(frame)
    # Ordinary independent perturbation really can violate the 18-window constraint.
    malformed = target.copy()
    candidate = next(c for c in target.filter(like="trinuc_").columns if (background[c] != target.at[0, c]).any())
    malformed.at[0, candidate] = background.loc[background[candidate] != target.at[0, candidate], candidate].iloc[0]
    assert malformed.filter(like="trinuc_").sum(axis=1).iloc[0] != 18
    print("PASS: independent trinucleotide masking can violate sequence constraints; all 32 grouped coalitions preserve complete sequence blocks and 18 windows.")

    coalition_values = model.predict(pd.concat(frames, ignore_index=True)).reshape(32, 32).mean(axis=1)
    values = {tuple(mask): value for mask, value in zip(masks, coalition_values)}
    exact = np.zeros(5)
    for j in range(5):
        for mask in masks:
            if mask[j]:
                continue
            with_j = mask.copy()
            with_j[j] = True
            size = int(mask.sum())
            weight = factorial(size) * factorial(4-size) / factorial(5)
            exact[j] += weight * (values[tuple(with_j)] - values[tuple(mask)])
    explanation, _ = engine.explain(**example_inputs(0))
    np.testing.assert_allclose(explanation.values, exact, atol=1e-8, rtol=0)
    print("PASS: SHAP agrees with independently calculated five-group Shapley values over all 32 coalitions.")

    # Same complete-ensemble callable and coherent masker, considered as an alternative.
    started = perf_counter()
    permutation = shap.PermutationExplainer(engine.predict_inputs, engine.masker, seed=42)
    approx = permutation(x[None, :], max_evals=110, batch_size=32, silent=True)[0]
    np.testing.assert_allclose(approx.base_values + approx.values.sum(), predict(model, **example_inputs(0)), atol=1e-6, rtol=0)
    print(f"Permutation benchmark (110 masks): {perf_counter()-started:.3f}s; production uses 32-coalition ExactExplainer, removing sampling error.")

    with patch.object(engine, "explainer", side_effect=AssertionError("Invalid input reached SHAP")) as mocked:
        for invalid in (example_inputs(0) | {"sequence": "ACGT"}, example_inputs(0) | {"leaf": None}):
            try:
                engine.explain(**invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid input accepted")
        mocked.assert_not_called()
    print("PASS: invalid input cannot trigger SHAP.")

    png = waterfall_png(explanation)
    image = Image.open(BytesIO(png))
    image.verify()
    assert len(png) > 10000
    # Preview is a test artifact, not an app dependency.
    Path("/tmp/crispr_shap_verified.png").write_bytes(png)

    with patch.object(EnsembleExplainer, "explain", autospec=True, side_effect=EnsembleExplainer.explain) as calls:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        app.button[0].click().run()
        app.button[1].click().run()
        calls.assert_not_called()
        assert not app.exception and app.metric[0].value == "35.78%"
        app.button(key="explain_prediction").click().run()
        assert calls.call_count == 1 and not app.exception and not app.error
        assert len(app.get("imgs")) == 1
        assert app.metric[0].value == "35.78%" and "shap_result" in app.session_state
        total_ui = app.session_state["shap_result"]["total_seconds"]
        app.run()
        assert calls.call_count == 1 and len(app.get("imgs")) == 1
        app.text_input(key="sequence").set_value(reference()["guide_sequences"][1]).run()
        assert not app.metric and not app.get("imgs") and "shap_result" not in app.session_state
        app.button[1].click().run()
        assert not app.get("imgs") and calls.call_count == 1
        app.button(key="explain_prediction").click().run()
        assert calls.call_count == 2 and not app.exception and not app.error
        app.selectbox(key="leaf").set_value("high").run()
        assert not app.get("imgs") and not app.metric
        app.text_input(key="sequence").set_value("ACGT").run()
        app.button[1].click().run()
        assert app.error and not app.metric and calls.call_count == 2
    print(f"PASS: waterfall is a valid PNG and renders in Streamlit; lazy execution, session reuse and sequence/context invalidation; first UI request={total_ui:.3f}s.")

    app.button[0].click().run()
    app.button[1].click().run()
    with patch.object(EnsembleExplainer, "explain", side_effect=RuntimeError("Simulated SHAP failure")):
        app.button(key="explain_prediction").click().run()
    assert app.error and not app.exception and app.metric[0].value == "35.78%"
    assert not app.get("imgs")
    print("PASS: simulated SHAP failure leaves the prediction visible and shows a clear error.")
    assert hashlib.sha256((ROOT / "final_ensemble.joblib").read_bytes()).hexdigest() == before
    print("PASS: saved model artifact unchanged.")
    print(f"ALL SHAP CHECKS PASSED. Cold calculation={times[0]:.3f}s; warm range={min(times[1:]):.3f}–{max(times[1:]):.3f}s; max reconstruction error={max(errors):.3e}.")


if __name__ == "__main__":
    verify()
