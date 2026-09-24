"""Exact grouped SHAP for the complete saved ensemble, without model changes."""
from io import BytesIO
import hashlib
import json
from threading import RLock
from time import perf_counter

import numpy as np
import pandas as pd
import shap

from predictor import ROOT, build_features, normalise_sequence, reference

INPUT_KEYS = ("sequence", "region", "chromatin", "leaf", "t0")
GROUP_NAMES = ("sequence_features", "feature", "within_atac_peak", "leaf_exp", "t0_exp")
LABELS = ("Guide sequence (all sequence features)", "Genomic region", "Chromatin accessibility", "Leaf expression", "T0 expression")
SEQUENCE_COLUMNS = tuple(c for c in reference()["feature_names"] if c not in GROUP_NAMES[1:])
PLOT_LOCK = RLock()


class CoherentMasker:
    """Swap whole sequences, then regenerate their 68 dependent model features.

    Hidden inputs come from the same real training row. Between-group mixtures
    remain interventional, potentially unobserved combinations, not causal claims.
    """
    def __init__(self, rows):
        self.data = np.asarray(rows, dtype=object)
        self.shape = self.data.shape

    def __call__(self, mask, x):
        return (np.where(np.asarray(mask, dtype=bool)[None, :], x[None, :], self.data),)


class EnsembleExplainer:
    def __init__(self, model, background_path=ROOT / "shap_background.json"):
        self.model = model
        self.lock = RLock()
        background = json.loads(background_path.read_text(encoding="utf-8"))
        if background["metadata"]["model_sha256"] != hashlib.sha256((ROOT / "final_ensemble.joblib").read_bytes()).hexdigest():
            raise ValueError("SHAP background was prepared for a different model artifact.")
        rows = background["inputs"]
        if len(rows) != 32:
            raise ValueError("Expected the verified 32-row training background.")
        generated = pd.concat([build_features(**row) for row in rows], ignore_index=True)
        expected = pd.DataFrame(background["features"], columns=reference()["feature_names"]).astype(reference()["feature_dtypes"])
        pd.testing.assert_frame_equal(generated, expected, check_exact=True)
        self.masker = CoherentMasker([[row[k] for k in INPUT_KEYS] for row in rows])
        self.explainer = shap.ExactExplainer(self.predict_inputs, self.masker, feature_names=list(GROUP_NAMES))
        self.expected_value = float(self.model.predict(generated).mean())

    def feature_frame(self, rows):
        """Vectorize repeated masked rows; reuse the unchanged feature builder."""
        records = np.asarray(rows, dtype=object)
        sequence_templates = {}
        values = []
        for sequence, region, chromatin, leaf, t0 in records:
            if sequence not in sequence_templates:
                sequence_templates[sequence] = build_features(sequence, region, chromatin, leaf, t0).iloc[0].to_dict()
            row = sequence_templates[sequence].copy()
            row.update(feature=region, within_atac_peak=chromatin, leaf_exp=leaf, t0_exp=t0)
            values.append(row)
        return pd.DataFrame(values, columns=reference()["feature_names"]).astype(reference()["feature_dtypes"])

    def predict_inputs(self, rows):
        return self.model.predict(self.feature_frame(rows))

    def explain(self, sequence, region, chromatin, leaf, t0):
        """Return (shap.Explanation, elapsed_seconds); validate before masking."""
        start = perf_counter()
        features = build_features(sequence, region, chromatin, leaf, t0)
        row = np.asarray([[normalise_sequence(sequence), region, chromatin, leaf, t0]], dtype=object)
        with self.lock:
            # Five groups need only 2**5 coalitions: exact, no permutation sampling.
            explanation = self.explainer(row, max_evals=32, batch_size=32, silent=True)[0]
        prediction = float(self.model.predict(features)[0])
        if not np.isfinite(explanation.values).all() or not np.isfinite(explanation.base_values):
            raise RuntimeError("Non-finite SHAP values.")
        if not np.isclose(float(explanation.base_values), self.expected_value, atol=1e-6, rtol=0):
            raise RuntimeError("SHAP baseline does not match the full ensemble background mean.")
        if not np.isclose(float(explanation.base_values) + explanation.values.sum(), prediction, atol=1e-6, rtol=0):
            raise RuntimeError("SHAP contributions do not reconstruct the ensemble prediction.")
        return explanation, perf_counter() - start


def waterfall_png(explanation):
    """Render five readable grouped contributions; preserve internal group names."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display = shap.Explanation(
        values=explanation.values, base_values=explanation.base_values,
        feature_names=list(LABELS),
    )
    with PLOT_LOCK:
        plt.figure()
        try:
            shap.plots.waterfall(display, max_display=5, show=False)
            figure = plt.gcf()
            figure.set_size_inches(10, 4.5)
            plt.xlabel("Model-predicted editing efficiency (%)")
            output = BytesIO()
            figure.savefig(output, format="png", dpi=150, bbox_inches="tight")
            return output.getvalue()
        finally:
            plt.close()


def contribution_summary(explanation):
    pairs = sorted(zip(LABELS, explanation.values), key=lambda pair: abs(pair[1]), reverse=True)
    def describe(sign):
        chosen = [(name, value) for name, value in pairs if sign * value > 1e-9][:3]
        return "; ".join(f"{name}: {value:+.2f} percentage points" for name, value in chosen) or "None."
    return describe(1), describe(-1)
