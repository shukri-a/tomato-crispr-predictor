"""Inference only: original fitted ensemble and notebook feature calculations."""

from functools import lru_cache
from pathlib import Path
import json
import math
import re

from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
import joblib
import pandas as pd
from sklearn.ensemble import VotingRegressor
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parent
REGIONS = ("Exo", "Int", "Pro")
EXPRESSIONS = ("low", "med", "high")


@lru_cache(maxsize=1)
def reference():
    with (ROOT / "model_reference.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def normalise_sequence(sequence):
    if not isinstance(sequence, str) or not sequence.strip():
        raise ValueError("Enter a 20-base guide RNA sequence in DNA format (A, C, G, T).")
    sequence = sequence.strip().upper()
    if len(sequence) != 20:
        raise ValueError(f"The guide must contain exactly 20 bases; received {len(sequence)}.")
    if re.fullmatch(r"[ACGT]{20}", sequence) is None:
        raise ValueError("Use only A, C, G and T. Remove internal spaces; U and ambiguous bases are not accepted.")
    return sequence


def build_features(sequence, region, chromatin, leaf, t0):
    """Convert five supplied inputs to the exact ordered, typed 72-column schema."""
    sequence = normalise_sequence(sequence)
    if region not in REGIONS:
        raise ValueError("Select genomic region: Exo, Int or Pro.")
    if type(chromatin) is not bool:
        raise ValueError("Select chromatin accessibility: True or False.")
    if leaf not in EXPRESSIONS:
        raise ValueError("Select leaf expression: low, med or high.")
    if t0 not in EXPRESSIONS:
        raise ValueError("Select T0 expression: low, med or high.")

    # Notebook cell 30: identical GC percentage and homopolymer calculation.
    max_run, current_run = 1, 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    values = {
        "feature": region, "leaf_exp": leaf, "t0_exp": t0,
        "first_base": sequence[0], "last_base": sequence[-1],
        "gc_content": gc_fraction(Seq(sequence)) * 100,
        "max_homopolymer_run": max_run,
        "within_atac_peak": chromatin,
    }
    schema = reference()
    trinuc_columns = [name for name in schema["feature_names"] if name.startswith("trinuc_")]
    # Notebook cell 151: count overlapping windows, not str.count().
    for name in trinuc_columns:
        trinuc = name.removeprefix("trinuc_")
        values[name] = sum(sequence[i:i + 3] == trinuc for i in range(len(sequence) - 2))
    if len(trinuc_columns) != 64 or sum(values[name] for name in trinuc_columns) != 18:
        raise RuntimeError("Trinucleotide schema/count check failed.")
    if len(schema["feature_names"]) != 72 or set(values) != set(schema["feature_names"]):
        raise RuntimeError("The reference feature schema is inconsistent.")
    return pd.DataFrame([values], columns=schema["feature_names"]).astype(schema["feature_dtypes"])


def load_model():
    """Load only the bundled trusted artifact; never fit or modify it."""
    model = joblib.load(ROOT / "final_ensemble.joblib")
    expected_types = ["Lasso", "CatBoostRegressor", "GradientBoostingRegressor", "RandomForestRegressor"]
    if not isinstance(model, VotingRegressor) or len(model.estimators_) != 4:
        raise RuntimeError("Expected the saved four-model VotingRegressor.")
    if model.n_features_in_ != 72 or list(model.feature_names_in_) != reference()["feature_names"]:
        raise RuntimeError("Saved model and reference feature schema differ.")
    if model.weights is not None and (len(model.weights) != 4 or any(w != model.weights[0] for w in model.weights) or model.weights[0] <= 0):
        raise RuntimeError("Expected equal voting weights.")
    for pipeline, expected_type in zip(model.estimators_, expected_types):
        if type(pipeline.named_steps["model"]).__name__ != expected_type:
            raise RuntimeError(f"Expected fitted {expected_type} pipeline.")
        check_is_fitted(pipeline.named_steps["preprocessor"])
        check_is_fitted(pipeline.named_steps["model"])
    return model


def predict(model, sequence, region, chromatin, leaf, t0):
    features = build_features(sequence, region, chromatin, leaf, t0)
    prediction = float(model.predict(features)[0])
    if not math.isfinite(prediction):
        raise RuntimeError("The model returned a non-finite prediction.")
    # Raw editing-percentage units: deliberately do not scale or clamp.
    return prediction


def example_inputs(index):
    data = reference()
    row = data["sample_inputs"][index]
    return {
        "sequence": data["guide_sequences"][index],
        "region": row["feature"], "chromatin": row["within_atac_peak"],
        "leaf": row["leaf_exp"], "t0": row["t0_exp"],
    }
