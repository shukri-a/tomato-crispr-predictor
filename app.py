"""Streamlit interface for the CRISPR editing-efficiency research prototype.

Run locally with:
    python -m streamlit run app.py
"""

import json
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd
import streamlit as st

from comparison import render_comparison
from predictor import (
    EXPRESSIONS,
    REGIONS,
    build_features,
    example_inputs,
    load_model,
    normalise_sequence,
    predict,
    reference,
)
from uat_cases import render_uat_loader


st.set_page_config(page_title="CRISPR Editing Efficiency Predictor", layout="centered")
st.title("CRISPR Editing Efficiency Predictor")
st.caption("Food Systems Collective — Research Prototype")
st.write(
    "Estimate tomato CRISPR-Cas9 on-target editing efficiency, understand what influenced the prediction, "
    "and compare candidate guides."
)


@st.cache_resource(show_spinner="Loading the trained model…")
def cached_model():
    """Load the fitted ensemble once and reuse it across Streamlit reruns."""
    return load_model()


@st.cache_resource(show_spinner=False)
def cached_explainer():
    """Build the SHAP explainer only when a user asks for an explanation."""
    from explanations import EnsembleExplainer

    return EnsembleExplainer(cached_model())


@st.cache_data
def load_pam_reference():
    """Load the guide-to-PAM lookup derived from the tomato reference dataset."""
    pam_df = pd.read_csv(Path(__file__).parent / "pam_reference.csv")
    pam_df["guide_sequence"] = pam_df["guide_sequence"].str.upper()
    return pam_df


def clear_result():
    """Clear results that belong to the current single-guide prediction."""
    for key in ("prediction", "prediction_inputs", "shap_result", "shap_error"):
        st.session_state.pop(key, None)


def find_reference_pam(sequence):
    """Return the recorded PAM for a guide when that guide exists in the reference lookup."""
    if not sequence:
        return None

    sequence = sequence.strip().upper()
    match = load_pam_reference()
    match = match[match["guide_sequence"] == sequence]

    if match.empty:
        return None

    return {
        "pam": match.iloc[0]["pam"],
        "strand": match.iloc[0]["strand"],
    }


def use_example():
    """Load one of the bundled software-reference examples into the normal input fields."""
    clear_result()
    for key, value in example_inputs(st.session_state.example).items():
        st.session_state[key] = value


def render_prediction_explanation():
    """Render and calculate the SHAP explanation for the active single-guide prediction."""
    st.subheader("Why did the model predict this value?")
    st.write(
        "SHAP compares this prediction with the model's reference prediction and shows which input groups "
        "moved the estimate higher or lower. Contributions are shown in percentage points."
    )
    st.caption(
        "The SHAP reference is the average ensemble prediction across 32 reproducibly sampled training examples. "
        "All sequence-derived features are kept together so the guide sequence is explained as one biological input."
    )

    if st.button("Explain prediction", key="explain_prediction"):
        st.session_state.pop("shap_result", None)
        st.session_state.pop("shap_error", None)

        try:
            with st.spinner("Calculating the explanation…"):
                started = perf_counter()
                explanation, seconds = cached_explainer().explain(*st.session_state.prediction_inputs)

                from explanations import contribution_summary, waterfall_png

                increases, decreases = contribution_summary(explanation)
                st.session_state.shap_result = {
                    "image": waterfall_png(explanation),
                    "increases": increases,
                    "decreases": decreases,
                    "baseline": float(explanation.base_values),
                    "seconds": seconds,
                    "total_seconds": perf_counter() - started,
                }
        except Exception:
            logging.exception("SHAP explanation failed")
            st.session_state.shap_error = (
                "The explanation could not be generated, but the prediction is still available. "
                "Check the SHAP dependencies and bundled training background; see the terminal for technical details."
            )

    if "shap_error" in st.session_state:
        st.error(st.session_state.shap_error)

    if "shap_result" in st.session_state:
        detail = st.session_state.shap_result
        st.image(detail["image"], width="stretch")
        st.write("**Increasing the estimate:** " + detail["increases"])
        st.write("**Decreasing the estimate:** " + detail["decreases"])
        st.caption(
            f"SHAP baseline: {detail['baseline']:.2f}% (average ensemble prediction across 32 training examples) "
            f"· Calculation: {detail['seconds']:.2f}s · Total including setup and plot: {detail['total_seconds']:.2f}s"
        )

    st.caption(
        "SHAP explains the behaviour of the combined four-model ensemble; it does not prove biological cause and effect. "
        "The explanation should support interpretation, not replace experimental validation."
    )


def render_pam_verification():
    """Show the reference PAM lookup for the currently predicted guide without affecting the model."""
    if "prediction_inputs" not in st.session_state:
        return

    pam_result = find_reference_pam(st.session_state.prediction_inputs[0])
    st.subheader("PAM Verification")

    if pam_result is None:
        st.info(
            "PAM could not be verified automatically because this guide is not present in the tomato reference dataset."
        )
        st.caption("PAM verification does not affect the model prediction.")
        return

    pam = pam_result["pam"]
    strand = pam_result["strand"]
    st.success(f"PAM identified: {pam} — matches the canonical SpCas9 NGG PAM pattern.")
    st.caption(
        f"Reference strand: {strand}. The PAM comes from the tomato reference dataset and is not used as an input "
        "to the prediction model."
    )
    st.caption("PAM verification does not affect the model prediction.")


def render_model_performance():
    """Show the fixed held-out evaluation metrics for the final fitted ensemble."""
    st.divider()
    st.subheader("Final Model Performance")

    performance_path = Path(__file__).parent / "model_performance.json"
    performance = json.loads(performance_path.read_text(encoding="utf-8"))

    st.caption(
        "Held-out test set: 84 rows from the original project split. These are fixed dataset-level results, "
        "not scores for the guide entered above."
    )
    st.table(
        {
            "Metric": ["MAE (percentage points)", "RMSE (percentage points)", "R²"],
            "Held-out test result": [f"{performance['metrics'][key]:.4f}" for key in ("MAE", "RMSE", "R2")],
        }
    )
    st.caption(
        "MAE is the average size of the prediction error. RMSE gives extra weight to larger errors. "
        "Lower is better for both. R² compares the model with a mean-only baseline; it is not an accuracy percentage."
    )
    st.caption(
        "These 84 rows were held out from final fitting, although earlier tuning/cross-validation used the full dataset. "
        "This is not external or unseen-gene validation."
    )


def render_single_guide():
    """Render the complete manual/UAT single-guide workflow in one place."""
    st.subheader("Predict / UAT")
    st.write(
        "Enter a guide manually or load an assigned UAT example. Both routes use the same five biological inputs, "
        "and experimental outcomes stay hidden during UAT."
    )

    render_uat_loader(clear_result)

    with st.expander("Optional demo example"):
        st.selectbox(
            "Demo example",
            range(len(reference()["guide_sequences"])),
            format_func=lambda i: f"Example {i + 1}",
            key="example",
        )
        st.button("Load Demo Example", on_click=use_example, key="demo_load")
        st.caption("These examples check software consistency and are separate from the UAT cases.")

    sequence = st.text_input(
        "Guide RNA sequence",
        key="sequence",
        placeholder="20 bases using A, C, G and T",
        help="20-base DNA guide sequence using A, C, G and T.",
    )
    region = st.selectbox(
        "Genomic region",
        REGIONS,
        index=None,
        key="region",
        placeholder="Select region",
        help="Where the guide targets the gene: exon, intron or promoter.",
        format_func=lambda value: {"Exo": "Exo — exon", "Int": "Int — intron", "Pro": "Pro — promoter"}[value],
    )
    chromatin = st.selectbox(
        "Chromatin accessibility / ATAC",
        (True, False),
        index=None,
        key="chromatin",
        placeholder="Select True or False",
        help="Whether the target site is in an open/accessible chromatin region.",
        format_func=lambda value: "True" if value else "False",
    )
    leaf = st.selectbox(
        "Leaf expression",
        EXPRESSIONS,
        index=None,
        key="leaf",
        placeholder="Select expression",
        help="Relative expression level of the target gene in leaf tissue.",
    )
    t0 = st.selectbox(
        "T0 expression",
        EXPRESSIONS,
        index=None,
        key="t0",
        placeholder="Select expression",
        help="Relative expression level of the target gene at the T0 stage.",
    )

    current_inputs = (sequence, region, chromatin, leaf, t0)
    if st.session_state.get("prediction_inputs") != current_inputs:
        clear_result()

    if st.button("Predict", type="primary", key="single_predict"):
        clear_result()
        try:
            build_features(sequence, region, chromatin, leaf, t0)
        except ValueError as exc:
            st.error(str(exc))
        else:
            try:
                result = predict(cached_model(), sequence, region, chromatin, leaf, t0)
            except Exception:
                logging.exception("Prediction failed")
                st.error(
                    "The prediction could not be completed. Check the bundled model and pinned dependencies; "
                    "see the terminal for technical details."
                )
            else:
                st.session_state.prediction = result
                st.session_state.prediction_inputs = current_inputs

    if "prediction" not in st.session_state:
        return

    result = st.session_state.prediction
    if len(set(normalise_sequence(sequence))) == 1:
        st.warning(
            "This guide contains the same base at all 20 positions. It is valid input, but the prediction should be "
            "interpreted with extra caution and confirmed experimentally."
        )

    st.metric("Model-predicted editing efficiency", f"{result:.2f}%")
    st.caption(
        f"Guide: {normalise_sequence(sequence)} · Region: {region} · ATAC: {chromatin} · Leaf: {leaf} · T0: {t0}"
    )

    if result < 0 or result > 100:
        st.warning(
            "The raw model output falls outside the biological 0–100% range. It has not been clipped, so it should "
            "not be interpreted as a feasible editing efficiency."
        )

    st.write("This is a model estimate and should be confirmed experimentally.")

    # Keep the explanation and PAM check with the prediction they belong to.
    render_prediction_explanation()
    render_pam_verification()


predict_tab, compare_tab = st.tabs(["Predict / UAT", "Compare Guides"])

with predict_tab:
    render_single_guide()
    render_model_performance()

with compare_tab:
    render_comparison(cached_model, cached_explainer)

st.divider()
st.caption(
    "Research proof of concept for tomato CRISPR-Cas9. Predictions may be inaccurate and require experimental validation. "
    "Demo/reference examples check software consistency; they do not establish independent predictive performance."
)
