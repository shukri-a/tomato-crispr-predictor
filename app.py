"""Run with: python -m streamlit run app.py"""

import logging
from time import perf_counter
import streamlit as st

from predictor import (
    EXPRESSIONS, REGIONS, build_features, example_inputs, load_model,
    normalise_sequence, predict, reference,
)

st.set_page_config(page_title="CRISPR Editing Efficiency Predictor", layout="centered")
st.title("CRISPR Editing Efficiency Predictor")
st.caption("Food Systems Collective — Research Prototype")
st.write("Enter a tomato guide and its biological context to estimate on-target editing efficiency.")


@st.cache_resource(show_spinner="Loading the trained model…")
def cached_model():
    return load_model()


@st.cache_resource(show_spinner=False)
def cached_explainer():
    # Optional explanation dependencies never block ordinary predictions.
    from explanations import EnsembleExplainer
    return EnsembleExplainer(cached_model())


def clear_result():
    for key in ("prediction", "prediction_inputs", "shap_result", "shap_error"):
        st.session_state.pop(key, None)


def use_example():
    clear_result()
    for key, value in example_inputs(st.session_state.example).items():
        st.session_state[key] = value


st.selectbox(
    "Reference example", range(len(reference()["guide_sequences"])),
    format_func=lambda i: f"Example {i + 1} — {reference()['guide_sequences'][i]}",
    key="example",
)
st.button("Try an Example", on_click=use_example)
st.caption("Loads a real reference guide and its recorded context. Click Predict to run it.")

sequence = st.text_input(
    "Guide RNA sequence", key="sequence",
    placeholder="20 bases using A, C, G and T",
    help="DNA format. Lowercase and surrounding whitespace are accepted. Enter the 20-base guide only.",
)
region = st.selectbox("Genomic region", REGIONS, index=None, key="region", placeholder="Select region")
chromatin = st.selectbox(
    "Chromatin accessibility (within_atac_peak)", (True, False),
    index=None, key="chromatin", placeholder="Select True or False",
    format_func=lambda value: "True" if value else "False",
)
leaf = st.selectbox("Leaf expression", EXPRESSIONS, index=None, key="leaf", placeholder="Select expression")
t0 = st.selectbox("T0 expression", EXPRESSIONS, index=None, key="t0", placeholder="Select expression")

current_inputs = (sequence, region, chromatin, leaf, t0)
if st.session_state.get("prediction_inputs") != current_inputs:
    clear_result()

if st.button("Predict", type="primary"):
    clear_result()
    try:
        # Validate before loading the model. No default biological context.
        build_features(sequence, region, chromatin, leaf, t0)
    except ValueError as exc:
        st.error(str(exc))
    else:
        try:
            result = predict(cached_model(), sequence, region, chromatin, leaf, t0)
        except Exception:
            logging.exception("Prediction failed")
            st.error("Prediction could not be completed. Check the bundled model and pinned dependencies; see the terminal for details.")
        else:
            st.session_state.prediction = result
            st.session_state.prediction_inputs = current_inputs

if "prediction" in st.session_state:
    result = st.session_state.prediction
    st.metric("Model-predicted editing efficiency", f"{result:.2f}%")
    st.caption(f"Guide: {normalise_sequence(sequence)} · Region: {region} · ATAC: {chromatin} · Leaf: {leaf} · T0: {t0}")
    if result < 0 or result > 100:
        st.warning("The raw prediction is outside the biological 0–100% range. It has not been clamped and should not be interpreted as a feasible efficiency.")
    st.write("This is a model estimate, not a guarantee of experimental success.")

    st.subheader("Why did the model predict this value?")
    st.write("SHAP shows how each input moves the model estimate above or below its average prediction for 32 training examples. Contributions are measured in percentage points.")
    st.caption("All sequence-derived features are explained together to preserve their dependencies.")
    if st.button("Explain prediction", key="explain_prediction"):
        st.session_state.pop("shap_result", None)
        st.session_state.pop("shap_error", None)
        try:
            with st.spinner("Calculating the ensemble explanation…"):
                started = perf_counter()
                engine = cached_explainer()
                explanation, seconds = engine.explain(*current_inputs)
                from explanations import contribution_summary, waterfall_png
                image = waterfall_png(explanation)
                increases, decreases = contribution_summary(explanation)
                st.session_state.shap_result = {
                    "image": image, "increases": increases, "decreases": decreases,
                    "baseline": float(explanation.base_values), "seconds": seconds,
                    "total_seconds": perf_counter() - started,
                }
        except Exception:
            logging.exception("SHAP explanation failed")
            st.session_state.shap_error = "Explanation unavailable. Your prediction is still valid as a model output. Check the SHAP dependencies and bundled training background; see the terminal for details."
    if "shap_error" in st.session_state:
        st.error(st.session_state.shap_error)
    if "shap_result" in st.session_state:
        detail = st.session_state.shap_result
        st.image(detail["image"], width="stretch")
        st.write("**Increasing the estimate:** " + detail["increases"])
        st.write("**Decreasing the estimate:** " + detail["decreases"])
        st.caption(f"Background mean: {detail['baseline']:.2f}% · Explanation calculation: {detail['seconds']:.2f}s · Total including setup and plot: {detail['total_seconds']:.2f}s")
    st.caption("SHAP describes the combined four-model ensemble's behaviour, not proven biological effects. These grouped comparisons can combine a sequence with context not observed experimentally; they are not experimental recommendations.")

st.divider()
from comparison import render_comparison
render_comparison(cached_model, cached_explainer)

st.divider()
st.caption(
    "Research proof of concept for tomato CRISPR-Cas9. Predictions may be inaccurate and require experimental validation. "
    "Reference examples check software consistency; they do not establish independent predictive performance."
)
