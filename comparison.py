"""Small comparison interface using the existing prediction and SHAP pipeline."""
import logging
from time import perf_counter

import pandas as pd
import streamlit as st

from predictor import EXPRESSIONS, REGIONS, build_features, example_inputs, normalise_sequence, predict

FIELDS = ("sequence", "region", "chromatin", "leaf", "t0")


def clear_comparison():
    for key in ("comparison_inputs", "comparison_results", "comparison_shap", "comparison_error"):
        st.session_state.pop(key, None)


def load_comparison_examples():
    clear_comparison()
    for i, label in enumerate("ABC"):
        for field, value in example_inputs(i).items():
            st.session_state[f"compare_{label}_{field}"] = value


def render_comparison(cached_model, cached_explainer):
    with st.expander("Compare Guides", expanded=False):
        st.write("Compare two or three candidates. Provide the biological context separately for each guide.")
        count = st.radio("Number of guides", (2, 3), horizontal=True, key="compare_count")
        st.button("Load comparison examples", key="compare_examples", on_click=load_comparison_examples)
        guides = []
        for label, column in zip("ABC"[:count], st.columns(count)):
            with column:
                st.markdown(f"**Guide {label}**")
                prefix = f"compare_{label}_"
                sequence = st.text_input("Guide RNA sequence", key=prefix + "sequence", placeholder="20 DNA bases")
                region = st.selectbox("Genomic region", REGIONS, index=None, key=prefix + "region", placeholder="Select region")
                chromatin = st.selectbox("Chromatin accessibility", (True, False), index=None, key=prefix + "chromatin", placeholder="Select True or False")
                leaf = st.selectbox("Leaf expression", EXPRESSIONS, index=None, key=prefix + "leaf", placeholder="Select expression")
                t0 = st.selectbox("T0 expression", EXPRESSIONS, index=None, key=prefix + "t0", placeholder="Select expression")
                guides.append((sequence, region, chromatin, leaf, t0))

        snapshot = tuple(guides)
        if st.session_state.get("comparison_inputs") != snapshot:
            clear_comparison()
        if st.button("Compare", key="compare_submit", type="primary"):
            clear_comparison()
            errors = []
            for label, inputs in zip("ABC", guides):
                try:
                    build_features(*inputs)
                except ValueError as exc:
                    errors.append(f"Guide {label}: {exc}")
            if errors:
                for error in errors:
                    st.error(error)
                st.info("Complete all included guides before comparing. Choose two guides to omit Guide C.")
            else:
                try:
                    model = cached_model()
                    rows = []
                    for label, inputs in zip("ABC", guides):
                        sequence, region, chromatin, leaf, t0 = inputs
                        rows.append({
                            "Guide": f"Guide {label}", "Sequence": normalise_sequence(sequence),
                            "Predicted efficiency (%)": predict(model, *inputs),
                            "Region": region, "ATAC peak": chromatin,
                            "Leaf": leaf, "T0": t0,
                        })
                    st.session_state.comparison_results = rows
                    st.session_state.comparison_inputs = snapshot
                except Exception:
                    logging.exception("Guide comparison failed")
                    st.error("Comparison could not be completed. See the terminal for details.")

        if "comparison_results" not in st.session_state:
            return
        rows = st.session_state.comparison_results
        contexts = {inputs[1:] for inputs in snapshot}
        if len(contexts) > 1:
            st.warning("These guides have different biological contexts. Prediction differences may reflect both the sequence and its context; this is not a sequence-only comparison.")
        else:
            st.caption("All included guides have the same biological context.")
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={"Predicted efficiency (%)": st.column_config.NumberColumn("Predicted efficiency (%)", format="%.2f")},
        )
        outside = [row["Guide"] for row in rows if not 0 <= row["Predicted efficiency (%)"] <= 100]
        if outside:
            st.warning(f"{', '.join(outside)}: raw prediction outside 0–100%. Values have not been clamped and are not biologically feasible efficiencies.")
        st.caption("Model estimates, not guarantees of experimental success. A higher prediction alone does not establish that a guide is experimentally superior.")

        selected = st.selectbox("Select guide for SHAP explanation", range(count), format_func=lambda i: rows[i]["Guide"], key="compare_selected")
        if st.session_state.get("comparison_shap", {}).get("selected") != selected:
            st.session_state.pop("comparison_shap", None)
        # Clear a failure message when switching guides as well.
        if st.session_state.get("comparison_error", {}).get("selected") != selected:
            st.session_state.pop("comparison_error", None)
        if st.button("Explain selected guide", key="compare_explain"):
            st.session_state.pop("comparison_shap", None)
            st.session_state.pop("comparison_error", None)
            try:
                with st.spinner("Calculating the selected guide's explanation…"):
                    started = perf_counter()
                    explanation, seconds = cached_explainer().explain(*snapshot[selected])
                    from explanations import contribution_summary, waterfall_png
                    increases, decreases = contribution_summary(explanation)
                    st.session_state.comparison_shap = {
                        "selected": selected, "image": waterfall_png(explanation),
                        "increases": increases, "decreases": decreases,
                        "baseline": float(explanation.base_values),
                        "seconds": seconds, "total_seconds": perf_counter() - started,
                    }
            except Exception:
                logging.exception("Comparison SHAP failed")
                st.session_state.comparison_error = {"selected": selected}
        if "comparison_error" in st.session_state:
            st.error("Explanation unavailable for this guide. Comparison predictions remain available; see the terminal for details.")
        if "comparison_shap" in st.session_state:
            detail = st.session_state.comparison_shap
            st.markdown(f"**Why did the model predict {rows[selected]['Predicted efficiency (%)']:.2f}% for {rows[selected]['Guide']}?**")
            st.caption(f"Sequence: {rows[selected]['Sequence']}")
            st.write("SHAP shows how five input groups move this ensemble prediction from the training-background mean. Contributions are in percentage points.")
            st.image(detail["image"], width="stretch")
            st.write("**Increasing the estimate:** " + detail["increases"])
            st.write("**Decreasing the estimate:** " + detail["decreases"])
            st.caption(f"Background mean: {detail['baseline']:.2f}% · Calculation: {detail['seconds']:.2f}s · Total: {detail['total_seconds']:.2f}s")
            st.caption("The existing grouped SHAP method explains the complete four-model ensemble. Sequence features stay together. SHAP describes model behaviour, not biological causation; some sequence/context combinations may be unobserved experimentally.")
