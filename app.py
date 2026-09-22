
# --------------------------------------------------
# IMPORT LIBRARIES
# --------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from itertools import product


# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Tomato CRISPR Editing Efficiency Predictor",
    page_icon="🧬",
    layout="centered"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

model_package = joblib.load("/content/crispr_model.joblib")

final_ensemble = model_package["model"]
model_features = model_package["model_features"]

# Load SHAP explainability information
shap_background = model_package["shap_background"]
shap_feature_names = model_package["shap_feature_names"]


# --------------------------------------------------
# SHAP EXPLAINABILITY SETUP
# --------------------------------------------------

def numeric_ensemble_predict(data):

    predictions = []

    for fitted_pipeline in final_ensemble.estimators_:

        fitted_model = fitted_pipeline.named_steps["model"]

        predictions.append(
            fitted_model.predict(data)
        )

    return np.mean(predictions, axis=0)


shap_explainer = shap.Explainer(
    numeric_ensemble_predict,
    shap_background,
    algorithm="permutation",
    feature_names=shap_feature_names
)


# --------------------------------------------------
# GUIDE RNA FUNCTIONS
# --------------------------------------------------

def validate_guide_sequence(sequence):

    sequence = sequence.strip().upper()

    if len(sequence) != 20:
        return False, "Guide RNA sequence must contain exactly 20 bases."

    if not all(base in "ATGC" for base in sequence):
        return False, "Guide RNA sequence can only contain A, T, G or C."

    return True, "Valid guide RNA sequence."


def calculate_gc_content(sequence):

    gc_count = sequence.count("G") + sequence.count("C")

    return gc_count / len(sequence)


def calculate_max_homopolymer_run(sequence):

    max_run = 1
    current_run = 1

    for i in range(1, len(sequence)):

        if sequence[i] == sequence[i - 1]:

            current_run += 1
            max_run = max(max_run, current_run)

        else:

            current_run = 1

    return max_run


def get_first_base(sequence):

    return sequence[0]


def get_last_base(sequence):

    return sequence[-1]


def calculate_trinucleotide_counts(sequence):

    # Create all 64 possible trinucleotides
    trinucleotides = [
        "".join(x)
        for x in product("ACGT", repeat=3)
    ]

    counts = {
        f"trinuc_{tri}": 0
        for tri in trinucleotides
    }

    # Count each trinucleotide in the guide sequence
    for i in range(len(sequence) - 2):

        tri = sequence[i:i + 3]

        counts[f"trinuc_{tri}"] += 1

    return counts


# --------------------------------------------------
# PREPARE SCIENTIST INPUT FOR MODEL
# --------------------------------------------------

def prepare_new_guide(
    guide_sequence,
    target_region,
    within_atac_peak,
    leaf_expression,
    t0_expression
):

    guide_sequence = guide_sequence.strip().upper()

    new_guide = {

        "feature": target_region,

        "leaf_exp": leaf_expression,

        "t0_exp": t0_expression,

        "within_atac_peak": within_atac_peak,

        "gc_content": calculate_gc_content(
            guide_sequence
        ),

        "max_homopolymer_run":
            calculate_max_homopolymer_run(
                guide_sequence
            ),

        "first_base": get_first_base(
            guide_sequence
        ),

        "last_base": get_last_base(
            guide_sequence
        )
    }

    # Automatically create the 64 trinucleotide features
    new_guide.update(
        calculate_trinucleotide_counts(
            guide_sequence
        )
    )

    new_guide_df = pd.DataFrame(
        [new_guide]
    )

    # Keep exactly the features expected by the model
    new_guide_df = new_guide_df[
        model_features
    ]

    return new_guide_df


# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------

st.title(
    "🧬 Tomato CRISPR Editing Efficiency Predictor"
)

st.write(
    "A machine learning tool for predicting CRISPR-Cas9 "
    "editing efficiency in tomato guide RNAs."
)

st.info(
    "Enter the guide RNA sequence and biological information "
    "below to generate a predicted editing efficiency."
)


guide_sequence = st.text_input(
    "Guide RNA Sequence",
    placeholder="Enter a 20-base sequence (A, T, G, C)",
    max_chars=20
)


target_region_display = st.selectbox(
    "Target Region",
    [
        "Exon",
        "Intron",
        "Promoter"
    ]
)


within_atac_display = st.radio(
    "Within ATAC Peak?",
    [
        "No",
        "Yes"
    ]
)


leaf_display = st.selectbox(
    "Leaf Expression",
    [
        "Low",
        "Medium",
        "High"
    ]
)


t0_display = st.selectbox(
    "T0 Expression",
    [
        "Low",
        "Medium",
        "High"
    ]
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "Predict Editing Efficiency",
    type="primary"
):

    valid, message = validate_guide_sequence(
        guide_sequence
    )

    if not valid:

        st.error(message)

    else:

        # Convert user-friendly labels to dataset labels
        region_mapping = {

            "Exon": "Exo",

            "Intron": "Int",

            "Promoter": "Pro"
        }

        expression_mapping = {

            "Low": "low",

            "Medium": "med",

            "High": "high"
        }

        target_region = region_mapping[
            target_region_display
        ]

        within_atac_peak = (
            within_atac_display == "Yes"
        )

        leaf_expression = expression_mapping[
            leaf_display
        ]

        t0_expression = expression_mapping[
            t0_display
        ]


        # --------------------------------------------------
        # PREPARE INPUT
        # --------------------------------------------------

        new_guide = prepare_new_guide(
            guide_sequence,
            target_region,
            within_atac_peak,
            leaf_expression,
            t0_expression
        )


        # --------------------------------------------------
        # GENERATE PREDICTION
        # --------------------------------------------------

        prediction = final_ensemble.predict(
            new_guide
        )[0]

        # Keep prediction within percentage range
        prediction = max(
            0,
            min(
                100,
                prediction
            )
        )


        # --------------------------------------------------
        # PREPARE SHAP INPUT
        # --------------------------------------------------

        fitted_preprocessor = (
            final_ensemble
            .estimators_[0]
            .named_steps["preprocessor"]
        )

        guide_numeric = (
            fitted_preprocessor
            .transform(new_guide)
        )


        # --------------------------------------------------
        # CALCULATE SHAP VALUES
        # --------------------------------------------------

        guide_shap_values = shap_explainer(
            guide_numeric,
            max_evals=200
        )


        # --------------------------------------------------
        # IDENTIFY STRONGEST CONTRIBUTORS
        # --------------------------------------------------

        shap_contributions = pd.DataFrame({

            "Feature":
                shap_feature_names,

            "SHAP Value":
                guide_shap_values.values[0]
        })

        shap_contributions[
            "Absolute Impact"
        ] = (
            shap_contributions[
                "SHAP Value"
            ].abs()
        )

        top_shap_features = (
            shap_contributions
            .sort_values(
                by="Absolute Impact",
                ascending=False
            )
            .head(5)
        )


        # --------------------------------------------------
        # DISPLAY PREDICTION
        # --------------------------------------------------

        st.success(
            f"Predicted Editing Efficiency: "
            f"{prediction:.2f}%"
        )

        st.caption(
            "This prediction is intended as a "
            "decision-support estimate for tomato "
            "CRISPR-Cas9 guide RNAs."
        )


        # --------------------------------------------------
        # DISPLAY SHAP EXPLANATION
        # --------------------------------------------------

        st.subheader(
            "Why did the model make this prediction?"
        )

        st.write(
            "The features below had the strongest influence "
            "on this guide's predicted editing efficiency:"
        )

        for _, row in top_shap_features.iterrows():

            feature_name = (
                row["Feature"]
                .replace(
                    "categorical__",
                    ""
                )
                .replace(
                    "boolean__",
                    ""
                )
                .replace(
                    "numerical__",
                    ""
                )
                .replace(
                    "trinucleotide__",
                    ""
                )
                .replace(
                    "trinuc_",
                    ""
                )
            )

            if row["SHAP Value"] > 0:

                direction = "increased"

            else:

                direction = "decreased"

            st.write(
                f"• **{feature_name}** — "
                f"{direction} the predicted "
                f"editing efficiency"
            )


        st.caption(
            "These explanations describe how the model made "
            "its prediction and do not establish biological causation."
        )
