"""Blinded UAT workflow using input-only cases; experimental outcomes stay outside the app."""

import csv
from functools import lru_cache
import re

import streamlit as st

from predictor import ROOT, build_features

COLUMNS = ("test_id", "guide_sequence", "genomic_region", "within_atac_peak", "leaf_exp", "t0_exp")
FIELDS = ("sequence", "region", "chromatin", "leaf", "t0")


@lru_cache(maxsize=1)
def load_cases():
    with (ROOT / "uat_test_cases.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("UAT CSV must contain only the six approved input columns.")
        cases = {}
        for row in reader:
            case_id = row["test_id"]
            if not re.fullmatch(r"G\d{2}-[ABC]", case_id) or case_id in cases:
                raise ValueError("UAT Test IDs must be unique and use the format G01-A.")
            if row["within_atac_peak"] not in ("True", "False"):
                raise ValueError(f"Invalid chromatin accessibility for {case_id}.")
            inputs = {
                "sequence": row["guide_sequence"],
                "region": row["genomic_region"],
                "chromatin": row["within_atac_peak"] == "True",
                "leaf": row["leaf_exp"],
                "t0": row["t0_exp"],
            }
            build_features(**inputs)  # Reuse the production validator unchanged.
            cases[case_id] = inputs

    if not cases:
        raise ValueError("No UAT inputs found.")

    groups = {}
    for case_id in sorted(cases):
        groups.setdefault(case_id.split("-")[0], []).append(case_id)
    for group, ids in groups.items():
        if len(ids) not in (2, 3) or ids != [f"{group}-{c}" for c in "ABC"[: len(ids)]]:
            raise ValueError(f"UAT group {group} must contain Guide A/B and optional C.")
    return cases, groups



def render_uat_loader(clear_single):
    """Populate the same editable fields used by manual prediction."""
    try:
        cases, _ = load_cases()
    except Exception as exc:
        st.error(f"UAT examples unavailable: {exc}. Manual entry remains available.")
        return

    def use_case():
        case_id = st.session_state.uat_case_id
        if case_id is not None:
            clear_single()
            for key, value in cases[case_id].items():
                st.session_state[key] = value

    selected = st.selectbox("UAT Test ID", list(cases), index=None,
                            placeholder="Select an assigned UAT example (optional)", key="uat_case_id")
    st.button("Load UAT example", key="uat_load_single", on_click=use_case, disabled=selected is None)
    st.caption("Loading replaces the five fields below. You can edit them before predicting; the Test ID identifies the source example only.")
