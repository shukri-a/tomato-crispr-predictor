"""Focused integration checks: python verify_comparison.py."""
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from explanations import EnsembleExplainer
from predictor import ROOT, example_inputs, load_model, predict, reference


def verify():
    model = load_model()
    with patch.object(EnsembleExplainer, "explain", autospec=True, side_effect=EnsembleExplainer.explain) as calls:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        app.button(key="compare_submit").click().run()
        assert len(app.error) == 2 and not app.dataframe and not app.exception
        calls.assert_not_called()
        app.button(key="compare_examples").click().run()
        app.button(key="compare_submit").click().run()
        assert not app.error and not app.exception
        table = app.dataframe[0].value
        assert len(table) == 2
        for i, row in table.iterrows():
            np.testing.assert_allclose(row["Predicted efficiency (%)"], predict(model, **example_inputs(i)), atol=1e-10, rtol=0)
        assert any("different biological contexts" in warning.value for warning in app.warning)
        calls.assert_not_called()
        print("PASS: two-guide reference comparison matches individual predictions; differing contexts flagged; SHAP not calculated automatically.")

        for i in (1, 0):
            app.selectbox(key="compare_selected").set_value(i).run()
            assert not app.get("imgs")
            app.button(key="compare_explain").click().run()
            assert not app.error and not app.exception and len(app.get("imgs")) == 1
            assert calls.call_args.args[1:] == tuple(example_inputs(i)[k] for k in ("sequence", "region", "chromatin", "leaf", "t0"))
            assert app.session_state["comparison_shap"]["selected"] == i
        called = calls.call_count
        app.run()
        assert calls.call_count == called and len(app.get("imgs")) == 1
        print("PASS: selected-guide SHAP receives the correct five inputs; switching selection clears the old explanation; ordinary reruns reuse it.")

        app.text_input(key="compare_A_sequence").set_value("  " + example_inputs(0)["sequence"].lower() + "  ").run()
        assert not app.dataframe and not app.get("imgs")
        app.button(key="compare_submit").click().run()
        np.testing.assert_allclose(app.dataframe[0].value.iloc[0]["Predicted efficiency (%)"], reference()["expected_predictions"][0], atol=1e-6, rtol=0)
        app.selectbox(key="compare_B_leaf").set_value(None).run()
        assert not app.dataframe and not app.get("imgs")
        app.button(key="compare_submit").click().run()
        assert any("Guide B" in error.value for error in app.error) and not app.dataframe
        assert calls.call_count == called
        print("PASS: sequence/context edits clear results; lowercase/whitespace accepted; missing context blocks the entire comparison.")

        app.radio(key="compare_count").set_value(3).run()
        app.button(key="compare_examples").click().run()
        app.button(key="compare_submit").click().run()
        assert not app.error and not app.exception and len(app.dataframe[0].value) == 3
        print("Three-guide demonstration:")
        for i, row in app.dataframe[0].value.iterrows():
            np.testing.assert_allclose(row["Predicted efficiency (%)"], predict(model, **example_inputs(i)), atol=1e-10, rtol=0)
            print(f"  {row['Guide']}: {row['Sequence']} -> {row['Predicted efficiency (%)']:.8f}%")
        app.selectbox(key="compare_selected").set_value(2).run()
        app.button(key="compare_explain").click().run()
        assert not app.error and not app.exception
        assert calls.call_args.args[1:] == tuple(example_inputs(2)[k] for k in ("sequence", "region", "chromatin", "leaf", "t0"))
        print("PASS: three-guide comparison matches individual predictions; Guide C SHAP receives Guide C inputs.")

        app.text_input(key="compare_C_sequence").set_value("ACGT").run()
        assert not app.dataframe and not app.get("imgs")
        app.button(key="compare_submit").click().run()
        assert any("Guide C" in error.value for error in app.error) and not app.dataframe
        app.radio(key="compare_count").set_value(2).run()
        app.button(key="compare_submit").click().run()
        assert not app.error and not app.exception and len(app.dataframe[0].value) == 2
        print("PASS: invalid included Guide C blocks comparison; switching to two guides excludes C explicitly and restores valid comparison.")

        for field in ("region", "chromatin", "leaf", "t0"):
            app.selectbox(key=f"compare_B_{field}").set_value(example_inputs(0)[field]).run()
        app.button(key="compare_submit").click().run()
        assert not app.error and not app.warning
        assert any("same biological context" in caption.value for caption in app.caption)
        print("PASS: identical-context comparisons are labeled correctly.")

    with patch.object(EnsembleExplainer, "explain", side_effect=RuntimeError("Simulated comparison explanation failure")):
        app.button(key="compare_explain").click().run()
    assert app.error and len(app.dataframe[0].value) == 2 and not app.exception
    # Single-guide state and actions remain independent of comparison fields.
    app.button[0].click().run()
    app.button[1].click().run()
    assert app.metric[0].value == "35.78%" and len(app.dataframe[0].value) == 2
    app.text_input(key="compare_A_sequence").set_value("invalid").run()
    assert app.metric[0].value == "35.78%" and not app.dataframe
    print("PASS: explanation failure preserves comparison results; comparison changes do not clear the single-guide prediction.")
    print("ALL COMPARISON CHECKS PASSED.")


if __name__ == "__main__":
    verify()
