"""Verify input-only UAT examples reuse the manual prediction/SHAP workflow."""
import csv
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from predictor import ROOT, load_model, predict
from uat_cases import COLUMNS, FIELDS, load_cases
from explanations import EnsembleExplainer


def verify():
    with (ROOT / 'uat_test_cases.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == COLUMNS
        rows = list(reader)
    cases, groups = load_cases()
    assert len(rows) == len(cases) == 14 and len(groups) == 6
    for name in ('uat_actual_outcomes.csv', 'uat_benchmark_results.csv', 'uat_benchmark_summary.md'):
        assert not (ROOT / name).exists()
    model = load_model()
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ['Predict / UAT', 'Compare Guides']
    for case_id, expected in cases.items():
        app.selectbox(key='uat_case_id').set_value(case_id).run()
        app.button(key='uat_load_single').click().run()
        assert app.text_input(key='sequence').value == expected['sequence']
        for field in FIELDS[1:]:
            assert app.selectbox(key=field).value == expected[field]
        assert not app.metric and 'shap_result' not in app.session_state
        app.button(key='single_predict').click().run()
        assert not app.error and not app.exception
        assert app.metric[0].value == f"{predict(model, **expected):.2f}%"
    print('PASS: all 14 UAT cases populate the same five manual fields and match individual predictions.')
    with patch.object(EnsembleExplainer, 'explain', autospec=True, side_effect=EnsembleExplainer.explain) as calls:
        app.button(key='explain_prediction').click().run()
        assert not app.error and not app.exception
        assert calls.call_args.args[1:] == tuple(cases['G06-C'][k] for k in FIELDS)
    assert 'shap_result' in app.session_state
    app.selectbox(key='uat_case_id').set_value('G01-A').run()
    app.button(key='uat_load_single').click().run()
    assert not app.metric and 'shap_result' not in app.session_state
    app.text_input(key='sequence').set_value('ACGT').run()
    app.button(key='single_predict').click().run()
    assert app.error and not app.metric
    app.button(key='uat_load_single').click().run()
    app.button(key='single_predict').click().run()
    assert not app.error and not app.exception
    print('PASS: UAT uses existing SHAP for the loaded guide; reload/edit clears stale results; validation unchanged.')
    # All existing group inputs remain usable in the unchanged comparison interface.
    for ids in groups.values():
        app.radio(key='compare_count').set_value(len(ids)).run()
        for label, case_id in zip('ABC', ids):
            for field, value in cases[case_id].items():
                widget = app.text_input if field == 'sequence' else app.selectbox
                widget(key=f'compare_{label}_{field}').set_value(value)
        app.run()
        app.button(key='compare_submit').click().run()
        assert not app.error and not app.exception
        result = app.dataframe[0].value
        for i, case_id in enumerate(ids):
            np.testing.assert_allclose(result.iloc[i]['Predicted efficiency (%)'], predict(model, **cases[case_id]), atol=1e-10, rtol=0)
    print('PASS: all six UAT groups work in Compare Guides with each recorded context.')
    print('PASS: only input-only UAT CSV bundled; hidden outcomes and benchmark files excluded.')
    print('ALL UAT WORKFLOW CHECKS PASSED.')


if __name__ == '__main__':
    verify()
