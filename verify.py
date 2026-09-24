"""Focused offline checks. Run: python verify.py (no training data needed)."""

import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from predictor import ROOT, build_features, example_inputs, load_model, predict, reference


def rejected(inputs):
    try:
        build_features(**inputs)
    except ValueError:
        return
    raise AssertionError(f"Invalid input was accepted: {inputs}")


def verify():
    print("Python:", platform.python_version())
    for package in ("streamlit", "scikit-learn", "catboost", "pandas", "numpy", "scipy", "biopython", "joblib"):
        print(f"{package}=={importlib.metadata.version(package)}")
    print("Model SHA256:", hashlib.sha256((ROOT / "final_ensemble.joblib").read_bytes()).hexdigest())
    data = reference()
    model = load_model()
    print("PASS: model loads; four equally weighted fitted pipelines; 72 features.")
    assert len(data["sample_inputs"]) == len(data["guide_sequences"]) == len(data["expected_predictions"]) == 5
    errors, frames = [], []
    for i, expected_prediction in enumerate(data["expected_predictions"]):
        inputs = example_inputs(i)
        actual = build_features(**inputs)
        expected = pd.DataFrame([data["sample_inputs"][i]], columns=data["feature_names"]).astype(data["feature_dtypes"])
        pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_dtype=True)
        assert actual.shape == (1, 72)
        assert int(actual.filter(like="trinuc_").sum(axis=1).iloc[0]) == 18
        prediction = predict(model, **inputs)
        np.testing.assert_allclose(prediction, expected_prediction, atol=1e-6, rtol=0)
        error = abs(prediction - expected_prediction)
        errors.append(error)
        frames.append(actual)
        print(f"PASS: example {i + 1}; all feature values/order/dtypes match; prediction={prediction:.15f}; abs_error={error:.3e}")
    np.testing.assert_allclose(model.predict(pd.concat(frames, ignore_index=True)), data["expected_predictions"], atol=1e-6, rtol=0)

    base = example_inputs(0)
    pd.testing.assert_frame_equal(build_features(**base), build_features(**(base | {"sequence": " \n" + base["sequence"].lower() + "\t "})))
    for sequence in (None, "", "A" * 19, "A" * 21, "A" * 19 + "N", "A" * 19 + "U", "A" * 9 + " " + "T" * 10):
        rejected(base | {"sequence": sequence})
    for key in ("region", "chromatin", "leaf", "t0"):
        for missing in (None, ""):
            rejected(base | {key: missing})
    for key, value in (("region", "Unknown"), ("chromatin", "False"), ("chromatin", 0), ("leaf", "medium"), ("t0", "unknown")):
        rejected(base | {key: value})
    for sequence, gc in (("A" * 20, 0.0), ("G" * 20, 100.0)):
        frame = build_features(**(base | {"sequence": sequence, "chromatin": True}))
        assert frame.at[0, "max_homopolymer_run"] == 20
        assert frame.at[0, "gc_content"] == gc
        assert frame.at[0, "trinuc_" + sequence[:3]] == 18
        assert bool(frame.at[0, "within_atac_peak"]) is True
    print("PASS: valid/lowercase/whitespace, invalid characters/lengths, missing/invalid context, True/False and overlapping homopolymers.")

    class FixedOutput:
        def __init__(self, value):
            self.value = value

        def predict(self, features):
            return [self.value]

    for value in (-1.0, 101.0):
        assert predict(FixedOutput(value), **base) == value
    try:
        predict(FixedOutput(float("nan")), **base)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Non-finite prediction accepted")
    print("PASS: out-of-range predictions preserved; non-finite predictions rejected.")

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    for key in ("region", "chromatin", "leaf", "t0"):
        assert app.selectbox(key=key).value is None
    app.button[1].click().run()
    assert app.error and not app.metric and not app.exception
    for i in range(5):
        app.selectbox(key="example").set_value(i).run()
        app.button[0].click().run()
        assert app.text_input(key="sequence").value == example_inputs(i)["sequence"]
        for key in ("region", "chromatin", "leaf", "t0"):
            assert app.selectbox(key=key).value == example_inputs(i)[key]
        app.button[1].click().run()
        assert not app.exception and not app.error
        assert app.metric[0].value == f"{data['expected_predictions'][i]:.2f}%"
    app.text_input(key="sequence").set_value("  " + example_inputs(4)["sequence"].lower() + "  ").run()
    assert not app.metric  # Editing an input clears the previous result.
    app.button[1].click().run()
    assert not app.error and app.metric[0].value == f"{data['expected_predictions'][4]:.2f}%"
    app.text_input(key="sequence").set_value("ACGT").run()
    app.button[1].click().run()
    assert app.error and not app.metric and not app.exception
    app.button[0].click().run()
    app.selectbox(key="leaf").set_value(None).run()
    app.button[1].click().run()
    assert app.error and not app.metric and not app.exception
    print("PASS: Streamlit renders; five example predictions, manual lowercase input, empty/invalid submissions and stale-result clearing work.")

    # Start a real local server in addition to executing the UI with AppTest.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryFile(mode="w+") as log:
        server = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
            "--server.address=127.0.0.1", f"--server.port={port}",
            "--server.headless=true", "--browser.gatherUsageStats=false",
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    log.seek(0)
                    raise AssertionError(log.read())
                try:
                    with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                        assert response.status == 200 and response.read() == b"ok"
                    with urlopen(f"http://127.0.0.1:{port}/", timeout=1) as response:
                        assert response.status == 200
                    break
                except OSError:
                    time.sleep(0.2)
            else:
                raise AssertionError("Streamlit did not start within 30 seconds")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    print("PASS: real local Streamlit server starts and responds successfully.")
    print(f"ALL CHECKS PASSED. Maximum reference prediction absolute error: {max(errors):.3e}")
    print("Reference checks verify implementation consistency, not independent predictive performance.")


if __name__ == "__main__":
    verify()
