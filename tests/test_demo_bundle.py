"""The deployed dashboard runs from the committed demo bundle alone (what Streamlit Community Cloud does).

Runs in CI, where data/processed does not exist, so every push checks that the live app still starts and
that each view, the what-if twin and the real-patient-trained model work from the bundle.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "app" / "dashboard.py"
BUNDLE = ROOT / "demo_bundle" / "processed" / "series.npz"

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not BUNDLE.exists(), reason="demo bundle not built")]


@pytest.fixture()
def bundle_app(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("TWIN_USE_DEMO_BUNDLE", "1")
    st.cache_resource.clear()           # never reuse a context built from data/processed by another test
    st.cache_data.clear()
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    yield at
    st.cache_resource.clear()
    st.cache_data.clear()


def test_runs_from_bundle_with_every_view(bundle_app) -> None:
    at = bundle_app
    assert not at.exception, [e.value for e in at.exception]
    assert any("demo bundle" in c.value for c in at.sidebar.caption)
    assert [t.label for t in at.tabs] == ["Live replay", "Glucose timeline", "Alert drivers", "What-if", "Patient overview"]
    at.button(key="start_whatif").click().run()
    next(r for r in at.radio if r.label == "Scenario").set_value("Extra insulin dose").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("80% band" in c.value for c in at.caption)


def test_real_patient_trained_model_ships_in_bundle(bundle_app) -> None:
    at = bundle_app
    radio = next(r for r in at.radio if r.label == "Risk model")
    real = [o for o in radio.options if o.startswith("Real-patient-trained")]
    assert real, "lgbm_cgm_shanghai is missing from demo_bundle (python scripts/make_demo_bundle.py)"
    radio.set_value(real[0]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("trained on 100 real patients" in c.value for c in at.sidebar.caption)
