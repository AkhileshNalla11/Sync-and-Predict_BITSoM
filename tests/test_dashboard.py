"""Stage 7 smoke test: the dashboard renders every view and every what-if preset without errors.

Needs the pipeline outputs (data/processed + trained models); skipped otherwise.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "app" / "dashboard.py"
NEEDED = [ROOT / "data" / "processed" / "series.npz",
          ROOT / "data" / "processed" / "models" / "lgbm_cgm_wearables_ehr_hypo.txt"]

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="run scripts/run_pipeline.sh first")]


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    return at


def _radio(at, label: str):
    return next(r for r in at.radio if r.label == label)


def test_renders_all_views_with_banner(app) -> None:
    assert not app.exception, [e.value for e in app.exception]
    banner = [m.value for m in app.markdown if 'class="gt-banner"' in m.value]
    assert banner and "SYNTHETIC DATA" in banner[0] and "not for clinical use" in banner[0]
    assert [t.label for t in app.tabs] == ["Glucose timeline", "Alert drivers", "What-if", "Patient overview"]
    assert any("Hypoglycaemia risk" in m.value for m in app.markdown)


@pytest.mark.parametrize("preset", ["Poor sleep night", "Skipped meal", "Extra insulin dose", "Late dinner (+2 h)", "Custom"])
def test_whatif_presets(app, preset: str) -> None:
    _radio(app, "Scenario").set_value(preset).run()
    assert not app.exception, [e.value for e in app.exception]


def test_choose_time_and_patient(app) -> None:
    _radio(app, "Prediction time (now)").set_value("Choose a time").run()
    assert not app.exception
    app.sidebar.checkbox[0].uncheck().run()                       # all test patients
    sb = next(s for s in app.selectbox if s.label == "Virtual patient (test set)")
    sb.set_value(sb.options[-1].split(" ")[0]).run()               # raw value = patient id (labels via format_func)
    assert not app.exception, [e.value for e in app.exception]
