"""Personalised twin: EHR-default parameters and recovery of known parameters on a tiny simulated patient."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.ehr_features import generate_cohort
from src.data.synthetic_gen import apply_cgm_sensor, generate_schedule
from src.twin import personalise as P

CFG = load_config()
DT = CFG["sampling"]["interval_minutes"]
CAL = CFG["simulation"]["calibration"]
WARM = 24 * 60 // DT
FIT_DAYS, HELD_DAYS = 7, 2


@pytest.fixture(scope="module")
def cohort() -> tuple[pd.DataFrame, pd.DataFrame]:
    ehr, _ = generate_cohort(60, np.random.default_rng(7))
    return ehr, P.ehr_default_params(ehr, CAL, DT)


def test_ehr_defaults_are_deterministic_population_values(cohort) -> None:
    ehr, default = cohort
    pd.testing.assert_frame_equal(default, P.ehr_default_params(ehr, CAL, DT))
    plain = ((~ehr.on_metformin) & (~ehr.on_insulin) & (~ehr.on_sulfonylurea)).to_numpy()
    assert plain.any()
    expect = np.clip(104 + 18 * (ehr.hba1c.to_numpy() - 6.5), 90, 230)        # g_target with no random draw
    np.testing.assert_allclose(default.g_basal.to_numpy()[plain], expect[plain])
    np.testing.assert_allclose(default.meal_gain, np.exp(0.22 * (ehr.hba1c - 7.5) + 0.03 * (ehr.bmi - 26.5)))
    assert (default.drug_sens > 0).all() and default.notna().all().all()


def test_fit_recovers_known_parameters(cohort) -> None:
    """Simulate a drug-treated patient with parameters away from the EHR default, fit them back on the first
    week, and check the held-out days.

    All four parameters are moved at once along a direction where g_basal and tau_clear_min partly trade off
    (see docs/twin_personalisation.md, identifiability), so those two only need to move toward the truth;
    drug sensitivity and carbohydrate gain are well identified and must be recovered closely.
    """
    ehr, default = cohort
    i = int(np.flatnonzero((ehr.on_sulfonylurea | ehr.on_insulin).to_numpy())[0])
    prior = default.iloc[[i]].reset_index(drop=True)
    true = prior.copy()
    factors = {"g_basal": 1.12, "meal_gain": 0.8, "tau_clear_min": 1.2, "drug_sens": 1.3}
    for n, f in factors.items():
        true[n] = true[n] * f
    s = generate_schedule(true.iloc[0], ehr.iloc[i], 1 + FIT_DAYS + HELD_DAYS, DT, np.random.default_rng(3))
    rng = np.random.default_rng(4)
    g = P.simulate_schedule(true, s, DT, CAL, noise=rng.normal(0, CAL["process_noise_sd"], (s.n, 1)))
    cgm = apply_cgm_sensor(g, DT, rng)[0, WARM:]

    fit = P.fit_parameters(s, cgm, prior, CAL, DT, WARM, FIT_DAYS * 24 * 60 // DT, fit_drug=True,
                           prior_weight=CFG["twin"]["personalise"]["prior_weight"])
    got, want, ehr_guess = fit.params.iloc[0], true.iloc[0], prior.iloc[0]
    assert fit.info["seconds"] < 10 and fit.info["fit_mae"] < 6          # close to the sensor noise floor
    for n in P.FIT_PARAMS:                                               # every parameter moves toward the truth
        assert abs(np.log(got[n] / want[n])) < abs(np.log(ehr_guess[n] / want[n])), n
    for n, tol in {"drug_sens": 0.12, "meal_gain": 0.15}.items():
        assert abs(got[n] / want[n] - 1) < tol, f"{n}: fitted {got[n]:.4g}, true {want[n]:.4g}"

    held = slice(FIT_DAYS * 24 * 60 // DT, None)                         # days never seen by the fit
    m = ~np.isnan(cgm[held])
    sims = P.sensor_view(P.simulate_schedule(pd.concat([fit.params, prior], ignore_index=True), s, DT, CAL), DT)
    mae_fit, mae_ehr = (np.mean(np.abs(x[WARM:][held][m] - cgm[held][m])) for x in sims)
    assert mae_fit < mae_ehr


def test_sensor_view_matches_simulator_lag() -> None:
    g = np.r_[np.full(10, 100.0), np.full(30, 200.0)][None]
    v = P.sensor_view(g, DT)
    assert v[0, 9] == pytest.approx(100) and 100 < v[0, 10] < 200 and v[0, -1] == pytest.approx(200, abs=1e-3)
    obs = P.observer_target(np.arange(50.0), warm=3, n_total=60, until=20, dt=DT)
    assert obs[3] == 2.0 and np.isnan(obs[3 + 19]) and obs[3 + 18] == 20.0           # only CGM up to 'until'
