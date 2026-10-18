"""Transfer-experiment helpers on a tiny random fixture (no real or synthetic pipeline data)."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.eval.metrics import auprc
from src.eval.transfer_cgmacros import (Transfer, balanced_weight, draw_participants, mean_over_draws,
                                        train_lgbm_transfer, transfer_proba)
from src.models.baselines import train_lgbm


@pytest.fixture(scope="module")
def cfg() -> dict:
    c = copy.deepcopy(load_config())
    c["models"]["lgbm"].update(min_child_samples=5, n_estimators=60, early_stopping_rounds=10, num_threads=1)
    return c


def _data(n: int, shift: float, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = (X.a + shift * X.b + rng.normal(scale=0.5, size=n) > 0).astype(int).to_numpy()
    return X, y


def test_plain_transfer_training_matches_train_lgbm(cfg):
    X, y = _data(400, 0.5, 0)
    b1 = train_lgbm(X[:300], y[:300], X[300:], y[300:], cfg)
    b2 = train_lgbm_transfer(X[:300], y[:300], X[300:], y[300:], cfg)
    np.testing.assert_allclose(b1.predict(X, num_iteration=b1.best_iteration), transfer_proba(b2, X), rtol=1e-10)


def test_finetune_boosts_from_the_init_scores(cfg):
    X, y = _data(400, 1.0, 1)
    init = np.where(X.a > 0, 1.0, -1.0)                       # a crude "pretrained" model, as log-odds
    b = train_lgbm_transfer(X[:300], y[:300], X[300:], y[300:], cfg, init_tr=init[:300], init_va=init[300:])
    p = transfer_proba(b, X[300:], init[300:])
    raw_new = b.predict(X[300:], num_iteration=b.best_iteration, raw_score=True)
    np.testing.assert_allclose(p, 1 / (1 + np.exp(-(init[300:] + raw_new))))
    assert b.best_iteration >= 1 and np.all((p > 0) & (p < 1))
    assert auprc(y[300:], p) >= auprc(y[300:], init[300:]) - 0.05   # the new trees correct, not destroy, the init


def test_balanced_weight():
    assert balanced_weight(1000, 50) == 20.0
    assert balanced_weight(10, 0) == 10.0


def test_draw_participants_nested_and_disjoint():
    rest = np.array([f"p{i}" for i in range(13)])
    val, trains = draw_participants(rest, (2, 4, 7, 10), 3, np.random.default_rng(0))
    assert len(val) == 3 and [len(trains[k]) for k in (2, 4, 7, 10)] == [2, 4, 7, 10]
    assert set(trains[2]) < set(trains[4]) < set(trains[7]) < set(trains[10])
    assert not set(val) & set(trains[10]) and set(val) | set(trains[10]) == set(rest)
    with pytest.raises(ValueError):
        draw_participants(rest, (11,), 3, np.random.default_rng(0))


def test_mean_over_draws():
    y = np.array([0, 1, 0, 1])
    S = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    f = mean_over_draws(auprc)
    assert f(y, S) == pytest.approx((auprc(y, S[:, 0]) + auprc(y, S[:, 1])) / 2)
    assert f(y, S[:, 0]) == pytest.approx(auprc(y, S[:, 0]))


def test_transfer_methods_score_only_the_test_participant(cfg):
    X, y = _data(600, 0.5, 2)
    pids = np.repeat([f"p{i}" for i in range(6)], 100)
    Xs, ys = _data(500, 0.0, 3)
    syn = train_lgbm(Xs[:400], ys[:400], Xs[400:], ys[400:], cfg)
    exp = Transfer(X, y, pids, Xs, ys, syn, cfg)
    for m in ("real_only", "finetune", "pooled", "pooled_unweighted"):
        s = exp.fit_score(m, np.array(["p0", "p1"]), np.array(["p2", "p3"]), np.array(["p5"]))
        assert s.shape == (100,) and np.all((s >= 0) & (s <= 1))
    lc = exp.learning_curve((1, 2), n_draws=2, methods=("real_only", "finetune"))
    assert set(lc) == {(m, k) for m in ("real_only", "finetune") for k in (1, 2)}
    assert all(S.shape == (600, 2) and np.isfinite(S).all() for S in lc.values())
