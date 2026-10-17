"""Stage 4 tests: fusion model shapes, static switch, losses, training, persistence of weights."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.config import load_config
from src.data.build_dataset import build_dataset
from src.data.ehr_features import STATIC_FEATURES
from src.data.synthetic_gen import generate_dataset
from src.data.wearable_features import SEQUENCE_FEATURES
from src.models.fusion_model import FusionNet, MultiTaskLoss, focal_loss
from src.models.train import WindowSource, load_model, predict, save_model, train_model

N_SEQ, N_STATIC, H = len(SEQUENCE_FEATURES), len(STATIC_FEATURES), 72


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def small_ds(cfg: dict) -> dict:
    return build_dataset(cfg, generate_dataset(cfg, 24, 3))


def _tiny_cfg(cfg: dict, tmp_path) -> dict:
    c = copy.deepcopy(cfg)
    c["fusion"].update(max_epochs=2, patience=1, seq_hidden=16, static_hidden=16, static_out=8,
                       shared_hidden=16, cnn_channels=8, batch_size=256)
    c["paths"]["models_dir"] = str(tmp_path)
    return c


@pytest.mark.parametrize("encoder", ["gru", "cnn"])
def test_forward_shapes(cfg: dict, encoder: str) -> None:
    c = copy.deepcopy(cfg)
    c["fusion"]["encoder"] = encoder
    model = FusionNet(N_SEQ, N_STATIC, c)
    out = model(torch.randn(5, H, N_SEQ), torch.randn(5, N_STATIC))
    assert set(out) == {"hyper", "hypo"} and all(v.shape == (5,) for v in out.values())


def test_sequence_only_ignores_static(cfg: dict) -> None:
    torch.manual_seed(0)
    model = FusionNet(N_SEQ, N_STATIC, cfg, use_static=False).eval()
    seq = torch.randn(4, H, N_SEQ)
    a = model(seq, torch.randn(4, N_STATIC))
    b = model(seq, torch.randn(4, N_STATIC))
    assert torch.equal(a["hypo"], b["hypo"])
    assert not hasattr(model, "static_encoder")


def test_fusion_uses_static(cfg: dict) -> None:
    torch.manual_seed(0)
    model = FusionNet(N_SEQ, N_STATIC, cfg).eval()
    seq = torch.randn(4, H, N_SEQ)
    assert not torch.equal(model(seq, torch.zeros(4, N_STATIC))["hypo"], model(seq, torch.ones(4, N_STATIC))["hypo"])
    with pytest.raises(ValueError):
        model(seq)


def test_focal_reduces_to_bce() -> None:
    logits, y = torch.randn(100), (torch.rand(100) < 0.3).float()
    assert focal_loss(logits, y, gamma=0.0).item() == pytest.approx(F.binary_cross_entropy_with_logits(logits, y).item(), rel=1e-5)
    assert focal_loss(logits, y, gamma=2.0).item() < focal_loss(logits, y, gamma=0.0).item()


def test_weighted_loss_upweights_rare_positives(cfg: dict) -> None:
    loss = MultiTaskLoss(cfg, hypo_prevalence=0.04)
    assert loss.pos_weight == pytest.approx((0.96 / 0.04) ** 0.5)
    y_miss = {"hyper": torch.zeros(1), "hypo": torch.ones(1)}        # a missed hypo ...
    y_false = {"hyper": torch.zeros(1), "hypo": torch.zeros(1)}      # ... costs more than a false alarm
    lg_low, lg_high = {"hyper": torch.zeros(1), "hypo": torch.tensor([-2.0])}, {"hyper": torch.zeros(1), "hypo": torch.tensor([2.0])}
    assert loss(lg_low, y_miss) > loss(lg_high, y_false)


def test_batches_align_static_with_patient(cfg: dict, small_ds: dict) -> None:
    src = WindowSource(small_ds, cfg)
    rows = np.array([0, 500, 1500])
    seq, static, y = src.batch(rows)
    assert seq.shape == (3, H, N_SEQ) and torch.isfinite(seq).all()
    for k, r in enumerate(rows):
        pid = small_ds["windows"].patient_id.iloc[r]
        np.testing.assert_allclose(static[k].numpy(), small_ds["static"].loc[pid].to_numpy(np.float32))
    assert torch.equal(y["hypo"], torch.tensor(small_ds["windows"].hypo.iloc[rows].to_numpy(np.float32)))


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["fusion_gru", "seqonly_cnn"])
def test_train_save_load_roundtrip(cfg: dict, small_ds: dict, tmp_path, variant: str) -> None:
    c = _tiny_cfg(cfg, tmp_path)
    model, meta = train_model(small_ds, c, variant, log=False)
    assert meta["best_epoch"] >= 0 and len(meta["history"]) >= 1
    assert meta["use_static"] == variant.startswith("fusion")
    save_model(model, meta, tmp_path)
    loaded, _ = load_model(variant, c)
    src = WindowSource(small_ds, c)
    rows = np.flatnonzero(small_ds["windows"].split.to_numpy() == "test")[:300]
    a, b = predict(model, src, rows), predict(loaded, src, rows)
    np.testing.assert_allclose(a["hypo"], b["hypo"], rtol=1e-6)
    assert ((a["hyper"] > 0) & (a["hyper"] < 1)).all()


@pytest.mark.slow
def test_training_is_deterministic(cfg: dict, small_ds: dict, tmp_path) -> None:
    c = _tiny_cfg(cfg, tmp_path)
    c["fusion"]["max_epochs"] = 1
    m1, _ = train_model(small_ds, c, "fusion_gru", log=False)
    m2, _ = train_model(small_ds, c, "fusion_gru", log=False)
    for (k, v1), v2 in zip(m1.state_dict().items(), m2.state_dict().values()):
        assert torch.equal(v1, v2), k
