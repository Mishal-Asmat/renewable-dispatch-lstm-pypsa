import numpy as np
import pytest

torch = pytest.importorskip("torch")

import lstm_model as lm


def test_model_output_shape():
    m = lm.ResidualLSTM(n_features=6, n_horizons=4, hidden_size=8, num_layers=2, dropout=0.1)
    out = m(torch.zeros(5, 48, 6))
    assert out.shape == (5, 4)


def test_training_reduces_loss_and_is_reproducible():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(256, 12, 6)).astype(np.float32)
    y = np.stack([X[:, -1, 0], 2 * X[:, -1, 1], -X[:, -1, 2], 0.5 * X[:, -1, 3]], axis=1).astype(np.float32)
    log1, log2 = [], []
    m1 = lm.train_one(X, y, X, y, seed=3, max_epochs=6, log=log1, tag="t")
    m2 = lm.train_one(X, y, X, y, seed=3, max_epochs=6, log=log2, tag="t")
    assert log1[-1]["train_loss"] < log1[0]["train_loss"]
    np.testing.assert_allclose(lm.predict(m1, X[:8]), lm.predict(m2, X[:8]), atol=1e-6)
