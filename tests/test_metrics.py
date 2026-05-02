import numpy as np

from bnn_sim.metrics import binary_auc, soma_rmse


def test_soma_rmse_uses_second_output_channel():
    prediction = np.array([[[0.0, 1.0], [0.0, 3.0]]], dtype=np.float32)
    target = np.array([[[0.0, 2.0], [0.0, 5.0]]], dtype=np.float32)

    assert np.isclose(soma_rmse(prediction, target), np.sqrt(2.5))


def test_binary_auc_handles_perfect_ranking():
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    scores = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)

    assert binary_auc(scores, labels) == 1.0
