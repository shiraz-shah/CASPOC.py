import numpy as np

from caspoc import SparsePLS


def test_sparse_pls_respects_keep_counts():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(40, 12))
    Y = rng.normal(size=(40, 5))

    model = SparsePLS(n_components=2, keep_x=[3, 4], keep_y=[2, 3]).fit(X, Y)

    assert model.x_weights_.shape == (12, 2)
    assert model.y_weights_.shape == (5, 2)
    assert np.count_nonzero(model.x_weights_[:, 0]) <= 3
    assert np.count_nonzero(model.x_weights_[:, 1]) <= 4
    assert np.count_nonzero(model.y_weights_[:, 0]) <= 2
    assert np.count_nonzero(model.y_weights_[:, 1]) <= 3


def test_sparse_pls_transform_and_predict_shapes():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 8))
    Y = rng.normal(size=(30, 3))

    model = SparsePLS(n_components=2, keep_x=4, keep_y=2).fit(X, Y)

    assert model.transform_x(X[:7]).shape == (7, 2)
    assert model.transform_y(Y[:7]).shape == (7, 2)
    assert model.predict(X[:7]).shape == (7, 3)
