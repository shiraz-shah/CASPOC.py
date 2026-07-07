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


def test_sparse_pls_predict_uses_latent_score_regression():
    rng = np.random.default_rng(14)
    latent = rng.normal(size=(35, 2))
    X = latent @ rng.normal(size=(2, 10)) + 0.1 * rng.normal(size=(35, 10))
    Y = latent @ rng.normal(size=(2, 2)) + 0.1 * rng.normal(size=(35, 2))

    model = SparsePLS(n_components=2, keep_x=[4, 4], keep_y=[2, 2]).fit(X, Y)

    expected = model.transform_x(X[:9]) @ model.score_coef_

    assert model.score_coef_.shape == (2, 2)
    np.testing.assert_allclose(model.predict(X[:9]), expected)
