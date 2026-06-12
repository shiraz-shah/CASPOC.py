import numpy as np

from caspoc import CASPOC


def test_caspoc_returns_expected_tables():
    rng = np.random.default_rng(12)
    latent = rng.normal(size=(36, 2))
    X = latent @ rng.normal(size=(2, 10)) + 0.2 * rng.normal(size=(36, 10))
    Y = latent @ rng.normal(size=(2, 4)) + 0.2 * rng.normal(size=(36, 4))

    model = CASPOC(
        n_components=2,
        n_repeats=3,
        n_folds=4,
        keep_x_options=[3],
        keep_y_options=[2],
        random_state=100,
    ).fit(X, Y)

    assert model.tune_correlations_.shape[0] == 6
    assert model.test_correlations_.shape[0] == 6
    assert {"Repeat", "KeepX", "KeepY", "Component", "Correlation", "Pvalue", "n"}.issubset(
        model.tune_correlations_.columns
    )
    assert model.train_loadings_x_["Variable"].nunique() == 10
    assert model.train_loadings_y_["Variable"].nunique() == 4
    assert model.tune_scores_x_.shape[0] == 3 * 4 * 9
    assert model.test_scores_y_.shape[0] == 3 * 4 * 9
    assert model.yhat_test_.shape[0] == 3 * 4 * 9
    assert model.result_.test_correlations is model.test_correlations_
