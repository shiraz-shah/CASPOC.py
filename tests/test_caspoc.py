import numpy as np
import pytest

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
    assert {
        "Repeat",
        "KeepX",
        "KeepY",
        "Component",
        "Correlation",
        "Pvalue",
        "n",
    }.issubset(model.tune_correlations_.columns)
    assert model.train_loadings_x_["Variable"].nunique() == 10
    assert model.train_loadings_y_["Variable"].nunique() == 4
    assert model.tune_scores_x_.shape[0] == 3 * 4 * 9
    assert model.test_scores_y_.shape[0] == 3 * 4 * 9
    assert model.yhat_test_.shape[0] == 3 * 4 * 9
    assert model.result_.test_correlations is model.test_correlations_


def test_caspoc_univariate_y_second_component_correlations_are_finite():
    rng = np.random.default_rng(13)
    latent = rng.normal(size=(45, 2))
    X = latent @ rng.normal(size=(2, 12)) + 0.2 * rng.normal(size=(45, 12))
    y = latent @ np.array([0.9, 0.35]) + 0.2 * rng.normal(size=45)

    model = CASPOC(
        n_components=2,
        n_repeats=3,
        n_folds=5,
        keep_x_options=[1, 3, 6],
        keep_y_options=[1],
        fix_x=[1],
        fix_y=[1],
        random_state=101,
    ).fit(X, y)

    comp2 = model.tune_correlations_.query("Component == 2")

    assert comp2["Correlation"].notna().all()


def test_caspoc_uses_stratified_folds_for_univariate_binary_y():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(40, 8))
    y = np.array([0, 1] * 20)

    model = CASPOC(
        n_repeats=2,
        n_folds=5,
        keep_x_options=[3],
        keep_y_options=[1],
        random_state=102,
    ).fit(X, y)

    assert model.y_type_ == "univariate binary"
    assert model.fold_strategy_ == "StratifiedKFold"
    assert model.fold_message_ == "Univariate binary Y. Using StratifiedKFold."
    for repeat_folds in model.folds_:
        for fold in repeat_folds:
            assert set(np.unique(y[fold], return_counts=True)[1]) == {4}


def test_caspoc_binary_y_requires_enough_samples_per_class():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(12, 5))
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    with pytest.raises(ValueError, match="each class must have at least n_folds"):
        CASPOC(
            n_repeats=1,
            n_folds=4,
            keep_x_options=[2],
            keep_y_options=[1],
        ).fit(X, y)


def test_caspoc_html_repr_includes_fit_metadata():
    rng = np.random.default_rng(16)
    X = rng.normal(size=(30, 6))
    y = np.array([0, 1] * 15)

    model = CASPOC(
        n_repeats=1,
        n_folds=5,
        keep_x_options=[2],
        keep_y_options=[1],
    ).fit(X, y)
    html = model._repr_html_()

    assert "CASPOC" in html
    assert "StratifiedKFold" in html
    assert "univariate binary" in html
