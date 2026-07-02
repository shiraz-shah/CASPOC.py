from __future__ import annotations

import pandas as pd


_MANUAL_PAGES: dict[str, list[dict[str, str]]] = {
    "CASPOC": [
        {
            "section": "purpose",
            "name": "CASPOC",
            "description": (
                "Python implementation of the CASPOC workflow from the R package "
                "jonathanth/caspoc: repeated double-split K-fold "
                "cross-validation around sparse PLS, with separate tuning and "
                "testing folds."
            ),
        },
        {
            "section": "purpose",
            "name": "tune/test split",
            "description": (
                "As in the R package, use one hold-out fold for tuning "
                "hyperparameters such as keepX, keepY, and n_components, while a "
                "separate hold-out fold remains available for downstream "
                "inference on unbiased test scores."
            ),
        },
        {
            "section": "input",
            "name": "X",
            "description": (
                "Predictor block. Accepts a 1D or 2D numeric array-like object. "
                "Pandas DataFrame columns are preserved as X variable names."
            ),
        },
        {
            "section": "input",
            "name": "Y",
            "description": (
                "Response block. Accepts a 1D or 2D numeric array-like object with "
                "the same number of rows as X. Pandas DataFrame columns are "
                "preserved as Y variable names."
            ),
        },
        {
            "section": "parameter",
            "name": "n_components",
            "description": (
                "Number of sparse PLS components to estimate. Named ncomp in the "
                "R CASPOC function."
            ),
        },
        {
            "section": "parameter",
            "name": "n_repeats",
            "description": (
                "Number of repeated cross-validation partitions. Named numRepeats "
                "in the R CASPOC function."
            ),
        },
        {
            "section": "parameter",
            "name": "n_folds",
            "description": (
                "Number of folds per repeat. Must be at least 3 because each split "
                "uses one tune fold, one test fold, and the remaining folds for "
                "training. Named numFolds in the R CASPOC function."
            ),
        },
        {
            "section": "parameter",
            "name": "keep_x_options",
            "description": (
                "Candidate numbers of X variables to retain per tuned component. "
                "If None, a default grid is generated from the number of X "
                "features, following the R package's 1-to-p grid for p <= 10 and "
                "roughly ten-step grid for larger p."
            ),
        },
        {
            "section": "parameter",
            "name": "keep_y_options",
            "description": (
                "Candidate numbers of Y variables to retain per tuned component. "
                "If None, a default grid is generated from the number of Y "
                "features, following the R package's 1-to-p grid for p <= 10 and "
                "roughly ten-step grid for larger p."
            ),
        },
        {
            "section": "parameter",
            "name": "fix_x",
            "description": (
                "Optional keepX values for leading components that should not be "
                "tuned. Remaining components use each value from keep_x_options. "
                "Named fixX in the R CASPOC function."
            ),
        },
        {
            "section": "parameter",
            "name": "fix_y",
            "description": (
                "Optional keepY values for leading components that should not be "
                "tuned. Remaining components use each value from keep_y_options. "
                "Named fixY in the R CASPOC function."
            ),
        },
        {
            "section": "parameter",
            "name": "random_state",
            "description": (
                "Seed used to generate reproducible repeated folds. Named "
                "base_seed in the R CASPOC function."
            ),
        },
        {
            "section": "parameter",
            "name": "manual_folds",
            "description": (
                "Optional argument to fit(). A list of repeats, where each repeat "
                "is a list of n_folds index arrays partitioning all samples, like "
                "the R manual_folds argument."
            ),
        },
        {
            "section": "correct use",
            "name": "tune before test",
            "description": (
                "Use tune_correlations_ to choose keepX/keepY. Use the matching "
                "rows in test_correlations_ only after that choice for unbiased "
                "held-out statistics."
            ),
        },
        {
            "section": "correct use",
            "name": "scaling",
            "description": (
                "Scaling is fit on the training samples inside each split and then "
                "applied to tune and test samples."
            ),
        },
    ],
    "CASPOC.outputs": [
        {
            "section": "attribute",
            "name": "tune_correlations_",
            "description": (
                "DataFrame with Repeat, KeepX, KeepY, Component, Correlation, "
                "Pvalue, and n for tune-set X/Y score correlations. Equivalent "
                "to results_tune_df in the R package."
            ),
        },
        {
            "section": "attribute",
            "name": "test_correlations_",
            "description": (
                "DataFrame with the same schema as tune_correlations_, computed on "
                "the independent test fold for each train/tune/test split. "
                "Equivalent to results_test_df in the R package."
            ),
        },
        {
            "section": "attribute",
            "name": "train_loadings_x_",
            "description": (
                "Long DataFrame of X weights from models fit on training samples. "
                "Includes comp columns, Variable, Repeat, keepX, keepY, and Fold. "
                "Equivalent to full_train_loadingsX in the R package."
            ),
        },
        {
            "section": "attribute",
            "name": "train_loadings_y_",
            "description": (
                "Long DataFrame of Y weights from models fit on training samples. "
                "Includes comp columns, Variable, Repeat, keepX, keepY, and Fold. "
                "Equivalent to full_train_loadingsY in the R package."
            ),
        },
        {
            "section": "attribute",
            "name": "tune_scores_x_ / tune_scores_y_",
            "description": (
                "Long DataFrames of held-out tune scores with Sample, component "
                "columns, Repeat, keepX, keepY, and Fold. Equivalent to "
                "full_tuneX and full_tuneY in the R package."
            ),
        },
        {
            "section": "attribute",
            "name": "test_scores_x_ / test_scores_y_",
            "description": (
                "Long DataFrames of held-out test scores with Sample, component "
                "columns, Repeat, keepX, keepY, and Fold. Equivalent to "
                "full_testX and full_testY in the R package."
            ),
        },
        {
            "section": "attribute",
            "name": "yhat_tune_ / yhat_test_",
            "description": (
                "Predicted Y values for tune and test samples, transformed back to "
                "the original Y scale for each split. Similar to full_yhat_tune "
                "and full_yhat_test in the R implementation."
            ),
        },
        {
            "section": "attribute",
            "name": "folds_",
            "description": (
                "The repeated fold index structure used by fit(). This can be "
                "reused as manual_folds to reproduce a split exactly."
            ),
        },
    ],
    "CASPOC.R": [
        {
            "section": "source",
            "name": "R package",
            "description": (
                "Original R implementation: github.com/jonathanth/caspoc. Its "
                "DESCRIPTION says the package wraps mixOmics sPLS in a "
                "double-split CV approach so keepX, keepY, and ncomp can be tuned "
                "in one hold-out set while another hold-out set remains unbiased "
                "for inference."
            ),
        },
        {
            "section": "name mapping",
            "name": "ncomp -> n_components",
            "description": "Number of sparse PLS components.",
        },
        {
            "section": "name mapping",
            "name": "numRepeats -> n_repeats",
            "description": "Number of repeated cross-validation partitions.",
        },
        {
            "section": "name mapping",
            "name": "numFolds -> n_folds",
            "description": "Number of folds per repeat.",
        },
        {
            "section": "name mapping",
            "name": "keepX_options -> keep_x_options",
            "description": "Candidate grid for retained X variables.",
        },
        {
            "section": "name mapping",
            "name": "keepY_options -> keep_y_options",
            "description": "Candidate grid for retained Y variables.",
        },
        {
            "section": "name mapping",
            "name": "fixX/fixY -> fix_x/fix_y",
            "description": "Fixed keep values for leading components.",
        },
        {
            "section": "name mapping",
            "name": "base_seed -> random_state",
            "description": "Seed for reproducible repeated folds.",
        },
        {
            "section": "output mapping",
            "name": "results_tune_df -> tune_correlations_",
            "description": (
                "Tune-fold Spearman correlations by repeat, keep grid, and "
                "component."
            ),
        },
        {
            "section": "output mapping",
            "name": "results_test_df -> test_correlations_",
            "description": (
                "Test-fold Spearman correlations by repeat, keep grid, and "
                "component."
            ),
        },
        {
            "section": "output mapping",
            "name": "full_tuneX/full_tuneY",
            "description": "Python equivalents are tune_scores_x_ and tune_scores_y_.",
        },
        {
            "section": "output mapping",
            "name": "full_testX/full_testY",
            "description": "Python equivalents are test_scores_x_ and test_scores_y_.",
        },
        {
            "section": "output mapping",
            "name": "full_train_loadingsX/full_train_loadingsY",
            "description": (
                "Python equivalents are train_loadings_x_ and train_loadings_y_."
            ),
        },
        {
            "section": "internal helper",
            "name": "deflate_sPLS_data",
            "description": (
                "The R package documents this as an internal function for "
                "deflating train, tune, and test data with training loadings when "
                "ncomp > 1. Python implements analogous score deflation inside "
                "SparsePLS.transform_x() and transform_y()."
            ),
        },
    ],
    "SparsePLS": [
        {
            "section": "purpose",
            "name": "SparsePLS",
            "description": (
                "Approximate sparse PLS estimator with mixOmics-style keepX/keepY "
                "cardinality thresholding, L2-normalized weights, and component "
                "deflation."
            ),
        },
        {
            "section": "parameter",
            "name": "n_components",
            "description": "Number of sparse PLS components to estimate.",
        },
        {
            "section": "parameter",
            "name": "keep_x",
            "description": (
                "Number of X variables retained per component. Use an int for the "
                "same value on all components, a list with length n_components, or "
                "None to keep all X variables."
            ),
        },
        {
            "section": "parameter",
            "name": "keep_y",
            "description": (
                "Number of Y variables retained per component. Use an int for the "
                "same value on all components, a list with length n_components, or "
                "None to keep all Y variables."
            ),
        },
        {
            "section": "method",
            "name": "fit(X, Y)",
            "description": (
                "Fit sparse PLS weights, loadings, scores, iteration counts, and a "
                "linear prediction coefficient matrix."
            ),
        },
        {
            "section": "method",
            "name": "transform(X, Y=None)",
            "description": (
                "Return X scores, or a tuple of X and Y scores when Y is supplied."
            ),
        },
        {
            "section": "method",
            "name": "predict(X)",
            "description": "Predict Y from X using the fitted coefficient matrix.",
        },
    ],
}


def manual_topics() -> list[str]:
    """Return available in-Python manual page topics."""
    return sorted(_MANUAL_PAGES)


def manual_page(topic: str = "CASPOC") -> pd.DataFrame:
    """Return a manual page as a pandas DataFrame.

    Parameters
    ----------
    topic
        One of the values returned by manual_topics().

    Returns
    -------
    pandas.DataFrame
        A table with section, name, and description columns.
    """
    if topic not in _MANUAL_PAGES:
        topics = ", ".join(manual_topics())
        raise ValueError(f"Unknown manual topic {topic!r}. Available topics: {topics}.")
    return pd.DataFrame(
        _MANUAL_PAGES[topic],
        columns=["section", "name", "description"],
    )
