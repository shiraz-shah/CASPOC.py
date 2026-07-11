from __future__ import annotations

from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from caspoc._utils import (
    as_2d_array,
    default_keep_options,
    feature_names,
    sample_index,
)
from caspoc.spls import SparsePLS


@dataclass
class CASPOCResult:
    tune_correlations: pd.DataFrame
    test_correlations: pd.DataFrame
    train_loadings_x: pd.DataFrame
    train_loadings_y: pd.DataFrame
    tune_scores_x: pd.DataFrame
    tune_scores_y: pd.DataFrame
    test_scores_x: pd.DataFrame
    test_scores_y: pd.DataFrame
    folds: list[list[np.ndarray]]
    yhat_tune: pd.DataFrame
    yhat_test: pd.DataFrame


class CASPOC:
    """Double-split repeated K-fold CV for approximate sparse PLS."""

    def __init__(
        self,
        n_components: int = 1,
        n_repeats: int = 11,
        n_folds: int = 10,
        keep_x_options: list[int] | None = None,
        keep_y_options: list[int] | None = None,
        fix_x: list[int] | None = None,
        fix_y: list[int] | None = None,
        random_state: int = 1,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        self.n_components = n_components
        self.n_repeats = n_repeats
        self.n_folds = n_folds
        self.keep_x_options = keep_x_options
        self.keep_y_options = keep_y_options
        self.fix_x = fix_x
        self.fix_y = fix_y
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, Y, manual_folds: list[list[np.ndarray]] | None = None):
        raw_x = X
        raw_y = Y
        X = as_2d_array(X, "X")
        Y = as_2d_array(Y, "Y")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples.")
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be at least 1.")
        if self.n_folds < 3:
            raise ValueError("n_folds must be at least 3 for train/tune/test splits.")
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1.")

        x_names = feature_names(raw_x, "X")
        y_names = feature_names(raw_y, "Y")
        samples = sample_index(raw_x, X.shape[0])

        keep_x_options = self.keep_x_options or default_keep_options(X.shape[1])
        keep_y_options = self.keep_y_options or default_keep_options(Y.shape[1])
        y_labels = self._binary_univariate_labels(Y)
        if manual_folds is not None:
            folds = manual_folds
            self.y_type_ = self._y_type(Y)
            self.fold_strategy_ = "manual"
            self.fold_message_ = "Manual folds supplied. Using manual_folds."
        elif y_labels is not None:
            folds = self._make_folds(X.shape[0], y_labels)
            self.y_type_ = "univariate binary"
            self.fold_strategy_ = "StratifiedKFold"
            self.fold_message_ = "Univariate binary Y. Using StratifiedKFold."
        else:
            folds = self._make_folds(X.shape[0])
            self.y_type_ = self._y_type(Y)
            self.fold_strategy_ = "KFold"
            self.fold_message_ = "Using KFold."
        self._validate_folds(folds, X.shape[0])

        tune_score_x_frames = []
        tune_score_y_frames = []
        test_score_x_frames = []
        test_score_y_frames = []
        loading_x_frames = []
        loading_y_frames = []
        tune_corr_rows = []
        test_corr_rows = []
        yhat_tune_frames = []
        yhat_test_frames = []

        for repeat_idx, repeat_folds in enumerate(folds, start=1):
            for keep_x_value in keep_x_options:
                for keep_y_value in keep_y_options:
                    fold_tune_x = []
                    fold_tune_y = []
                    fold_test_x = []
                    fold_test_y = []

                    for fold_idx in range(self.n_folds):
                        tune_idx = np.asarray(repeat_folds[fold_idx], dtype=int)
                        test_idx = np.asarray(
                            repeat_folds[(fold_idx + 1) % self.n_folds], dtype=int
                        )
                        excluded = {fold_idx, (fold_idx + 1) % self.n_folds}
                        train_idx = np.concatenate(
                            [
                                np.asarray(values, dtype=int)
                                for idx, values in enumerate(repeat_folds)
                                if idx not in excluded
                            ]
                        )

                        scaler_x = StandardScaler()
                        scaler_y = StandardScaler()
                        train_x = scaler_x.fit_transform(X[train_idx])
                        train_y = scaler_y.fit_transform(Y[train_idx])
                        tune_x = scaler_x.transform(X[tune_idx])
                        tune_y = scaler_y.transform(Y[tune_idx])
                        test_x = scaler_x.transform(X[test_idx])
                        test_y = scaler_y.transform(Y[test_idx])

                        keep_x = self._component_keep(keep_x_value, self.fix_x)
                        keep_y = self._component_keep(keep_y_value, self.fix_y)
                        model = SparsePLS(
                            n_components=self.n_components,
                            keep_x=keep_x,
                            keep_y=keep_y,
                            max_iter=self.max_iter,
                            tol=self.tol,
                        ).fit(train_x, train_y)

                        tune_x_scores, tune_y_scores = model.transform(tune_x, tune_y)
                        test_x_scores, test_y_scores = model.transform(test_x, test_y)

                        fold_tune_x.append(tune_x_scores)
                        fold_tune_y.append(tune_y_scores)
                        fold_test_x.append(test_x_scores)
                        fold_test_y.append(test_y_scores)

                        meta = {
                            "Repeat": repeat_idx,
                            "keepX": keep_x_value,
                            "keepY": keep_y_value,
                            "Fold": fold_idx + 1,
                        }
                        tune_score_x_frames.append(
                            self._score_frame(tune_x_scores, tune_idx, samples, meta)
                        )
                        tune_score_y_frames.append(
                            self._score_frame(tune_y_scores, tune_idx, samples, meta)
                        )
                        test_score_x_frames.append(
                            self._score_frame(test_x_scores, test_idx, samples, meta)
                        )
                        test_score_y_frames.append(
                            self._score_frame(test_y_scores, test_idx, samples, meta)
                        )
                        loading_x_frames.append(
                            self._loading_frame(model.x_weights_, x_names, meta)
                        )
                        loading_y_frames.append(
                            self._loading_frame(model.y_weights_, y_names, meta)
                        )
                        yhat_tune_frames.append(
                            self._prediction_frame(
                                scaler_y.inverse_transform(model.predict(tune_x)),
                                tune_idx,
                                samples,
                                y_names,
                                meta,
                            )
                        )
                        yhat_test_frames.append(
                            self._prediction_frame(
                                scaler_y.inverse_transform(model.predict(test_x)),
                                test_idx,
                                samples,
                                y_names,
                                meta,
                            )
                        )

                    tune_x_all = np.vstack(fold_tune_x)
                    tune_y_all = np.vstack(fold_tune_y)
                    test_x_all = np.vstack(fold_test_x)
                    test_y_all = np.vstack(fold_test_y)
                    for component in range(self.n_components):
                        tune_corr_rows.append(
                            self._corr_row(
                                repeat_idx,
                                keep_x_value,
                                keep_y_value,
                                component + 1,
                                tune_x_all[:, component],
                                tune_y_all[:, component],
                            )
                        )
                        test_corr_rows.append(
                            self._corr_row(
                                repeat_idx,
                                keep_x_value,
                                keep_y_value,
                                component + 1,
                                test_x_all[:, component],
                                test_y_all[:, component],
                            )
                        )

        result = CASPOCResult(
            tune_correlations=pd.DataFrame(tune_corr_rows),
            test_correlations=pd.DataFrame(test_corr_rows),
            train_loadings_x=pd.concat(loading_x_frames, ignore_index=True),
            train_loadings_y=pd.concat(loading_y_frames, ignore_index=True),
            tune_scores_x=pd.concat(tune_score_x_frames, ignore_index=True),
            tune_scores_y=pd.concat(tune_score_y_frames, ignore_index=True),
            test_scores_x=pd.concat(test_score_x_frames, ignore_index=True),
            test_scores_y=pd.concat(test_score_y_frames, ignore_index=True),
            folds=folds,
            yhat_tune=pd.concat(yhat_tune_frames, ignore_index=True),
            yhat_test=pd.concat(yhat_test_frames, ignore_index=True),
        )
        self.result_ = result
        self.tune_correlations_ = result.tune_correlations
        self.test_correlations_ = result.test_correlations
        self.train_loadings_x_ = result.train_loadings_x
        self.train_loadings_y_ = result.train_loadings_y
        self.tune_scores_x_ = result.tune_scores_x
        self.tune_scores_y_ = result.tune_scores_y
        self.test_scores_x_ = result.test_scores_x
        self.test_scores_y_ = result.test_scores_y
        self.folds_ = result.folds
        self.yhat_tune_ = result.yhat_tune
        self.yhat_test_ = result.yhat_test
        return self

    def _make_folds(
        self,
        n_samples: int,
        y_labels: np.ndarray | None = None,
    ) -> list[list[np.ndarray]]:
        folds = []
        for repeat_idx in range(1, self.n_repeats + 1):
            if y_labels is None:
                splitter = KFold(
                    n_splits=self.n_folds,
                    shuffle=True,
                    random_state=self.random_state + repeat_idx,
                )
                split = splitter.split(np.arange(n_samples))
            else:
                splitter = StratifiedKFold(
                    n_splits=self.n_folds,
                    shuffle=True,
                    random_state=self.random_state + repeat_idx,
                )
                split = splitter.split(np.arange(n_samples), y_labels)
            folds.append([test_idx for _, test_idx in split])
        return folds

    def _binary_univariate_labels(self, Y: np.ndarray) -> np.ndarray | None:
        if Y.shape[1] != 1:
            return None

        labels = Y[:, 0]
        classes, counts = np.unique(labels, return_counts=True)
        if classes.size != 2:
            return None
        if counts.min() < self.n_folds:
            raise ValueError(
                "Univariate binary Y was detected, but each class must have at "
                "least n_folds samples for stratified folds."
            )
        return labels

    def _y_type(self, Y: np.ndarray) -> str:
        if Y.shape[1] != 1:
            return "multivariate"
        if np.unique(Y[:, 0]).size == 2:
            return "univariate binary"
        return "univariate"

    def _repr_html_(self) -> str:
        rows = [
            ("n_components", self.n_components),
            ("n_repeats", self.n_repeats),
            ("n_folds", self.n_folds),
        ]
        if hasattr(self, "fold_strategy_"):
            rows.extend(
                [
                    ("fit status", "fitted"),
                    ("Y type", self.y_type_),
                    ("fold strategy", self.fold_strategy_),
                    ("fold message", self.fold_message_),
                ]
            )
        else:
            rows.append(("fit status", "not fitted"))

        body = "".join(
            "<tr>"
            f"<th>{escape(str(name))}</th>"
            f"<td>{escape(str(value))}</td>"
            "</tr>"
            for name, value in rows
        )
        return (
            "<div style='border:1px solid #9ec5fe; border-radius:6px; "
            "background:#eef6ff; padding:12px; max-width:520px'>"
            "<div style='font-weight:700; color:#084298; margin-bottom:8px'>"
            "CASPOC</div>"
            "<table style='border-collapse:collapse'>"
            "<tbody>"
            f"{body}"
            "</tbody>"
            "</table>"
            "</div>"
        )

    def _validate_folds(self, folds: list[list[np.ndarray]], n_samples: int) -> None:
        if len(folds) != self.n_repeats:
            raise ValueError("manual_folds outer length must match n_repeats.")
        for repeat_folds in folds:
            if len(repeat_folds) != self.n_folds:
                raise ValueError("Each manual_folds repeat must contain n_folds folds.")
            combined = np.sort(
                np.concatenate([np.asarray(fold) for fold in repeat_folds])
            )
            if not np.array_equal(combined, np.arange(n_samples)):
                raise ValueError(
                    "Each repeat's folds must partition all sample indices."
                )

    def _component_keep(self, value: int, fixed: list[int] | None) -> list[int]:
        if fixed is None:
            return [int(value)] * self.n_components
        if len(fixed) > self.n_components:
            raise ValueError("Fixed keep vectors cannot exceed n_components.")
        return [int(x) for x in fixed] + [int(value)] * (
            self.n_components - len(fixed)
        )

    def _score_frame(self, scores, indices, samples, meta):
        frame = pd.DataFrame(
            scores,
            columns=[f"comp{i}" for i in range(1, self.n_components + 1)],
        )
        frame.insert(0, "Sample", [samples[idx] for idx in indices])
        for key, value in meta.items():
            frame[key] = value
        return frame

    def _loading_frame(self, weights, variables, meta):
        frame = pd.DataFrame(
            weights,
            columns=[f"comp{i}" for i in range(1, self.n_components + 1)],
        )
        frame["Variable"] = variables
        for key, value in meta.items():
            frame[key] = value
        return frame

    def _prediction_frame(self, predictions, indices, samples, y_names, meta):
        frame = pd.DataFrame(predictions, columns=y_names)
        frame.insert(0, "Sample", [samples[idx] for idx in indices])
        for key, value in meta.items():
            frame[key] = value
        return frame

    def _corr_row(self, repeat, keep_x, keep_y, component, x_scores, y_scores):
        corr = spearmanr(x_scores, y_scores)
        return {
            "Repeat": repeat,
            "KeepX": keep_x,
            "KeepY": keep_y,
            "Component": component,
            "Correlation": float(corr.statistic),
            "Pvalue": float(corr.pvalue),
            "n": len(x_scores),
        }
