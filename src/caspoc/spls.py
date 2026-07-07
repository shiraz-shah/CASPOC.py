from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from caspoc._utils import as_2d_array, mixomics_keep_threshold, normalize_vector


@dataclass
class SparsePLSFit:
    x_weights: np.ndarray
    y_weights: np.ndarray
    x_loadings: np.ndarray
    y_loadings: np.ndarray
    x_scores: np.ndarray
    y_scores: np.ndarray
    coef_: np.ndarray
    n_iter: list[int]


class SparsePLS(BaseEstimator, TransformerMixin):
    """Approximate sparse PLS with mixOmics-style keepX/keepY thresholding."""

    def __init__(
        self,
        n_components: int = 2,
        keep_x: int | list[int] | None = None,
        keep_y: int | list[int] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        self.n_components = n_components
        self.keep_x = keep_x
        self.keep_y = keep_y
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, Y):
        X = as_2d_array(X, "X")
        Y = as_2d_array(Y, "Y")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples.")
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1.")

        keep_x = self._expand_keep(self.keep_x, X.shape[1], "keep_x")
        keep_y = self._expand_keep(self.keep_y, Y.shape[1], "keep_y")

        x_res = X.copy()
        y_res = Y.copy()

        x_weights = []
        y_weights = []
        x_loadings = []
        y_loadings = []
        x_scores = []
        y_scores = []
        n_iter = []

        for comp in range(self.n_components):
            u, v, iters = self._fit_component(
                x_res, y_res, keep_x[comp], keep_y[comp]
            )
            t = x_res @ u
            s = y_res @ v

            denom = float(t.T @ t)
            if denom <= np.finfo(float).eps:
                p = np.zeros(x_res.shape[1])
                q = np.zeros(y_res.shape[1])
            else:
                p = (x_res.T @ t) / denom
                q = (y_res.T @ t) / denom

            x_res = x_res - np.outer(t, p)
            y_res = y_res - np.outer(t, q)

            x_weights.append(u)
            y_weights.append(v)
            x_loadings.append(p)
            y_loadings.append(q)
            x_scores.append(t)
            y_scores.append(s)
            n_iter.append(iters)

        self.x_weights_ = np.column_stack(x_weights)
        self.y_weights_ = np.column_stack(y_weights)
        self.x_loadings_ = np.column_stack(x_loadings)
        self.y_loadings_ = np.column_stack(y_loadings)
        self.x_scores_ = np.column_stack(x_scores)
        self.y_scores_ = np.column_stack(y_scores)
        self.keep_x_ = keep_x
        self.keep_y_ = keep_y
        self.n_iter_ = n_iter

        self.coef_ = np.linalg.pinv(X) @ Y
        self.score_coef_ = np.linalg.pinv(self.x_scores_) @ Y
        self.fit_ = SparsePLSFit(
            x_weights=self.x_weights_,
            y_weights=self.y_weights_,
            x_loadings=self.x_loadings_,
            y_loadings=self.y_loadings_,
            x_scores=self.x_scores_,
            y_scores=self.y_scores_,
            coef_=self.score_coef_,
            n_iter=self.n_iter_,
        )
        return self

    def transform(self, X, Y=None):
        if Y is None:
            return self.transform_x(X)
        return self._deflated_pair_scores(as_2d_array(X, "X"), as_2d_array(Y, "Y"))

    def transform_x(self, X):
        return self._deflated_scores(as_2d_array(X, "X"), self.x_weights_)

    def transform_y(self, Y):
        return self._deflated_scores(as_2d_array(Y, "Y"), self.y_weights_)

    def predict(self, X):
        scores = self.transform_x(X)
        return scores @ self.score_coef_

    def _fit_component(self, X, Y, keep_x: int, keep_y: int):
        cross_cov = X.T @ Y
        left, _, right_t = np.linalg.svd(cross_cov, full_matrices=False)
        u = normalize_vector(left[:, 0])
        v = normalize_vector(right_t.T[:, 0])

        for iteration in range(1, self.max_iter + 1):
            old_u = u.copy()
            old_v = v.copy()

            u = X.T @ (Y @ v)
            u = normalize_vector(mixomics_keep_threshold(u, keep_x))

            v = Y.T @ (X @ u)
            v = normalize_vector(mixomics_keep_threshold(v, keep_y))

            diff = max(np.sum((u - old_u) ** 2), np.sum((v - old_v) ** 2))
            if diff < self.tol:
                return u, v, iteration

        return u, v, self.max_iter

    def _deflated_scores(self, data: np.ndarray, weights: np.ndarray) -> np.ndarray:
        residual = data.copy()
        scores = []
        for comp in range(weights.shape[1]):
            score = residual @ weights[:, comp]
            denom = float(score.T @ score)
            if denom > np.finfo(float).eps:
                loading = (residual.T @ score) / denom
                residual = residual - np.outer(score, loading)
            scores.append(score)
        return np.column_stack(scores)

    def _deflated_pair_scores(
        self, X: np.ndarray, Y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x_residual = X.copy()
        y_residual = Y.copy()
        x_scores = []
        y_scores = []

        for comp in range(self.n_components):
            x_score = x_residual @ self.x_weights_[:, comp]
            y_score = y_residual @ self.y_weights_[:, comp]

            denom = float(x_score.T @ x_score)
            if denom > np.finfo(float).eps:
                x_loading = (x_residual.T @ x_score) / denom
                y_loading = (y_residual.T @ x_score) / denom
                x_residual = x_residual - np.outer(x_score, x_loading)
                y_residual = y_residual - np.outer(x_score, y_loading)

            x_scores.append(x_score)
            y_scores.append(y_score)

        return np.column_stack(x_scores), np.column_stack(y_scores)

    def _expand_keep(self, keep, n_features: int, name: str) -> list[int]:
        if keep is None:
            values = [n_features] * self.n_components
        elif isinstance(keep, int):
            values = [keep] * self.n_components
        else:
            values = list(keep)
            if len(values) != self.n_components:
                raise ValueError(f"{name} must have length n_components.")

        if any(value < 1 or value > n_features for value in values):
            raise ValueError(f"{name} values must be between 1 and {n_features}.")
        return [int(value) for value in values]
