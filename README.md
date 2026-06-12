# caspoc.py

`caspoc.py` is a native Python implementation of the CASPOC workflow:
double-split repeated K-fold cross-validation around an approximate sparse PLS
model.

The sparse PLS implementation is inspired by `mixOmics::spls`, but it is not a
numerically exact port. It uses iterative loading updates, mixOmics-style
`keepX`/`keepY` soft thresholding, L2 normalization, and component deflation.

## Reference data

Two public `mixOmics` datasets are useful for validation:

- `mixOmics::liver.toxicity`: used in the upstream `spls` examples.
- `mixOmics::breast.TCGA`: used in the original CASPOC documentation example.

These are good candidates for cross-language comparison tests if an R
environment with `mixOmics` and `caspoc` is available. The Python package also
includes synthetic tests that do not require R.

## Example

```python
from caspoc import CASPOC

model = CASPOC(
    n_components=2,
    n_repeats=3,
    n_folds=5,
    keep_x_options=[10, 25],
    keep_y_options=[3, 5],
    random_state=42,
).fit(X, Y)

model.tune_correlations_
model.test_correlations_
model.train_loadings_x_
```
