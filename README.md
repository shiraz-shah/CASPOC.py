# caspoc.py

`caspoc.py` is a native Python implementation of the CASPOC workflow:
double-split repeated K-fold cross-validation around an approximate sparse PLS
model.

The sparse PLS implementation is inspired by `mixOmics::spls`, but it is not a
numerically exact port. It uses iterative loading updates, mixOmics-style
`keepX`/`keepY` soft thresholding, L2 normalization, and component deflation.

## Installation

```bash
pip install git+https://github.com/shiraz-shah/CASPOC.py.git
```

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
