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

## In-Python manual pages

The package includes small manual tables for notebook and console use:

```python
from caspoc import manual_page, manual_topics

manual_topics()
manual_page("CASPOC")
manual_page("CASPOC.outputs")
manual_page("CASPOC.R")
manual_page("SparsePLS")
```

Use `tune_correlations_` to choose `keepX`/`keepY`, then use the matching rows
in `test_correlations_` for held-out downstream statistics.

The `CASPOC.R` topic maps the original R package argument and output names
from `jonathanth/caspoc` to this Python API.
