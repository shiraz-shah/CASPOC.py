# caspoc.py

`caspoc.py` is a native Python implementation of the CASPOC workflow:
double-split repeated K-fold cross-validation around an approximate sparse PLS
model.

The sparse PLS implementation is inspired by `mixOmics::spls`, but it is not a
numerically exact port. The R-version is still the reference. It uses iterative
loading updates, mixOmics-style `keepX`/`keepY` soft thresholding, L2 normalization,
and component deflation.

It can fit a multivariate X against a multivariate Y, but works great for univariate Y
too. Here, it provides a more efficient alternative to nested cross-validation.

Use `tune_correlations_` to choose `keepX`/`keepY`, then use the matching rows
in `test_correlations_` for held-out downstream statistics.

## Installation

```bash
pip install git+https://github.com/shiraz-shah/CASPOC.py.git
```

# Minimal workflow:
- Train the first component
- Inspect tuning-performance to pick best `keep_x` and `keep_y`
- Train the next component while fixing the first component to the above
- Inspect tuning performance to pick best `keep_x` and `keep_y` for the second component
- Continue until no more signal can be recovered

Once the you've decided on the final number of parametres and components:
- Fetch component loadings from `.train_loadings_x_`
- Fetch unbiased test set scores from `.yhat_test_`

# Minimal code example with univariate y
```python
from caspoc import CASPOC

model = CASPOC(n_components=1)
model.fit(X, y)
sns.boxplot(x='KeepX', y='Correlation', data=model.tune_correlations_)
```
<img width="575" height="430" alt="image" src="https://github.com/user-attachments/assets/49763bd2-5164-4e30-94ed-17c2b36e5a8a" />

```python
model = CASPOC(n_components = 2, fix_x = [1])
model.fit(x, y)
sns.boxplot(x='KeepX', y='Correlation', data=model.tune_correlations_.query("Component == 2"))
```
<img width="575" height="429" alt="image" src="https://github.com/user-attachments/assets/ad15c207-6e2d-4e0c-8768-0141b4dfa71e" />

```python
model = CASPOC(n_components = 2, fix_x = [1, 30])
...
...
# final model, two components, 1 and 30 features respectively
model = CASPOC(n_components = 2, fix_x = [1], keep_x_options = [30])
model.fit(x, y)
# assessing model predictions for held-out test set aginst true y
roc_auc_score(y, model.yhat_test_.query("Repeat == 1").set_index("Sample")["Y1"].loc[y.index])
spearmanr(y, model.yhat_test_..query("Repeat == 1").set_index("Sample")["Y1"].loc[y.index])
```

## In-Python manual pages

The package includes small manual tables for that are especially useful for
AI coding agents to learn how to use CASPOC.

```python
from caspoc import manual_page, manual_topics

manual_topics()
manual_page("CASPOC")
manual_page("CASPOC.outputs")
manual_page("CASPOC.R")
manual_page("SparsePLS")
```

The `CASPOC.R` topic maps the original R package argument and output names
from `jonathanth/caspoc` to this Python API.
