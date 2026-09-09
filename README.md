# k-NN Decision-Boundary Streamlit App

This small app visualizes the decision regions of a k-nearest-neighbors (k-NN)
classifier on a two-dimensional synthetic data set.

## Run locally

```bash
uv sync
uv run streamlit run app.py
```

## Controls

- **Number of neighbors (`k`)** controls how local or smooth the classifier is.
- **Distance-weighted voting** switches between uniform voting and giving closer
  neighbors more influence.
- **Distance metric** compares Euclidean, Manhattan, Chebyshev, and a
  Minkowski distance with `p = 3`.
