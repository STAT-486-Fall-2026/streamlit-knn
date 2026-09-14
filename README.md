# k-NN Decision-Boundary Streamlit App

This small app visualizes the decision regions of a k-nearest-neighbors (k-NN)
classifier on a two-dimensional synthetic data set.

## Set up with Conda

Create the environment and install the Python requirements:

```bash
conda env create -f environment.yml
conda activate streamlit-knn
```

Run the app:

```bash
streamlit run app.py
```

If the environment already exists, update it with:

```bash
conda env update -f environment.yml --prune
```

## Controls

- **Number of neighbors (`k`)** controls how local or smooth the classifier is.
- **Distance-weighted voting** switches between uniform voting and giving closer
  neighbors more influence.
- **Distance metric** compares Euclidean, Manhattan, Chebyshev, and a
  Minkowski distance with `p = 3`, plus Cosine distance.
- **Standardize features** centers each feature and scales it to unit variance
  before fitting, which is useful when features have different measurement
  scales.
- The plot shows training observations as circles and held-out test
  observations as triangles; both training and test accuracy are reported.

## Available datasets

The app keeps **Moons** as the default and provides three other reproducible
two-dimensional choices:

- **Concentric circles**: nested nonlinear classes that test radial geometry.
- **Anisotropic blobs**: stretched and rotated clusters that expose scale and
  axis-orientation effects.
- **Linear separation**: mostly linearly separated classes with a small amount
  of noise, useful as a simple baseline.
- **Directional rays**: classes are separated by a small angle while radius
  varies across several orders of magnitude, making Cosine distance a natural
  choice. Leave standardization off because it changes the angular geometry.

The plot uses circles for training observations and triangles for held-out test
observations. Misclassified test observations are overlaid with a red x. Both
training and test accuracy are reported.
