"""Interactive k-nearest-neighbors decision-boundary demo."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import make_blobs, make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="k-NN Decision Boundaries",
    page_icon="🧭",
    layout="wide",
)


DATASET_DESCRIPTIONS = {
    "Moons": "Two interleaving half-circles; a nonlinear neighborhood problem.",
    "Concentric circles": "Nested circles; tests nonlinear boundaries and radial geometry.",
    "Anisotropic blobs": "Stretched and rotated blobs; emphasizes axis and scale effects.",
    "Linear separation": "Mostly linearly separated classes with a small amount of noise.",
    "Directional rays": "Classes differ by angle while radius is noise; ideal for Cosine distance.",
}


@st.cache_data
def make_demo_data(
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a selected reproducible data set with a held-out test split."""

    if dataset_name == "Moons":
        X, y = make_moons(n_samples=280, noise=0.22, random_state=12)
    elif dataset_name == "Concentric circles":
        X, y = make_circles(
            n_samples=280,
            factor=0.45,
            noise=0.08,
            random_state=12,
        )
    elif dataset_name == "Anisotropic blobs":
        X, y = make_blobs(
            n_samples=280,
            centers=[(-1.4, -1.0), (1.4, 1.0)],
            cluster_std=0.7,
            random_state=12,
        )
        X = X @ np.array([[1.9, -0.8], [0.2, 0.45]])
    elif dataset_name == "Linear separation":
        X, y = make_classification(
            n_samples=280,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=1.4,
            flip_y=0.04,
            random_state=12,
        )
    elif dataset_name == "Directional rays":
        rng = np.random.default_rng(12)
        n_per_class = 60
        angles = np.concatenate(
            [
                rng.normal(np.deg2rad(25), np.deg2rad(2), n_per_class),
                rng.normal(np.deg2rad(35), np.deg2rad(2), n_per_class),
            ]
        )
        radii = np.exp(
            rng.uniform(np.log(0.01), np.log(1000), 2 * n_per_class)
        )
        X = np.column_stack(
            [
                radii * np.cos(angles),
                radii * np.sin(angles),
            ]
        )
        y = np.repeat([0, 1], n_per_class)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=21,
        stratify=y,
    )
    return X_train, y_train, X_test, y_test


DISTANCE_METRICS = {
    "Euclidean (L2)": {"metric": "minkowski", "p": 2},
    "Manhattan (L1)": {"metric": "manhattan"},
    "Chebyshev (L∞)": {"metric": "chebyshev"},
    "Minkowski (p = 3)": {"metric": "minkowski", "p": 3},
    "Cosine": {"metric": "cosine"},
}

CLASS_COLORS = ["#E69F00", "#56B4E9"]
BACKGROUND_COLORS = ["#FFF4D6", "#E7F5FC"]


def plot_decision_boundary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_predictions: np.ndarray,
    model: KNeighborsClassifier,
) -> plt.Figure:
    """Plot predicted regions, training points, and held-out test points."""

    x_padding = 0.45
    y_padding = 0.45
    X_all = np.vstack([X_train, X_test])
    x_min, x_max = X_all[:, 0].min() - x_padding, X_all[:, 0].max() + x_padding
    y_min, y_max = X_all[:, 1].min() - y_padding, X_all[:, 1].max() + y_padding

    # A moderately dense grid keeps the boundary smooth without slowing the app.
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    predictions = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.contourf(
        xx,
        yy,
        predictions,
        levels=[-0.5, 0.5, 1.5],
        colors=BACKGROUND_COLORS,
        alpha=0.9,
    )
    ax.contour(
        xx,
        yy,
        predictions,
        levels=[0.5],
        colors=["#333333"],
        linewidths=1.5,
    )

    for class_value, color in enumerate(CLASS_COLORS):
        train_points = y_train == class_value
        ax.scatter(
            X_train[train_points, 0],
            X_train[train_points, 1],
            s=48,
            facecolors="white",
            edgecolors=color,
            linewidths=1.8,
            label=f"Class {class_value} training",
            zorder=3,
        )

        test_points = y_test == class_value
        ax.scatter(
            X_test[test_points, 0],
            X_test[test_points, 1],
            s=62,
            marker="^",
            facecolors=color,
            edgecolors="white",
            linewidths=0.8,
            label=f"Class {class_value} test",
            zorder=3,
        )

    misclassified = test_predictions != y_test
    if np.any(misclassified):
        ax.scatter(
            X_test[misclassified, 0],
            X_test[misclassified, 1],
            s=110,
            marker="x",
            c="#D55E00",
            linewidths=2.2,
            label="Misclassified test",
            zorder=5,
        )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("k-NN classification regions")
    ax.legend(frameon=True, loc="upper right")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


st.title("k-NN Decision Boundaries")
st.write(
    "Explore how the number of neighbors, voting rule, and distance metric "
    "change a k-nearest-neighbors classifier."
)

with st.sidebar:
    st.header("Data controls")
    dataset_name = st.selectbox(
        "Dataset",
        options=list(DATASET_DESCRIPTIONS),
        index=0,
        help="Moons is the default; the alternatives highlight different metric behavior.",
    )
    st.caption(DATASET_DESCRIPTIONS[dataset_name])

    st.header("Model controls")
    k = st.slider(
        "Number of neighbors (k)",
        min_value=1,
        max_value=35,
        value=7,
        help="Small k values produce more local, flexible boundaries.",
    )
    weighted = st.toggle(
        "Distance-weighted voting",
        value=False,
        help=(
            "When enabled, nearby observations receive more voting influence. "
            "When disabled, every neighbor receives the same vote."
        ),
    )
    distance_name = st.selectbox(
        "Distance metric",
        options=list(DISTANCE_METRICS),
        index=0,
    )
    standardize = st.toggle(
        "Standardize features",
        value=False,
        help=(
            "Subtract each feature's mean and divide by its standard deviation "
            "before fitting. This prevents large-scale features from dominating "
            "the distance calculation."
        ),
    )

X_train, y_train, X_test, y_test = make_demo_data(dataset_name)

if standardize:
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

weights = "distance" if weighted else "uniform"
model = KNeighborsClassifier(
    n_neighbors=k,
    weights=weights,
    **DISTANCE_METRICS[distance_name],
)
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)
test_accuracy = np.mean(test_predictions == y_test)

plot_column, summary_column = st.columns([3.2, 1])

with plot_column:
    figure = plot_decision_boundary(
        X_train,
        y_train,
        X_test,
        y_test,
        test_predictions,
        model,
    )
    st.pyplot(figure, clear_figure=True, width="stretch")

with summary_column:
    st.subheader("Current model")
    st.metric("k", k)
    st.metric("Training accuracy", f"{model.score(X_train, y_train):.1%}")
    st.metric("Test accuracy", f"{test_accuracy:.1%}")
    st.write(f"**Voting:** {'distance-weighted' if weighted else 'uniform'}")
    st.write(f"**Distance:** {distance_name}")
    st.write(f"**Features:** {'standardized' if standardize else 'original scale'}")
    st.info(
        "The colored background shows the class predicted at each location. "
        "The dark curve is the approximate decision boundary. Triangles are "
        "held-out test points; circles are training points. Red x marks "
        "indicate misclassified test points."
    )

st.caption(
    "This demonstration uses a fixed two-dimensional synthetic data set with a "
    "held-out test split. Moons is the default; use the dataset selector to "
    "compare alternative geometries."
)
