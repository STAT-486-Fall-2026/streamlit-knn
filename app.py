"""Interactive k-nearest-neighbors decision-boundary demo."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier


st.set_page_config(
    page_title="k-NN Decision Boundaries",
    page_icon="🧭",
    layout="wide",
)


@st.cache_data
def make_demo_data() -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible, mildly noisy two-class data set."""

    return make_moons(n_samples=220, noise=0.22, random_state=12)


DISTANCE_METRICS = {
    "Euclidean (L2)": {"metric": "minkowski", "p": 2},
    "Manhattan (L1)": {"metric": "manhattan"},
    "Chebyshev (L∞)": {"metric": "chebyshev"},
    "Minkowski (p = 3)": {"metric": "minkowski", "p": 3},
}

CLASS_COLORS = ["#E69F00", "#56B4E9"]
BACKGROUND_COLORS = ["#FFF4D6", "#E7F5FC"]


def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    model: KNeighborsClassifier,
) -> plt.Figure:
    """Plot the training points and the model's predicted regions."""

    x_padding = 0.45
    y_padding = 0.45
    x_min, x_max = X[:, 0].min() - x_padding, X[:, 0].max() + x_padding
    y_min, y_max = X[:, 1].min() - y_padding, X[:, 1].max() + y_padding

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
        class_points = y == class_value
        ax.scatter(
            X[class_points, 0],
            X[class_points, 1],
            s=48,
            facecolors="white",
            edgecolors=color,
            linewidths=1.8,
            label=f"Class {class_value}",
            zorder=3,
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

X, y = make_demo_data()

with st.sidebar:
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

weights = "distance" if weighted else "uniform"
model = KNeighborsClassifier(
    n_neighbors=k,
    weights=weights,
    **DISTANCE_METRICS[distance_name],
)
model.fit(X, y)

plot_column, summary_column = st.columns([3.2, 1])

with plot_column:
    figure = plot_decision_boundary(X, y, model)
    st.pyplot(figure, clear_figure=True, use_container_width=True)

with summary_column:
    st.subheader("Current model")
    st.metric("k", k)
    st.metric("Training accuracy", f"{model.score(X, y):.1%}")
    st.write(f"**Voting:** {'distance-weighted' if weighted else 'uniform'}")
    st.write(f"**Distance:** {distance_name}")
    st.info(
        "The colored background shows the class predicted at each location. "
        "The dark curve is the approximate decision boundary."
    )

st.caption(
    "This demonstration uses a fixed two-dimensional synthetic data set so that "
    "the effect of each control is easy to see."
)
